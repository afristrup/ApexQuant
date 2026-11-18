"""GPTQ and AWQ through llm-compressor, calibrated on the given windows."""

from __future__ import annotations

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM


def quantize_llmcompressor(path: str, method: str, bits: int, group: int,
                           windows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every parameter of the calibrated model, with weights baked to their grid.

    Parameters are read through ``align_module_device`` because llm-compressor
    keeps them behind offload hooks; writing ``.data`` in place does not stick.

    GPTQ is symmetric at three and four bits and asymmetric at two, where a
    symmetric grid wastes a level. AWQ rescales the norms ahead of each linear
    as well as the linears, so the whole state dict is returned.
    """
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.utils import align_module_device
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    args = QuantizationArgs(num_bits=bits, type="int", symmetric=method == "gptq" and bits >= 3,
                            strategy="group", group_size=group)
    groups = {"group_0": QuantizationScheme(targets=["Linear"], weights=args)}
    if method == "gptq":
        recipe = [GPTQModifier(config_groups=groups, ignore=["lm_head"])]
    elif method == "awq":
        recipe = [AWQModifier(), QuantizationModifier(config_groups=groups, ignore=["lm_head"])]
    else:
        raise ValueError(f"unknown method {method!r}")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    data = Dataset.from_dict({"input_ids": windows.tolist(),
                              "attention_mask": torch.ones_like(windows).tolist()})
    oneshot(model=model, dataset=data, recipe=recipe, max_seq_length=windows.shape[1],
            num_calibration_samples=windows.shape[0])
    with torch.device("meta"):
        names = {n for n, _ in AutoModelForCausalLM.from_config(model.config).named_parameters()}
    out, baked = {}, 0
    with torch.no_grad():
        for prefix, mod in model.named_modules():
            with align_module_device(mod):
                for leaf, p in mod.named_parameters(recurse=False):
                    name = f"{prefix}.{leaf}" if prefix else leaf
                    if name not in names:
                        continue
                    value = p.data
                    scheme = getattr(mod, "quantization_scheme", None)
                    if leaf == "weight" and scheme is not None and scheme.weights is not None:
                        value = fake_quantize(value, mod.weight_scale, mod.weight_zero_point,
                                              scheme.weights)
                        baked += 1
                    out[name] = value.to("cpu", copy=True)
    if not baked:
        raise RuntimeError("llm-compressor attached no weight quantization")
    return out
