# clay

Clay v1.5 foundation model encoder, de-Lightning'd and Triton-fused: the frozen
Earth-observation encoder `apexquant`'s Table 4/5 evaluates (see the sibling
`../apexquant/` package).

This is a straight torch port of the [Clay Foundation
Model](https://clay-foundation.github.io/) network. The pieces that made the
original a Lightning project — the module wrapper, the datamodule, the
callbacks, the masked-autoencoder training objective with its frozen DINOv2
teacher — are not here, because this package exists to define the model and
run it forward, not to train it. Every op picks a fused Triton kernel or a
torch reference through `clay.ops.use_triton`, which never imports `triton`
unless it's installed, so the whole encoder also runs on CPU.

## Layout

- `src/clay/model/` — the network, one module per file (`encoder.py`,
  `decoder.py`, `dynamic_embedding.py`, `waves_transformer.py`, ...).
- `src/clay/ops/` — each op (attention, layer norm, linear+GELU, sin/cos
  positional encodings), choosing its Triton kernel or torch reference.
- `src/clay/kernels/` — the Triton kernels themselves.
- `src/clay/inference/` — loading published weights and producing embeddings.
- `configs/metadata.yaml` — per-sensor band wavelengths and GSD (Apache-2.0,
  copied from the upstream project), needed by `load_sensor_metadata`.
- `tests/` — parity tests against the original Lightning implementation.

## Quick start

```python
import torch
from clay import build_encoder, embed, load_checkpoint_state_dict, load_sensor_metadata

encoder = build_encoder(size="large")  # Clay v1.5's architecture: dim=1024, depth=24
load_checkpoint_state_dict(encoder, "path/to/clay-v1.5.ckpt")
encoder.eval()

waves = load_sensor_metadata("sentinel-2-l1c")
datacube = {
    "pixels": torch.randn(1, len(waves), 224, 224),
    "time": torch.zeros(1, 4),
    "latlon": torch.zeros(1, 4),
    "gsd": 10.0,
    "waves": waves,
}
tokens = embed(encoder, datacube)  # (1, 1 + num_patches, dim); token 0 is the class token
```

`apexquant`'s `experiments/progressive_trellis_clay.py` builds on exactly this
entry point: quantize the encoder's linears in place (swap in
`ProgressiveTrellisLinear`), then call `embed` and read what the model
actually produces at each nested tier.

## Sizes

Four published architectures (`src/clay/model/sizes.py`); Clay v1.5 is `large`:

| size | encoder dim | depth | heads |
|---|---:|---:|---:|
| tiny | 192 | 6 | 4 |
| small | 384 | 6 | 6 |
| base | 768 | 12 | 12 |
| large (Clay v1.5) | 1024 | 24 | 16 |

The decoder stays small at every size — it exists only to make the
masked-autoencoder pretraining objective well posed and is discarded at
inference, which is why `embed()` rejects an encoder still configured for
masking (`mask_ratio` must be 0, `shuffle` off).

## Testing

```bash
uv sync --all-extras
uv run pytest
```

Most of the suite (sin/cos parity, Triton-kernel-vs-torch-reference parity,
inference shape checks) runs standalone. Four tests —
`test_module_parity.py`, `test_network_parity.py`, `test_cuda_parity.py`,
`test_sincos_parity.py`'s CUDA arm — compare this port against the original
vendored Lightning implementation as an oracle; that vendor tree isn't part of
this package (it's a separate, large, upstream-licensed checkout), so those
tests skip with a message pointing at where it would need to live rather than
failing on a missing import. `test_real_weights.py` skips unless a published
Clay v1.5 checkpoint is already in the local Hugging Face cache — it's the one
test that runs the port against real weights end to end when it can.

## Relationship to `apexquant`

This package is `apexquant`'s sibling, not a published dependency: the Clay
experiments (`progressive_trellis_clay.py`, `mixed_depth_trellis_clay.py`,
`trellis_clay_speed_memory.py`, `allocation_probe_clay.py`) import it via
`apexquant`'s `pyproject.toml` `[tool.uv.sources]` entry, which points at
`../clay-model` as an editable local path. Nothing under `src/apexquant`
itself imports `clay` — only the experiment scripts that need a real encoder do.
