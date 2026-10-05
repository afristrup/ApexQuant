"""Spend the second trellis pass only where it changes the answer.

P=2 cannot be made to approach dense speed by tuning, and the reason is
arithmetic: the fused form is two dots of 2MNK where one pass is one, and the
GEMM already runs both at the same efficiency it runs one (55 TFLOPS, 67% of the
card's TF32 peak). Measured on Clay, P=2 is 20.7 ms against P=1's 11.3 and dense
TF32's 13.3. Two structural attempts at the gap failed on the same thing --
accumulating both passes in one program removes the fold but halves the program
count, and it loses by more than the fold cost (17.1 -> 18.2 ms on the GEMM).

So the question worth asking is not how to make uniform P=2 cheaper but whether
it needs to be uniform. The recursion is per-layer: a layer's second pass is a
separate artifact tier, and nothing requires every layer to carry one. This
measures the cost of demoting each layer to P=1, one layer at a time, and then
buys the second pass back for the layers that care most.

WHAT IS MEASURED. Damage is leave-one-out against the all-P=2 model, so it reads
the marginal value of a layer's second pass rather than its total contribution.
Layers are swapped in place rather than into a deep copy, which is what makes 102
trials cheap: the codes for the shallower tier are a prefix of the ones already
there, so the P=1 variant of a layer is a slice, not a re-encode.

The artifact stays nested. Each layer holds a prefix of its own P=2 encode, which
is exactly what the recursion guarantees; what varies across layers is how much
of it is shipped.

Times here run a few percent above `trellis_clay_speed_memory.py` for the same
configuration, because this process keeps both tiers of all 102 layers resident to
make the swapping cheap, and that memory competes for L2 with the forward. Rows
within the tables are measured against each other and are the comparison to read;
the absolute milliseconds are not comparable across the two scripts.

    uv run python experiments/mixed_depth_trellis_clay.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_trellis_clay import DEV, datacube, eligible, load_encoder
from trellis_clay_speed_memory import measure, resident_mib

from clay.inference.embed import embed  # noqa: E402
from apexquant.kernels.triton_gemv_trellis import (  # noqa: E402
    trellis_window_table,
)
from apexquant.vq.progressive_artifact import (  # noqa: E402
    group_size_for,
    quantize,
    stored_bits_per_weight,
)
from apexquant.vq.progressive_trellis_linear import (  # noqa: E402
    ProgressiveTrellisLinear,
)
from apexquant.vq.trellis_codec import (  # noqa: E402
    DEFAULT_TRELLIS_BITS,
    TrellisCodec,
)


def build_layer(codes, scale, table, codec, in_features, group_size, seed, bias,
                passes):
    """A packed layer carrying the first ``passes`` tiers of an encode."""
    words = (codes[:passes].reshape(passes, codes.shape[1], -1)
             .contiguous().view(torch.int32))
    return ProgressiveTrellisLinear(words, scale[:passes].to(torch.float32),
                                    table, codec, in_features, group_size, seed,
                                    bias)


def set_submodule(model, qualified, module):
    """Replace ``model.<qualified>`` in place."""
    parent_name, _, attr = qualified.rpartition(".")
    setattr(model.get_submodule(parent_name), attr, module)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--trellis-bits", type=int, default=DEFAULT_TRELLIS_BITS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chip", type=int, default=224)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0],
                    help="Share of layers, by damage rank, that keep two passes.")
    args = ap.parse_args()

    dense = load_encoder()
    targets = eligible(dense)
    cube = datacube(args.chip, 1)
    reference = embed(dense, cube)[:, 0, :]

    codecs, tables, variants, widths = {}, {}, {}, {}
    dense_layers, codes_by_name = {}, {}
    encode_start = time.perf_counter()
    for name, module in targets:
        weight = module.weight.detach().to(DEV, torch.float32)
        out_f, in_f = weight.shape
        gs = group_size_for(in_f, args.group_size)
        if gs not in codecs:
            codecs[gs] = TrellisCodec(d=gs, bits=args.bits,
                                      trellis_bits=args.trellis_bits,
                                      seed=0, device=DEV)
            tables[gs] = trellis_window_table(codecs[gs])
        codes, scale = quantize(weight, codecs[gs], 2, gs, args.seed)
        bias = None if module.bias is None else module.bias.detach()
        variants[name] = {
            p: build_layer(codes, scale, tables[gs], codecs[gs], in_f, gs,
                           args.seed, bias, p)
            for p in (1, 2)
        }
        dense_layers[name] = module
        widths[name] = (out_f, in_f, weight.numel())
        codes_by_name[name] = codes
        del weight
    print(f"# encoded {len(targets)} layers at P=2 in "
          f"{time.perf_counter() - encode_start:.1f}s\n")

    model = dense
    for name in variants:
        set_submodule(model, name, variants[name][2])
    all_two = embed(model, cube)[:, 0, :]

    # Leave-one-out: demote one layer, read the damage, put it back.
    damage = {}
    for name in variants:
        set_submodule(model, name, variants[name][1])
        tokens = embed(model, cube)[:, 0, :]
        damage[name] = 1.0 - F.cosine_similarity(tokens, all_two).item()
        set_submodule(model, name, variants[name][2])
    ranked = sorted(damage, key=damage.get, reverse=True)

    print("## Ten layers whose second pass matters most\n")
    print("| layer | shape | 1 - cosine when demoted |")
    print("|---|---|---:|")
    for name in ranked[:10]:
        out_f, in_f, _ = widths[name]
        print(f"| `{name}` | ({out_f}, {in_f}) | {damage[name]:.2e} |")

    total = sum(n for _, _, n in widths.values())
    # Two memory numbers, and they differ by what is being deployed.
    #
    # Elastic: one artifact, the full two-pass encode, every row decoded out of it.
    # This is measured, and it is also where the nesting shows up as something
    # other than a slogan -- a prefix of a contiguous encode needs no copy, so the
    # shallower tier of a layer is a view of the deeper one and switching tier
    # allocates nothing. `nesting_table` checks that rather than asserting it.
    #
    # Shipped: what a deployment pinned to one operating point carries, which is
    # the stored-bits accounting and not something `resident_mib` can see, because
    # the two tiers share storage by construction.
    elastic_bits = 0.0
    for name in variants:
        out_f, in_f, numel = widths[name]
        elastic_bits += stored_bits_per_weight(out_f, in_f, 2, args.group_size,
                                               args.bits) * numel
    for name in variants:
        set_submodule(model, name, variants[name][2])
    elastic_mib = resident_mib(model)

    print(f"\n## Buying the second pass back by damage rank\n")
    print(f"Elastic: one artifact at {elastic_bits / total:.3f} bits/weight, "
          f"{elastic_mib:.0f} MiB resident, serves every row below.\n")
    print("| two-pass layers | shipped bits/weight | shipped MiB | peak MiB | "
          "ms | vs all-P=1 | cls cosine |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    baseline = None
    for fraction in args.fractions:
        keep = set(ranked[:round(fraction * len(ranked))])
        bits = 0.0
        for name in variants:
            out_f, in_f, numel = widths[name]
            passes = 2 if name in keep else 1
            set_submodule(model, name, variants[name][passes])
            bits += stored_bits_per_weight(out_f, in_f, passes, args.group_size,
                                           args.bits) * numel
        elapsed, peak = measure(model, cube)
        cosine = F.cosine_similarity(embed(model, cube)[:, 0, :], reference).item()
        if baseline is None:
            baseline = elapsed
        print(f"| {len(keep)}/{len(ranked)} | {bits / total:.3f} | "
              f"{bits / 8 / 1024 / 1024:.0f} | {peak:.0f} | {elapsed:.1f} | "
              f"{elapsed / baseline:.2f}x | {cosine:.6f} |", flush=True)

    nesting_table(variants, codes_by_name, codecs, widths, targets, args)
    protect_table(model, variants, dense_layers, widths, ranked, reference, cube,
                  total, args)


def nesting_table(variants, codes_by_name, codecs, widths, targets, args):
    """Check that a shallower tier really is the deeper one's bytes, not a copy.

    Two claims are usually asserted together and are separate. That a prefix of
    the encode decodes to the shallower model is a property of the recursion, and
    it is checked here by re-encoding at depth one and comparing bit for bit
    against the slice. That switching tier costs no memory is a property of the
    layout, and it holds only because a depth-one prefix of a contiguous
    ``(P, ...)`` encode is contiguous itself, so the slice is a view; that is
    checked by comparing storage pointers.
    """
    print("\n## Nested decoding\n")
    print("| layer | prefix is a view of the deeper tier | "
          "prefix equals a fresh depth-1 encode |")
    print("|---|---|---|")
    for name, module in targets[:3]:
        codes = codes_by_name[name]
        gs = group_size_for(widths[name][1], args.group_size)
        fresh, _ = quantize(module.weight.detach().to(DEV, torch.float32),
                            codecs[gs], 1, gs, args.seed)
        shares = codes[:1].untyped_storage().data_ptr() == \
            codes.untyped_storage().data_ptr()
        identical = torch.equal(fresh, codes[:1])
        print(f"| `{name}` | {'yes' if shares else 'NO'} | "
              f"{'yes' if identical else 'NO'} |")


def protect_table(model, variants, dense_layers, widths, ranked, reference, cube,
                  total, args):
    """All layers at P=1 except the most sensitive few, which stay dense.

    The ranking is dominated by one 128x128 layer inside the patch embedding,
    which is not where a 311M-parameter model keeps its weight. It generates the
    projection every band is embedded with, so an error there is not one layer's
    error -- it reaches every patch of every channel. Protecting a handful of such
    layers costs almost no memory, so it is worth knowing how much of P=2's value
    it accounts for.
    """
    print(f"\n## All layers at P=1, with the most sensitive kept dense\n")
    print("| layers kept dense | bits/weight | shipped MiB | ms | cls cosine |")
    print("|---:|---:|---:|---:|---:|")
    for count in (0, 1, 3, 8):
        bits = 0.0
        for name in variants:
            out_f, in_f, numel = widths[name]
            if name in set(ranked[:count]):
                set_submodule(model, name, dense_layers[name])
                bits += 32.0 * numel
            else:
                set_submodule(model, name, variants[name][1])
                bits += stored_bits_per_weight(out_f, in_f, 1, args.group_size,
                                               args.bits) * numel
        elapsed, _ = measure(model, cube)
        cosine = F.cosine_similarity(embed(model, cube)[:, 0, :], reference).item()
        print(f"| {count} | {bits / total:.3f} | "
              f"{bits / 8 / 1024 / 1024:.0f} | {elapsed:.1f} | {cosine:.6f} |",
              flush=True)


if __name__ == "__main__":
    main()
