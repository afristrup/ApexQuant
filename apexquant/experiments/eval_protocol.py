"""The QuIP#/QTIP/YAQA evaluation protocol, reimplemented.

The vendored scripts in `eval/` cannot run our artifacts: they load checkpoints
through the upstream `lib/`, which expects that project's `QuantizedLinear`
format. They are also GPL-3.0, while this repository is MIT (see
`eval/README.md`). So rather than adapt their code, this module reimplements the
*protocol* from scratch, which is what comparability actually requires: the same
data, the same tokenisation, the same sequence length and the same reduction.

Protocol, matched to upstream:

* **wikitext2** -- `wikitext/wikitext-2-raw-v1`, split `test`, the whole corpus
  joined with `"\\n\\n"` and tokenised in one call, then cut into
  `numel // seqlen` non-overlapping chunks with the trailing remainder dropped.
* **c4** -- `allenai/c4`, validation shard
  `en/c4-validation.00000-of-00008.json.gz`: 256 documents drawn at random
  (fixed seed 0, as upstream hardcodes for the validation split), each
  contributing one random `seqlen`-length window, concatenated.
* **seqlen 8192**, which is the upstream default and the length their published
  Llama tables use. This differs from the 2048 used elsewhere in this
  repository: perplexity falls with context length, so numbers at the two
  lengths are NOT comparable and must never be placed in one table.
* **perplexity** -- token-weighted mean cross-entropy, exponentiated. Upstream
  averages per-chunk means instead; with equal-length chunks and the remainder
  dropped the two agree exactly.
* **KL** -- `KLDivLoss(reduction='batchmean')` between `log_softmax` of the
  quantised logits and `softmax` of the reference logits, which is YAQA's
  headline metric and needs both models resident at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_SEQLEN = 8192


def get_test_tokens(dataset: str, tokenizer, seqlen: int = DEFAULT_SEQLEN):
    """Tokenised evaluation corpus, cut into equal chunks.

    Args:
        dataset: ``"wikitext2"`` or ``"c4"``.
        tokenizer: The model's tokenizer.
        seqlen: Chunk length.

    Returns:
        Long tensor ``(nsamples, seqlen)``.
    """
    import random

    from datasets import load_dataset

    if dataset == "wikitext2":
        # Canonical namespaced id; `wikitext` alone no longer resolves under
        # datasets>=4. Same corpus, same revision content.
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                            split="test")
        ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
        n = ids.numel() // seqlen
        if n == 0:
            raise ValueError(f"corpus shorter than one {seqlen}-token chunk")
        return ids[0, : n * seqlen].view(n, seqlen)

    if dataset == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        rng = random.Random(0)
        chunks = []
        while len(chunks) < 256:
            row = data[rng.randint(0, len(data) - 1)]["text"]
            enc = tokenizer(row, return_tensors="pt").input_ids
            if enc.shape[1] < seqlen:
                continue
            i = rng.randint(0, enc.shape[1] - seqlen - 1)
            chunks.append(enc[0, i:i + seqlen])
        return torch.stack(chunks)

    raise ValueError(f"unknown dataset {dataset!r}")


@torch.no_grad()
def perplexity(model, tokens: torch.Tensor, device="cuda",
               score_from: int | None = None, chunk_tokens: int = 2048) -> float:
    """Token-weighted perplexity over pre-chunked tokens.

    ``score_from`` selects which positions inside each window are scored, and it
    is the single largest protocol difference between this harness and the GGUF
    community's numbers. Default ``None`` scores every position, which is what
    the QuIP#/QTIP tables this repository is otherwise matched to do.
    ``llama-perplexity`` instead scores only the second half of each window, so
    that every scored token has at least ``n_ctx/2`` tokens of context rather
    than a handful; pass ``score_from=tokens.shape[1] // 2`` for that.

    The difference is not small and it is not a wash. Measured on TinyLlama over
    128 windows of 512, scoring all positions gives $11.01$ and scoring the last
    half gives $9.03$, an 18% reduction, because the early positions in a window
    are predicted from almost no context and carry most of the loss. Two numbers
    produced under different settings of this argument must never be compared.
    """
    model.eval()
    start = 1 if not score_from else max(int(score_from), 1)
    nll, ntok = 0.0, 0
    for i in range(tokens.shape[0]):
        chunk = tokens[i].unsqueeze(0).to(device)
        logits = model(chunk, use_cache=False).logits[0]
        target = chunk[0, 1:]
        # Summed over slices of the window rather than the whole of it. Written
        # flat, `.float()` and `log_softmax` each materialise a tensor the size
        # of the logits, which at Llama-3.1's 128k vocabulary is 3.9 GiB apiece
        # at seqlen 8192 against 1.0 at 2048. `kl_divergence_stats` below already
        # chunks for this reason; these two did not, and a 70B scored at 8192
        # returned perplexity in the tens of thousands, varying between runs on
        # identical inputs. The chunking is necessary and is NOT the whole
        # fault: with it, a trellis artifact at P=3 returns 3.85 at 8192 against
        # 4.21 at 2048 on the same 16384 tokens, the right direction and size,
        # but a scalar artifact at the same length still returns 21389 at P=1 and
        # 17127 at P=2, where four stored bits must be near lossless and 7B
        # models give 5.6. Something about seqlen 8192 remains unexplained, so
        # score at 2048 until it is. Position s predicts token s+1, so the slices of the
        # logits and of the targets share an offset and the result is unchanged.
        parts = []
        for t0 in range(0, logits.shape[0] - 1, chunk_tokens):
            sl = logits[t0:t0 + chunk_tokens].float().log_softmax(dim=-1)
            tg = target[t0:t0 + chunk_tokens]
            parts.append(-sl.gather(-1, tg.unsqueeze(-1)).squeeze(-1))
            del sl
        tokwise = torch.cat(parts)
        sel = tokwise[start - 1:]
        nll += float(sel.sum())
        ntok += sel.numel()
        del logits, parts, tokwise
    return float(torch.tensor(nll / ntok).exp())


@torch.no_grad()
def kl_to_reference(model, reference, tokens: torch.Tensor,
                    device="cuda", chunk_tokens: int = 2048) -> float:
    """Mean batchmean KL from a reference model's next-token distribution.

    Both models must be resident simultaneously, which for a 70B pair means well
    beyond a single 80 GB card.
    """
    model.eval()
    reference.eval()
    total = 0.0
    for i in range(tokens.shape[0]):
        chunk = tokens[i].unsqueeze(0).to(device)
        q = model(chunk, use_cache=False).logits.reshape(-1, model.config.vocab_size)
        p = reference(chunk, use_cache=False).logits.reshape(-1, q.shape[-1])
        # Chunked for the same reason as `perplexity`: holding both models'
        # logits plus a log_softmax and a softmax of each is four vocabulary-
        # sized fp32 tensors at once, 15.7 GiB at seqlen 8192 on a 128k
        # vocabulary. batchmean divides by the number of rows, so summing the
        # per-row divergence and dividing once at the end is the same number.
        rows, acc = q.shape[0], 0.0
        for t0 in range(0, rows, chunk_tokens):
            lq = q[t0:t0 + chunk_tokens].float().log_softmax(dim=-1)
            lp = p[t0:t0 + chunk_tokens].float().log_softmax(dim=-1)
            acc += float((lp.exp() * (lp - lq)).sum())
            del lq, lp
        total += acc / rows
        del q, p
    return total / tokens.shape[0]


def kl_divergence_stats(model, reference, tokens: torch.Tensor, device="cuda",
                        percentiles=(99.0, 99.9), chunk_tokens: int = 4096,
                        score_from: int | None = None) -> dict:
    """Per-token KL statistics: mean, tail percentiles and max.

    ``kl_to_reference`` returns the mean, which is what YAQA reports. The GGUF
    community reports something different and complementary: llama.cpp's
    ``--kl-divergence`` prints a ``99.9%`` percentile alongside the mean, and
    Unsloth's Qwen3.5 tables lead with it. The two answer different questions. The
    mean says how far the quantized model sits from the original on a typical
    token; the tail says how badly it fails on its worst ones, which is the
    statistic that moves when a quantizer breaks a small number of important
    predictions rather than blurring all of them.

    Reporting only the mean hides exactly the failure our own construction is
    least able to see. The trace identity says our objective controls the
    magnitude of the weight error and is indifferent to its direction, and a
    direction-sensitive failure shows up in the tail long before it moves the
    mean.

    One protocol note that decides comparability. The published GGUF numbers are
    measured on 512-token windows; ours default to 8192. Perplexity and KL both
    move with context length, so pass ``seqlen=512`` to :func:`get_test_tokens`
    when the intent is to sit in the same table as those numbers, and never mix
    the two lengths in one comparison.

    Args:
        model: The quantized model.
        reference: The unquantized model. Both must be resident.
        tokens: ``(nsamples, seqlen)`` from :func:`get_test_tokens`.
        device: Where to run.
        percentiles: Tail percentiles to report, as percents.
        score_from: First position scored inside each window, as in
            :func:`perplexity`. The GGUF numbers window their KL the same way
            they window perplexity, so a comparison against them has to set this
            too; early positions carry both more loss and more divergence.
        chunk_tokens: Positions per softmax chunk, to bound the transient. At a
            248k vocab a full 8192-position logit pair is 16 GiB in float32.

    Returns:
        ``{"mean": float, "p99.0": float, "p99.9": float, "max": float,
        "tokens": int}``, all in nats.
    """
    model.eval()
    reference.eval()
    kept = []
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(tokens.shape[0]):
            chunk = tokens[i].unsqueeze(0).to(device)
            q = model(chunk, use_cache=False).logits.squeeze(0)
            p = reference(chunk, use_cache=False).logits.squeeze(0)
            if score_from:
                q = q[int(score_from):]
                p = p[int(score_from):]
            for s in range(0, q.shape[0], chunk_tokens):
                lq = q[s:s + chunk_tokens].float().log_softmax(dim=-1)
                lp = p[s:s + chunk_tokens].float().log_softmax(dim=-1)
                kl = (lp.exp() * (lp - lq)).sum(dim=-1)
                total += float(kl.sum())
                count += kl.numel()
                kept.append(kl.to("cpu", torch.float32))
                del lq, lp, kl
            del q, p
    allkl = torch.cat(kept)
    out = {"mean": total / max(count, 1), "max": float(allkl.max()), "tokens": count}
    for pc in percentiles:
        out[f"p{pc}"] = float(torch.quantile(allkl.double(), pc / 100.0))
    return out
