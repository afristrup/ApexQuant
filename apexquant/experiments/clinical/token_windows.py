"""Fixed-length token windows drawn from a document list."""

from __future__ import annotations

import random

import torch


def token_windows(tokenizer, texts: list[str], count: int, seq_len: int,
                  seed: int) -> torch.Tensor:
    """``count`` windows of ``seq_len`` tokens, BOS first, documents shuffled by ``seed``.

    Documents are joined by blank lines and cut into consecutive windows, so a
    window can span documents, as in the usual calibration recipe. A tokenizer
    without a BOS token (Qwen3.5) gets plain windows, which is how it is trained.
    """
    order = list(texts)
    random.Random(seed).shuffle(order)
    bos_id = tokenizer.bos_token_id
    body = seq_len - (bos_id is not None)
    ids: list[int] = []
    for text in order:
        ids.extend(tokenizer(text + "\n\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= count * body:
            break
    if len(ids) < count * body:
        raise RuntimeError(f"{len(ids)} tokens, need {count * body}")
    grid = torch.tensor(ids[:count * body]).view(count, body)
    if bos_id is None:
        return grid
    return torch.cat([torch.full((count, 1), bos_id), grid], dim=1)
