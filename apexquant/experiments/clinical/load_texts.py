"""Raw documents for every corpus the study reads."""

from __future__ import annotations

from itertools import islice

from datasets import load_dataset

from .settings import PUBMED, PUBMED_REVISION, PUBMED_SHARDS, PUBMED_SPLITS


def load_texts(source: str, split: str, limit: int) -> list[str]:
    """Up to ``limit`` non-empty documents from ``source``.

    PubMed is split by PMID modulo ten into disjoint oracle, dev and eval parts,
    so no abstract used to calibrate or size an update is ever scored.
    """
    if source == "pubmed":
        keep = PUBMED_SPLITS[split]
        rows = load_dataset(PUBMED, revision=PUBMED_REVISION, data_files=PUBMED_SHARDS,
                            split="train", streaming=True)
        docs = (r["content"] for r in rows if int(r["PMID"]) % 10 in keep)
    elif source == "c4":
        rows = load_dataset("allenai/c4", data_files="en/c4-validation.0000*-of-00008.json.gz",
                            split="train", streaming=True)
        docs = (r["text"] for r in rows)
    elif source == "wikitext":
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return ["".join(rows["text"])]
    elif source == "code":
        rows = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
        docs = (r["content"] for r in rows)
    elif source == "danish":
        rows = load_dataset("wikimedia/wikipedia", "20231101.da", split="train", streaming=True)
        docs = (r["text"] for r in rows)
    else:
        raise ValueError(f"unknown source {source!r}")
    texts = [t for t in islice((d for d in docs if d.strip()), limit)]
    if len(texts) < limit:
        raise RuntimeError(f"{source}/{split}: {len(texts)} documents, wanted {limit}")
    return texts
