"""Stable character n-gram hashing (fastText-style) for the tiny encoder.

Raw byte embeddings are cheap but weak on language meaning: every byte is an
island. Hashed character n-grams give the tiny model subword features --
"amber" and "AMBER!" hash to the same buckets -- while staying small enough
to train on a laptop CPU.
"""

from __future__ import annotations

import zlib

PAD_ID = 0  # reserved; hashed features live in [1, buckets]


def char_ngrams(text: str, ngram_range: tuple[int, int] = (3, 6)) -> list[str]:
    """Lower-cased, boundary-marked character n-grams of ``text``."""
    marked = f"<{text.strip().lower()}>"
    low, high = ngram_range
    grams: list[str] = []
    for n in range(low, high + 1):
        if len(marked) < n:
            continue
        grams.extend(marked[i : i + n] for i in range(len(marked) - n + 1))
    return grams


def ngram_ids(
    text: str,
    buckets: int,
    ngram_range: tuple[int, int] = (3, 6),
    max_features: int = 512,
) -> list[int]:
    """Stable hashed feature ids for ``text``; ``PAD_ID`` never appears."""
    ids = [
        zlib.crc32(gram.encode("utf-8")) % (buckets - 1) + 1
        for gram in char_ngrams(text, ngram_range)
    ]
    if len(ids) > max_features:
        # Evenly spaced subsample so long contexts keep head, middle and tail.
        step = len(ids) / max_features
        ids = [ids[int(i * step)] for i in range(max_features)]
    return ids or [PAD_ID + 1]
