"""Compile ``artifacts/ngram_reference.json`` into a binary a Worker can actually hold.

The reference is 18.6 MB of JSON: 16,818 unigrams, 263,395 bigrams, 512,992 trigrams, with
every key a whitespace-joined string. Parsed into JavaScript objects that is well past a
Worker's 128 MB isolate, and most of the weight is string keys we do not need at runtime --
lookups only ever ask "what is the count of this (a, b, c)".

So: intern the vocabulary once (16,819 tokens fits comfortably in a uint16 id), turn every
n-gram key into integer ids, sort, and binary-search at query time. ~6 MB of typed arrays,
about 2 MB over the wire compressed, and roughly 8 MB resident.

**This is a re-encoding, not an approximation.** No pruning, no hashing, no Bloom filter --
every count survives exactly, because ``novel_trigram_rate`` is a membership test whose
false positives would silently lower a feature the classifier has a weight for, and the
calibration downstream was fitted against exact values. ``verify()`` reloads the binary and
checks all 793,205 entries round-trip before the file is written.

    python edge/scripts/build_ngram_bin.py

Layout (little-endian throughout):

    magic      8   b"PALNGRM1"
    header    28   7 x uint32: nVocab, nUni, nBi, nTri, totalTokens, vocabSize, nDocuments
    vocabLen   4   uint32, byte length of the vocab blob
    vocab      -   "\n"-joined utf-8 tokens, in id order
    (pad to 4)
    uniCount   -   uint32[nVocab]              count of each token, 0 if absent
    biKey      -   uint32[nBi]                 (a << 16) | b, ascending
    biCount    -   uint32[nBi]
    triHi      -   uint32[nTri]                a           } sorted by (hi, lo)
    triLo      -   uint32[nTri]                (b << 16)|c }
    triCount   -   uint32[nTri]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "ngram_reference.json"
OUT = ROOT / "edge" / "assets" / "ngram.bin"

MAGIC = b"PALNGRM1"
BOUNDARY = "<s>"


def build() -> bytes:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    unigrams: dict[str, int] = payload["unigrams"]
    bigrams: dict[str, int] = payload["bigrams"]
    trigrams: dict[str, int] = payload["trigrams"]

    # Id 0 is reserved for the boundary token so the JS side never has to look it up by
    # string. Everything else keeps the JSON's own key order, which is stable.
    vocab: list[str] = [BOUNDARY]
    index: dict[str, int] = {BOUNDARY: 0}
    for word in unigrams:
        if word not in index:
            index[word] = len(vocab)
            vocab.append(word)

    for key in list(bigrams) + list(trigrams):
        for part in key.split(" "):
            if part not in index:
                # Should only ever be <s>, which is already interned. Anything else means
                # the reference was written by a fitter this compiler does not understand.
                raise SystemExit(f"unknown token {part!r} in an n-gram key")

    if len(vocab) > 0xFFFF:
        raise SystemExit(f"vocabulary of {len(vocab)} exceeds the uint16 id space")

    uni_counts = [0] * len(vocab)
    for word, count in unigrams.items():
        uni_counts[index[word]] = count

    bi: list[tuple[int, int]] = []
    for key, count in bigrams.items():
        a, b = key.split(" ")
        bi.append(((index[a] << 16) | index[b], count))
    bi.sort(key=lambda kv: kv[0])

    tri: list[tuple[int, int, int]] = []
    for key, count in trigrams.items():
        a, b, c = key.split(" ")
        tri.append((index[a], (index[b] << 16) | index[c], count))
    tri.sort(key=lambda kv: (kv[0], kv[1]))

    blob = "\n".join(vocab).encode("utf-8")
    parts: list[bytes] = [
        MAGIC,
        struct.pack(
            "<7I",
            len(vocab),
            len(unigrams),
            len(bi),
            len(tri),
            payload["total_tokens"],
            payload["vocab_size"],
            payload["n_documents"],
        ),
        struct.pack("<I", len(blob)),
        blob,
        b"\0" * ((-len(blob)) % 4),  # keep the typed arrays 4-byte aligned
        struct.pack(f"<{len(uni_counts)}I", *uni_counts),
        struct.pack(f"<{len(bi)}I", *[k for k, _ in bi]),
        struct.pack(f"<{len(bi)}I", *[c for _, c in bi]),
        struct.pack(f"<{len(tri)}I", *[h for h, _, _ in tri]),
        struct.pack(f"<{len(tri)}I", *[lo for _, lo, _ in tri]),
        struct.pack(f"<{len(tri)}I", *[c for _, _, c in tri]),
    ]
    return b"".join(parts)


def verify(data: bytes) -> None:
    """Reload the binary and check every entry round-trips. A silent lossy encoding here
    would show up downstream as a detector that is subtly miscalibrated and looks fine."""
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    off = len(MAGIC)
    n_vocab, n_uni, n_bi, n_tri, total, vsize, ndocs = struct.unpack_from("<7I", data, off)
    off += 28
    (blob_len,) = struct.unpack_from("<I", data, off)
    off += 4
    vocab = data[off : off + blob_len].decode("utf-8").split("\n")
    off += blob_len + ((-blob_len) % 4)

    def take(n: int) -> tuple[int, ...]:
        nonlocal off
        out = struct.unpack_from(f"<{n}I", data, off)
        off += 4 * n
        return out

    uni = take(n_vocab)
    bi_key, bi_count = take(n_bi), take(n_bi)
    tri_hi, tri_lo, tri_count = take(n_tri), take(n_tri), take(n_tri)

    assert len(vocab) == n_vocab, "vocab blob length disagrees with the header"
    assert (total, vsize, ndocs) == (
        payload["total_tokens"],
        payload["vocab_size"],
        payload["n_documents"],
    )
    assert n_uni == len(payload["unigrams"])

    got_uni = {vocab[i]: uni[i] for i in range(n_vocab) if uni[i]}
    assert got_uni == payload["unigrams"], "unigram counts do not round-trip"

    got_bi = {
        f"{vocab[bi_key[i] >> 16]} {vocab[bi_key[i] & 0xFFFF]}": bi_count[i] for i in range(n_bi)
    }
    assert got_bi == payload["bigrams"], "bigram counts do not round-trip"

    got_tri = {
        f"{vocab[tri_hi[i]]} {vocab[tri_lo[i] >> 16]} {vocab[tri_lo[i] & 0xFFFF]}": tri_count[i]
        for i in range(n_tri)
    }
    assert got_tri == payload["trigrams"], "trigram counts do not round-trip"

    assert all(bi_key[i] <= bi_key[i + 1] for i in range(n_bi - 1)), "bigram keys not sorted"
    assert all(
        (tri_hi[i], tri_lo[i]) <= (tri_hi[i + 1], tri_lo[i + 1]) for i in range(n_tri - 1)
    ), "trigram keys not sorted"

    print(f"verified {n_uni} unigrams, {n_bi} bigrams, {n_tri} trigrams round-trip exactly")


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    data = build()
    verify(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"{OUT.relative_to(ROOT)}  {len(data) / 1e6:.2f} MB  (from {SRC.stat().st_size / 1e6:.2f} MB JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
