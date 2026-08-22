"""
NA head glycosylation feature utilities.
"""

from __future__ import annotations

from typing import Iterable, Set


def scan_n_linked_glycosylation(seq: str) -> set[int]:
    """
    Return zero-based motif start positions for N-X-S/T motifs where X != P.

    Positions are reported in the input sequence coordinate system. Gaps and
    ambiguous residues do not match the motif.
    """

    seq = (seq or "").upper()
    motifs: set[int] = set()
    for idx in range(max(len(seq) - 2, 0)):
        first, second, third = seq[idx], seq[idx + 1], seq[idx + 2]
        if first == "N" and second not in {"P", "-"} and third in {"S", "T"}:
            motifs.add(idx)
    return motifs


def na_head_glycan_jaccard(
    seq_a: str,
    seq_b: str,
    head_positions: Iterable[int],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Jaccard distance between NA head glycosylation motif sets.
    """

    head: Set[int] = set(head_positions)
    glycans_a = scan_n_linked_glycosylation(seq_a) & head
    glycans_b = scan_n_linked_glycosylation(seq_b) & head
    union = glycans_a | glycans_b
    if not union:
        return 0.0
    symmetric_difference = glycans_a ^ glycans_b
    return len(symmetric_difference) / (len(union) + epsilon)


def na_head_glycan_mismatch(
    seq_a: str,
    seq_b: str,
    head_positions: Iterable[int] | None = None,
) -> float:
    """
    Return 1.0 when NA head N-linked glycan motif sets differ, else 0.0.
    """

    glycans_a = scan_n_linked_glycosylation(seq_a)
    glycans_b = scan_n_linked_glycosylation(seq_b)
    if head_positions is not None:
        head: Set[int] = set(head_positions)
        glycans_a = glycans_a & head
        glycans_b = glycans_b & head
    return 0.0 if glycans_a == glycans_b else 1.0
