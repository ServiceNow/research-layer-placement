"""Token sequence utilities for comparing and analyzing token sequences."""

from collections.abc import Hashable, Sequence


def find_differing_ranges[T: Hashable](
    sequences: Sequence[Sequence[T]],
) -> list[tuple[int, int]]:
    """Find (start, end) range of differing elements for each sequence.

    Given multiple sequences, identifies the subspan where they differ by finding:
    - Common prefix: elements identical at the start of ALL sequences
    - Common suffix: elements identical at the END of all sequences

    The differing range is everything between prefix and suffix. This is useful
    for multiple-choice scoring (find answer tokens), diff algorithms, etc.

    Works with any hashable element type (ints, strings, None, tuples, etc.).

    Args:
        sequences: List of sequences to compare. Elements must be hashable.

    Returns:
        List of (start, end) tuples, one per sequence. Differing range is [start, end).

    Examples:
        >>> # Single differing element at same position
        >>> seqs = [[1, 2, 3, 10, 5], [1, 2, 3, 11, 5], [1, 2, 3, 12, 5]]
        >>> find_differing_ranges(seqs)
        [(3, 4), (3, 4), (3, 4)]

        >>> # Variable length differences
        >>> seqs = [[1, 2, 10, 5], [1, 2, 11, 12, 5]]
        >>> find_differing_ranges(seqs)
        [(2, 3), (2, 4)]

        >>> # All identical (edge case) - empty ranges
        >>> seqs = [[1, 2, 3], [1, 2, 3]]
        >>> find_differing_ranges(seqs)
        [(3, 3), (3, 3)]

        >>> # Works with any hashable type
        >>> seqs = [['a', 'b', 'X'], ['a', 'b', 'Y']]
        >>> find_differing_ranges(seqs)
        [(2, 3), (2, 3)]
    """
    if not sequences:
        return []

    # Helper to safely get element at position
    def get_elem(seq: Sequence[T], pos: int) -> T | None:
        if pos < len(seq):
            return seq[pos]
        return None

    # Find longest common prefix (elements identical at start of ALL sequences)
    min_len = min(len(s) for s in sequences)
    prefix_len = 0
    for i in range(min_len):
        elems_at_i = set(get_elem(s, i) for s in sequences)
        # All same - extend prefix
        if len(elems_at_i) == 1:
            prefix_len = i + 1
        else:
            break

    # Find longest common suffix (elements identical at END of all sequences)
    # Don't overlap with prefix
    suffix_len = 0
    max_suffix_check = min_len - prefix_len
    for i in range(1, max_suffix_check + 1):
        elems_at_neg_i = set(s[-i] for s in sequences)
        # All same
        if len(elems_at_neg_i) == 1:
            suffix_len = i
        else:
            break

    # For each sequence, differing elements are [prefix_len, len - suffix_len)
    ranges: list[tuple[int, int]] = []
    for s in sequences:
        start = prefix_len
        end = len(s) - suffix_len
        # Ensure valid range
        if end < start:
            end = start
        ranges.append((start, end))

    return ranges
