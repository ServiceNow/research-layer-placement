"""Tests for token_utils, particularly find_differing_ranges."""

import pytest

from place_layers.core.token_utils import find_differing_ranges


class TestFindAnswerRanges:
    """Tests for find_differing_ranges function."""

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert find_differing_ranges([]) == []

    def test_single_sequence(self):
        """Single sequence - nothing to diff against, so all tokens are 'common'.

        In practice, this shouldn't happen (multiple-choice needs 2+ choices).
        The function returns an empty range since there's no difference to score.
        """
        seqs = [[1, 2, 3, 4, 5]]
        result = find_differing_ranges(seqs)
        # All tokens are "same" (only one sequence), prefix=5, suffix=0
        assert result == [(5, 5)]

    def test_identical_sequences(self):
        """All identical sequences - empty answer ranges."""
        seqs = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        result = find_differing_ranges(seqs)
        # All same, so prefix=3, suffix=0, range=[3,3) for each
        assert result == [(3, 3), (3, 3), (3, 3)]

    def test_single_token_answers_same_length(self):
        """Typical MMLU case: single-token answers A, B, C, D at same position."""
        # [context..., answer, end_token]
        seqs = [
            [1, 2, 3, 10, 5],  # answer=10
            [1, 2, 3, 11, 5],  # answer=11
            [1, 2, 3, 12, 5],  # answer=12
            [1, 2, 3, 13, 5],  # answer=13
        ]
        result = find_differing_ranges(seqs)
        # prefix=[1,2,3] (len=3), suffix=[5] (len=1)
        # answer range = [3, len-1) = [3, 4) for each
        assert result == [(3, 4), (3, 4), (3, 4), (3, 4)]

    def test_variable_length_answers(self):
        """Variable length answers: 1-token vs 2-token."""
        # [context, answer..., end]
        seqs = [
            [1, 2, 10, 5],        # 1-token answer: [10]
            [1, 2, 11, 12, 5],    # 2-token answer: [11, 12]
        ]
        result = find_differing_ranges(seqs)
        # prefix=[1,2] (len=2), suffix=[5] (len=1)
        # seq0: [2, 4-1) = [2, 3)
        # seq1: [2, 5-1) = [2, 4)
        assert result == [(2, 3), (2, 4)]

    def test_variable_length_three_choices(self):
        """Three choices with different lengths."""
        seqs = [
            [1, 2, 10, 5],           # 1-token: [10]
            [1, 2, 11, 12, 5],       # 2-token: [11, 12]
            [1, 2, 13, 14, 15, 5],   # 3-token: [13, 14, 15]
        ]
        result = find_differing_ranges(seqs)
        # prefix=[1,2] (len=2), suffix=[5] (len=1)
        assert result == [(2, 3), (2, 4), (2, 5)]

    def test_no_common_suffix(self):
        """Different end tokens - no common suffix."""
        seqs = [
            [1, 2, 3, 10, 50],
            [1, 2, 3, 11, 51],
            [1, 2, 3, 12, 52],
        ]
        result = find_differing_ranges(seqs)
        # prefix=[1,2,3] (len=3), suffix=0 (end tokens differ)
        # answer range = [3, len) for each
        assert result == [(3, 5), (3, 5), (3, 5)]

    def test_no_common_prefix(self):
        """Different start tokens - no common prefix."""
        seqs = [
            [10, 2, 3, 5],
            [11, 2, 3, 5],
            [12, 2, 3, 5],
        ]
        result = find_differing_ranges(seqs)
        # prefix=0 (first tokens differ), suffix=[5] (len=1)?
        # Actually [2,3,5] is common suffix... no wait, they differ at pos 0
        # prefix=0, need to check suffix from end
        # pos -1: all 5 ✓
        # pos -2: all 3 ✓
        # pos -3: all 2 ✓
        # pos -4 would be pos 0: 10,11,12 differ
        # But suffix cannot overlap entire sequence, max check = min_len - prefix = 4 - 0 = 4
        # So suffix could be [2,3,5] = 3 tokens
        # answer range = [0, 4-3) = [0, 1)
        assert result == [(0, 1), (0, 1), (0, 1)]

    def test_with_none_values(self):
        """None values (first token typically) should be treated as matching."""
        seqs = [
            [None, 2, 3, 10, 5],  # type: ignore[list-item]
            [None, 2, 3, 11, 5],  # type: ignore[list-item]
        ]
        result = find_differing_ranges(seqs)
        # All sequences have None at pos 0, so it's part of the common prefix
        # prefix=[None, 2, 3] (len=3), suffix=[5] (len=1)
        # answer range = [3, 5-1) = [3, 4)
        assert result == [(3, 4), (3, 4)]

    def test_none_at_start_then_common(self):
        """None at start, then common tokens, then different answer."""
        seqs = [
            [None, 2, 3, 10, 5],  # type: ignore[list-item]
            [None, 2, 3, 11, 5],  # type: ignore[list-item]
        ]
        result = find_differing_ranges(seqs)
        # All sequences have same None at pos 0 - it's treated as matching
        # prefix=[None, 2, 3] len=3, suffix=[5] len=1
        # ranges: [3, 4), [3, 4)
        assert result == [(3, 4), (3, 4)]

    def test_longer_common_prefix_and_suffix(self):
        """Realistic chat template with longer prefix and suffix."""
        # Simulating: [BOS, sys_start, ..., sys_end, user_start, ..., question, user_end,
        #              asst_start, ANSWER, asst_end, EOS]
        prefix = [1, 100, 101, 102, 103, 200, 201, 202, 203, 300]  # 10 tokens
        suffix = [400, 2]  # asst_end, EOS = 2 tokens

        seqs = [
            prefix + [10] + suffix,       # 1-token answer
            prefix + [11, 12] + suffix,   # 2-token answer
            prefix + [13] + suffix,       # 1-token answer
        ]
        result = find_differing_ranges(seqs)
        # prefix_len=10, suffix_len=2
        # seq0: [10, 13-2) = [10, 11)
        # seq1: [10, 14-2) = [10, 12)
        # seq2: [10, 13-2) = [10, 11)
        assert result == [(10, 11), (10, 12), (10, 11)]

    def test_two_identical_one_different(self):
        """Two identical sequences, one different."""
        seqs = [
            [1, 2, 10, 5],
            [1, 2, 10, 5],  # same as first
            [1, 2, 11, 5],  # different answer
        ]
        result = find_differing_ranges(seqs)
        # prefix=[1,2] (all same), pos 2 differs (10,10,11 - not all same)
        # suffix=[5]
        assert result == [(2, 3), (2, 3), (2, 3)]

    def test_prefix_suffix_no_overlap(self):
        """Ensure prefix and suffix don't overlap."""
        # Minimal case: 2 tokens, both same
        seqs = [[1, 2], [1, 2]]
        result = find_differing_ranges(seqs)
        # prefix=2, suffix can't extend (would overlap)
        # ranges: [2, 2) = empty
        assert result == [(2, 2), (2, 2)]

    def test_minimal_difference(self):
        """Minimal case: just one differing token."""
        seqs = [[10], [11]]
        result = find_differing_ranges(seqs)
        # prefix=0, suffix=0 (nothing common)
        # ranges: [0, 1)
        assert result == [(0, 1), (0, 1)]
