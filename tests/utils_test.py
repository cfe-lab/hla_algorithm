from collections.abc import Iterable, Sequence

import numpy as np
import pytest

from easyhla.utils import (
    bin2nuc,
    calc_padding,
    check_bases,
    count_forgiving_mismatches,
    count_strict_mismatches,
    nuc2bin,
)


class TestBinaryNucleotideTranslation:
    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("ACGT", (1, 2, 4, 8)),
            ("YANK", (10, 1, 15, 12)),
            ("TARDY", (8, 1, 5, 13, 10)),
            ("MRMAN", (3, 5, 3, 1, 15)),
            ("MYSWARD", (3, 10, 6, 9, 1, 5, 13)),
            # This is where I pull out scrabblewordfinder.org
            ("GANTRY", (4, 1, 15, 8, 5, 10)),
            ("SKYWATCH", (6, 12, 10, 9, 1, 8, 2, 11)),
            ("THWACK", (8, 11, 9, 1, 2, 12)),
            ("VAN", (7, 1, 15)),
            ("ABRA", (1, 14, 5, 1)),
        ],
    )
    def test_nuc2bin_bin2nuc_good_cases(
        self, sequence_str: str, sequence_list: Sequence[int]
    ):
        """
        Test that we can convert back and forth between a list of binary values
        and strings.
        """
        result_str = bin2nuc(sequence_list)
        assert result_str == sequence_str
        result_list = nuc2bin(sequence_str)
        assert result_list == sequence_list

    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("E", (0,)),
            ("a", (0,)),
            ("123", (0, 0, 0)),
            ("AC_TT_G", (1, 2, 0, 8, 8, 0, 4)),
        ],
    )
    def test_nuc2bin_bad_characters(
        self, sequence_str: str, sequence_list: Sequence[int]
    ):
        """
        Translating characters that aren't in the mapping turns them into 0s.
        """
        result_list = nuc2bin(sequence_str)
        assert result_list == sequence_list

    @pytest.mark.parametrize(
        "sequence_list, sequence_str",
        [
            (np.array([0]), "_"),
            (np.array([16]), "_"),
            (np.array([17]), "_"),
            (np.array([250]), "_"),
            (np.array([1, 2, 0, 8, 8, 0, 4]), "AC_TT_G"),
            (np.array([1, 14, 2, 100, 8, 17, 4]), "ABC_T_G"),
        ],
    )
    def test_bin2nuc_bad_indices(self, sequence_list: Sequence[int], sequence_str: str):
        """
        Indices not in our mapping become underscores.
        """
        result_str = bin2nuc(sequence_list)
        assert result_str == sequence_str


@pytest.mark.parametrize(
    "sequence_1, sequence_2, exp_result",
    [
        # All the normal bases:
        ([1], [1], 0),
        ([1], [2], 1),
        ([1], [4], 1),
        ([1], [8], 1),
        ([2], [1], 1),
        ([2], [2], 0),
        ([2], [4], 1),
        ([2], [8], 1),
        ([4], [1], 1),
        ([4], [2], 1),
        ([4], [4], 0),
        ([4], [8], 1),
        ([8], [1], 1),
        ([8], [2], 1),
        ([8], [4], 1),
        ([8], [8], 0),
        # Testing mixtures:
        ([5], [1], 0),
        ([5], [8], 1),
        ([1], [5], 0),
        ([8], [5], 1),
        ([12], [3], 1),
        ([12], [5], 0),
        ([15], [7], 0),
        ([7], [15], 0),
        # Longer sequences:
        ([1, 2, 4, 8], [1, 2, 4, 8], 0),
        ([1, 2, 4, 8], [1, 2, 4, 4], 1),
        ([1, 2, 4, 8], [8, 4, 2, 1], 4),
        ([1, 2, 4, 8], [5, 2, 6, 12], 0),
        ([5, 2, 6, 12], [1, 2, 4, 8], 0),
        ([1, 2, 7, 2], [2, 3, 6, 2], 1),
    ],
)
def test_count_strict_mismatches(
    sequence_1: list[int], sequence_2: list[int], exp_result: int
):
    result = count_strict_mismatches(sequence_1, sequence_2)
    assert result == exp_result


@pytest.mark.parametrize(
    "sequence_1, sequence_2, exp_result",
    [
        # All the normal bases:
        ([1], [1], 0),
        ([1], [2], 1),
        ([1], [4], 1),
        ([1], [8], 1),
        ([2], [1], 1),
        ([2], [2], 0),
        ([2], [4], 1),
        ([2], [8], 1),
        ([4], [1], 1),
        ([4], [2], 1),
        ([4], [4], 0),
        ([4], [8], 1),
        ([8], [1], 1),
        ([8], [2], 1),
        ([8], [4], 1),
        ([8], [8], 0),
        # Mixture compared to normal base:
        ([5], [1], 0),
        ([1], [5], 0),
        ([5], [8], 1),
        ([8], [5], 1),
        # Mixture compared to self:
        ([3], [3], 0),
        # Mixture subsumed by mixture:
        ([3], [7], 0),
        ([7], [3], 0),
        # Mixture orthogonal to other mixture:
        ([3], [12], 1),
        ([12], [3], 1),
        # Mixture partially overlaps other mixture:
        ([12], [5], 1),
        ([5], [12], 1),
        ([13], [3], 1),
        ([3], [13], 1),
        # Longer sequences:
        ([1, 2, 4, 8], [1, 2, 4, 8], 0),
        ([1, 2, 4, 8], [1, 2, 4, 4], 1),
        ([1, 2, 4, 8], [8, 4, 2, 1], 4),
        ([1, 2, 4, 8], [5, 2, 6, 12], 0),
        ([5, 2, 6, 12], [1, 2, 4, 8], 0),
        ([1, 2, 7, 2, 12, 5], [2, 3, 6, 2, 5, 10], 3),
    ],
)
def test_count_forgiving_mismatches(
    sequence_1: list[int], sequence_2: list[int], exp_result: int
):
    result = count_forgiving_mismatches(sequence_1, sequence_2)
    assert result == exp_result


@pytest.mark.parametrize(
    "sequence_1, sequence_2, expected_error",
    [
        ([1, 2], [1, 2, 14], "Sequences must be the same length"),
        ([], [], "Sequences must be non-empty"),
    ],
)
def test_count_forgiving_mismatches_exception_cases(
    sequence_1: list[int], sequence_2: list[int], expected_error: str
):
    with pytest.raises(ValueError) as excinfo:
        count_forgiving_mismatches(sequence_1, sequence_2)
    assert expected_error in str(excinfo.value)


@pytest.mark.parametrize(
    "sequence, exp_good",
    [
        ("A", True),
        ("C", True),
        ("G", True),
        ("T", True),
        ("R", True),
        ("Y", True),
        ("K", True),
        ("M", True),
        ("S", True),
        ("W", True),
        ("V", True),
        ("H", True),
        ("D", True),
        ("B", True),
        ("N", True),
        ("Z", False),
        ("a", False),
        ("k", False),
        ("AZ", False),
        ("aZ", False),
        ("CZ", False),
        ("cZ", False),
        ("GZ", False),
        ("gZ", False),
        ("TZ", False),
        ("tZ", False),
        ("ZGR", False),
        ("CYMSWANDB", True),
        ("CYMsWANdB", False),
        ("ZYMSWANDB", False),
        ("CYMSWANDBZ", False),
    ],
)
def test_check_bases(sequence: str, exp_good: bool):
    if exp_good:
        check_bases(seq=sequence)
    else:
        with pytest.raises(ValueError):
            check_bases(seq=sequence)


@pytest.mark.parametrize(
    "standard, sequence, exp_left_pad, exp_right_pad",
    [
        ([1, 2, 4, 8], [1, 2, 4, 8], 0, 0),
        ([1, 2, 4, 8, 8, 8, 8], [1, 2, 4, 8], 0, 3),
        ([8, 8, 1, 2, 4, 8, 8, 8, 8], [1, 2, 4, 8], 2, 3),
        ([8, 8, 8, 8, 8, 8, 8, 1, 2, 4, 8], [1, 2, 4, 8], 7, 0),
        ([8, 8, 1, 2, 4, 8, 8, 4, 8], [1, 8, 2, 4, 8], 1, 3),
        # In the case of a tie, the first match is chosen:
        # Here, 10, 0 is equally good:
        ([8, 8, 1, 2, 4, 8, 8, 8, 8, 8, 1, 2, 4, 8, 8, 8, 8], [1, 2, 4, 8], 2, 11),
        # Here, 10, 0 is equally good:
        ([8, 1, 1, 8, 1, 8, 8, 8, 8, 8, 1, 8, 1, 1], [1, 1, 1, 1], 1, 9),
        # The best match is properly found:
        ([8, 8, 1, 1, 8, 1, 8, 8, 8, 8, 1, 1, 1, 1, 8], [1, 1, 1, 1], 10, 1),
        ([8, 8, 1, 1, 1, 1, 8, 8, 8, 8, 1, 1, 8, 1, 8], [1, 1, 1, 1], 2, 9),
        # Mixtures are handled properly:
        ([3, 5, 12], [1, 1, 8], 0, 0),
        ([14, 14, 3, 5, 12, 11], [1, 1, 8], 2, 1),
        ([3, 5, 12, 4, 6, 4], [1, 1, 8], 0, 3),
        ([4, 4, 6, 5, 3, 9], [1, 1, 8], 3, 0),
        # Matches with mixtures are treated equally to matches without mixtures:
        ([2, 3, 5, 12, 2, 2, 1, 1, 8], [1, 1, 8], 1, 5),
        ([3, 5, 12, 2, 2, 1, 1, 8, 1], [1, 1, 8], 0, 6),
        ([1, 1, 3, 2, 2, 3, 5, 12, 1], [1, 1, 8], 5, 1),
        ([1, 1, 3, 2, 2, 3, 5, 12], [1, 1, 8], 5, 0),
    ],
)
def test_calc_padding(
    standard: Iterable[int],
    sequence: Iterable[int],
    exp_left_pad: int,
    exp_right_pad: int,
):
    std = np.array(standard)
    seq = np.array(sequence)
    left_pad, right_pad = calc_padding(std, seq)
    assert left_pad == exp_left_pad
    assert right_pad == exp_right_pad
