from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from pytest_mock import MockerFixture

from easyhla.utils import (
    EXON_NAME,
    HLA_LOCUS,
    GroupedAllele,
    allele_integer_coordinates,
    bin2nuc,
    calc_padding,
    check_bases,
    collate_standards,
    count_forgiving_mismatches,
    count_strict_mismatches,
    get_acceptable_match,
    group_identical_alleles,
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


@pytest.mark.parametrize(
    "sequence, reference, mismatch_threshold, expected_score, expected_best_match",
    [
        pytest.param(
            "",
            "",
            20,
            0,
            None,
            id="empty_sequence_and_reference",
        ),
        pytest.param(
            "A" * 100,
            "A" * 100,
            20,
            0,
            "A" * 100,
            id="exact_match_same_length",
        ),
        pytest.param(
            "A" * 20 + "CTCT" * 20,
            "A" * 20,
            20,
            0,
            "A" * 20,
            id="exact_match_at_beginning",
        ),
        pytest.param(
            "CTCT" * 20 + "A" * 20,
            "A" * 20,
            0,
            0,
            "A" * 20,
            id="exact_match_at_end",
        ),
        pytest.param(
            "CTCT" * 10 + "A" * 20 + "CTCT" * 10,
            "A" * 20,
            0,
            0,
            "A" * 20,
            id="exact_match_in_middle",
        ),
        pytest.param(
            "C" * 40 + "A" * 20 + "T" * 40,
            "A" * 20,
            20,
            19,
            "C" * 19 + "A",
            id="acceptable_match_gives_up_quickly",
        ),
        pytest.param(
            "T" * 40 + "ACGT" * 5 + "C" * 40,
            "ACGT" * 5,
            3,
            0,
            "ACGT" * 5,
            id="typical_case",
        ),
    ],
)
def test_get_acceptable_match(
    sequence: str,
    reference: str,
    mismatch_threshold: int,
    expected_score: int,
    expected_best_match: str,
):
    score: int
    best_match: Optional[str]
    score, best_match = get_acceptable_match(sequence, reference, mismatch_threshold)
    assert score == expected_score
    assert best_match == expected_best_match


@pytest.mark.parametrize(
    "sequence, reference",
    [
        pytest.param("A" * 100, "A" * 101, id="reference_too_long_edge_case"),
        pytest.param("A" * 100, "A" * 150, id="reference_too_long_typical_case"),
    ],
)
def test_get_acceptable_match_error_case(sequence: str, reference: str):
    with pytest.raises(ValueError):
        get_acceptable_match("A" * 100, "A" * 101, 20)


@pytest.mark.parametrize(
    "allele, expected_coords",
    [
        pytest.param("A*01", (1,), id="single_coordinate"),
        pytest.param("B*57:02", (57, 2), id="two_coordinates"),
        pytest.param("B*57:02:15G", (57, 2, 15), id="three_coordinates"),
        pytest.param("B*57:02:15:02", (57, 2, 15, 2), id="four_coordinates"),
    ],
)
def test_allele_integer_coordinates(allele: str, expected_coords: tuple[int, ...]):
    assert allele_integer_coordinates(allele) == expected_coords


EXON_REFERENCES: dict[HLA_LOCUS, dict[EXON_NAME, str]] = {
    "A": {
        "exon2": "ACGT" * 10,
        "exon3": "TGCA" * 10,
    },
    "B": {
        "exon2": "CGTA" * 10,
        "exon3": "ATGC" * 10,
    },
    "C": {
        "exon2": "TACG" * 10,
        "exon3": "GCAT" * 10,
    },
}


@pytest.mark.parametrize(
    (
        "srs, exon_references, overall_mismatch_threshold, "
        "acceptable_match_search_threshold, report_interval, use_logging, "
        "expected_a, expected_b, expected_c, expected_logging_calls"
    ),
    [
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            32,
            0,
            1000,
            False,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [],
            [],
            [],
            id="one_good_a_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            32,
            0,
            1000,
            True,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [],
            [],
            [],
            id="one_good_a_standard_with_logging_but_no_messages",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            32,
            0,
            1,
            True,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [],
            [],
            ["Processing sequence 1 of 1...."],
            id="one_good_a_standard_with_logging_and_status_update_message",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("CGTA" * 10 + "C" * 40 + "ATGC" * 10),
                    id="HLA_1",
                    description="HLA_1 B*57:01:04 120 bp",
                ),
            ],
            EXON_REFERENCES,
            32,
            0,
            1000,
            False,
            [],
            [("B*57:01:04", "CGTA" * 10, "ATGC" * 10)],
            [],
            [],
            id="one_good_b_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TACG" * 10 + "A" * 117 + "GCAT" * 10 + "A" * 10),
                    id="HLA_1",
                    description="HLA_1 C*22:33:44:55N 207 bp",
                ),
            ],
            EXON_REFERENCES,
            32,
            0,
            1000,
            False,
            [],
            [],
            [("C*22:33:44:55N", "TACG" * 10, "GCAT" * 10)],
            [],
            id="one_good_c_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TTTT" * 10 + "CCCC" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="one_bad_a_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TTTT" * 10 + "AAAA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            True,
            [],
            [],
            [],
            ['Rejecting "A*01:01:01:01": 30 exon2 mismatches, 29 exon3 mismatches.'],
            id="one_bad_a_standard_with_logging_and_no_status_update",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TTTT" * 10 + "AAAA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1,
            True,
            [],
            [],
            [],
            [
                "Processing sequence 1 of 1....",
                'Rejecting "A*01:01:01:01": 30 exon2 mismatches, 29 exon3 mismatches.',
            ],
            id="one_bad_a_standard_with_logging_and_with_status_update",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TTTT" * 10 + "CCCC" * 10),
                    id="HLA_1",
                    description="HLA_1 B*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="one_bad_b_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("TTTT" * 10 + "CCCC" * 10),
                    id="HLA_1",
                    description="HLA_1 C*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="one_bad_c_standard",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 40),
                    id="HLA_1",
                    description="HLA_1 D*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="non_abc_standard_skipped",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TTTT" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="good_exon2_bad_exon3_no_logging",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TTTT" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            True,
            [],
            [],
            [],
            ['Rejecting "A*01:01:01:01": 0 exon2 mismatches, 21 exon3 mismatches.'],
            id="good_exon2_bad_exon3_with_logging_no_status_update",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TTTT" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1,
            True,
            [],
            [],
            [],
            [
                "Processing sequence 1 of 1....",
                'Rejecting "A*01:01:01:01": 0 exon2 mismatches, 21 exon3 mismatches.',
            ],
            id="good_exon2_bad_exon3_with_logging_with_status_update",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("C" * 80 + "ATGC" * 10),
                    id="HLA_1",
                    description="HLA_1 B*57:01:04 120 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [],
            [],
            [],
            [],
            id="good_exon3_bad_exon2_no_logging",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("C" * 80 + "ATGC" * 10),
                    id="HLA_1",
                    description="HLA_1 B*57:01:04 120 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            True,
            [],
            [],
            [],
            ['Rejecting "B*57:01:04": 20 exon2 mismatches, 0 exon3 mismatches.'],
            id="good_exon3_bad_exon2_with_logging_no_status_update",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
                SeqRecord(
                    Seq("C" * 80 + "ATGC" * 10),
                    id="HLA_2",
                    description="HLA_2 B*57:01:04 120 bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 10 + "AAAA" * 10),
                    id="HLA_3",
                    description="HLA_3 C*22:33:43 160bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 9 + "GCAA"),
                    id="HLA_4",
                    description="HLA_4 C*22:33:44 120bp",
                ),
                SeqRecord(
                    Seq("C" * 20 + "AGTA" + "CGTA" * 9 + "T" * 80 + "ATGC" * 10),
                    id="HLA_5",
                    description="HLA_5 B*57:01:06 180 bp",
                ),
                SeqRecord(
                    Seq("ACGT" * 80),
                    id="HLA_6",
                    description="HLA_5 E*101:11:10:11N 180 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [("B*57:01:06", "AGTA" + "CGTA" * 9, "ATGC" * 10)],
            [
                ("C*22:33:43", "TACG" * 10, "GCAT" * 10),
                ("C*22:33:44", "TACG" * 10, "GCAT" * 9 + "GCAA"),
            ],
            [],
            id="typical_case_no_logging",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
                SeqRecord(
                    Seq("C" * 80 + "ATGC" * 10),
                    id="HLA_2",
                    description="HLA_2 B*57:01:04 120 bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 10 + "AAAA" * 10),
                    id="HLA_3",
                    description="HLA_3 C*22:33:43 160bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 9 + "GCAA"),
                    id="HLA_4",
                    description="HLA_4 C*22:33:44 120bp",
                ),
                SeqRecord(
                    Seq("C" * 20 + "AGTA" + "CGTA" * 9 + "T" * 80 + "ATGC" * 10),
                    id="HLA_5",
                    description="HLA_5 B*57:01:06 180 bp",
                ),
                SeqRecord(
                    Seq("ACGT" * 80),
                    id="HLA_6",
                    description="HLA_5 E*101:11:10:11N 180 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            1000,
            False,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [("B*57:01:06", "AGTA" + "CGTA" * 9, "ATGC" * 10)],
            [
                ("C*22:33:43", "TACG" * 10, "GCAT" * 10),
                ("C*22:33:44", "TACG" * 10, "GCAT" * 9 + "GCAA"),
            ],
            [
                'Rejecting "B*57:01:04": 20 exon2 mismatches, 0 exon3 mismatches.',
            ],
            id="typical_case_with_logging_no_status_updates",
        ),
        pytest.param(
            [
                SeqRecord(
                    Seq("ACGT" * 10 + "TGCA" * 10),
                    id="HLA_1",
                    description="HLA_1 A*01:01:01:01 80 bp",
                ),
                SeqRecord(
                    Seq("C" * 80 + "ATGC" * 10),
                    id="HLA_2",
                    description="HLA_2 B*57:01:04 120 bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 10 + "AAAA" * 10),
                    id="HLA_3",
                    description="HLA_3 C*22:33:43 160bp",
                ),
                SeqRecord(
                    Seq("TACG" * 10 + "C" * 40 + "GCAT" * 9 + "GCAA"),
                    id="HLA_4",
                    description="HLA_4 C*22:33:44 120bp",
                ),
                SeqRecord(
                    Seq("C" * 20 + "AGTA" + "CGTA" * 9 + "T" * 80 + "ATGC" * 10),
                    id="HLA_5",
                    description="HLA_5 B*57:01:06 180 bp",
                ),
                SeqRecord(
                    Seq("ACGT" * 80),
                    id="HLA_6",
                    description="HLA_5 E*101:11:10:11N 180 bp",
                ),
            ],
            EXON_REFERENCES,
            5,
            0,
            2,
            False,
            [("A*01:01:01:01", "ACGT" * 10, "TGCA" * 10)],
            [("B*57:01:06", "AGTA" + "CGTA" * 9, "ATGC" * 10)],
            [
                ("C*22:33:43", "TACG" * 10, "GCAT" * 10),
                ("C*22:33:44", "TACG" * 10, "GCAT" * 9 + "GCAA"),
            ],
            [
                "Processing sequence 2 of 6....",
                'Rejecting "B*57:01:04": 20 exon2 mismatches, 0 exon3 mismatches.',
                "Processing sequence 4 of 6....",
                "Processing sequence 6 of 6....",
            ],
            id="typical_case_with_logging_with_status_updates",
        ),
    ],
)
def test_collate_standards(
    srs: Iterable[SeqRecord],
    exon_references: dict[HLA_LOCUS, dict[EXON_NAME, str]],
    overall_mismatch_threshold: int,
    acceptable_match_search_threshold: int,
    report_interval: Optional[int],
    use_logging: bool,
    expected_a: list[tuple[str, str, str]],
    expected_b: list[tuple[str, str, str]],
    expected_c: list[tuple[str, str, str]],
    expected_logging_calls: list[str],
    mocker: MockerFixture,
):
    mock_logger: Optional[mocker.MagicMock] = None
    if use_logging:
        mock_logger = mocker.MagicMock()
    result: dict[HLA_LOCUS, list[tuple[str, str, str]]] = collate_standards(
        srs,
        exon_references,
        mock_logger,
        overall_mismatch_threshold,
        acceptable_match_search_threshold,
        report_interval,
    )
    assert result["A"] == expected_a
    assert result["B"] == expected_b
    assert result["C"] == expected_c

    if use_logging:
        if len(expected_logging_calls) > 0:
            mock_logger.info.assert_has_calls(
                [mocker.call(x) for x in expected_logging_calls],
                any_order=False,
            )
        else:
            mock_logger.assert_not_called()


@pytest.mark.parametrize(
    "alleles, expected_result",
    [
        pytest.param(
            ["C*25:26:49:33N"],
            "C*25:26:49:33N",
            id="single_allele_four_coordinates",
        ),
        pytest.param(
            ["B*01:22:43"],
            "B*01:22:43",
            id="single_allele_three_coordinates",
        ),
        pytest.param(
            ["B*01:22:43", "B*01:1145:233"],
            "B*01:22:43G",
            id="two_alleles_no_shortening_edge_case",
        ),
        pytest.param(
            ["B*01:122", "B*01:125N"],
            "B*01:122G",
            id="two_alleles_no_shortening_normal_case",
        ),
        pytest.param(
            ["B*01:02:03:04N", "B*01:125N"],
            "B*01:02:03G",
            id="two_alleles_with_shortening",
        ),
    ],
)
def test_grouped_allele_get_group_name(
    alleles: list[str],
    expected_result: str,
):
    ga: GroupedAllele = GroupedAllele("AGCTAGCTAGCT", "TGCATGCATGCA", alleles)
    assert ga.get_group_name() == expected_result


@pytest.mark.parametrize(
    "allele_infos, use_logging, expected_result, expected_logging_calls",
    [
        pytest.param(
            [("A*01:01:01:01", "AAA", "CCC")],
            False,
            {
                "A*01:01:01:01": GroupedAllele("AAA", "CCC", ["A*01:01:01:01"]),
            },
            [],
            id="single_allele_no_logging",
        ),
        pytest.param(
            [("A*01:01:01:01", "AAA", "CCC")],
            True,
            {
                "A*01:01:01:01": GroupedAllele("AAA", "CCC", ["A*01:01:01:01"]),
            },
            [],
            id="single_allele_with_logging",
        ),
        pytest.param(
            [
                ("B*01:01:01:01", "AAA", "CCC"),
                ("B*57:01:02", "TTT", "CCT"),
            ],
            False,
            {
                "B*01:01:01:01": GroupedAllele("AAA", "CCC", ["B*01:01:01:01"]),
                "B*57:01:02": GroupedAllele("TTT", "CCT", ["B*57:01:02"]),
            },
            [],
            id="two_alleles_no_grouping_no_logging",
        ),
        pytest.param(
            [
                ("B*01:01:01:01", "AAA", "CCC"),
                ("B*57:01:02", "TTT", "CCT"),
            ],
            True,
            {
                "B*01:01:01:01": GroupedAllele("AAA", "CCC", ["B*01:01:01:01"]),
                "B*57:01:02": GroupedAllele("TTT", "CCT", ["B*57:01:02"]),
            },
            [],
            id="two_alleles_no_grouping_with_logging",
        ),
        pytest.param(
            [
                ("B*01:01:01:01", "AAA", "CCC"),
                ("B*57:01:02", "AAA", "CCC"),
            ],
            False,
            {
                "B*01:01:01G": GroupedAllele(
                    "AAA",
                    "CCC",
                    ["B*01:01:01:01", "B*57:01:02"],
                ),
            },
            [],
            id="two_alleles_grouped_no_logging",
        ),
        pytest.param(
            [
                ("B*01:01:01:01", "AAA", "CCC"),
                ("B*57:01:02", "AAA", "CCC"),
            ],
            True,
            {
                "B*01:01:01G": GroupedAllele(
                    "AAA",
                    "CCC",
                    ["B*01:01:01:01", "B*57:01:02"],
                ),
            },
            ["[B*01:01:01:01, B*57:01:02] -> B*01:01:01G"],
            id="two_alleles_grouped_with_logging",
        ),
        pytest.param(
            [
                ("B*01:01:01:01", "AAA", "CCC"),
                ("B*01:02:07", "ACT", "TAT"),
                ("B*57:01:02", "AAA", "CCC"),
                ("B*58:02:02:02N", "TAC", "TTT"),
                ("B*59:112", "ACT", "TAT"),
                ("B*101:101:101:101", "AAA", "CCC"),
            ],
            True,
            {
                "B*01:01:01G": GroupedAllele(
                    "AAA",
                    "CCC",
                    ["B*01:01:01:01", "B*57:01:02", "B*101:101:101:101"],
                ),
                "B*01:02:07G": GroupedAllele(
                    "ACT",
                    "TAT",
                    ["B*01:02:07", "B*59:112"],
                ),
                "B*58:02:02:02N": GroupedAllele(
                    "TAC",
                    "TTT",
                    ["B*58:02:02:02N"],
                ),
            },
            [
                "[B*01:01:01:01, B*57:01:02, B*101:101:101:101] -> B*01:01:01G",
                "[B*01:02:07, B*59:112] -> B*01:02:07G",
            ],
            id="typical_case",
        ),
    ],
)
def test_group_identical_alleles(
    allele_infos: list[tuple[str, str, str]],
    use_logging: bool,
    expected_result: dict[str, GroupedAllele],
    expected_logging_calls: list[str],
    mocker: MockerFixture,
):
    mock_logger: Optional[mocker.MagicMock] = None
    if use_logging:
        mock_logger = mocker.MagicMock()
    result: dict[str, GroupedAllele] = group_identical_alleles(
        allele_infos, mock_logger
    )
    assert result == expected_result

    if use_logging:
        if len(expected_logging_calls) > 0:
            mock_logger.info.assert_has_calls(
                [mocker.call(x) for x in expected_logging_calls],
                any_order=False,
            )
        else:
            mock_logger.assert_not_called()
