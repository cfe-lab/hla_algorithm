from typing import Optional

import numpy as np
import pytest

from hla_algorithm.models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
)


class TestHLASequence:
    @pytest.mark.parametrize(
        (
            "hla_sequence, expected_sequence_for_interpretation, "
            "expected_exon2_str, expected_intron_str, expected_exon3_str"
        ),
        [
            pytest.param(
                HLASequence(
                    two=(1, 2, 4, 8),
                    intron=(),
                    three=(14, 13, 11, 7),
                    name="test_sequence",
                    locus="B",
                    num_sequences_used=1,
                ),
                (1, 2, 4, 8, 14, 13, 11, 7),
                "ACGT",
                "",
                "BDHV",
                id="no_intron",
            ),
            pytest.param(
                HLASequence(
                    two=(1, 2, 4, 8),
                    intron=(2, 4, 5, 6),
                    three=(14, 13, 11, 7),
                    name="test_sequence",
                    locus="A",
                    num_sequences_used=1,
                ),
                (1, 2, 4, 8, 14, 13, 11, 7),
                "ACGT",
                "CGRS",
                "BDHV",
                id="with_intron",
            ),
        ],
    )
    def test_basic_properties(
        self,
        hla_sequence: HLASequence,
        expected_sequence_for_interpretation: tuple[int, ...],
        expected_exon2_str: str,
        expected_intron_str: str,
        expected_exon3_str: str,
    ):
        assert (
            hla_sequence.sequence_for_interpretation
            == expected_sequence_for_interpretation
        )
        assert hla_sequence.exon2_str == expected_exon2_str
        assert hla_sequence.intron_str == expected_intron_str
        assert hla_sequence.exon3_str == expected_exon3_str


class TestHLAStandard:
    @pytest.mark.parametrize(
        "two, three, expected_tuple, expected_array",
        [
            pytest.param(
                (),
                (),
                (),
                np.array([]),
                id="empty_sequence",
            ),
            pytest.param(
                (4,),
                (),
                (4,),
                np.array([4]),
                id="single_base_sequence",
            ),
            pytest.param(
                (1, 2, 4, 8),
                (11, 15, 12, 8),
                (1, 2, 4, 8, 11, 15, 12, 8),
                np.array([1, 2, 4, 8, 11, 15, 12, 8]),
                id="typical_sequence",
            ),
        ],
    )
    def test_sequence_np(
        self,
        two: tuple[int, ...],
        three: tuple[int, ...],
        expected_tuple: tuple[int, ...],
        expected_array: np.ndarray,
    ):
        hla_standard: HLAStandard = HLAStandard(
            allele="B*01:23:45", two=two, three=three
        )
        assert hla_standard.sequence == expected_tuple
        assert np.array_equal(hla_standard.sequence_np, expected_array)


class TestHLACombinedStandard:
    @pytest.mark.parametrize(
        "allele_pairs, exp_allele_pair_str",
        [
            ([("A*01:23:45", "A*33:44:55:66")], "A*01:23:45 - A*33:44:55:66"),
            (
                [
                    ("B*37:42:47G", "B*55:24"),
                    ("B*37:42:99:01N", "B*54:02:03"),
                ],
                "B*37:42:47G - B*55:24|B*37:42:99:01N - B*54:02:03",
            ),
            (
                [
                    ("C*88:99:110C", "C*22:23:25"),
                    ("C*38:83", "C*54:40"),
                    ("C*01:23:04", "C*01:23:04"),
                ],
                "C*88:99:110C - C*22:23:25|C*38:83 - C*54:40|C*01:23:04 - C*01:23:04",
            ),
        ],
    )
    def test_get_allele_pair_str(
        self,
        allele_pairs: list[tuple[str, str]],
        exp_allele_pair_str: str,
    ):
        cs: HLACombinedStandard = HLACombinedStandard(
            standard_bin=(1, 5, 11, 2, 6, 8),
            possible_allele_pairs=tuple(allele_pairs),
        )
        assert cs.get_allele_pair_str() == exp_allele_pair_str


class TestHLAMismatch:
    @pytest.mark.parametrize(
        "index, sequence_base, standard_base, expected_str",
        [
            (55, "A", "C", "55:A->C"),
            (199, "C", "R", "199:C->R"),
            (9, "T", "V", "9:T->V"),
        ],
    )
    def test_string(
        self,
        index: int,
        sequence_base: str,
        standard_base: str,
        expected_str: str,
    ):
        mismatch: HLAMismatch = HLAMismatch(
            index=index,
            sequence_base=sequence_base,
            standard_base=standard_base,
        )
        assert str(mismatch) == expected_str


@pytest.mark.parametrize(
    "raw_mismatches, expected_result",
    [
        pytest.param([], 0, id="no_mismatches"),
        pytest.param(
            [HLAMismatch(index=100, sequence_base="A", standard_base="T")],
            1,
            id="one_mismatches",
        ),
        pytest.param(
            [
                HLAMismatch(index=100, sequence_base="A", standard_base="T"),
                HLAMismatch(index=150, sequence_base="R", standard_base="W"),
                HLAMismatch(index=150, sequence_base="C", standard_base="Y"),
            ],
            3,
            id="several_mismatches",
        ),
    ],
)
def test_hla_match_details_mismatch_count(
    raw_mismatches: list[HLAMismatch], expected_result: int
):
    assert HLAMatchDetails(mismatches=raw_mismatches).mismatch_count == expected_result


class TestHLAProteinPair:
    @pytest.mark.parametrize(
        "raw_lesser, raw_greater",
        [
            pytest.param(
                ("01", "01", "01", "01"),
                ("01", "01", "01", "11"),
            ),
            pytest.param(
                ("01", "01", "01", "01"),
                ("01", "01", "31", "01"),
            ),
            pytest.param(
                ("01", "01", "01", "01"),
                ("01", "41", "01", "01"),
            ),
            pytest.param(
                ("01", "01", "01", "01"),
                ("91", "41", "01", "01"),
            ),
            pytest.param(
                ("15", "84", "99", "92"),
                ("16", "41", "01", "01"),
            ),
            pytest.param(
                ("15", "84", "89", "92"),
                ("15", "91", "91", "01"),
            ),
            pytest.param(
                ("15", "84", "89", "92"),
                ("110", "91", "91", "01"),
                id="three_digits_sorted_by_number_and_not_alphabetically_first_first",
            ),
            pytest.param(
                ("15", "84", "89", "92"),
                ("15", "111", "91", "01"),
                id="three_digits_sorted_by_number_and_not_alphabetically_first_second",
            ),
            pytest.param(
                ("15", "84", "89", "92"),
                ("15", "84", "101", "01"),
                id="three_digits_sorted_by_number_and_not_alphabetically_second_first",
            ),
            pytest.param(
                ("15", "84", "89", "92"),
                ("15", "84", "89", "100"),
                id="three_digits_sorted_by_number_and_not_alphabetically_second_second",
            ),
        ],
    )
    def test_strictly_less_than(
        self,
        raw_lesser: tuple[str, str, str, str],
        raw_greater: tuple[str, str, str, str],
    ):
        first_pair: HLAProteinPair = HLAProteinPair(
            first_field_1=raw_lesser[0],
            first_field_2=raw_lesser[1],
            second_field_1=raw_lesser[2],
            second_field_2=raw_lesser[3],
        )
        second_pair: HLAProteinPair = HLAProteinPair(
            first_field_1=raw_greater[0],
            first_field_2=raw_greater[1],
            second_field_1=raw_greater[2],
            second_field_2=raw_greater[3],
        )
        assert first_pair < second_pair
        assert not (second_pair < first_pair)

    def test_equal_pairs(self):
        protein_pair: HLAProteinPair = HLAProteinPair(
            first_field_1="15",
            first_field_2="85",
            second_field_1="57",
            second_field_2="02",
        )
        assert not (protein_pair < protein_pair)

    @pytest.mark.parametrize(
        "raw_allele_1, raw_allele_2, expected_result",
        [
            pytest.param(
                "57:01",
                "57:03",
                None,
                id="good_allele_pair",
            ),
            pytest.param(
                "unmapped",
                "57:03",
                HLAProteinPair.NonAlleleException(first_unmapped=True),
                id="first_unmapped",
            ),
            pytest.param(
                "deprecated",
                "57:03",
                HLAProteinPair.NonAlleleException(first_deprecated=True),
                id="first_deprecated",
            ),
            pytest.param(
                "57:01",
                "unmapped",
                HLAProteinPair.NonAlleleException(second_unmapped=True),
                id="second_unmapped",
            ),
            pytest.param(
                "57:01",
                "deprecated",
                HLAProteinPair.NonAlleleException(second_deprecated=True),
                id="second_deprecated",
            ),
            pytest.param(
                "unmapped",
                "unmapped",
                HLAProteinPair.NonAlleleException(
                    first_unmapped=True, second_unmapped=True
                ),
                id="both_unmapped",
            ),
            pytest.param(
                "unmapped",
                "deprecated",
                HLAProteinPair.NonAlleleException(
                    first_unmapped=True, second_deprecated=True
                ),
                id="unmapped_deprecated",
            ),
            pytest.param(
                "deprecated",
                "unmapped",
                HLAProteinPair.NonAlleleException(
                    first_deprecated=True, second_unmapped=True
                ),
                id="deprecated_unmapped",
            ),
            pytest.param(
                "deprecated",
                "deprecated",
                HLAProteinPair.NonAlleleException(
                    first_deprecated=True, second_deprecated=True
                ),
                id="both_deprecated",
            ),
        ],
    )
    def test_non_allele_exception_from_frequency_entry(
        self,
        raw_allele_1: str,
        raw_allele_2: str,
        expected_result: Optional[HLAProteinPair.NonAlleleException],
    ):
        result: Optional[HLAProteinPair.NonAlleleException] = (
            HLAProteinPair.NonAlleleException.from_frequency_entry(
                raw_allele_1, raw_allele_2
            )
        )
        if expected_result is None:
            assert result is None
        else:
            for field in (
                "first_unmapped",
                "first_deprecated",
                "second_unmapped",
                "second_deprecated",
            ):
                assert getattr(result, field) == getattr(expected_result, field)

    def test_from_frequency_entry_good_case(self):
        result: HLAProteinPair = HLAProteinPair.from_frequency_entry("57:01", "56:220")
        expected_result: HLAProteinPair = HLAProteinPair(
            first_field_1="57",
            first_field_2="01",
            second_field_1="56",
            second_field_2="220",
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "raw_first_allele, raw_second_allele, expected_exception",
        [
            pytest.param(
                "unmapped",
                "56:220",
                HLAProteinPair.NonAlleleException(first_unmapped=True),
                id="first_unmapped",
            ),
            pytest.param(
                "57:02",
                "deprecated",
                HLAProteinPair.NonAlleleException(second_deprecated=True),
                id="second_deprecated",
            ),
            pytest.param(
                "deprecated",
                "unmapped",
                HLAProteinPair.NonAlleleException(
                    first_deprecated=True, second_unmapped=True
                ),
                id="deprecated_unmapped",
            ),
        ],
    )
    def test_from_frequency_entry_exceptions(
        self,
        raw_first_allele: str,
        raw_second_allele: str,
        expected_exception: HLAProteinPair.NonAlleleException,
    ):
        with pytest.raises(HLAProteinPair.NonAlleleException) as e:
            HLAProteinPair.from_frequency_entry(raw_first_allele, raw_second_allele)
            for field in (
                "first_unmapped",
                "first_deprecated",
                "second_unmapped",
                "second_deprecated",
            ):
                assert getattr(e, field) == getattr(expected_exception, field)


class TestAllelePairs:
    @pytest.mark.parametrize(
        "raw_alleles, exp_result",
        [
            ([("A*11:01", "A*26:01")], False),
            ([("C*12:34:56:78B", "C*12:34:56:78B")], True),
            (
                [
                    ("B*57:01", "B*57:08"),
                    ("B*57:10", "B*57:13:224"),
                    ("B*58:55:22", "B*58"),
                ],
                False,
            ),
            (
                [
                    ("A*11:01", "A*26:01"),
                    ("A*13:13:13:13", "A*13:13:13:13"),
                    ("A*17:223", "A*17:222"),
                ],
                True,
            ),
        ],
    )
    def test_is_homozygous(
        self,
        raw_alleles: list[tuple[str, str]],
        exp_result: bool,
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_alleles)
        assert ap.is_homozygous() == exp_result

    @pytest.mark.parametrize(
        "raw_allele_pairs, expected_result, expected_result_digits_only",
        [
            (
                [("A*01:23:45N", "A*22:33:44:55G")],
                [(["A*01", "23", "45N"], ["A*22", "33", "44", "55G"])],
                [(["01", "23", "45"], ["22", "33", "44", "55"])],
            ),
            (
                [("C*88:110:111", "C*15:25:115")],
                [(["C*88", "110", "111"], ["C*15", "25", "115"])],
                [(["88", "110", "111"], ["15", "25", "115"])],
            ),
            (
                [
                    ("B*57:01:02", "B*57:01:02"),
                    ("B*56:45:22:33", "B*54:111N"),
                    ("B*12:100:37G", "B*22:100:101G"),
                ],
                [
                    (["B*57", "01", "02"], ["B*57", "01", "02"]),
                    (["B*56", "45", "22", "33"], ["B*54", "111N"]),
                    (["B*12", "100", "37G"], ["B*22", "100", "101G"]),
                ],
                [
                    (["57", "01", "02"], ["57", "01", "02"]),
                    (["56", "45", "22", "33"], ["54", "111"]),
                    (["12", "100", "37"], ["22", "100", "101"]),
                ],
            ),
        ],
    )
    def test_get_paired_gene_coordinates(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        expected_result: list[tuple[list[str], list[str]]],
        expected_result_digits_only: list[tuple[list[str], list[str]]],
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.get_paired_gene_coordinates(False) == expected_result
        assert ap.get_paired_gene_coordinates(True) == expected_result_digits_only

    @pytest.mark.parametrize(
        "raw_allele_pairs, expected_result",
        [
            (
                [("A*01:23:45N", "A*22:33:44:55G")],
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="23",
                        second_field_1="22",
                        second_field_2="33",
                    )
                },
            ),
            (
                [("C*88:110:111", "C*15:25:115")],
                {
                    HLAProteinPair(
                        first_field_1="88",
                        first_field_2="110",
                        second_field_1="15",
                        second_field_2="25",
                    ),
                },
            ),
            (
                [
                    ("B*57:01:02", "B*57:01:02"),
                    ("B*56:45:22:33", "B*54:111N"),
                    ("B*12:100:37G", "B*22:100:101G"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="57",
                        first_field_2="01",
                        second_field_1="57",
                        second_field_2="01",
                    ),
                    HLAProteinPair(
                        first_field_1="56",
                        first_field_2="45",
                        second_field_1="54",
                        second_field_2="111",
                    ),
                    HLAProteinPair(
                        first_field_1="12",
                        first_field_2="100",
                        second_field_1="22",
                        second_field_2="100",
                    ),
                },
            ),
            (
                [("B*57:01:02", "B*57:01:02"), ("B*57:01:04", "B*57:01:07")],
                {
                    HLAProteinPair(
                        first_field_1="57",
                        first_field_2="01",
                        second_field_1="57",
                        second_field_2="01",
                    ),
                },
            ),
            (
                [
                    ("B*57:01:02", "B*57:01:02"),
                    ("B*23:45:66N", "B*24:22:33:100"),
                    ("B*57:01:04", "B*57:01:07"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="57",
                        first_field_2="01",
                        second_field_1="57",
                        second_field_2="01",
                    ),
                    HLAProteinPair(
                        first_field_1="23",
                        first_field_2="45",
                        second_field_1="24",
                        second_field_2="22",
                    ),
                },
            ),
        ],
    )
    def test_get_protein_pairs(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        expected_result: list[tuple[list[str], list[str]]],
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.get_protein_pairs() == expected_result

    @pytest.mark.parametrize(
        "raw_allele_pairs, expected_result",
        [
            ([("A*01:23:45N", "A*22:33:44:55G")], False),
            ([("A*01:23", "A*01:23")], False),
            (
                [
                    ("B*57:01:23", "B*54:02"),
                    ("B*57:02:15", "B*54:22"),
                ],
                False,
            ),
            (
                [("B*57:02:01", "B*54:02:03:06N"), ("B*56:02:01", "B*54:02:03:06N")],
                True,
            ),
            (
                [("B*57:02:01", "B*54:02:03:06N"), ("B*57:02:01", "B*22:11:12")],
                True,
            ),
            (
                [
                    ("A*02:01:01G", "A*03:01:01G"),
                    ("A*02:01:52", "A*03:01:03"),
                    ("A*02:01:02", "A*03:01:12"),
                    ("A*02:01:36", "A*03:01:38"),
                    ("A*02:237", "A*03:05:01"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24:01", "A*03:17:01"),
                    ("A*02:195", "A*03:23:01"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35:01", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20:01", "A*03:157"),
                ],
                False,
            ),
            (
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                True,
            ),
            (
                [
                    ("C*11:01:01G", "C*26:01:01G"),
                    ("C*13:02", "C*26:04"),
                    ("C*11:01:07", "C*26:01:17"),
                    ("C*11:19", "C*26:13"),
                ],
                True,
            ),
        ],
    )
    def test_is_ambiguous(
        self, raw_allele_pairs: list[tuple[str, str]], expected_result: bool
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.is_ambiguous() == expected_result

    @pytest.mark.parametrize(
        "raw_alleles, frequencies, exp_result",
        [
            (
                [
                    ("A*11:01", "A*26:01"),
                    ("A*11:01", "A*26:01"),
                    ("A*11:19", "A*26:13"),
                ],
                {},
                [
                    ("A*11:01", "A*26:01"),
                    ("A*11:01", "A*26:01"),
                    ("A*11:19", "A*26:13"),
                ],
            ),
            (
                [
                    ("A*11:01", "A*26:01"),
                    ("A*11:40", "A*23:01"),
                ],
                {},
                [("A*11:01", "A*26:01")],
            ),
            (
                [
                    ("A*11:01", "A*12:01"),
                    ("A*11:01", "A*12:01"),
                    ("A*11:40", "A*13:01"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="11",
                        first_field_2="40",
                        second_field_1="13",
                        second_field_2="01",
                    ): 4,
                    HLAProteinPair(
                        first_field_1="11",
                        first_field_2="01",
                        second_field_1="12",
                        second_field_2="01",
                    ): 2,
                },
                [("A*11:40", "A*13:01")],
            ),
            (
                [
                    ("A*11:01", "A*12:01"),
                    ("A*13:01", "A*12:44"),
                    ("A*13:40", "A*12:01"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="11",
                        first_field_2="01",
                        second_field_1="12",
                        second_field_2="01",
                    ): 1,
                    HLAProteinPair(
                        first_field_1="13",
                        first_field_2="01",
                        second_field_1="12",
                        second_field_2="44",
                    ): 0,
                    HLAProteinPair(
                        first_field_1="13",
                        first_field_2="40",
                        second_field_1="12",
                        second_field_2="01",
                    ): 10,
                },
                [("A*13:01", "A*12:44"), ("A*13:40", "A*12:01")],
            ),
        ],
    )
    def test_get_unambiguous_allele_pairs(
        self,
        raw_alleles: list[tuple[str, str]],
        frequencies: dict[HLAProteinPair, int],
        exp_result: list[tuple[str, str]],
    ):
        allele_pairs = AllelePairs(allele_pairs=raw_alleles)
        result = allele_pairs.get_unambiguous_allele_pairs(frequencies)
        assert result == exp_result

    @pytest.mark.parametrize(
        "raw_allele_pairs, frequencies, expected_result, expected_unambiguous_set",
        [
            pytest.param(
                [("B*57:01", "B*59:03")],
                {},
                "B*57:01 - B*59:03",
                {("B*57:01", "B*59:03")},
                id="single_allele_pair_two_coordinates",
            ),
            pytest.param(
                [("C*01:02:03", "C*03:04:05")],
                {},
                "C*01:02:03 - C*03:04:05",
                {("C*01:02:03", "C*03:04:05")},
                id="single_allele_pair_three_coordinates",
            ),
            pytest.param(
                [("C*01:02:03", "C*03:04")],
                {},
                "C*01:02:03 - C*03:04",
                {("C*01:02:03", "C*03:04")},
                id="single_allele_pair_three_and_two_coordinates",
            ),
            pytest.param(
                [("C*01:02", "C*03:04:05:06")],
                {},
                "C*01:02 - C*03:04:05:06",
                {("C*01:02", "C*03:04:05:06")},
                id="single_allele_pair_two_and_four_coordinates",
            ),
            pytest.param(
                [("C*01:02:03:04", "C*03:04:05:06")],
                {},
                "C*01:02:03:04 - C*03:04:05:06",
                {("C*01:02:03:04", "C*03:04:05:06")},
                id="single_allele_pair_four_coordinates",
            ),
            pytest.param(
                [("A*01:02:03:04N", "A*11:22:33:44G")],
                {},
                "A*01:02:03:04 - A*11:22:33:44",
                {("A*01:02:03:04N", "A*11:22:33:44G")},
                id="single_allele_pair_strip_trailing_letters",
            ),
            pytest.param(
                [("B*57:02:03:04N", "B*59:01:03"), ("B*56:01:01", "B*58:03:03:03N")],
                {},
                "B*56:01:01 - B*58:03:03:03",
                {("B*56:01:01", "B*58:03:03:03N")},
                id="find_best_without_frequencies",
            ),
            pytest.param(
                [("B*57:02:03:04N", "B*59:01:03"), ("B*56:01:01", "B*58:03:03:03N")],
                {
                    HLAProteinPair(
                        first_field_1="57",
                        first_field_2="02",
                        second_field_1="59",
                        second_field_2="01",
                    ): 15,
                    HLAProteinPair(
                        first_field_1="56",
                        first_field_2="01",
                        second_field_1="58",
                        second_field_2="03",
                    ): 150,
                },
                "B*56:01:01 - B*58:03:03:03",
                {("B*56:01:01", "B*58:03:03:03N")},
                id="best_with_frequencies_matches_best_without",
            ),
            pytest.param(
                [("B*57:02:03:04N", "B*59:01:03"), ("B*56:01:01", "B*58:03:03:03N")],
                {
                    HLAProteinPair(
                        first_field_1="57",
                        first_field_2="02",
                        second_field_1="59",
                        second_field_2="01",
                    ): 150,
                    HLAProteinPair(
                        first_field_1="56",
                        first_field_2="01",
                        second_field_1="58",
                        second_field_2="03",
                    ): 15,
                },
                "B*57:02:03:04 - B*59:01:03",
                {("B*57:02:03:04N", "B*59:01:03")},
                id="best_with_frequencies_overrides_best_without",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                ],
                {},
                "A*02 - A*03",
                {
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                },
                id="several_pairs_no_frequencies",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="02",
                        second_field_1="09",
                        second_field_2="10",
                    ): 150,
                },
                "A*02 - A*03",
                {
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                },
                id="frequencies_not_relevant",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="02",
                        first_field_2="237",
                        second_field_1="03",
                        second_field_2="05",
                    ): 150,
                },
                "A*02 - A*03",
                {
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24", "A*03:17"),
                    ("A*02:195", "A*03:23"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                },
                id="frequencies_do_not_affect_decision",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                    ("A*04:123", "A*22:33"),
                    ("A*04:123:22", "A*22:33:45:66N"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="02",
                        first_field_2="237",
                        second_field_1="03",
                        second_field_2="05",
                    ): 150,
                },
                "A*02 - A*03",
                {
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                },
                id="ambiguous_set_no_frequencies",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                    ("A*04:123", "A*22:33"),
                    ("A*04:123:22", "A*22:33:45:66N"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="02",
                        first_field_2="237",
                        second_field_1="03",
                        second_field_2="05",
                    ): 150,
                },
                "A*02 - A*03",
                {
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                },
                id="ambiguous_set_frequencies_agree_with_lowest_numbered",
            ),
            pytest.param(
                [
                    ("A*02:01", "A*03:01"),
                    ("A*02:237", "A*03:05"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20", "A*03:157"),
                    ("A*04:123", "A*22:33"),
                    ("A*04:123:22", "A*22:33:45:66N"),
                ],
                {
                    HLAProteinPair(
                        first_field_1="04",
                        first_field_2="123",
                        second_field_1="22",
                        second_field_2="33",
                    ): 150,
                },
                "A*04:123 - A*22:33",
                {
                    ("A*04:123", "A*22:33"),
                    ("A*04:123:22", "A*22:33:45:66N"),
                },
                id="ambiguous_set_frequencies_dictate_best_allele_choice",
            ),
        ],
    )
    def test_best_common_allele_pair_str(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        frequencies: dict[HLAProteinPair, int],
        expected_result: str,
        expected_unambiguous_set: set[tuple[str, str]],
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        result_pair_str: str
        result_unambiguous_set: set[tuple[str, str]]
        result_pair_str, result_unambiguous_set = ap.best_common_allele_pair_str(
            frequencies
        )
        assert result_pair_str == expected_result
        assert result_unambiguous_set == expected_unambiguous_set

    @pytest.mark.parametrize(
        "combined_standards, exp_alleles",
        [
            (
                [
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(
                            ("A*02:01:01G", "A*03:01:01G"),
                            ("A*02:01:52", "A*03:01:03"),
                            ("A*02:01:02", "A*03:01:12"),
                            ("A*02:01:36", "A*03:01:38"),
                            ("A*02:237", "A*03:05:01"),
                            ("A*02:26", "A*03:07"),
                            ("A*02:34", "A*03:08"),
                            ("A*02:90", "A*03:09"),
                            ("A*02:24:01", "A*03:17:01"),
                            ("A*02:195", "A*03:23:01"),
                            ("A*02:338", "A*03:95"),
                            ("A*02:35:01", "A*03:108"),
                            ("A*02:86", "A*03:123"),
                            ("A*02:20:01", "A*03:157"),
                        ),
                    )
                ],
                [
                    ("A*02:01:01G", "A*03:01:01G"),
                    ("A*02:01:02", "A*03:01:12"),
                    ("A*02:01:36", "A*03:01:38"),
                    ("A*02:01:52", "A*03:01:03"),
                    ("A*02:195", "A*03:23:01"),
                    ("A*02:20:01", "A*03:157"),
                    ("A*02:237", "A*03:05:01"),
                    ("A*02:24:01", "A*03:17:01"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:35:01", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:90", "A*03:09"),
                ],
            ),
            (
                [
                    HLACombinedStandard(
                        standard_bin=(1, 4, 5, 9),
                        possible_allele_pairs=(
                            ("A*11:01:01G", "A*26:01:01G"),
                            ("A*11:01:07", "A*26:01:17"),
                            ("A*11:19", "A*26:13"),
                            ("A*11:40", "A*66:01G"),
                        ),
                    )
                ],
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
            ),
            (
                [
                    HLACombinedStandard(
                        standard_bin=(1, 4, 5, 9),
                        possible_allele_pairs=(
                            ("A*11:01:01G", "A*26:01:01G"),
                            ("A*11:01:07", "A*26:01:17"),
                            ("A*11:19", "A*26:13"),
                            ("A*11:40", "A*66:01G"),
                        ),
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 5, 9),
                        possible_allele_pairs=(
                            ("A*22:33:44:55G", "A*01:02:03"),
                            ("A*24:25:26", "A*27:28:32"),
                            ("A*32:42", "A*113:110:02:13N"),
                        ),
                    ),
                ],
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                    ("A*22:33:44:55G", "A*01:02:03"),
                    ("A*24:25:26", "A*27:28:32"),
                    ("A*32:42", "A*113:110:02:13N"),
                ],
            ),
        ],
    )
    def test_get_allele_pairs(
        self,
        combined_standards: list[HLACombinedStandard],
        exp_alleles: list[tuple[str, str]],
    ):
        result_alleles = AllelePairs.get_allele_pairs(combined_standards)
        assert result_alleles.allele_pairs == exp_alleles

    @pytest.mark.parametrize(
        "raw_allele_pairs, exp_result",
        [
            pytest.param(
                [],
                [],
                id="empty_list",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                ],
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                ],
                id="single_element",
            ),
            pytest.param(
                [
                    ("A*11:01:01", "A*26:01:01"),
                    ("A*12:01:01", "A*26:01:01"),
                ],
                [
                    ("A*11:01:01", "A*26:01:01"),
                    ("A*12:01:01", "A*26:01:01"),
                ],
                id="two_elements_trivial_sort",
            ),
            pytest.param(
                [
                    ("A*12:01:01", "A*26:01:01"),
                    ("A*11:01:01", "A*26:01:01"),
                ],
                [
                    ("A*11:01:01", "A*26:01:01"),
                    ("A*12:01:01", "A*26:01:01"),
                ],
                id="two_elements_nontrivial_sort",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*25:01:01"),
                    ("A*11:01:01", "A*26:01:01"),
                ],
                [
                    ("A*11:01:01", "A*26:01:01"),
                    ("A*11:01:01G", "A*25:01:01"),
                ],
                id="two_elements_letter_vs_no_letter",
            ),
            pytest.param(
                [
                    ("A*11:01:01N", "A*25:01:01"),
                    ("A*11:01:01G", "A*26:01:01"),
                ],
                [
                    ("A*11:01:01G", "A*26:01:01"),
                    ("A*11:01:01N", "A*25:01:01"),
                ],
                id="two_elements_letter_tiebreak",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01N"),
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:40", "A*66:01G"),
                ],
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:01G", "A*26:01:01N"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:40", "A*66:01G"),
                ],
                id="typical_case",
            ),
        ],
    )
    def test_sort_pairs(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        exp_result: list[tuple[str, str]],
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.sort_pairs() == exp_result

    @pytest.mark.parametrize(
        "raw_allele_pairs, sorted, max_length, exp_stringification",
        [
            pytest.param(
                [
                    ("A*02:01:01G", "A*03:01:01G"),
                    ("A*02:01:52", "A*03:01:03"),
                    ("A*02:01:02", "A*03:01:12"),
                    ("A*02:01:36", "A*03:01:38"),
                    ("A*02:237", "A*03:05:01"),
                    ("A*02:26", "A*03:07"),
                    ("A*02:34", "A*03:08"),
                    ("A*02:90", "A*03:09"),
                    ("A*02:24:01", "A*03:17:01"),
                    ("A*02:195", "A*03:23:01"),
                    ("A*02:338", "A*03:95"),
                    ("A*02:35:01", "A*03:108"),
                    ("A*02:86", "A*03:123"),
                    ("A*02:20:01", "A*03:157"),
                ],
                False,
                3900,
                # NOTE: This is one string concatenated together
                (
                    "A*02:01:01G - A*03:01:01G;"
                    "A*02:01:52 - A*03:01:03;"
                    "A*02:01:02 - A*03:01:12;"
                    "A*02:01:36 - A*03:01:38;"
                    "A*02:237 - A*03:05:01;"
                    "A*02:26 - A*03:07;"
                    "A*02:34 - A*03:08;"
                    "A*02:90 - A*03:09;"
                    "A*02:24:01 - A*03:17:01;"
                    "A*02:195 - A*03:23:01;"
                    "A*02:338 - A*03:95;"
                    "A*02:35:01 - A*03:108;"
                    "A*02:86 - A*03:123;"
                    "A*02:20:01 - A*03:157"
                ),
                id="typical_case_no_truncation",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                False,
                3900,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "A*11:40 - A*66:01G"
                ),
                id="no_truncation_no_sorting",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                True,
                3900,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "A*11:40 - A*66:01G"
                ),
                id="no_truncation_trivial_sorting",
            ),
            pytest.param(
                [
                    ("A*11:19", "A*26:13"),
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:40", "A*66:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                ],
                True,
                3900,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "A*11:40 - A*66:01G"
                ),
                id="no_truncation_meaningful_sorting",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01N"),
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:40", "A*66:01G"),
                ],
                True,
                3900,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:01G - A*26:01:01N;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:40 - A*66:01G"
                ),
                id="no_truncation_sorting_with_letter_tiebreak",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                False,
                60,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "...TRUNCATED"
                ),
                id="with_truncation",
            ),
            pytest.param(
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                False,
                25,
                "A*11:01:01G - A*26:01:01G;...TRUNCATED",
                id="with_strong_truncation",
            ),
        ],
    )
    def test_stringify(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        sorted: bool,
        max_length: int,
        exp_stringification: str,
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.stringify(sorted, max_length) == exp_stringification

    @pytest.mark.parametrize(
        "raw_allele_pairs, allele_name, expected_result",
        [
            pytest.param(
                [("B*57:01", "B*59:03")],
                "B*57:01",
                True,
                id="single_pair_name_present_as_first_exact_match",
            ),
            pytest.param(
                [("B*57:01", "B*59:03")],
                "B*59:03",
                True,
                id="single_pair_name_present_as_second_exact_match",
            ),
            pytest.param(
                [("B*57:01:03", "B*59:03")],
                "B*57:01",
                True,
                id="single_pair_name_present_as_first_inexact_match",
            ),
            pytest.param(
                [("B*57:01:03", "B*59:03")],
                "B*57",
                True,
                id="single_pair_name_present_as_first_inexact_match_only_one_coordinate",
            ),
            pytest.param(
                [("B*57:01", "B*59:03:02:11G")],
                "B*59:03:02",
                True,
                id="single_pair_name_present_as_second_inexact_match",
            ),
            pytest.param(
                [("B*57:01", "B*59:03:02:11G")],
                "B*5",
                True,
                id="single_pair_name_present_as_second_inexact_match_only_one_coordinate",
            ),
            pytest.param(
                [("A*57:01", "A*59:03")],
                "A*57:01:04",
                False,
                id="single_pair_no_starts_with_match",
            ),
            pytest.param(
                [("A*57:01", "A*59:03")],
                "A*01:23:45",
                False,
                id="single_pair_no_match_at_all",
            ),
            pytest.param(
                [
                    ("C*11:22:33", "C*55:52:01:03G"),
                    ("C*11:24:01", "C*55:01:53:04"),
                    ("C*11:24:02", "C*55:01:54"),
                ],
                "C*55:01",
                True,
                id="typical_case_with_match",
            ),
            pytest.param(
                [
                    ("B*11:22:33", "B*57:01:01:03G"),
                    ("B*111:112:01", "B*55:02:88:89"),
                    ("B*52:25:52", "B*55:01:54"),
                ],
                "B*57",
                True,
                id="typical_case_with_match_only_one_coordinate",
            ),
            pytest.param(
                [
                    ("C*11:22:33", "C*55:52:01:03G"),
                    ("C*11:24:01", "C*55:01:53:04"),
                    ("C*11:24:02", "C*55:01:54"),
                ],
                "C*54:22",
                False,
                id="typical_case_without_match",
            ),
        ],
    )
    def test_contains_allele(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        allele_name: str,
        expected_result: bool,
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.contains_allele(allele_name) == expected_result


@pytest.fixture
def hla_sequence() -> HLASequence:
    return HLASequence(
        two=(2, 2, 1, 2),  # "CCTC"
        intron=(),
        three=(1, 4, 4, 2, 8),  # "AGGCT"
        name="dummy_seq",
        locus="B",
        num_sequences_used=1,
    )


class TestHLAInterpretation:
    @pytest.mark.parametrize(
        (
            "raw_matches, frequencies, expected_mismatch_count, "
            "expected_best_matches, expected_allele_pairs, "
            "expected_best_rep_ap, expected_common_ap_str, expected_best_rep_cs"
        ),
        [
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=15, sequence_base="A", standard_base="R"),
                            HLAMismatch(index=17, sequence_base="A", standard_base="R"),
                            HLAMismatch(index=88, sequence_base="G", standard_base="C"),
                            HLAMismatch(
                                index=111, sequence_base="G", standard_base="T"
                            ),
                            HLAMismatch(
                                index=205, sequence_base="R", standard_base="Y"
                            ),
                        ]
                    ),
                },
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="01",
                        second_field_1="02",
                        second_field_2="02",
                    ): 150,
                    HLAProteinPair(
                        first_field_1="22",
                        first_field_2="33",
                        second_field_1="22",
                        second_field_2="34",
                    ): 15,
                },
                5,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    )
                },
                {("A*01:01:01", "A*02:02:02")},
                ("A*01:01:01", "A*02:02:02"),
                "A*01:01:01 - A*02:02:02",
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                ),
                id="single_matching_combined_standard",
            ),
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=15, sequence_base="A", standard_base="R"),
                            HLAMismatch(index=17, sequence_base="A", standard_base="R"),
                            HLAMismatch(index=88, sequence_base="G", standard_base="C"),
                            HLAMismatch(
                                index=111, sequence_base="G", standard_base="T"
                            ),
                            HLAMismatch(
                                index=205, sequence_base="R", standard_base="Y"
                            ),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=22, sequence_base="R", standard_base="C"),
                            HLAMismatch(
                                index=222, sequence_base="A", standard_base="R"
                            ),
                        ],
                    ),
                },
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="01",
                        second_field_1="02",
                        second_field_2="02",
                    ): 150,
                    HLAProteinPair(
                        first_field_1="22",
                        first_field_2="33",
                        second_field_1="22",
                        second_field_2="34",
                    ): 15,
                },
                2,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    )
                },
                {("A*10:01:01", "A*20:02:02")},
                ("A*10:01:01", "A*20:02:02"),
                "A*10:01:01 - A*20:02:02",
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                ),
                id="two_matches_no_tie",
            ),
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=55, sequence_base="A", standard_base="G")
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:03"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=48, sequence_base="R", standard_base="C")
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=45, sequence_base="T", standard_base="C"),
                            HLAMismatch(index=57, sequence_base="R", standard_base="Y"),
                            HLAMismatch(
                                index=122, sequence_base="R", standard_base="G"
                            ),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(("A*10:01:10", "A*22:22:22"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=100, sequence_base="A", standard_base="T")
                        ]
                    ),
                },
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="01",
                        second_field_1="02",
                        second_field_2="02",
                    ): 150,
                    HLAProteinPair(
                        first_field_1="10",
                        first_field_2="01",
                        second_field_1="20",
                        second_field_2="02",
                    ): 1500,
                },
                1,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:03"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(("A*10:01:10", "A*22:22:22"),),
                    ),
                },
                {
                    ("A*01:01:01", "A*02:02:02"),
                    ("A*10:01:01", "A*20:02:03"),
                    ("A*10:01:10", "A*22:22:22"),
                },
                ("A*10:01:01", "A*20:02:03"),
                "A*10:01:01 - A*20:02:03",
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:01", "A*20:02:03"),),
                ),
                id="typical_case_single_element_unambiguous_set_of_allele_pairs",
            ),
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=55, sequence_base="A", standard_base="G")
                        ]
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=48, sequence_base="R", standard_base="C")
                        ]
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=45, sequence_base="T", standard_base="C"),
                            HLAMismatch(index=57, sequence_base="R", standard_base="Y"),
                            HLAMismatch(
                                index=122, sequence_base="R", standard_base="G"
                            ),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(
                            ("A*10:01:10", "A*20:01"),
                            ("A*10:01:10", "A*22:22:22"),
                        ),
                    ): HLAMatchDetails(
                        mismatches=[
                            HLAMismatch(index=100, sequence_base="A", standard_base="T")
                        ]
                    ),
                },
                {
                    HLAProteinPair(
                        first_field_1="01",
                        first_field_2="01",
                        second_field_1="02",
                        second_field_2="02",
                    ): 150,
                    HLAProteinPair(
                        first_field_1="10",
                        first_field_2="01",
                        second_field_1="20",
                        second_field_2="02",
                    ): 1500,
                },
                1,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(
                            ("A*10:01:10", "A*20:01"),
                            ("A*10:01:10", "A*22:22:22"),
                        ),
                    ),
                },
                {
                    ("A*01:01:01", "A*02:02:02"),
                    ("A*10:01:15", "A*20:02:03"),
                    ("A*10:01:10", "A*20:01"),
                    ("A*10:01:10", "A*22:22:22"),
                },
                ("A*10:01:10", "A*20:01"),
                "A*10:01 - A*20",
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("A*10:01:10", "A*20:01"),
                        ("A*10:01:10", "A*22:22:22"),
                    ),
                ),
                id="typical_case_nontrivial_unambiguous_set_of_allele_pairs",
            ),
        ],
    )
    def test_basic_methods(
        self,
        hla_sequence: HLASequence,
        frequencies: dict[HLAProteinPair, int],
        raw_matches: dict[HLACombinedStandard, HLAMatchDetails],
        expected_mismatch_count: int,
        expected_best_matches: set[HLACombinedStandard],
        expected_allele_pairs: set[tuple[str, str]],
        expected_best_rep_ap: tuple[str, str],
        expected_common_ap_str: str,
        expected_best_rep_cs: HLACombinedStandard,
    ):
        interp: HLAInterpretation = HLAInterpretation(
            hla_sequence=hla_sequence,
            matches=raw_matches,
            allele_frequencies=frequencies,
        )
        assert interp.locus == "B"
        assert interp.lowest_mismatch_count() == expected_mismatch_count
        assert interp.best_matches() == expected_best_matches
        assert (
            set(interp.best_matching_allele_pairs().allele_pairs)
            == expected_allele_pairs
        )

        best_rep_ap: tuple[str, str]
        common_ap_str: str
        best_rep_cs: HLACombinedStandard
        best_rep_ap, common_ap_str, best_rep_cs = interp.best_common_allele_pair()
        assert best_rep_ap == expected_best_rep_ap
        assert common_ap_str == expected_common_ap_str
        assert best_rep_cs == expected_best_rep_cs

    @pytest.mark.parametrize(
        "b5701_standards, expected_result",
        [
            pytest.param(
                None,
                None,
                id="no_standards_specified",
            ),
            pytest.param(
                [
                    HLAStandard(
                        allele="B*01:23:45",
                        two=(2, 2, 1, 2),
                        three=(1, 4, 4, 2, 8),
                    ),
                ],
                0,
                id="perfect_match",
            ),
            pytest.param(
                [
                    HLAStandard(
                        allele="B*01:23:45",
                        two=(2, 1, 1, 2),
                        three=(1, 4, 8, 2, 8),
                    ),
                ],
                2,
                id="some_unambiguous_mismatches",
            ),
            pytest.param(
                [
                    HLAStandard(
                        allele="B*01:23:45",
                        two=(2, 6, 1, 2),
                        three=(1, 4, 8, 5, 8),
                    ),
                ],
                2,
                id="some_ambiguous_mismatches",
            ),
            pytest.param(
                [
                    HLAStandard(
                        allele="B*01:23:45",
                        two=(2, 5, 3, 2),
                        three=(1, 4, 8, 10, 9),
                    ),
                ],
                2,
                id="ambiguous_and_unambiguous_mismatches",
            ),
            pytest.param(
                [
                    HLAStandard(
                        allele="B*01:23:45",
                        two=(2, 2, 1, 2),
                        three=(1, 4, 4, 2, 8),
                    ),
                    HLAStandard(
                        allele="B*01:23:46",
                        two=(2, 1, 1, 2),
                        three=(1, 4, 8, 2, 8),
                    ),
                    HLAStandard(
                        allele="B*01:23:4",
                        two=(2, 5, 3, 2),
                        three=(1, 4, 8, 10, 9),
                    ),
                ],
                0,
                id="minimum_distance_chosen",
            ),
        ],
    )
    def test_distance_from_b5701(
        self,
        hla_sequence: HLASequence,
        b5701_standards: Optional[list[HLAStandard]],
        expected_result: Optional[int],
    ):
        interp: HLAInterpretation = HLAInterpretation(
            hla_sequence=hla_sequence,
            matches={},
            allele_frequencies={},
            b5701_standards=b5701_standards,
        )
        assert interp.distance_from_b7501() == expected_result

    @pytest.mark.parametrize(
        "raw_matches, expected_result",
        [
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("B*01:01:01", "B*02:02:02"),),
                    ): HLAMatchDetails(mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 9, 4),
                        possible_allele_pairs=(
                            ("B*01:03:22", "B*02:07:05"),
                            ("B*01:03:25", "B*02:07:05"),
                        ),
                    ): HLAMatchDetails(mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 9, 4),
                        possible_allele_pairs=(
                            ("B*21:55:07:33N", "B*21:55:07:33N"),
                            ("B*21:55:07:33N", "B*21:55:42"),
                        ),
                    ): HLAMatchDetails(mismatches=[]),
                },
                False,
                id="typical_case_not_b5701",
            ),
            pytest.param(
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(
                            ("B*22:33:44", "B*56:02:51"),
                            ("B*57:01:04", "B*57:01:03"),
                        ),
                    ): HLAMatchDetails(mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 9, 4),
                        possible_allele_pairs=(
                            ("B*02:03:25", "B*02:03:27"),
                            ("B*13:31:13", "B*13:31:13"),
                        ),
                    ): HLAMatchDetails(mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 9, 4),
                        possible_allele_pairs=(
                            ("B*22:55:07:33N", "B*21:55:33"),
                            ("B*22:55:07:33N", "B*21:55:42"),
                        ),
                    ): HLAMatchDetails(mismatches=[]),
                },
                True,
                id="typical_case_is_b5701",
            ),
        ],
    )
    def test_is_b5701(
        self,
        hla_sequence: HLASequence,
        raw_matches: dict[HLACombinedStandard, HLAMatchDetails],
        expected_result: Optional[bool],
    ):
        interp: HLAInterpretation = HLAInterpretation(
            hla_sequence=hla_sequence,
            matches=raw_matches,
            allele_frequencies={},
            b5701_standards=None,
        )
        assert interp.is_b5701() == expected_result
