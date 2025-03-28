import numpy
import pytest

from easyhla.models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
)


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
        "index, observed_base, expected_bases, expected_str",
        [
            (55, "A", ["C"], "55:A->C"),
            (199, "C", ["A", "G"], "199:C->A/G"),
            (9, "T", ["A", "C", "G"], "9:T->A/C/G"),
        ],
    )
    def test_string(
        self,
        index: int,
        observed_base: str,
        expected_bases: list[str],
        expected_str: str,
    ):
        mismatch: HLAMismatch = HLAMismatch(
            index=index,
            observed_base=observed_base,
            expected_bases=expected_bases,
        )
        assert str(mismatch) == expected_str


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
        "raw_allele_pairs, frequencies, expected_result",
        [
            ([("B*57:01", "B*59:03")], {}, "B*57:01 - B*59:03"),
            ([("C*01:02:03", "C*03:04:05")], {}, "C*01:02:03 - C*03:04:05"),
            ([("C*01:02:03", "C*03:04")], {}, "C*01:02:03 - C*03:04"),
            ([("C*01:02", "C*03:04:05:06")], {}, "C*01:02 - C*03:04:05:06"),
            ([("C*01:02:03:04", "C*03:04:05:06")], {}, "C*01:02:03:04 - C*03:04:05:06"),
            (
                [("A*01:02:03:04N", "A*11:22:33:44G")],
                {},
                "A*01:02:03:04 - A*11:22:33:44",
            ),
            (
                [("B*57:02:03:04N", "B*59:01:03"), ("B*56:01:01", "B*58:03:03:03N")],
                {},
                "B*56:01:01 - B*58:03:03:03",
            ),
            (
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
            ),
            (
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
            ),
            (  # with no frequencies
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
            ),
            (  # has frequencies, but they don't cover the alleles we see
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
            ),
            (  # has frequencies but they don't affect the decision
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
            ),
            (  # ambiguous set, no frequencies
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
            ),
            (  # ambiguous set, has frequencies which happen to coincide with the lowest-numbering
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
            ),
            (  # ambiguous set, has frequencies which dictate the "best allele" choice
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
            ),
        ],
    )
    def test_best_common_allele_pair(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        frequencies: dict[HLAProteinPair, int],
        expected_result: str,
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        result = ap.best_common_allele_pair_str(frequencies)
        assert result == expected_result

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
        "raw_allele_pairs, max_length, exp_stringification",
        [
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
            ),
            (
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                3900,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "A*11:40 - A*66:01G"
                ),
            ),
            (
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                60,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "...TRUNCATED"
                ),
            ),
            (
                [
                    ("A*11:01:01G", "A*26:01:01G"),
                    ("A*11:01:07", "A*26:01:17"),
                    ("A*11:19", "A*26:13"),
                    ("A*11:40", "A*66:01G"),
                ],
                25,
                "A*11:01:01G - A*26:01:01G;...TRUNCATED",
            ),
        ],
    )
    def test_stringify(
        self,
        raw_allele_pairs: list[tuple[str, str]],
        max_length: int,
        exp_stringification: str,
    ):
        ap: AllelePairs = AllelePairs(allele_pairs=raw_allele_pairs)
        assert ap.stringify(max_length) == exp_stringification


@pytest.fixture
def hla_sequence() -> HLASequence:
    return HLASequence(
        two="CCTC",
        intron="",
        three="AGGCT",
        sequence=numpy.array([1, 4, 2, 4, 8]),
        name="dummy_seq",
        num_sequences_used=1,
    )


class TestHLAInterpretation:
    @pytest.mark.parametrize(
        "raw_matches, expected_mismatch_count, expected_best_matches, expected_allele_pairs",
        [
            (
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(mismatch_count=5, mismatches=[]),
                },
                5,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    )
                },
                {("A*01:01:01", "A*02:02:02")},
            ),
            (
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(mismatch_count=5, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ): HLAMatchDetails(mismatch_count=2, mismatches=[]),
                },
                2,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    )
                },
                {("A*10:01:01", "A*20:02:02")},
            ),
            (
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ): HLAMatchDetails(mismatch_count=3, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                },
                1,
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 4),
                        possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 4, 9, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ),
                    HLACombinedStandard(
                        standard_bin=(2, 4, 10, 2),
                        possible_allele_pairs=(("A*10:01:01", "A*20:02:02"),),
                    ),
                },
                {
                    ("A*01:01:01", "A*02:02:02"),
                    ("A*10:01:01", "A*20:02:02"),
                    ("A*10:01:01", "A*20:02:02"),
                },
            ),
        ],
    )
    def test_best_matches(
        self,
        hla_sequence: HLASequence,
        raw_matches: dict[HLACombinedStandard, HLAMatchDetails],
        expected_mismatch_count: int,
        expected_best_matches: set[HLACombinedStandard],
        expected_allele_pairs: set[tuple[str, str]],
    ):
        interp: HLAInterpretation = HLAInterpretation(
            hla_sequence=hla_sequence,
            matches=raw_matches,
        )
        assert interp.lowest_mismatch_count() == expected_mismatch_count
        assert interp.best_matches() == expected_best_matches
        assert (
            set(interp.best_matching_allele_pairs().allele_pairs)
            == expected_allele_pairs
        )
