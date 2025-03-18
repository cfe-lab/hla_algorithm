import pytest

from easyhla.models import (
    AllelePairs,
)


class TestModels:
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
                    "11|40,13|01": 4,
                    "11|01,12|01": 2,
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
                    "11|01,12|01": 1,
                    "13|01,12|44": 0,
                    "13|40,12|01": 10,
                },
                [("A*13:01", "A*12:44"), ("A*13:40", "A*12:01")],
            ),
        ],
    )
    def test_get_unambiguous_allele_set(
        self,
        raw_alleles: list[tuple[str, str]],
        frequencies: dict[str, int],
        exp_result: list[tuple[str, str]],
    ):
        allele_pairs = AllelePairs(allele_pairs=raw_alleles)
        result = allele_pairs.get_unambiguous_allele_pairs(frequencies)
        assert result == exp_result

    @pytest.mark.parametrize(
        "allele_pairs, frequencies, exp_result_clean, exp_homozygous, exp_ambiguous, exp_proteins_as_strings, exp_gene_coordinates",
        [
            (
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
                False,
                False,
                {
                    "02|01,03|01",
                    "02|237,03|05",
                    "02|26,03|07",
                    "02|34,03|08",
                    "02|90,03|09",
                    "02|24,03|17",
                    "02|195,03|23",
                    "02|338,03|95",
                    "02|35,03|108",
                    "02|86,03|123",
                    "02|20,03|157",
                },
                [
                    (["A*02", "01"], ["A*03", "01"]),
                    (["A*02", "01"], ["A*03", "01"]),
                    (["A*02", "01"], ["A*03", "01"]),
                    (["A*02", "01"], ["A*03", "01"]),
                    (["A*02", "237"], ["A*03", "05"]),
                    (["A*02", "26"], ["A*03", "07"]),
                    (["A*02", "34"], ["A*03", "08"]),
                    (["A*02", "90"], ["A*03", "09"]),
                    (["A*02", "24"], ["A*03", "17"]),
                    (["A*02", "195"], ["A*03", "23"]),
                    (["A*02", "338"], ["A*03", "95"]),
                    (["A*02", "35"], ["A*03", "108"]),
                    (["A*02", "86"], ["A*03", "123"]),
                    (["A*02", "20"], ["A*03", "157"]),
                ],
            ),
            (
                [
                    ("A*11:01", "A*26:01"),
                    ("A*11:01", "A*26:01"),
                    ("A*11:19", "A*26:13"),
                ],
                {},
                "A*11 - A*26",
                False,
                False,
                {"11|01,26|01", "11|19,26|13"},
                [
                    (["A*11", "01"], ["A*26", "01"]),
                    (["A*11", "01"], ["A*26", "01"]),
                    (["A*11", "19"], ["A*26", "13"]),
                ],
            ),
            (
                [
                    ("A*11:01", "A*26:01"),
                    ("A*11:40", "A*26:01G"),
                ],
                {},
                "A*11 - A*26",
                False,
                False,
                {"11|01,26|01", "11|40,26|01"},
                [
                    (["A*11", "01"], ["A*26", "01"]),
                    (["A*11", "40"], ["A*26", "01G"]),
                ],
            ),
            (
                [
                    ("A*11:01", "A*11:01"),
                    ("A*11:40", "A*11:01G"),
                ],
                {},
                "A*11 - A*11",
                True,
                False,
                {"11|01,11|01", "11|40,11|01"},
                [
                    (["A*11", "01"], ["A*11", "01"]),
                    (["A*11", "40"], ["A*11", "01G"]),
                ],
            ),
            (
                [
                    ("A*11:01", "A*12:01"),
                    ("A*11:01", "A*12:01"),
                    ("A*11:40", "A*13:01"),
                ],
                {"11|01,12|01": 15},
                "A*11:01 - A*12:01",
                False,
                True,
                {"11|01,12|01", "11|40,13|01"},
                [
                    (["A*11", "01"], ["A*12", "01"]),
                    (["A*11", "01"], ["A*12", "01"]),
                    (["A*11", "40"], ["A*13", "01"]),
                ],
            ),
        ],
    )
    def test_alleles(
        self,
        allele_pairs: list[tuple[str, str]],
        frequencies: dict[str, int],
        exp_result_clean: list[str],
        exp_homozygous: bool,
        exp_ambiguous: bool,
        exp_proteins_as_strings: set[str],
        exp_gene_coordinates: list[tuple[list[str], list[str]]],
    ):
        result = AllelePairs(allele_pairs=allele_pairs)

        assert result.is_ambiguous() == exp_ambiguous
        assert result.is_homozygous() == exp_homozygous
        assert result.best_common_allele_pair_str(frequencies) == exp_result_clean
        assert result.get_proteins_as_strings() == exp_proteins_as_strings
        assert result.get_paired_gene_coordinates() == exp_gene_coordinates

    @pytest.mark.parametrize(
        "allele_pairs, exp_result",
        [
            (
                [(f"A*{i:04}:02", f"B*{i:04}:01") for i in range(1000)],
                (
                    ";".join([f"A*{i:04}:02 - B*{i:04}:01" for i in range(178)])
                    + ";...TRUNCATED"
                ),
            ),
            (
                [(f"A*{i:04}:02", f"B*{i:04}:01") for i in range(100)],
                ";".join([f"A*{i:04}:02 - B*{i:04}:01" for i in range(100)]),
            ),
        ],
        ids=["Truncated Result", "Non-truncated Result"],
    )
    def test_alleles_stringify(
        self, allele_pairs: list[tuple[str, str]], exp_result: str
    ):
        result = AllelePairs(allele_pairs=allele_pairs)
        assert result.stringify() == exp_result
