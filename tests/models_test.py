import pytest
from datetime import datetime
from contextlib import nullcontext as does_not_raise
from typing import List, Optional, Dict, Tuple, Any

from src.easyhla.easyhla import EasyHLA, DATE_FORMAT
from src.easyhla.models import (
    HLAStandard,
    HLAStandardMatch,
    HLACombinedStandardResult,
    Alleles,
)


class TestModels:
    @pytest.mark.parametrize(
        "alleles, exp_result_clean, exp_homozygous, exp_ambiguous, exp_ambiguous_collection, exp_collection",
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
                "A*02 - A*03",
                False,
                False,
                {
                    "02|01,03|01": 0,
                    "02|237,03|05": 0,
                    "02|26,03|07": 0,
                    "02|34,03|08": 0,
                    "02|90,03|09": 0,
                    "02|24,03|17": 0,
                    "02|195,03|23": 0,
                    "02|338,03|95": 0,
                    "02|35,03|108": 0,
                    "02|86,03|123": 0,
                    "02|20,03|157": 0,
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
                "A*11 - A*26",
                False,
                False,
                {"11|01,26|01": 0, "11|19,26|13": 0},
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
                "A*11 - A*26",
                False,
                False,
                {"11|01,26|01": 0, "11|40,26|01": 0},
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
                "A*11 - A*11",
                True,
                False,
                {"11|01,11|01": 0, "11|40,11|01": 0},
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
                "A*11",  # Strictly speaking, this would be a failure since it should be a pair
                False,
                True,
                {"11|01,12|01": 0, "11|40,13|01": 0},
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
        alleles: List[Tuple[str, str]],
        exp_result_clean: List[str],
        exp_homozygous: bool,
        exp_ambiguous: bool,
        exp_ambiguous_collection: set[str],
        exp_collection: List[Tuple[List[str], List[str]]],
    ):
        result = Alleles(alleles=alleles)

        assert result.is_ambiguous() == exp_ambiguous
        assert result.is_homozygous() == exp_homozygous
        assert result.stringify_clean() == exp_result_clean
        assert result.get_ambiguous_collection() == exp_ambiguous_collection
        assert result.get_collection() == exp_collection

    @pytest.mark.parametrize(
        "alleles, exp_result",
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
    def test_alleles_stringify(self, alleles: List[Tuple[str, str]], exp_result: str):
        result = Alleles(alleles=alleles)
        assert result.stringify() == exp_result
