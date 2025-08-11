import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, cast
from unittest.mock import MagicMock, _Call

import numpy as np
import pytest
import yaml
from pytest_mock import MockerFixture

from hla_algorithm.hla_algorithm import HLA_LOCUS, HLAAlgorithm, LoadedStandards
from hla_algorithm.models import (
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)
from hla_algorithm.utils import GroupedAllele, HLARawStandard, StoredHLAStandards

# from .conftest import compare_ref_vs_test


HLA_STANDARDS: dict[HLA_LOCUS, HLARawStandard] = {
    "A": HLARawStandard(
        allele="A*01:01:01G",
        exon2=(
            "GCTCCCACTCCATGAGGTATTTCTTCACATCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCGCCGT"
            "GGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGCCAGAAGATGGAGCCGCGGGCG"
            "CCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCAGGAGACACGGAATATGAAGGCCCACTCACAGACTG"
            "ACCGAGCGAACCTGGGGACCCTGCGCGGCTACTACAACCAGAGCGAGGACG"
        ),
        exon3=(
            "GTTCTCACACCATCCAGATAATGTATGGCTGCGACGTGGGGCCGGACGGGCGCTTCCTCCGCGGGTACCGGCA"
            "GGACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCTTGGACCGCGGCGGACATGGCA"
            "GCTCAGATCACCAAGCGCAAGTGGGAGGCGGTCCATGCGGCGGAGCAGCGGAGAGTCTACCTGGAGGGCCGGT"
            "GCGTGGACGGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCACGG"
        ),
    ),
    "B": HLARawStandard(
        allele="B*07:02:01G",
        exon2=(
            "GCTCCCACTCCATGAGGTATTTCTACACCTCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCATCTCAGT"
            "GGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGAGAGGAGCCGCGGGCG"
            "CCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCGGAACACACAGATCTACAAGGCCCAGGCACAGACTG"
            "ACCGAGAGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCGAGGCCG"
        ),
        exon3=(
            "GGTCTCACACCCTCCAGAGCATGTACGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGCATGACCA"
            "GTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCGCCGCGGACACGGCG"
            "GCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGAGAGCCTACCTGGAGGGCGAGT"
            "GCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGACAAGCTGGAGCGCGCTG"
        ),
    ),
    "C": HLARawStandard(
        allele="C*01:02:01G",
        exon2=(
            "GCTCCCACTCCATGAAGTATTTCTTCACATCCGTGTCCCGGCCTGGCCGCGGAGAGCCCCGCTTCATCTCAGT"
            "GGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGTCCGAGAGGGGAGCCGCGGGCG"
            "CCGTGGGTGGAGCAGGAGGGGCCGGAGTATTGGGACCGGGAGACACAGAAGTACAAGCGCCAGGCACAGACTG"
            "ACCGAGTGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCGAGGCCG"
        ),
        exon3=(
            "GGTCTCACACCCTCCAGTGGATGTGTGGCTGCGACCTGGGGCCCGACGGGCGCCTCCTCCGCGGGTATGACCA"
            "GTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCGCCGCGGACACCGCG"
            "GCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGAGAGCCTACCTGGAGGGCACGT"
            "GCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGCTGCAGCGCGCGG"
        ),
    ),
}

HLA_FREQUENCIES: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = {
    "A": {
        HLAProteinPair(
            first_field_1="22",
            first_field_2="33",
            second_field_1="14",
            second_field_2="23",
        ): 1,
    },
    "B": {
        HLAProteinPair(
            first_field_1="57",
            first_field_2="01",
            second_field_1="57",
            second_field_2="03",
        ): 1,
    },
    "C": {
        HLAProteinPair(
            first_field_1="40",
            first_field_2="43",
            second_field_1="25",
            second_field_2="29",
        ): 1,
    },
}


@pytest.fixture(scope="module")
def hla_algorithm():
    standards: dict[HLA_LOCUS, dict[str, HLAStandard]] = {
        locus: {
            HLA_STANDARDS[locus].allele: HLAStandard.from_raw_standard(
                HLA_STANDARDS[locus]
            )
        }
        for locus in cast(tuple[HLA_LOCUS, HLA_LOCUS, HLA_LOCUS], ("A", "B", "C"))
    }
    dummy_loaded_standards: LoadedStandards = {
        "tag": "v0.1.0-dummy-test",
        "last_updated": datetime(2025, 5, 30, 12, 0, 0),
        "standards": standards,
    }
    return HLAAlgorithm(
        loaded_standards=dummy_loaded_standards,
        hla_frequencies=HLA_FREQUENCIES,
    )


@pytest.mark.parametrize(
    "sequence, matching_standards, thresholds, exp_result",
    [
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [0, 1, 5],
            [((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch"))],
            id="one_combo_all_matches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std1",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                )
            ],
            [0, 1, 2, 5],
            [((1, 2, 4, 4), 1, ("std1", "std1"))],
            id="one_combo_retained_regardless_of_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                )
            ],
            [0, 1, 3, 4, 5, 10],
            [((8, 4, 2, 1), 4, ("std_allmismatch", "std_allmismatch"))],
            id="only_combo_retained_regardless_of_threshold_more_mismatches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_twomatch",
                    two=(1, 4),
                    three=(2, 8),
                    mismatch=2,
                ),
            ],
            [0, 1, 2, 3, 5],
            [((1, 4, 2, 8), 2, ("std_twomatch", "std_twomatch"))],
            id="one_combo_two_mismatches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_onemismatch",
                    two=(1, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [0],
            [((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch"))],
            id="combo_with_mismatch_above_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_onemismatch",
                    two=(1, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [0],
            [
                (
                    (1, 4, 4, 8),
                    1,
                    ("std_onemismatch", "std_onemismatch"),
                ),
                (
                    (1, 6, 4, 8),
                    1,
                    ("std_allmatch", "std_onemismatch"),
                ),
                (
                    (1, 2, 4, 8),
                    0,
                    ("std_allmatch", "std_allmatch"),
                ),
            ],
            id="combo_with_mismatch_above_threshold_previous_bests_reported",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_onemismatch",
                    two=(1, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [1, 2, 5],
            [
                (
                    (1, 2, 4, 8),
                    0,
                    ("std_allmatch", "std_allmatch"),
                ),
                (
                    (1, 6, 4, 8),
                    1,
                    ("std_allmatch", "std_onemismatch"),
                ),
                (
                    (1, 4, 4, 8),
                    1,
                    ("std_onemismatch", "std_onemismatch"),
                ),
            ],
            id="several_combos_all_below_threshold",
        ),
        pytest.param(
            (9, 6, 4, 6),
            [
                HLAStandardMatch(
                    allele="std1",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std2",
                    two=(8, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [0, 1, 2],
            [
                (
                    (1, 2, 4, 4),
                    3,
                    ("std1", "std1"),
                ),
                (
                    (9, 6, 4, 12),
                    1,
                    ("std1", "std2"),
                ),
            ],
            id="first_above_threshold_second_below_rest_rejected",
        ),
        pytest.param(
            (9, 6, 4, 6),
            [
                HLAStandardMatch(
                    allele="std1",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std2",
                    two=(8, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [3, 4, 5],
            [
                (
                    (1, 2, 4, 4),
                    3,
                    ("std1", "std1"),
                ),
                (
                    (9, 6, 4, 12),
                    1,
                    ("std1", "std2"),
                ),
                (
                    (8, 4, 4, 8),
                    3,
                    ("std2", "std2"),
                ),
            ],
            id="all_combos_have_mismatches_below_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [0],
            [((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch"))],
            id="more_standards_only_first_one_below_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [0],
            [
                ((1, 2, 4, 4), 1, ("std_1mismatch", "std_1mismatch")),
                ((1, 2, 4, 12), 1, ("std_1mismatch", "std_allmatch")),
                ((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch")),
            ],
            id="more_standards_some_reported_early_late_ones_rejected",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [0],
            [
                ((8, 4, 2, 1), 4, ("std_allmismatch", "std_allmismatch")),
                ((9, 6, 6, 5), 4, ("std_1mismatch", "std_allmismatch")),
                ((1, 2, 4, 4), 1, ("std_1mismatch", "std_1mismatch")),
                ((1, 2, 4, 12), 1, ("std_1mismatch", "std_allmatch")),
                ((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch")),
            ],
            id="running_threshold_is_reduced_in_steps",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [4, 5, 10],
            [
                ((1, 2, 4, 8), 0, ("std_allmatch", "std_allmatch")),
                ((1, 2, 4, 12), 1, ("std_1mismatch", "std_allmatch")),
                ((1, 2, 4, 4), 1, ("std_1mismatch", "std_1mismatch")),
                ((9, 6, 6, 9), 4, ("std_allmatch", "std_allmismatch")),
                ((9, 6, 6, 5), 4, ("std_1mismatch", "std_allmismatch")),
                ((8, 4, 2, 1), 4, ("std_allmismatch", "std_allmismatch")),
            ],
            id="all_combos_below_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std1",
                    two=(2, 2),
                    three=(4, 4),
                    mismatch=2,
                ),
                HLAStandardMatch(
                    allele="std2",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std3",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std4",
                    two=(2, 2),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [2, 3, 10],
            [
                ((2, 2, 4, 4), 2, ("std1", "std1")),
                ((3, 2, 4, 12), 2, ("std1", "std2")),
                ((1, 2, 4, 8), 0, ("std2", "std2")),
                ((3, 2, 4, 4), 2, ("std1", "std3")),
                ((1, 2, 4, 12), 1, ("std2", "std3")),
                ((1, 2, 4, 4), 1, ("std3", "std3")),
                ((2, 2, 4, 12), 2, ("std1", "std4")),
                ((3, 2, 4, 8), 1, ("std2", "std4")),
                ((3, 2, 4, 12), 2, ("std3", "std4")),
                ((2, 2, 4, 8), 1, ("std4", "std4")),
            ],
            id="several_standards_produce_same_sequence",
        ),
    ],
)
def test_combine_standards_stepper(
    sequence: Sequence[int],
    matching_standards: list[HLAStandardMatch],
    thresholds: list[int],
    exp_result: list[tuple[tuple[int, ...], int, tuple[str, str]]],
):
    for threshold in thresholds:
        result: list[tuple[tuple[int, ...], int, tuple[str, str]]] = list(
            HLAAlgorithm.combine_standards_stepper(
                matching_stds=matching_standards,
                seq=sequence,
                mismatch_threshold=threshold,
            )
        )
        assert result == exp_result


@pytest.mark.parametrize(
    "sequence, matching_standards, thresholds, exp_result",
    [
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [None, 0, 1, 5],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
            },
            id="one_combo_all_matches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_twomatch",
                    two=(1, 4),
                    three=(2, 8),
                    mismatch=2,
                ),
            ],
            [None, 0, 1, 2, 3, 5],
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 2, 8),
                    possible_allele_pairs=(("std_twomatch", "std_twomatch"),),
                ): 2,
            },
            id="one_combo_two_mismatches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_allmatch2",
                    two=(1, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [None, 0],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
            },
            id="combo_with_mismatch_above_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_allmatch2",
                    two=(1, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [1, 2, 5],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(1, 6, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch2"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(1, 4, 4, 8),
                    possible_allele_pairs=(("std_allmatch2", "std_allmatch2"),),
                ): 1,
            },
            id="several_combos_all_below_threshold",
        ),
        pytest.param(
            (9, 6, 4, 6),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch2",
                    two=(8, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [None, 0, 1, 2],
            {
                HLACombinedStandard(
                    standard_bin=(9, 6, 4, 12),
                    possible_allele_pairs=(("std_1mismatch2", "std_allmatch"),),
                ): 1,
            },
            id="best_match_has_mismatch_others_rejected",
        ),
        #
        pytest.param(
            (9, 6, 4, 6),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch2",
                    two=(8, 4),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [3, 4, 5],
            {
                HLACombinedStandard(
                    standard_bin=(9, 6, 4, 12),
                    possible_allele_pairs=(("std_1mismatch2", "std_allmatch"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 3,
                HLACombinedStandard(
                    standard_bin=(8, 4, 4, 8),
                    possible_allele_pairs=(("std_1mismatch2", "std_1mismatch2"),),
                ): 3,
            },
            id="all_combos_have_mismatches_below_threshold",
        ),
        #
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                )
            ],
            [None, 0, 1, 2, 5],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                ): 1
            },
            id="one_combo_retained_regardless_of_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                )
            ],
            [None, 0, 1, 3, 4, 5, 10],
            {
                HLACombinedStandard(
                    standard_bin=(8, 4, 2, 1),
                    possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                ): 4,
            },
            id="only_combo_retained_regardless_of_threshold_more_mismatches",
        ),
        #
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [None, 0],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
            },
            id="several_combos_only_one_below_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [1, 2, 3],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 12),
                    possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                ): 1,
            },
            id="several_combos_only_ones_below_threshold_retained",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            [4, 5, 10],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 12),
                    possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(9, 6, 6, 9),
                    possible_allele_pairs=(("std_allmatch", "std_allmismatch"),),
                ): 4,
                HLACombinedStandard(
                    standard_bin=(9, 6, 6, 5),
                    possible_allele_pairs=(("std_1mismatch", "std_allmismatch"),),
                ): 4,
                HLACombinedStandard(
                    standard_bin=(8, 4, 2, 1),
                    possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                ): 4,
            },
            id="several_combos_all_below_threshold_retained",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [None, 0],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
            },
            id="standard_as_second_part_of_combo_worse_than_already_known_combo_skipped_only_best_retained",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [1, 2, 3],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 12),
                    possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                ): 1,
            },
            id="standard_as_second_part_of_combo_worse_than_already_known_combo_skipped_under_threshold_retained",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandardMatch(
                    allele="std1",
                    two=(2, 2),
                    three=(4, 4),
                    mismatch=2,
                ),
                HLAStandardMatch(
                    allele="std2",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std3",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std4",
                    two=(2, 2),
                    three=(4, 8),
                    mismatch=1,
                ),
            ],
            [2, 3, 10],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std2", "std2"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(2, 2, 4, 4),
                    possible_allele_pairs=(("std1", "std1"),),
                ): 2,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 4),
                    possible_allele_pairs=(("std3", "std3"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(2, 2, 4, 8),
                    possible_allele_pairs=(("std4", "std4"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(3, 2, 4, 12),
                    possible_allele_pairs=(
                        ("std1", "std2"),
                        ("std3", "std4"),
                    ),
                ): 2,
                HLACombinedStandard(
                    standard_bin=(3, 2, 4, 4),
                    possible_allele_pairs=(("std1", "std3"),),
                ): 2,
                HLACombinedStandard(
                    standard_bin=(2, 2, 4, 12),
                    possible_allele_pairs=(("std1", "std4"),),
                ): 2,
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 12),
                    possible_allele_pairs=(("std2", "std3"),),
                ): 1,
                HLACombinedStandard(
                    standard_bin=(3, 2, 4, 8),
                    possible_allele_pairs=(("std2", "std4"),),
                ): 1,
            },
            id="several_standards_produce_same_sequence",
        ),
    ],
)
def test_combine_standards(
    sequence: list[int],
    matching_standards: list[HLAStandardMatch],
    thresholds: list[Optional[int]],
    exp_result: dict[int, list[int]],
):
    for threshold in thresholds:
        result = HLAAlgorithm.combine_standards(
            matching_stds=matching_standards,
            seq=sequence,
            mismatch_threshold=threshold,
        )
        assert result == exp_result


@pytest.mark.parametrize(
    "std_bin, seq_bin, locuses, expected_result",
    [
        pytest.param(
            [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8],
            [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8],
            ["A", "B", "C"],
            [],
            id="no_mismatches",
        ),
        pytest.param(
            [1, 2, 4, 12, 1, 2, 5, 8, 1, 2, 13, 8],
            [1, 2, 4, 12, 1, 2, 5, 8, 1, 2, 13, 8],
            ["A", "B", "C"],
            [],
            id="no_mismatches_with_mixtures",
        ),
        pytest.param(
            [1, 2, 4, 8],
            [4, 2, 4, 8],
            ["A", "B", "C"],
            [HLAMismatch(index=1, observed_base="G", expected_base="A")],
            id="mismatch_at_beginning",
        ),
        pytest.param(
            [1, 2, 4, 8],
            [1, 2, 4, 1],
            ["A", "B", "C"],
            [HLAMismatch(index=4, observed_base="A", expected_base="T")],
            id="mismatch_at_end",
        ),
        pytest.param(
            [1, 2, 4, 8],
            [1, 4, 4, 8],
            ["A", "B", "C"],
            [HLAMismatch(index=2, observed_base="G", expected_base="C")],
            id="mismatch_in_middle",
        ),
        pytest.param(
            [1, 2, 4, 8],
            [5, 2, 4, 8],
            ["A", "B", "C"],
            [HLAMismatch(index=1, observed_base="R", expected_base="A")],
            id="mixture_seq_to_unambiguous_std_mismatch",
        ),
        pytest.param(
            [1, 2, 11, 8],
            [1, 2, 4, 8],
            ["A", "B", "C"],
            [HLAMismatch(index=3, observed_base="G", expected_base="H")],
            id="unambiguous_seq_to_mixture_std_mismatch",
        ),
        pytest.param(
            [1, 2, 4, 3],
            [1, 2, 4, 5],
            ["A", "B", "C"],
            [HLAMismatch(index=4, observed_base="R", expected_base="M")],
            id="mixture_seq_to_mixture_std_mismatch",
        ),
        pytest.param(
            [1] * 270 + [4] * 276,
            [1] * 200 + [4] + [1] * 69 + [4] * 276,
            ["A", "B", "C"],
            [HLAMismatch(index=201, observed_base="G", expected_base="A")],
            id="indexing_not_modified_before_position_270",
        ),
        pytest.param(
            [1] * 269 + [3] + [4] * 276,
            [1] * 270 + [4] * 276,
            ["A", "B", "C"],
            [HLAMismatch(index=270, observed_base="A", expected_base="M")],
            id="indexing_not_modified_at_position_270",
        ),
        pytest.param(
            [1] * 270 + [4] * 276,
            [1] * 270 + [14] + [4] * 275,
            ["A"],
            [HLAMismatch(index=512, observed_base="B", expected_base="G")],
            id="locus_a_indexing_modified_at_position_271",
        ),
        pytest.param(
            [1] * 270 + [14] + [4] * 275,
            [1] * 270 + [4] * 276,
            ["B", "C"],
            [HLAMismatch(index=271, observed_base="G", expected_base="B")],
            id="locus_b_c_indexing_not_modified_at_position_271",
        ),
        pytest.param(
            [1] * 270 + [4] * 276,
            [1] * 270 + [4] * 100 + [11] + [4] * 175,
            ["A"],
            [HLAMismatch(index=612, observed_base="H", expected_base="G")],
            id="locus_a_indexing_modified_after_position_270",
        ),
        pytest.param(
            [1] * 270 + [4] * 100 + [11] + [4] * 175,
            [1] * 270 + [4] * 276,
            ["B", "C"],
            [HLAMismatch(index=371, observed_base="G", expected_base="H")],
            id="locus_b_c_indexing_not_modified_after_position_270",
        ),
        pytest.param(
            [1] * 170 + [3] + [1] * 99 + [11] + [4] * 99 + [4] * 50 + [1] + [4] * 125,
            [1] * 270 + [4] * 100 + [4] * 50 + [11] + [4] * 125,
            ["A"],
            [
                HLAMismatch(index=171, observed_base="A", expected_base="M"),
                HLAMismatch(index=512, observed_base="G", expected_base="H"),
                HLAMismatch(index=662, observed_base="H", expected_base="A"),
            ],
            id="locus_b_c_several_mismatches",
        ),
        pytest.param(
            [1] * 170 + [3] + [1] * 99 + [11] + [4] * 99 + [4] * 50 + [1] + [4] * 125,
            [1] * 270 + [4] * 100 + [4] * 50 + [11] + [4] * 125,
            ["B", "C"],
            [
                HLAMismatch(index=171, observed_base="A", expected_base="M"),
                HLAMismatch(index=271, observed_base="G", expected_base="H"),
                HLAMismatch(index=421, observed_base="H", expected_base="A"),
            ],
            id="locus_b_c_several_mismatches",
        ),
    ],
)
def test_get_mismatches_good_cases(
    std_bin: Iterable[int],
    seq_bin: Iterable[int],
    locuses: Iterable[HLA_LOCUS],
    expected_result: list[HLAMismatch],
):
    for locus in locuses:
        result: list[HLAMismatch] = HLAAlgorithm.get_mismatches(
            tuple(std_bin), cast(Sequence[int], np.array(seq_bin)), locus
        )
        assert result == expected_result


@pytest.mark.parametrize(
    "std_bin, seq_bin, expected_error",
    [
        pytest.param(
            [],
            [],
            "standard must be non-trivial",
            id="empty_sequence_and_standard",
        ),
        pytest.param(
            [],
            [1, 2, 4],
            "standard must be non-trivial",
            id="empty_standard_nontrivial_sequence",
        ),
        pytest.param(
            [1],
            [1, 2, 4],
            "standard and sequence must be the same length",
            id="longer_sequence",
        ),
        pytest.param(
            [1] * 100,
            [1, 2, 4],
            "standard and sequence must be the same length",
            id="longer_standard",
        ),
    ],
)
def test_get_mismatches_errors(
    std_bin: Iterable[int],
    seq_bin: Iterable[int],
    expected_error: str,
):
    for locus in ["A", "B", "C"]:
        with pytest.raises(ValueError) as excinfo:
            HLAAlgorithm.get_mismatches(
                tuple(std_bin),
                cast(Sequence[int], np.array(seq_bin)),
                cast(HLA_LOCUS, locus),
            )
        assert expected_error in str(excinfo.value)


@pytest.mark.parametrize(
    "sequence, threshold, raw_standards, expected_interpretation",
    [
        pytest.param(
            HLASequence(
                two=(1, 2),
                intron=(),
                three=(4, 8),
                name="E1",
                locus="C",
                num_sequences_used=1,
            ),
            5,
            {
                "A": [],
                "B": [],
                "C": [
                    HLAStandard(
                        allele="std_allmatch",
                        two=(1, 2),
                        three=(4, 8),
                    ),
                    HLAStandard(
                        allele="std_1mismatch",
                        two=(1, 2),
                        three=(4, 4),
                    ),
                    HLAStandard(
                        allele="std_allmismatch",
                        two=(8, 4),
                        three=(2, 1),
                    ),
                ],
            },
            HLAInterpretation(
                hla_sequence=HLASequence(
                    two=(1, 2),
                    intron=(),
                    three=(4, 8),
                    name="E1",
                    locus="C",
                    num_sequences_used=1,
                ),
                matches={
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 12),
                        possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                    ): HLAMatchDetails(
                        mismatch_count=1,
                        mismatches=[
                            HLAMismatch(index=4, expected_base="K", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                    ): HLAMatchDetails(
                        mismatch_count=1,
                        mismatches=[
                            HLAMismatch(index=4, expected_base="G", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 9),
                        possible_allele_pairs=(("std_allmatch", "std_allmismatch"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="W", observed_base="A"),
                            HLAMismatch(index=2, expected_base="S", observed_base="C"),
                            HLAMismatch(index=3, expected_base="S", observed_base="G"),
                            HLAMismatch(index=4, expected_base="W", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 5),
                        possible_allele_pairs=(("std_1mismatch", "std_allmismatch"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="W", observed_base="A"),
                            HLAMismatch(index=2, expected_base="S", observed_base="C"),
                            HLAMismatch(index=3, expected_base="S", observed_base="G"),
                            HLAMismatch(index=4, expected_base="R", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(8, 4, 2, 1),
                        possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="T", observed_base="A"),
                            HLAMismatch(index=2, expected_base="G", observed_base="C"),
                            HLAMismatch(index=3, expected_base="C", observed_base="G"),
                            HLAMismatch(index=4, expected_base="A", observed_base="T"),
                        ],
                    ),
                },
                allele_frequencies=HLA_FREQUENCIES["C"],
            ),
            id="typical_case_non_b",
        ),
        pytest.param(
            HLASequence(
                two=(1, 2),
                intron=(),
                three=(4, 8),
                name="E1",
                locus="B",
                num_sequences_used=1,
            ),
            5,
            {
                "A": [],
                "B": [
                    HLAStandard(
                        allele="B*57:01:01G",
                        two=(1, 2),
                        three=(4, 8),
                    ),
                    HLAStandard(
                        allele="B*57:01:02",
                        two=(1, 2),
                        three=(4, 4),
                    ),
                    HLAStandard(
                        allele="B*57:01:03",
                        two=(8, 4),
                        three=(2, 1),
                    ),
                ],
                "C": [],
            },
            HLAInterpretation(
                hla_sequence=HLASequence(
                    two=(1, 2),
                    intron=(),
                    three=(4, 8),
                    name="E1",
                    locus="B",
                    num_sequences_used=1,
                ),
                matches={
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("B*57:01:01G", "B*57:01:01G"),),
                    ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 12),
                        possible_allele_pairs=(("B*57:01:01G", "B*57:01:02"),),
                    ): HLAMatchDetails(
                        mismatch_count=1,
                        mismatches=[
                            HLAMismatch(index=4, expected_base="K", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("B*57:01:02", "B*57:01:02"),),
                    ): HLAMatchDetails(
                        mismatch_count=1,
                        mismatches=[
                            HLAMismatch(index=4, expected_base="G", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 9),
                        possible_allele_pairs=(("B*57:01:01G", "B*57:01:03"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="W", observed_base="A"),
                            HLAMismatch(index=2, expected_base="S", observed_base="C"),
                            HLAMismatch(index=3, expected_base="S", observed_base="G"),
                            HLAMismatch(index=4, expected_base="W", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 5),
                        possible_allele_pairs=(("B*57:01:02", "B*57:01:03"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="W", observed_base="A"),
                            HLAMismatch(index=2, expected_base="S", observed_base="C"),
                            HLAMismatch(index=3, expected_base="S", observed_base="G"),
                            HLAMismatch(index=4, expected_base="R", observed_base="T"),
                        ],
                    ),
                    HLACombinedStandard(
                        standard_bin=(8, 4, 2, 1),
                        possible_allele_pairs=(("B*57:01:03", "B*57:01:03"),),
                    ): HLAMatchDetails(
                        mismatch_count=4,
                        mismatches=[
                            HLAMismatch(index=1, expected_base="T", observed_base="A"),
                            HLAMismatch(index=2, expected_base="G", observed_base="C"),
                            HLAMismatch(index=3, expected_base="C", observed_base="G"),
                            HLAMismatch(index=4, expected_base="A", observed_base="T"),
                        ],
                    ),
                },
                allele_frequencies=HLA_FREQUENCIES["B"],
                b5701_standards=[
                    HLAStandard(
                        allele="B*57:01:01G",
                        two=(1, 2),
                        three=(4, 8),
                    ),
                    HLAStandard(
                        allele="B*57:01:02",
                        two=(1, 2),
                        three=(4, 4),
                    ),
                    HLAStandard(
                        allele="B*57:01:03",
                        two=(8, 4),
                        three=(2, 1),
                    ),
                ],
            ),
            id="typical_case_b",
        ),
    ],
)
def test_interpret_good_cases(
    sequence: HLASequence,
    threshold: int,
    raw_standards: dict[HLA_LOCUS, list[HLAStandard]],
    expected_interpretation: HLAInterpretation,
    hla_algorithm: HLAAlgorithm,
    mocker: MockerFixture,
):
    # Replace the standards with the ones in the test.
    standards: dict[HLA_LOCUS, dict[str, HLAStandard]] = {
        "A": {},
        "B": {},
        "C": {},
    }
    for locus in ("A", "B", "C"):
        standards[locus] = {std.allele: std for std in raw_standards[locus]}
    hla_algorithm.hla_standards = standards

    # Spy on the internals to make sure they're called correctly.
    get_matching_standards_spy: MagicMock = mocker.spy(
        hla_algorithm, "get_matching_standards"
    )
    combine_standards_spy: MagicMock = mocker.spy(hla_algorithm, "combine_standards")
    get_mismatches_spy: MagicMock = mocker.spy(hla_algorithm, "get_mismatches")

    result: HLAInterpretation = hla_algorithm.interpret(sequence, threshold=threshold)
    assert result == expected_interpretation

    # Using assert_called_once_with doesn't work here as we feed it a generator
    # and the comparison fails; we have to manually convert the value to a list
    # to be able to compare them.
    get_matching_standards_spy.assert_called_once()
    gms_call_args: _Call = get_matching_standards_spy.call_args
    assert len(gms_call_args.args) == 2
    assert len(gms_call_args.kwargs) == 0
    assert gms_call_args.args[0] == sequence.sequence_for_interpretation
    assert list(gms_call_args.args[1]) == list(standards[sequence.locus].values())

    matching_standards: list[HLAStandardMatch] = get_matching_standards_spy.spy_return

    combine_standards_spy.assert_called_once_with(
        matching_standards,
        sequence.sequence_for_interpretation,
        mismatch_threshold=threshold,
    )
    all_combos: dict[HLACombinedStandard, int] = combine_standards_spy.spy_return

    get_mismatches_spy.assert_has_calls(
        [
            mocker.call(
                x.standard_bin, sequence.sequence_for_interpretation, sequence.locus
            )
            for x in all_combos.keys()
        ],
        any_order=False,
    )


@pytest.mark.parametrize(
    "sequence, threshold, raw_standards",
    [
        pytest.param(
            HLASequence(
                two=(1, 2, 4, 8, 10, 2),
                intron=(),
                three=(4, 8, 5, 7, 11, 1),
                name="E1",
                locus="C",
                num_sequences_used=1,
            ),
            5,
            {
                "A": [],
                "B": [],
                "C": [
                    HLAStandard(
                        allele="std_1",
                        two=(2, 4, 8, 1, 10, 2),
                        three=(8, 1, 5, 7, 11, 1),
                    ),
                    HLAStandard(
                        allele="std_2",
                        two=(8, 4, 2, 1, 10, 2),
                        three=(4, 8, 10, 11, 4, 1),
                    ),
                    HLAStandard(
                        allele="std_3",
                        two=(1, 2, 4, 4, 5, 8),
                        three=(8, 8, 5, 8, 11, 4),
                    ),
                ],
            },
            id="no_matching_standards",
        ),
    ],
)
def test_interpret_error_cases(
    sequence: HLASequence,
    threshold: int,
    raw_standards: dict[HLA_LOCUS, list[HLAStandard]],
    hla_algorithm: HLAAlgorithm,
    mocker: MockerFixture,
):
    # Replace the standards with the ones in the test.
    for locus in ("A", "B", "C"):
        hla_algorithm.hla_standards[locus] = {std.allele: std for std in raw_standards[locus]}

    # Spy on the internals to make sure they're called correctly.
    get_matching_standards_spy: MagicMock = mocker.spy(
        hla_algorithm, "get_matching_standards"
    )
    combine_standards_spy: MagicMock = mocker.spy(hla_algorithm, "combine_standards")
    get_mismatches_spy: MagicMock = mocker.spy(hla_algorithm, "get_mismatches")

    with pytest.raises(HLAAlgorithm.NoMatchingStandards):
        hla_algorithm.interpret(sequence, threshold=threshold)

    # Using assert_called_once_with doesn't work here as we feed it a generator
    # and the comparison fails; we have to manually convert the value to a list
    # to be able to compare them.
    get_matching_standards_spy.assert_called_once()
    gms_call_args: _Call = get_matching_standards_spy.call_args
    assert len(gms_call_args.args) == 2
    assert len(gms_call_args.kwargs) == 0
    assert gms_call_args.args[0] == sequence.sequence_for_interpretation
    assert list(gms_call_args.args[1]) == list(
        hla_algorithm.hla_standards[sequence.locus].values()
    )

    combine_standards_spy.assert_not_called()
    get_mismatches_spy.assert_not_called()


READ_HLA_STANDARDS_TYPICAL_CASE_INPUT: dict[HLA_LOCUS, list[GroupedAllele]] = {
    "A": [
        GroupedAllele(
            exon2="G" * 270,
            exon3="A" * 276,
            alleles=["A*01:23:45:67N"],
        ),
        GroupedAllele(
            alleles=["A*55:66:77", "A*72:01:02"],
            exon2="GT" * 135,
            exon3="AC" * 138,
        ),
    ],
    "B": [
        GroupedAllele(
            exon2="A" * 270,
            exon3="T" * 276,
            alleles=["B*01:23:45:67N", "B*01:24:44"],
        ),
        GroupedAllele(
            alleles=["B*57:01:02"],
            exon2="C" * 270,
            exon3="GT" * 138,
        ),
        GroupedAllele(
            alleles=["B*100:101:111"],
            exon2="AT" * 135,
            exon3="G" * 276,
        ),
    ],
    "C": [
        GroupedAllele(
            alleles=["C*101:102:103", "C*105:106:107:108N"],
            exon2="T" * 270,
            exon3="A" * 276,
        ),
        GroupedAllele(
            alleles=["C*77:01:03"],
            exon2="TG" * 135,
            exon3="CA" * 138,
        ),
    ],
}


READ_HLA_STANDARDS_TYPICAL_CASE_OUTPUT: dict[HLA_LOCUS, dict[str, HLAStandard]] = {
    "A": {
        "A*01:23:45:67N": HLAStandard(
            allele="A*01:23:45:67N",
            two=(4,) * 270,
            three=(1,) * 276,
        ),
        "A*55:66:77G": HLAStandard(
            allele="A*55:66:77G",
            two=(4, 8) * 135,
            three=(1, 2) * 138,
        ),
    },
    "B": {
        "B*01:23:45G": HLAStandard(
            allele="B*01:23:45G",
            two=(1,) * 270,
            three=(8,) * 276,
        ),
        "B*57:01:02": HLAStandard(
            allele="B*57:01:02",
            two=(2,) * 270,
            three=(4, 8) * 138,
        ),
        "B*100:101:111": HLAStandard(
            allele="B*100:101:111",
            two=(1, 8) * 135,
            three=(4,) * 276,
        ),
    },
    "C": {
        "C*101:102:103G": HLAStandard(
            allele="C*101:102:103G",
            two=(8,) * 270,
            three=(1,) * 276,
        ),
        "C*77:01:03": HLAStandard(
            allele="C*77:01:03",
            two=(8, 4) * 135,
            three=(2, 1) * 138,
        ),
    },
}


@pytest.mark.parametrize(
    "raw_standards, raw_expected_result",
    [
        pytest.param(
            {"A": [], "B": [], "C": []},
            {"A": {}, "B": {}, "C": {}},
            id="empty_file",
        ),
        pytest.param(
            {
                "A": [
                    GroupedAllele(
                        exon2="A" * 270,
                        exon3="T" * 276,
                        alleles=["A*01:23:45:67N"],
                    )
                ],
                "B": [],
                "C": [],
            },
            {
                "A": {
                    "A*01:23:45:67N": HLAStandard(
                        allele="A*01:23:45:67N",
                        two=(1,) * 270,
                        three=(8,) * 276,
                    ),
                },
                "B": {},
                "C": {},
            },
            id="single_entry",
        ),
        pytest.param(
            {
                "A": [],
                "B": [
                    GroupedAllele(
                        exon2="A" * 270,
                        exon3="T" * 276,
                        alleles=["B*01:23:45:67N", "B*01:24:44"],
                    ),
                    GroupedAllele(
                        alleles=["B*57:01:02"],
                        exon2="C" * 270,
                        exon3="GT" * 138,
                    ),
                    GroupedAllele(
                        alleles=["B*100:101:111"],
                        exon2="AT" * 135,
                        exon3="G" * 276,
                    ),
                ],
                "C": [],
            },
            {
                "A": {},
                "B": {
                    "B*01:23:45G": HLAStandard(
                        allele="B*01:23:45G",
                        two=(1,) * 270,
                        three=(8,) * 276,
                    ),
                    "B*57:01:02": HLAStandard(
                        allele="B*57:01:02",
                        two=(2,) * 270,
                        three=(4, 8) * 138,
                    ),
                    "B*100:101:111": HLAStandard(
                        allele="B*100:101:111",
                        two=(1, 8) * 135,
                        three=(4,) * 276,
                    ),
                },
                "C": {},
            },
            id="multiple_entries",
        ),
        pytest.param(
            READ_HLA_STANDARDS_TYPICAL_CASE_INPUT,
            READ_HLA_STANDARDS_TYPICAL_CASE_OUTPUT,
            id="typical_case",
        ),
    ],
)
def test_read_hla_standards(
    raw_standards: dict[HLA_LOCUS, list[GroupedAllele]],
    raw_expected_result: dict[HLA_LOCUS, dict[str, HLAStandard]],
    tmp_path: Path,
    mocker: MockerFixture,
):
    # Convert the raw expected results into a LoadedStandards:
    expected_result: LoadedStandards = {
        "tag": "0.1.0-dummy-test",
        "last_updated": datetime(2025, 5, 30, 12, 0, 0),
        "standards": raw_expected_result,
    }
    # Build a YAML string from the raw standards:
    stored_standards: StoredHLAStandards = StoredHLAStandards(
        tag="0.1.0-dummy-test",
        commit_hash="foobar",
        last_updated=datetime(2025, 5, 30, 12, 0, 0),
        standards=raw_standards,
    )
    standards_file_str: str = yaml.safe_dump(stored_standards.model_dump())
    read_result: LoadedStandards = HLAAlgorithm.read_hla_standards(
        StringIO(standards_file_str)
    )
    assert read_result == expected_result

    # Also try reading it from a file.
    p = tmp_path / "hla_standards.yaml"
    p.write_text(standards_file_str)
    dirname_return_mock: MagicMock = mocker.MagicMock()
    mocker.patch.object(os.path, "dirname", return_value=dirname_return_mock)
    mocker.patch.object(os.path, "join", return_value=str(p))
    load_result: LoadedStandards = HLAAlgorithm.load_default_hla_standards()
    assert load_result == expected_result


READ_HLA_FREQUENCIES_TYPICAL_CASE_INPUT: list[str] = [
    "14:23,22:33,57:01,57:03,25:29,40:43",
    "17:34,88:82,52:02,56:11,19:82,19:82",
    "54:32,98:76,57:01,57:03,11:11,22:22",
    "14:23,deprecated,57:01,57:03,25:29,40:43",
    "54:32,98:76,57:01,57:03,19:82,19:82",
    "54:32,98:74,57:02,57:03,11:11,22:22",
    "54:32,98:76,57:01,57:03,19:82,unmapped",
    "54:32,98:74,unmapped,deprecated,11:11,22:22",
]

READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = {
    "A": {
        HLAProteinPair(
            first_field_1="14",
            first_field_2="23",
            second_field_1="22",
            second_field_2="33",
        ): 1,
        HLAProteinPair(
            first_field_1="17",
            first_field_2="34",
            second_field_1="88",
            second_field_2="82",
        ): 1,
        HLAProteinPair(
            first_field_1="54",
            first_field_2="32",
            second_field_1="98",
            second_field_2="76",
        ): 3,
        HLAProteinPair(
            first_field_1="54",
            first_field_2="32",
            second_field_1="98",
            second_field_2="74",
        ): 2,
    },
    "B": {
        HLAProteinPair(
            first_field_1="57",
            first_field_2="01",
            second_field_1="57",
            second_field_2="03",
        ): 5,
        HLAProteinPair(
            first_field_1="52",
            first_field_2="02",
            second_field_1="56",
            second_field_2="11",
        ): 1,
        HLAProteinPair(
            first_field_1="57",
            first_field_2="02",
            second_field_1="57",
            second_field_2="03",
        ): 1,
    },
    "C": {
        HLAProteinPair(
            first_field_1="25",
            first_field_2="29",
            second_field_1="40",
            second_field_2="43",
        ): 2,
        HLAProteinPair(
            first_field_1="19",
            first_field_2="82",
            second_field_1="19",
            second_field_2="82",
        ): 2,
        HLAProteinPair(
            first_field_1="11",
            first_field_2="11",
            second_field_1="22",
            second_field_2="22",
        ): 3,
    },
}


@pytest.mark.parametrize(
    "raw_hlas_observed, expected_locus_a, expected_locus_b, expected_locus_c",
    [
        pytest.param(
            ["14:23,22:33,57:01,57:03,25:29,40:43"],
            {
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="23",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="40",
                    second_field_2="43",
                ): 1,
            },
            id="single_row",
        ),
        pytest.param(
            ["14:23,unmapped,57:01,57:03,25:29,40:43"],
            {},
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="40",
                    second_field_2="43",
                ): 1,
            },
            id="single_row_a_non_allele",
        ),
        pytest.param(
            ["14:23,22:33,deprecated,57:03,25:29,40:43"],
            {
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="23",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
            },
            {},
            {
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="40",
                    second_field_2="43",
                ): 1,
            },
            id="single_row_b_non_allele",
        ),
        pytest.param(
            ["14:23,22:33,57:01,57:03,deprecated,unmapped"],
            {
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="23",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 1,
            },
            {},
            id="single_row_c_non_allele",
        ),
        pytest.param(
            ["unmapped,deprecated,57:01,unmapped,deprecated,40:43"],
            {},
            {},
            {},
            id="single_row_all_non_alleles",
        ),
        pytest.param(
            [
                "14:23,22:33,57:01,57:03,25:29,40:43",
                "14:34,22:33,57:01,56:11,25:29,51:50",
            ],
            {
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="23",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="34",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 1,
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="56",
                    second_field_2="11",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="40",
                    second_field_2="43",
                ): 1,
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="51",
                    second_field_2="50",
                ): 1,
            },
            id="partial_matches_distinguished",
        ),
        pytest.param(
            [
                "14:23,22:33,57:01,57:03,25:29,40:43",
                "17:34,88:82,52:02,56:11,19:82,19:82",
                "54:32,98:76,57:01,57:03,11:11,22:22",
            ],
            {
                HLAProteinPair(
                    first_field_1="14",
                    first_field_2="23",
                    second_field_1="22",
                    second_field_2="33",
                ): 1,
                HLAProteinPair(
                    first_field_1="17",
                    first_field_2="34",
                    second_field_1="88",
                    second_field_2="82",
                ): 1,
                HLAProteinPair(
                    first_field_1="54",
                    first_field_2="32",
                    second_field_1="98",
                    second_field_2="76",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 2,
                HLAProteinPair(
                    first_field_1="52",
                    first_field_2="02",
                    second_field_1="56",
                    second_field_2="11",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="25",
                    first_field_2="29",
                    second_field_1="40",
                    second_field_2="43",
                ): 1,
                HLAProteinPair(
                    first_field_1="19",
                    first_field_2="82",
                    second_field_1="19",
                    second_field_2="82",
                ): 1,
                HLAProteinPair(
                    first_field_1="11",
                    first_field_2="11",
                    second_field_1="22",
                    second_field_2="22",
                ): 1,
            },
            id="multiple_rows_locus_b_greater_than_one",
        ),
        pytest.param(
            READ_HLA_FREQUENCIES_TYPICAL_CASE_INPUT,
            READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT["A"],
            READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT["B"],
            READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT["C"],
            id="typical_case",
        ),
    ],
)
def test_read_hla_frequencies(
    raw_hlas_observed: list[str],
    expected_locus_a: dict[HLAProteinPair, int],
    expected_locus_b: dict[HLAProteinPair, int],
    expected_locus_c: dict[HLAProteinPair, int],
    tmp_path: Path,
    mocker: MockerFixture,
):
    frequencies_str: str = (
        "a_first,a_second,b_first,b_second,c_first,c_second\n"
        + "\n".join(raw_hlas_observed)
        + "\n"
    )
    expected_results: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = {
        "A": expected_locus_a,
        "B": expected_locus_b,
        "C": expected_locus_c,
    }
    result: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = HLAAlgorithm.read_hla_frequencies(
        StringIO(frequencies_str)
    )
    assert result == expected_results

    # Now try loading these from a file.
    p = tmp_path / "hla_frequencies.csv"
    p.write_text(frequencies_str)
    dirname_return_mock: MagicMock = mocker.MagicMock()
    mocker.patch.object(os.path, "dirname", return_value=dirname_return_mock)
    mocker.patch.object(os.path, "join", return_value=str(p))
    load_result: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = (
        HLAAlgorithm.load_default_hla_frequencies()
    )
    assert load_result == expected_results


@pytest.fixture
def fake_loaded_standards(mocker: MockerFixture) -> LoadedStandards:
    return {
        "tag": "0.1.0-dummy-test",
        "last_updated": datetime(2025, 5, 30, 12, 0, 0),
        "standards": mocker.MagicMock(),
    }


def test_init_no_defaults(
    fake_loaded_standards: LoadedStandards, mocker: MockerFixture
):
    fake_frequencies: MagicMock = mocker.MagicMock()

    hla_algorithm: HLAAlgorithm = HLAAlgorithm(fake_loaded_standards, fake_frequencies)
    assert hla_algorithm.tag == fake_loaded_standards["tag"]
    assert hla_algorithm.last_updated == fake_loaded_standards["last_updated"]
    assert hla_algorithm.hla_standards == fake_loaded_standards["standards"]
    assert hla_algorithm.hla_frequencies == fake_frequencies


def test_init_all_defaults(
    fake_loaded_standards: LoadedStandards, mocker: MockerFixture
):
    fake_frequencies: MagicMock = mocker.MagicMock()

    _: MagicMock = mocker.patch.object(
        HLAAlgorithm, "load_default_hla_standards", return_value=fake_loaded_standards
    )
    __: MagicMock = mocker.patch.object(
        HLAAlgorithm, "load_default_hla_frequencies", return_value=fake_frequencies
    )

    hla_algorithm: HLAAlgorithm = HLAAlgorithm()
    assert hla_algorithm.tag == fake_loaded_standards["tag"]
    assert hla_algorithm.last_updated == fake_loaded_standards["last_updated"]
    assert hla_algorithm.hla_standards == fake_loaded_standards["standards"]
    assert hla_algorithm.hla_frequencies == fake_frequencies


@pytest.fixture
def fake_stored_standards() -> StoredHLAStandards:
    return StoredHLAStandards(
        tag="0.1.0-dummy-test",
        commit_hash="foobar",
        last_updated=datetime(2025, 6, 2, 12, 0, 0),
        standards=READ_HLA_STANDARDS_TYPICAL_CASE_INPUT,
    )


def test_use_config_no_defaults(
    fake_stored_standards: StoredHLAStandards, tmp_path: Path
):
    standards_path: Path = tmp_path / "hla_standards.yaml"
    standards_path.write_text(yaml.safe_dump(fake_stored_standards.model_dump()))

    fake_frequencies_str: str = (
        "a_first,a_second,b_first,b_second,c_first,c_second\n"
        + "\n".join(READ_HLA_FREQUENCIES_TYPICAL_CASE_INPUT)
        + "\n"
    )
    freq_path: Path = tmp_path / "hla_frequencies.csv"
    freq_path.write_text(fake_frequencies_str)

    hla_algorithm: HLAAlgorithm = HLAAlgorithm.use_config(
        os.fspath(standards_path), os.fspath(freq_path)
    )
    assert hla_algorithm.tag == fake_stored_standards.tag
    assert hla_algorithm.last_updated == fake_stored_standards.last_updated
    assert hla_algorithm.hla_standards == READ_HLA_STANDARDS_TYPICAL_CASE_OUTPUT
    assert hla_algorithm.hla_frequencies == READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT


def test_use_config_all_defaults(
    fake_stored_standards: StoredHLAStandards, tmp_path: Path, mocker: MockerFixture
):
    standards_path: Path = tmp_path / "hla_standards.yaml"
    standards_path.write_text(yaml.safe_dump(fake_stored_standards.model_dump()))

    fake_frequencies_str: str = (
        "a_first,a_second,b_first,b_second,c_first,c_second\n"
        + "\n".join(READ_HLA_FREQUENCIES_TYPICAL_CASE_INPUT)
        + "\n"
    )

    freq_path: Path = tmp_path / "hla_frequencies.csv"
    freq_path.write_text(fake_frequencies_str)

    mocker.patch.object(
        os.path, "join", side_effect=[str(standards_path), str(freq_path)]
    )

    hla_algorithm: HLAAlgorithm = HLAAlgorithm.use_config()
    assert hla_algorithm.tag == fake_stored_standards.tag
    assert hla_algorithm.last_updated == fake_stored_standards.last_updated
    assert hla_algorithm.hla_standards == READ_HLA_STANDARDS_TYPICAL_CASE_OUTPUT
    assert hla_algorithm.hla_frequencies == READ_HLA_FREQUENCIES_TYPICAL_CASE_OUTPUT


@pytest.mark.parametrize(
    "sequence, hla_stds, mismatch_threshold, exp_result",
    [
        #
        pytest.param(
            (1, 2, 4, 8),
            [HLAStandard(allele="std_allmismatch", two=(1, 2), three=(4, 8))],
            5,
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                )
            ],
            id="one_standard_no_mismatches",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [HLAStandard(allele="std_allmismatch", two=(1, 2), three=(4, 4))],
            5,
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                )
            ],
            id="one_standard_one_mismatch",
        ),
        pytest.param(
            (1, 3, 4, 8),
            [HLAStandard(allele="std_mixturematch", two=(1, 2), three=(4, 8))],
            5,
            [
                HLAStandardMatch(
                    allele="std_mixturematch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                )
            ],
            id="mixture_match",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [HLAStandard(allele="std_allmismatch", two=(8, 4), three=(2, 1))],
            5,
            [
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                )
            ],
            id="one_standard_all_mismatch",
        ),
        pytest.param(
            (1, 2, 4, 8, 3, 5, 7, 9),
            [
                HLAStandard(
                    allele="std_mismatch_over_threshold",
                    two=(1, 2, 8, 4),
                    three=(4, 8, 8, 1),
                )
            ],
            5,
            [],
            id="one_standard_mismatch_above_threshold",
        ),
        pytest.param(
            (1, 2, 4, 8),
            [
                HLAStandard(allele="std_allmatch", two=(1, 2), three=(4, 8)),
                HLAStandard(allele="std_1mismatch", two=(1, 2), three=(4, 4)),
                HLAStandard(allele="std_allmismatch", two=(8, 4), three=(2, 1)),
            ],
            5,
            [
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_1mismatch",
                    two=(1, 2),
                    three=(4, 4),
                    mismatch=1,
                ),
                HLAStandardMatch(
                    allele="std_allmismatch",
                    two=(8, 4),
                    three=(2, 1),
                    mismatch=4,
                ),
            ],
            id="several_standards_below_threshold",
        ),
        pytest.param(
            (1, 3, 4, 8, 2, 5, 4, 1),
            [
                HLAStandard(
                    allele="std_mixturematch",
                    two=(1, 2, 4, 8),
                    three=(2, 1, 4, 1),
                ),
                HLAStandard(
                    allele="std_2mismatch",
                    two=(1, 4, 4, 4),
                    three=(2, 4, 4, 1),
                ),
                HLAStandard(
                    allele="std_allmismatch",
                    two=(8, 4, 2, 1),
                    three=(1, 8, 8, 8),
                ),
                HLAStandard(
                    allele="std_4mismatch",
                    two=(8, 4, 2, 1),
                    three=(2, 1, 4, 1),
                ),
            ],
            5,
            [
                HLAStandardMatch(
                    allele="std_mixturematch",
                    two=(1, 2, 4, 8),
                    three=(2, 1, 4, 1),
                    mismatch=0,
                ),
                HLAStandardMatch(
                    allele="std_2mismatch",
                    two=(1, 4, 4, 4),
                    three=(2, 4, 4, 1),
                    mismatch=2,
                ),
                HLAStandardMatch(
                    allele="std_4mismatch",
                    two=(8, 4, 2, 1),
                    three=(2, 1, 4, 1),
                    mismatch=4,
                ),
            ],
            id="typical_case",
        ),
    ],
)
def test_get_matching_standards(
    sequence: np.ndarray,
    hla_stds: Iterable[HLAStandard],
    mismatch_threshold: int,
    exp_result: list[HLAStandardMatch],
):
    result: list[HLAStandardMatch] = HLAAlgorithm.get_matching_standards(
        seq=cast(Sequence[int], sequence),
        hla_stds=hla_stds,
        mismatch_threshold=mismatch_threshold,
    )
    print(result)
    assert result == exp_result


# @pytest.mark.parametrize("easyhla", ["C"], indirect=True)
# class TestEasyHLADiscreteHLALocusC:
#     """
#     Testing EasyHLA where tests require HLA-C.
#     """
#     @pytest.mark.integration
#     def test_run(self, easyhla: EasyHLA):
#         """
#         Integration test, assert that pyEasyHLA produces an identical output to
#         the original Ruby output.
#         """
#         input_file = os.path.dirname(__file__) + "/input/test.fasta"
#         ref_output_file = os.path.dirname(__file__) + "/output/hla-c-output.csv"
#         output_file = os.path.dirname(__file__) + "/output/test.csv"

#         easyhla.run(
#             input_file,
#             output_file,
#             0,
#         )

#         compare_ref_vs_test(
#             easyhla=easyhla,
#             reference_output_file=ref_output_file,
#             output_file=output_file,
#         )


# @pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
# class TestEasyHLA:
#     @pytest.mark.integration
#     @pytest.mark.slow
#     def test_run(self, easyhla: EasyHLA):
#         """
#         Integration test, assert that pyEasyHLA produces an identical output to
#         the original Ruby output.
#         """

#         input_file = (
#             os.path.dirname(__file__) + f"/input/hla-{easyhla.locus.lower()}-seqs.fasta"
#         )
#         ref_output_file = (
#             os.path.dirname(__file__)
#             + f"/output/hla-{easyhla.locus.lower()}-output-ref.csv"
#         )
#         output_file = (
#             os.path.dirname(__file__) + f"/output/hla-{easyhla.locus.lower()}-test.csv"
#         )

#         if not os.path.exists(input_file):
#             pytest.skip("Input sequence does not exist!")
#         if not os.path.exists(ref_output_file):
#             pytest.skip("Reference output does not exist!")

#         start_time = datetime.now()
#         print(f"Test started at {start_time.isoformat()}")

#         easyhla.run(
#             input_file,
#             output_file,
#             0,
#         )

#         end_time = datetime.now()

#         print(f"Interpretation ended at {end_time.isoformat()}")

#         compare_ref_vs_test(
#             easyhla=easyhla,
#             reference_output_file=ref_output_file,
#             output_file=output_file,
#         )

#         end_compare_time = datetime.now()

#         print(f"Test ended at {end_compare_time.isoformat()}")

#         print(f"Time elapsed: {(end_compare_time - start_time).total_seconds()}")
#         print(
#             f"Time elapsed for interpretation: {(end_time - start_time).total_seconds()}"
#         )
#         print(
#             f"Time elapsed for output comparison: {(end_compare_time - end_time).total_seconds()}"
#         )
#     pass
