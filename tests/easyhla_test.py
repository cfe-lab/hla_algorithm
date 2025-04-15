import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from pydantic import BaseModel
from pytest_mock import MockerFixture

from easyhla.easyhla import EXON_NAME, HLA_LOCI, EasyHLA
from easyhla.models import (
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)
from easyhla.utils import nuc2bin

# from .conftest import compare_ref_vs_test


class DummyStandard(BaseModel):
    allele: str
    exon2: str
    exon3: str


HLA_STANDARDS: dict[HLA_LOCI, DummyStandard] = {
    "A": DummyStandard(
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
    "B": DummyStandard(
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
    "C": DummyStandard(
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

HLA_FREQUENCIES: dict[HLA_LOCI, HLAProteinPair] = {
    "A": HLAProteinPair(
        first_field_1="22",
        first_field_2="33",
        second_field_1="14",
        second_field_2="23",
    ),
    "B": HLAProteinPair(
        first_field_1="57",
        first_field_2="01",
        second_field_1="57",
        second_field_2="03",
    ),
    "C": HLAProteinPair(
        first_field_1="40",
        first_field_2="43",
        second_field_1="25",
        second_field_2="29",
    ),
}


def get_dummy_easyhla(locus: HLA_LOCI) -> EasyHLA:
    # We only need one standard as it only uses the first standard to pad
    # our inputs against.
    current_standard: DummyStandard = HLA_STANDARDS[locus]
    dummy_standards: dict[str, HLAStandard] = {
        current_standard.allele: HLAStandard(
            allele=current_standard.allele,
            two=nuc2bin(current_standard.exon2),
            three=nuc2bin(current_standard.exon3),
        )
    }
    dummy_frequencies: dict[HLAProteinPair, int] = {HLA_FREQUENCIES[locus]: 1}
    return EasyHLA(
        locus,
        hla_standards=dummy_standards,
        hla_frequencies=dummy_frequencies,
        last_modified=datetime(2025, 4, 8),
    )


@pytest.fixture(scope="module")
def easyhla(request: pytest.FixtureRequest):
    return get_dummy_easyhla(request.param)


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
                    allele="std_allmatch",
                    two=(1, 4),
                    three=(2, 8),
                    mismatch=2,
                ),
            ],
            [None, 0, 1, 2, 3, 5],
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 2, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
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
                    allele="std_first_last_mismatch",
                    two=(2, 2),
                    three=(4, 4),
                    mismatch=2,
                ),
                HLAStandardMatch(
                    allele="std_produces_identical_combo",
                    two=(3, 2),
                    three=(4, 12),
                    mismatch=2,
                ),
                HLAStandardMatch(
                    allele="std_allmatch",
                    two=(1, 2),
                    three=(4, 8),
                    mismatch=0,
                ),
            ],
            [2, 3, 10],
            {
                HLACombinedStandard(
                    standard_bin=(1, 2, 4, 8),
                    possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                ): 0,
                HLACombinedStandard(
                    standard_bin=(2, 2, 4, 4),
                    possible_allele_pairs=(
                        ("std_first_last_mismatch", "std_first_last_mismatch"),
                    ),
                ): 2,
                HLACombinedStandard(
                    standard_bin=(3, 2, 4, 12),
                    possible_allele_pairs=(
                        ("std_allmatch", "std_first_last_mismatch"),
                        ("std_allmatch", "std_produces_identical_combo"),
                        ("std_first_last_mismatch", "std_produces_identical_combo"),
                        (
                            "std_produces_identical_combo",
                            "std_produces_identical_combo",
                        ),
                    ),
                ): 2,
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
        result = EasyHLA.combine_standards(
            matching_stds=matching_standards,
            seq=sequence,
            mismatch_threshold=threshold,
        )
        assert result == exp_result


@pytest.mark.parametrize(
    (
        "sr, unmatched, expected_id, expected_is_exon, expected_matched, "
        "expected_exon2, expected_exon3, expected_unmatched"
    ),
    [
        pytest.param(
            SeqRecord(id="fullseq1", seq=Seq("ACGT")),
            {"exon2": {}, "exon3": {}},
            "fullseq1",
            False,
            False,
            "",
            "",
            {"exon2": {}, "exon3": {}},
            id="full_seq_no_possible_matches",
        ),
        pytest.param(
            SeqRecord(id="fullseq1", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("CCC")},
                "exon3": {},
            },
            "fullseq1",
            False,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("CCC")},
                "exon3": {},
            },
            id="full_seq_one_rejected_exon2_match",
        ),
        pytest.param(
            SeqRecord(id="fullseq1", seq=Seq("ACGT")),
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            "fullseq1",
            False,
            False,
            "",
            "",
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            id="full_seq_one_rejected_exon3_match",
        ),
        pytest.param(
            SeqRecord(id="fullseq1", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {"E2_exon3": Seq("CCC")},
            },
            "fullseq1",
            False,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {"E2_exon3": Seq("CCC")},
            },
            id="full_seq_one_rejected_exon2_and_exon3_match",
        ),
        pytest.param(
            SeqRecord(id="E1", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {},
            },
            "E1",
            False,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {},
            },
            id="full_seq_does_not_match_unmatched_exon2",
        ),
        pytest.param(
            SeqRecord(id="E1", seq=Seq("ACGT")),
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            "E1",
            False,
            False,
            "",
            "",
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            id="full_seq_does_not_match_unmatched_exon3",
        ),
        pytest.param(
            SeqRecord(id="E1", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            "E1",
            False,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("ATA")},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            id="full_seq_does_not_match_unmatched_exon2_or_exon_3",
        ),
        # exon2 tests:
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {"exon2": {}, "exon3": {}},
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("ACGT")},
                "exon3": {},
            },
            id="exon2_unmatched_added_to_empty_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {
                "exon2": {"E2_exon2": Seq("CCC")},
                "exon3": {},
            },
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {
                    "E1_exon2": Seq("ACGT"),
                    "E2_exon2": Seq("CCC"),
                },
                "exon3": {},
            },
            id="exon2_unmatched_added_to_nonempty_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {
                "exon2": {},
                "exon3": {"E2_exon3": Seq("CCC")},
            },
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {"E1_exon2": Seq("ACGT")},
                "exon3": {"E2_exon3": Seq("CCC")},
            },
            id="exon2_does_not_match",
        ),
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            "E1",
            True,
            True,
            "ACGT",
            "CCC",
            {"exon2": {}, "exon3": {}},
            id="exon2_match_empties_unmatched_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {
                "exon2": {"E3_exon2": Seq("GCC")},
                "exon3": {"E1_exon3": Seq("CCC")},
            },
            "E1",
            True,
            True,
            "ACGT",
            "CCC",
            {
                "exon2": {"E3_exon2": Seq("GCC")},
                "exon3": {},
            },
            id="exon2_match_leaves_other_unmatched_undisturbed",
        ),
        pytest.param(
            SeqRecord(id="E1_exon2", seq=Seq("ACGT")),
            {
                "exon2": {"E5_exon2": Seq("CCTCCC")},
                "exon3": {
                    "E1_exon3": Seq("CCC"),
                    "E7_exon3": Seq("GCC"),
                },
            },
            "E1",
            True,
            True,
            "ACGT",
            "CCC",
            {
                "exon2": {"E5_exon2": Seq("CCTCCC")},
                "exon3": {"E7_exon3": Seq("GCC")},
            },
            id="exon2_match_leaves_nonempty_unmatched_dict",
        ),
        # exon3 tests:
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {"exon2": {}, "exon3": {}},
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {},
                "exon3": {"E1_exon3": Seq("ACGT")},
            },
            id="exon3_unmatched_added_to_empty_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {
                "exon2": {},
                "exon3": {"E2_exon3": Seq("CCC")},
            },
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {},
                "exon3": {
                    "E1_exon3": Seq("ACGT"),
                    "E2_exon3": Seq("CCC"),
                },
            },
            id="exon3_unmatched_added_to_nonempty_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {
                "exon2": {"E2_exon2": Seq("CCC")},
                "exon3": {},
            },
            "E1",
            True,
            False,
            "",
            "",
            {
                "exon2": {"E2_exon2": Seq("CCC")},
                "exon3": {"E1_exon3": Seq("ACGT")},
            },
            id="exon3_does_not_match",
        ),
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("CCC")},
                "exon3": {},
            },
            "E1",
            True,
            True,
            "CCC",
            "ACGT",
            {"exon2": {}, "exon3": {}},
            id="exon3_match_empties_unmatched_dict",
        ),
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {
                "exon2": {"E1_exon2": Seq("GCC")},
                "exon3": {"E3_exon3": Seq("CCC")},
            },
            "E1",
            True,
            True,
            "GCC",
            "ACGT",
            {
                "exon2": {},
                "exon3": {"E3_exon3": Seq("CCC")},
            },
            id="exon3_match_leaves_other_unmatched_undisturbed",
        ),
        pytest.param(
            SeqRecord(id="E1_exon3", seq=Seq("ACGT")),
            {
                "exon2": {
                    "E1_exon2": Seq("CCC"),
                    "E7_exon2": Seq("GCC"),
                },
                "exon3": {"E5_exon3": Seq("CCTCCC")},
            },
            "E1",
            True,
            True,
            "CCC",
            "ACGT",
            {
                "exon2": {"E7_exon2": Seq("GCC")},
                "exon3": {"E5_exon3": Seq("CCTCCC")},
            },
            id="exon3_match_leaves_nonempty_unmatched_dict",
        ),
    ],
)
def test_pair_exons_helper(
    sr: SeqRecord,
    unmatched: dict[EXON_NAME, dict[str, Seq]],
    expected_id: str,
    expected_is_exon: bool,
    expected_matched: bool,
    expected_exon2: str,
    expected_exon3: str,
    expected_unmatched: dict[EXON_NAME, dict[str, Seq]],
):
    result: tuple[str, bool, bool, str, str] = EasyHLA.pair_exons_helper(sr, unmatched)
    assert result[0] == expected_id
    assert result[1] == expected_is_exon
    assert result[2] == expected_matched
    assert result[3] == expected_exon2
    assert result[4] == expected_exon3
    assert unmatched == expected_unmatched


@pytest.mark.parametrize(
    "raw_sequence_records, locus, expected_paired, expected_unmatched",
    [
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["C"].exon2
                    + "A" * 100
                    + "Z"
                    + "A" * 140
                    + HLA_STANDARDS["C"].exon3,
                )
            ],
            "C",
            [],
            {"exon2": {}, "exon3": {}},
            id="base_check_fails",
        ),
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["A"].exon2 + "A" * 241 + HLA_STANDARDS["A"].exon3,
                )
            ],
            "A",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["A"].exon2),
                    intron=nuc2bin("A" * 241),
                    three=nuc2bin(HLA_STANDARDS["A"].exon3),
                    name="E1",
                    num_sequences_used=1,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="single_full_sequence",
        ),
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["A"].exon2
                    + "A" * 1000
                    + HLA_STANDARDS["A"].exon3,  # too long
                )
            ],
            "A",
            [],
            {"exon2": {}, "exon3": {}},
            id="locus_a_length_check_fails",
        ),
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["B"].exon2 + "A" * 241 + HLA_STANDARDS["B"].exon3,
                )
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=nuc2bin("A" * 241),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E1",
                    num_sequences_used=1,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="locus_b_single_full_sequence_min_length",
        ),
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["B"].exon2 + "A" * 250 + HLA_STANDARDS["B"].exon3,
                )
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=nuc2bin("A" * 250),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E1",
                    num_sequences_used=1,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="locus_b_single_full_sequence_max_length",
        ),
        pytest.param(
            [
                (
                    "E1",
                    HLA_STANDARDS["B"].exon2 + "A" * 245 + HLA_STANDARDS["B"].exon3,
                )
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=nuc2bin("A" * 245),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E1",
                    num_sequences_used=1,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="locus_b_single_full_sequence_middle_of_length_range",
        ),
        pytest.param(
            [("E1_exon2", HLA_STANDARDS["B"].exon2)],
            "B",
            [],
            {"exon2": {"E1_exon2": Seq(HLA_STANDARDS["B"].exon2)}, "exon3": {}},
            id="locus_b_single_exon2_sequence",
        ),
        pytest.param(
            [("E1_exon3", HLA_STANDARDS["B"].exon3)],
            "B",
            [],
            {"exon2": {}, "exon3": {"E1_exon3": Seq(HLA_STANDARDS["B"].exon3)}},
            id="locus_b_single_exon3_sequence",
        ),
        pytest.param(
            [
                ("E1_exon2", HLA_STANDARDS["B"].exon2),
                ("E1_exon3", HLA_STANDARDS["B"].exon3),
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E1",
                    num_sequences_used=2,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="locus_b_one_properly_paired_sequence_exon2_first",
        ),
        pytest.param(
            [
                ("E1_exon3", HLA_STANDARDS["B"].exon3),
                ("E1_exon2", HLA_STANDARDS["B"].exon2),
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E1",
                    num_sequences_used=2,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="locus_b_one_properly_paired_sequence_exon3_first",
        ),
        pytest.param(
            [
                (
                    "E1_exon2",
                    HLA_STANDARDS["B"].exon2[0:100],  # too short
                )
            ],
            "B",
            [],
            {"exon2": {}, "exon3": {}},
            id="locus_b_exon2_failed_length_check",
        ),
        pytest.param(
            [
                ("E1_exon2_short", HLA_STANDARDS["C"].exon2[0:250]),
                ("E1_exon3", HLA_STANDARDS["C"].exon3),
            ],
            "C",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["C"].exon2[0:250] + "N" * 20),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["C"].exon3),
                    name="E1",
                    num_sequences_used=2,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="exon2_properly_padded",
        ),
        pytest.param(
            [
                ("E1_exon2", HLA_STANDARDS["C"].exon2),
                ("E1_exon3_short", HLA_STANDARDS["C"].exon3[0:270]),
            ],
            "C",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["C"].exon2),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["C"].exon3[0:270] + "N" * 6),
                    name="E1",
                    num_sequences_used=2,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="exon3_properly_padded",
        ),
        pytest.param(
            [
                ("E1_exon2_short", HLA_STANDARDS["C"].exon2[0:250]),
                ("E1_exon3_short", HLA_STANDARDS["C"].exon3[0:270]),
            ],
            "C",
            [
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["C"].exon2[0:250] + "N" * 20),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["C"].exon3[0:270] + "N" * 6),
                    name="E1",
                    num_sequences_used=2,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="both_exons_properly_padded",
        ),
        pytest.param(
            [
                ("E1_exon2_short", HLA_STANDARDS["B"].exon2[0:250]),
            ],
            "B",
            [],
            {
                "exon2": {"E1_exon2_short": Seq(HLA_STANDARDS["B"].exon2[0:250])},
                "exon3": {},
            },
            id="unmatched_exon2_not_padded",
        ),
        pytest.param(
            [
                ("E1_exon3_short", HLA_STANDARDS["B"].exon3[0:250]),
            ],
            "B",
            [],
            {
                "exon2": {},
                "exon3": {"E1_exon3_short": Seq(HLA_STANDARDS["B"].exon3[0:250])},
            },
            id="unmatched_exon3_not_padded",
        ),
        pytest.param(
            [
                (
                    "E1_full_short",
                    HLA_STANDARDS["B"].exon2[15:]
                    + "A" * 241
                    + HLA_STANDARDS["B"].exon3[0:265],
                ),
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin("N" * 15 + HLA_STANDARDS["B"].exon2[15:]),
                    intron=(1,) * 241,
                    three=nuc2bin(HLA_STANDARDS["B"].exon3[0:265] + "N" * 11),
                    name="E1_full_short",
                    num_sequences_used=1,
                ),
            ],
            {"exon2": {}, "exon3": {}},
            id="short_full_sequence_properly_padded",
        ),
        pytest.param(
            [
                (
                    "E1_full_short",
                    HLA_STANDARDS["B"].exon2[15:]
                    + "A" * 241
                    + HLA_STANDARDS["B"].exon3[0:265],
                ),
                ("E2_exon2_short", HLA_STANDARDS["B"].exon2[0:250]),
                (
                    "E3_full",
                    HLA_STANDARDS["B"].exon2[15:]
                    + "A" * 241
                    + HLA_STANDARDS["B"].exon3,  # wrong length
                ),
                ("E2_exon3", HLA_STANDARDS["B"].exon3),
                (
                    "E4_full",
                    HLA_STANDARDS["B"].exon2 + "A" * 241 + HLA_STANDARDS["B"].exon3,
                ),
                ("E5_exon3", HLA_STANDARDS["B"].exon3[0:200]),  # wrong length
                ("E6_exon2", HLA_STANDARDS["B"].exon2),
                ("E5_exon2", HLA_STANDARDS["B"].exon2),  # will go unmatched
                ("E6_exon3", HLA_STANDARDS["B"].exon3),
                ("E7_exon3_short", "A" * 270),
            ],
            "B",
            [
                HLASequence(
                    two=nuc2bin("N" * 15 + HLA_STANDARDS["B"].exon2[15:]),
                    intron=(1,) * 241,
                    three=nuc2bin(HLA_STANDARDS["B"].exon3[0:265] + "N" * 11),
                    name="E1_full_short",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2[0:250] + "N" * 20),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E2",
                    num_sequences_used=2,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(1,) * 241,
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E4_full",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E6",
                    num_sequences_used=2,
                ),
            ],
            {
                "exon2": {"E5_exon2": Seq(HLA_STANDARDS["B"].exon2)},
                "exon3": {"E7_exon3_short": Seq("A" * 270)},
            },
            id="typical_case",
        ),
    ],
)
def test_pair_exons(
    raw_sequence_records: list[tuple[str, str]],
    locus: HLA_LOCI,
    expected_paired: list[HLASequence],
    expected_unmatched: dict[EXON_NAME, dict[str, Seq]],
):
    easyhla: EasyHLA = get_dummy_easyhla(locus)
    paired_seqs: list[HLASequence]
    unmatched: dict[EXON_NAME, dict[str, Seq]]

    srs: list[SeqRecord] = [
        SeqRecord(id=id, seq=Seq(sequence)) for id, sequence in raw_sequence_records
    ]
    paired_seqs, unmatched = easyhla.pair_exons(srs)
    assert paired_seqs == expected_paired
    assert unmatched == expected_unmatched


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
    locuses: Iterable[HLA_LOCI],
    expected_result: list[HLAMismatch],
):
    for locus in locuses:
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        result: list[HLAMismatch] = easyhla.get_mismatches(
            tuple(std_bin), np.array(seq_bin)
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
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        with pytest.raises(ValueError) as excinfo:
            easyhla.get_mismatches(tuple(std_bin), np.array(seq_bin))
        assert expected_error in str(excinfo.value)


@pytest.mark.parametrize(
    "sequence, locus, threshold, raw_standards, expected_interpretation",
    [
        pytest.param(
            HLASequence(
                two=(1, 2),
                intron=(),
                three=(4, 8),
                name="E1",
                num_sequences_used=1,
            ),
            "C",
            5,
            [
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
            HLAInterpretation(
                hla_sequence=HLASequence(
                    two=(1, 2),
                    intron=(),
                    three=(4, 8),
                    name="E1",
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
            ),
            id="typical_case_non_b",
        ),
        pytest.param(
            HLASequence(
                two=(1, 2),
                intron=(),
                three=(4, 8),
                name="E1",
                num_sequences_used=1,
            ),
            "B",
            5,
            [
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
            HLAInterpretation(
                hla_sequence=HLASequence(
                    two=(1, 2),
                    intron=(),
                    three=(4, 8),
                    name="E1",
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
    locus: HLA_LOCI,
    threshold: int,
    raw_standards: list[HLAStandard],
    expected_interpretation: HLAInterpretation,
    mocker: MockerFixture,
):
    easyhla: EasyHLA = get_dummy_easyhla(locus)
    # Replace the standards with the ones in the test.
    standards: dict[str, HLAStandard] = {std.allele: std for std in raw_standards}
    easyhla.hla_standards = standards

    # Spy on the internals to make sure they're called correctly.
    get_matching_standards_spy: mocker.MagicMock = mocker.spy(
        easyhla, "get_matching_standards"
    )
    combine_standards_spy: mocker.MagicMock = mocker.spy(easyhla, "combine_standards")
    get_mismatches_spy: mocker.MagicMock = mocker.spy(easyhla, "get_mismatches")

    result: HLAInterpretation = easyhla.interpret(sequence, threshold=threshold)
    assert result == expected_interpretation

    # Using assert_called_once_with doesn't work here as we feed it a generator
    # and the comparison fails; we have to manually convert the value to a list
    # to be able to compare them.
    get_matching_standards_spy.assert_called_once()
    gms_call_args: mocker.call = get_matching_standards_spy.call_args
    assert len(gms_call_args.args) == 2
    assert len(gms_call_args.kwargs) == 0
    assert gms_call_args.args[0] == sequence.sequence_for_interpretation
    assert list(gms_call_args.args[1]) == list(standards.values())

    matching_standards: list[HLAStandardMatch] = get_matching_standards_spy.spy_return

    combine_standards_spy.assert_called_once_with(
        matching_standards,
        sequence.sequence_for_interpretation,
        mismatch_threshold=threshold,
    )
    all_combos: dict[HLACombinedStandard, int] = combine_standards_spy.spy_return

    get_mismatches_spy.assert_has_calls(
        [
            mocker.call(x.standard_bin, sequence.sequence_for_interpretation)
            for x in all_combos.keys()
        ],
        any_order=False,
    )


@pytest.mark.parametrize(
    "sequence, locus, threshold, raw_standards",
    [
        pytest.param(
            HLASequence(
                two=(1, 2, 4, 8, 10, 2),
                intron=(),
                three=(4, 8, 5, 7, 11, 1),
                name="E1",
                num_sequences_used=1,
            ),
            "B",
            5,
            [
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
            id="no_matching_standards",
        ),
    ],
)
def test_interpret_error_cases(
    sequence: HLASequence,
    locus: HLA_LOCI,
    threshold: int,
    raw_standards: list[HLAStandard],
    mocker: MockerFixture,
):
    easyhla: EasyHLA = get_dummy_easyhla(locus)
    # Replace the standards with the ones in the test.
    standards: dict[str, HLAStandard] = {std.allele: std for std in raw_standards}
    easyhla.hla_standards = standards

    # Spy on the internals to make sure they're called correctly.
    get_matching_standards_spy: mocker.MagicMock = mocker.spy(
        easyhla, "get_matching_standards"
    )
    combine_standards_spy: mocker.MagicMock = mocker.spy(easyhla, "combine_standards")
    get_mismatches_spy: mocker.MagicMock = mocker.spy(easyhla, "get_mismatches")

    with pytest.raises(EasyHLA.NoMatchingStandards):
        easyhla.interpret(sequence, threshold=threshold)

    # Using assert_called_once_with doesn't work here as we feed it a generator
    # and the comparison fails; we have to manually convert the value to a list
    # to be able to compare them.
    get_matching_standards_spy.assert_called_once()
    gms_call_args: mocker.call = get_matching_standards_spy.call_args
    assert len(gms_call_args.args) == 2
    assert len(gms_call_args.kwargs) == 0
    assert gms_call_args.args[0] == sequence.sequence_for_interpretation
    assert list(gms_call_args.args[1]) == list(standards.values())

    combine_standards_spy.assert_not_called()
    get_mismatches_spy.assert_not_called()


def test_unknown_hla_locus():
    """
    Assert we raise a value error if we put in an unknown HLA locus.
    """
    with pytest.raises(ValueError):
        EasyHLA("D")


def test_known_hla_locus_lowercase():
    """
    Assert we raise a value error if we put in an HLA locus with wrong case.
    """
    with pytest.raises(ValueError):
        EasyHLA("a")


@pytest.mark.parametrize(
    "raw_standards, raw_expected_result",
    [
        pytest.param(
            [],
            [],
            id="empty_file",
        ),
        pytest.param(
            [("A*01:23:45:67N", "A" * 270, "T" * 276)],
            [
                HLAStandard(
                    allele="A*01:23:45:67N",
                    two=(1,) * 270,
                    three=(8,) * 276,
                ),
            ],
            id="single_entry",
        ),
        pytest.param(
            [
                ("B*01:23:45:67N", "A" * 270, "T" * 276),
                ("B*57:01:02", "C" * 270, "GT" * 138),
                ("B*100:101:111", "AT" * 135, "G" * 276),
            ],
            [
                HLAStandard(
                    allele="B*01:23:45:67N",
                    two=(1,) * 270,
                    three=(8,) * 276,
                ),
                HLAStandard(
                    allele="B*57:01:02",
                    two=(2,) * 270,
                    three=(4, 8) * 138,
                ),
                HLAStandard(
                    allele="B*100:101:111",
                    two=(1, 8) * 135,
                    three=(4,) * 276,
                ),
            ],
            id="multiple_entries",
        ),
    ],
)
def test_read_hla_standards(
    raw_standards: list[tuple[str, str, str]],
    raw_expected_result: list[HLAStandard],
    tmp_path: Path,
    mocker: MockerFixture,
):
    # Convert the raw expected results into a dict:
    expected_result: dict[str, HLAStandard] = {
        std.allele: std for std in raw_expected_result
    }
    # Build a string from the raw standards:
    standards_file_str: str = ""
    for allele, exon2, exon3 in raw_standards:
        standards_file_str += f"{allele},{exon2},{exon3}\n"
    read_result: list[HLAStandard] = EasyHLA.read_hla_standards(
        StringIO(standards_file_str)
    )
    assert read_result == expected_result

    # Also, try loading these from a file.
    for locus in ("A", "B", "C"):
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        p = tmp_path / "hla_std.csv"
        p.write_text(standards_file_str)
        dirname_return_mock: mocker.MagicMock = mocker.MagicMock()
        mocker.patch.object(os.path, "dirname", return_value=dirname_return_mock)
        mocker.patch.object(os.path, "join", return_value=str(p))
        load_result: list[HLAStandard] = easyhla.load_default_hla_standards()
        assert load_result == expected_result


@pytest.mark.parametrize(
    "raw_hlas_observed, expected_locus_a, expected_locus_b, expected_locus_c",
    [
        pytest.param(
            ["1423,2233,5701,5703,2529,4043"],
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
            [
                "1423,2233,5701,5703,2529,4043",
                "1434,2233,5701,5611,2529,5150",
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
                "1423,2233,5701,5703,2529,4043",
                "1734,8882,5202,5611,1982,1982",
                "5432,9876,5701,5703,1111,2222",
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
            [
                "1423,2233,5701,5703,2529,4043",
                "1734,8882,5202,5611,1982,1982",
                "5432,9876,5701,5703,1111,2222",
                "5432,9876,5701,5703,1982,1982",
                "5432,9874,5702,5703,1111,2222",
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
                ): 2,
                HLAProteinPair(
                    first_field_1="54",
                    first_field_2="32",
                    second_field_1="98",
                    second_field_2="74",
                ): 1,
            },
            {
                HLAProteinPair(
                    first_field_1="57",
                    first_field_2="01",
                    second_field_1="57",
                    second_field_2="03",
                ): 3,
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
                ): 2,
                HLAProteinPair(
                    first_field_1="11",
                    first_field_2="11",
                    second_field_1="22",
                    second_field_2="22",
                ): 2,
            },
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
    frequencies_str: str = "\n".join(raw_hlas_observed) + "\n"
    expected_results: dict[HLA_LOCI, dict[HLAProteinPair, int]] = {
        "A": expected_locus_a,
        "B": expected_locus_b,
        "C": expected_locus_c,
    }
    for locus in ("A", "B", "C"):
        result: dict[HLAProteinPair, int] = EasyHLA.read_hla_frequencies(
            locus, StringIO(frequencies_str)
        )
        assert result == expected_results[locus]

    # Now try loading these from a file.
    p = tmp_path / "hla_frequencies.csv"
    p.write_text(frequencies_str)

    for locus in ("A", "B", "C"):
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        dirname_return_mock: mocker.MagicMock = mocker.MagicMock()
        mocker.patch.object(os.path, "dirname", return_value=dirname_return_mock)
        mocker.patch.object(os.path, "join", return_value=str(p))
        load_result: dict[HLAProteinPair, int] = easyhla.load_default_hla_frequencies()
        assert load_result == expected_results[locus]


def test_load_default_last_modified(tmp_path: Path, mocker: MockerFixture):
    """
    Assert we can load our mtime and that it is represented correctly.
    """
    fake_mtime_path: Path = tmp_path / "hla_nuc.fasta.mtime"
    fake_mtime_path.write_text("Thu Apr 10 14:43:30 UTC 2025")

    dirname_return_mock: mocker.MagicMock = mocker.MagicMock()
    mocker.patch.object(os.path, "dirname", return_value=dirname_return_mock)
    mocker.patch.object(os.path, "join", return_value=str(fake_mtime_path))

    result: datetime = EasyHLA.load_default_last_modified()

    # Note that the timezone is lost when the date is read back in.
    dummy_last_modified: datetime = datetime(2025, 4, 10, 14, 43, 30)
    assert result == dummy_last_modified


def test_init_no_defaults(mocker: MockerFixture):
    fake_standards: mocker.MagicMock = mocker.MagicMock()
    fake_frequencies: mocker.MagicMock = mocker.MagicMock()
    fake_last_modified: datetime = datetime(2025, 4, 10, 15, 55, 0)

    for locus in ("A", "B", "C"):
        easyhla: EasyHLA = EasyHLA(
            locus, fake_standards, fake_frequencies, fake_last_modified
        )
        assert easyhla.locus == locus
        assert easyhla.hla_standards == fake_standards
        assert easyhla.hla_frequencies == fake_frequencies
        assert easyhla.last_modified == fake_last_modified


def test_init_all_defaults(mocker: MockerFixture):
    fake_standards: mocker.MagicMock = mocker.MagicMock()
    fake_frequencies: mocker.MagicMock = mocker.MagicMock()
    fake_last_modified: datetime = datetime(2025, 4, 10, 15, 55, 0)

    mocker.MagicMock = mocker.patch.object(
        EasyHLA, "load_default_hla_standards", return_value=fake_standards
    )
    mocker.MagicMock = mocker.patch.object(
        EasyHLA, "load_default_hla_frequencies", return_value=fake_frequencies
    )
    mocker.MagicMock = mocker.patch.object(
        EasyHLA, "load_default_last_modified", return_value=fake_last_modified
    )

    for locus in ("A", "B", "C"):
        easyhla: EasyHLA = EasyHLA(locus)
        assert easyhla.locus == locus
        assert easyhla.hla_standards == fake_standards
        assert easyhla.hla_frequencies == fake_frequencies
        assert easyhla.last_modified == fake_last_modified


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
        EasyHLA.check_bases(seq=sequence)
    else:
        with pytest.raises(ValueError):
            EasyHLA.check_bases(seq=sequence)


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
    left_pad, right_pad = EasyHLA.calc_padding(std, seq)
    assert left_pad == exp_left_pad
    assert right_pad == exp_right_pad


@pytest.mark.parametrize(
    "std_bin, seq_bin, exon, exp_raw_result",
    [
        # Cases with zero padding introduced:
        (
            [1, 2, 4, 8],
            [1, 2, 4, 8],
            "exon2",
            [1, 2, 4, 8],
        ),
        (
            [1, 2, 4, 8],
            [1, 2, 4, 8],
            "exon3",
            [1, 2, 4, 8],
        ),
        (
            [1, 2, 4, 8],
            [1, 2, 4, 8],
            None,
            [1, 2, 4, 8],
        ),
        # Integration tests with exon2:
        (
            [1, 2, 4, 8] + [1] * (266 + EasyHLA.EXON3_LENGTH),
            [1, 2, 4, 8],
            "exon2",
            [1, 2, 4, 8, *([15] * 266)],
        ),
        (
            [1] * 100 + [5, 6, 4, 12] + [1] * (166 + EasyHLA.EXON3_LENGTH),
            [4, 4, 4, 4],
            "exon2",
            [*([15] * 100), 4, 4, 4, 4, *([15] * 166)],
        ),
        (
            [1] * 266 + [6, 6, 6, 6] + [1] * EasyHLA.EXON3_LENGTH,
            [4, 5, 4, 5],
            "exon2",
            [*([15] * 266), 4, 5, 4, 5],
        ),
        # Only the exon2 portion of the standard is considered:
        (
            [1] * 47 + [1, 2, 4] + [1] * 220 + [2] * 150 + [1, 2, 4, 8] + [1] * 122,
            [1, 2, 4, 8],
            "exon2",
            [*([15] * 47), 1, 2, 4, 8, *([15] * 219)],
        ),
        # The better match is picked:
        (
            [1] * 22
            + [4, 4, 4]
            + [1] * 46
            + [4, 4, 4, 4]
            + [1] * (195 + EasyHLA.EXON3_LENGTH),
            [4, 4, 4, 4],
            "exon2",
            [*([15] * 71), 4, 4, 4, 4, *([15] * 195)],
        ),
        (
            [2] * 21
            + [4, 6, 4, 7]
            + [1] * 46
            + [4, 4, 2, 4]
            + [1] * (195 + EasyHLA.EXON3_LENGTH),
            [5, 5, 5, 7],
            "exon2",
            [*([15] * 21), 5, 5, 5, 7, *([15] * (50 + 195))],
        ),
        # Integration tests with exon3
        (
            [4] * EasyHLA.EXON2_LENGTH + [1, 2, 4, 8] + [1] * 272,
            [1, 2, 4, 8],
            "exon3",
            [1, 2, 4, 8, *([15] * 272)],
        ),
        (
            [4] * (EasyHLA.EXON2_LENGTH + 50) + [1, 2, 4, 8] + [1] * 222,
            [1, 2, 4, 8],
            "exon3",
            [*([15] * 50), 1, 2, 4, 8, *([15] * 222)],
        ),
        (
            [4] * EasyHLA.EXON2_LENGTH + [1] * 272 + [1, 2, 4, 8],
            [1, 2, 4, 8],
            "exon3",
            [*([15] * 272), 1, 2, 4, 8],
        ),
        # Only the exon3 portion of the standard is considered:
        (
            [1] * 46 + [1, 2, 4, 8] + [1] * 220 + [2] * 150 + [1, 2, 4, 8] + [1] * 122,
            [1, 2, 4, 8],
            "exon3",
            [*([15] * 150), 1, 2, 4, 8, *([15] * 122)],
        ),
        # Integration test with intron:
        (
            [4] * 100 + [1, 2, 4, 8] + [1] * (166 + 296) + [8, 4, 2, 1] + [4] * 76,
            [1, 2, 4, 8] + [1] * (166 + 296) + [8, 4, 2, 1],
            None,
            [
                *([15] * 100),
                1,
                2,
                4,
                8,
                *([1] * (166 + 296)),
                8,
                4,
                2,
                1,
                *([15] * 76),
            ],
        ),
    ],
)
def test_pad_short(
    std_bin: Sequence[int],
    seq_bin: Sequence[int],
    exon: Optional[EXON_NAME],
    exp_raw_result: Sequence[int],
):
    result = EasyHLA.pad_short(std_bin, seq_bin, exon)
    # Debug code for future users
    print(
        result,
        sum([1 for a in result if a == 1]),
        sum([1 for a in result if a == 15]),
        len(result),
    )
    print(
        np.array(exp_raw_result),
        sum([1 for a in exp_raw_result if a == 1]),
        sum([1 for a in exp_raw_result if a == 15]),
        len(exp_raw_result),
    )
    assert np.array_equal(result, np.array(exp_raw_result))


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
    exp_result: Iterable[HLAStandardMatch],
):
    result = EasyHLA.get_matching_standards(
        seq=sequence, hla_stds=hla_stds, mismatch_threshold=mismatch_threshold
    )  # type: ignore
    print(result)
    assert result == exp_result


CHECK_LENGTH_HLA_A_TEST_CASES = [
    pytest.param(EasyHLA.HLA_A_LENGTH, "myseq_regular", True, id="regular_good_case"),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon2",
        True,
        id="regular_good_case_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon3",
        True,
        id="regular_good_case_ignores_exon3_name",
    ),
    pytest.param(1000, "myseq_regular", False, id="sequence_too_long"),
    pytest.param(1000, "myseq_exon2", False, id="sequence_too_long_ignores_exon2_name"),
    pytest.param(1000, "myseq_exon3", False, id="sequence_too_long_ignores_exon3_name"),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 1,
        "myseq_regular",
        False,
        id="sequence_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 1,
        "myseq_exon2",
        False,
        id="sequence_too_long_edge_case_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 1,
        "myseq_exon3",
        False,
        id="sequence_too_long_edge_case_ignores_exon3_name",
    ),
    pytest.param(300, "myseq_regular", False, id="sequence_too_short"),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon2",
        False,
        id="sequence_too_short_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_exon3",
        False,
        id="sequence_too_short_ignores_exon3_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_regular",
        False,
        id="sequence_too_short_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_exon2",
        False,
        id="sequence_too_short_edge_case_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_exon3",
        False,
        id="sequence_too_short_edge_case_ignores_exon3_name",
    ),
    pytest.param(
        300,
        "myseq_short",
        True,
        id="shortseq_good_case",
    ),
    pytest.param(
        300,
        "myseq_exon2_short",
        True,
        id="shortseq_good_case_ignores_exon2_name",
    ),
    pytest.param(
        300,
        "myseq_exon3_short",
        True,
        id="shortseq_good_case_ignores_exon3_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_short",
        True,
        id="shortseq_good_case_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_exon2_short",
        True,
        id="shortseq_good_case_edge_case_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH - 1,
        "myseq_exon3_short",
        True,
        id="shortseq_good_case_edge_case_ignores_exon3_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 100,
        "myseq_short",
        False,
        id="shortseq_too_long",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 100,
        "myseq_short_exon2",
        False,
        id="shortseq_too_long_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH + 100,
        "myseq_short_exon3",
        False,
        id="shortseq_too_long_ignores_exon3_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_short",
        False,
        id="shortseq_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon2_short",
        False,
        id="shortseq_too_long_edge_case_ignores_exon2_name",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon3_short",
        False,
        id="shortseq_too_long_edge_case_ignores_exon3_name",
    ),
]


@pytest.mark.parametrize(
    "sequence_length, name, expected_to_pass",
    CHECK_LENGTH_HLA_A_TEST_CASES,
)
def test_check_length_hla_type_a(
    sequence_length: int, name: str, expected_to_pass: bool
):
    easyhla: EasyHLA = get_dummy_easyhla("A")
    sequence: str = "A" * sequence_length
    if expected_to_pass:
        easyhla.check_length(seq=sequence, name=name)
    else:
        with pytest.raises(ValueError):
            easyhla.check_length(seq=sequence, name=name)


CHECK_LENGTH_HLA_BC_TEST_CASES = [
    # exon2 tests:
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon2",
        True,
        id="regular_exon2_good_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon2_exon3",
        True,
        id="regular_exon2_good_case_ignores_exon3_name",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH - 1,
        "myseq_exon2",
        False,
        id="regular_exon2_too_short_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH - 100,
        "myseq_exon2",
        False,
        id="regular_exon2_too_short_typical_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH + 1,
        "myseq_exon2",
        False,
        id="regular_exon2_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_exon2",
        False,
        id="regular_exon2_too_long_exon3_length",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon2",
        False,
        id="regular_exon2_too_long_a_length",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH + 100,
        "myseq_exon2",
        False,
        id="regular_exon2_too_long_typical_case",
    ),
    # exon3 tests:
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_exon3",
        True,
        id="regular_exon3_good_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH - 1,
        "myseq_exon3",
        False,
        id="regular_exon3_too_short_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon3",
        False,
        id="regular_exon3_too_short_exon2_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH - 100,
        "myseq_exon3",
        False,
        id="regular_exon3_too_short_typical_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH + 1,
        "myseq_exon3",
        False,
        id="regular_exon3_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon3",
        False,
        id="regular_exon3_too_long_a_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH + 100,
        "myseq_exon3",
        False,
        id="regular_exon3_too_long_typical_case",
    ),
    # Full sequence tests:
    pytest.param(
        EasyHLA.MIN_HLA_BC_LENGTH,
        "myseq_full",
        True,
        id="regular_full_good_case_lower_edge_case",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH,
        "myseq_full",
        True,
        id="regular_full_good_case_upper_edge_case",
    ),
    pytest.param(
        790,
        "myseq_full",
        True,
        id="regular_full_good_case_middle_of_range",
    ),
    pytest.param(
        EasyHLA.MIN_HLA_BC_LENGTH - 1,
        "myseq_full",
        False,
        id="regular_full_too_short_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_full",
        False,
        id="regular_full_too_short_exon2_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_full",
        False,
        id="regular_full_too_short_exon3_length",
    ),
    pytest.param(
        EasyHLA.MIN_HLA_BC_LENGTH - 100,
        "myseq_full",
        False,
        id="regular_full_too_short_typical_case",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH + 1,
        "myseq_full",
        False,
        id="regular_full_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH + 100,
        "myseq_full",
        False,
        id="regular_full_too_long_typical_case",
    ),
    # Short exon2 tests:
    pytest.param(
        EasyHLA.EXON2_LENGTH - 1,
        "myseq_exon2_short",
        True,
        id="short_exon2_good_case_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH - 100,
        "myseq_exon2_short",
        True,
        id="short_exon2_good_case_typical",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon2_short",
        False,
        id="short_exon2_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_exon2_short",
        False,
        id="short_exon2_too_long_exon3_length",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon2_short",
        False,
        id="short_exon2_too_long_a_length",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH + 10,
        "myseq_exon2_short",
        False,
        id="short_exon2_too_long_typical_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH - 1,
        "myseq_exon2_short",
        False,
        id="short_exon2_ignores_exon3_name",
    ),
    # Short exon3 tests:
    pytest.param(
        EasyHLA.EXON3_LENGTH - 1,
        "myseq_exon3_short",
        True,
        id="short_exon3_good_case_edge_case",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH - 100,
        "myseq_exon3_short",
        True,
        id="short_exon3_good_case_typical",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_exon3_short",
        True,
        id="short_exon3_good_case_exon2_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_exon3_short",
        False,
        id="short_exon3_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.HLA_A_LENGTH,
        "myseq_exon3_short",
        False,
        id="short_exon3_too_long_a_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH + 10,
        "myseq_exon3_short",
        False,
        id="short_exon3_too_long_typical_case",
    ),
    # Full sequence tests:
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH - 1,
        "myseq_short",
        True,
        id="short_full_good_case_edge_case",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH - 100,
        "myseq_short",
        True,
        id="short_full_good_case_typical",
    ),
    pytest.param(
        EasyHLA.MIN_HLA_BC_LENGTH,
        "myseq_short",
        True,
        id="short_full_good_case_min_bc_length",
    ),
    pytest.param(
        EasyHLA.EXON2_LENGTH,
        "myseq_short",
        True,
        id="short_full_good_case_exon2_length",
    ),
    pytest.param(
        EasyHLA.EXON3_LENGTH,
        "myseq_short",
        True,
        id="short_full_good_case_exon3_length",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH,
        "myseq_short",
        False,
        id="short_full_too_long_edge_case",
    ),
    pytest.param(
        EasyHLA.MAX_HLA_BC_LENGTH + 150,
        "myseq_short",
        False,
        id="short_full_too_long_typical_case",
    ),
]


@pytest.mark.parametrize(
    "sequence_length, name, expected_to_pass",
    CHECK_LENGTH_HLA_BC_TEST_CASES,
)
def test_check_length_hla_type_b_and_c(
    sequence_length: int, name: str, expected_to_pass: bool
):
    for locus in ("B", "C"):
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        sequence: str = "A" * sequence_length
        if expected_to_pass:
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)


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
