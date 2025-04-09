import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import pytz
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from pydantic import BaseModel
from pytest_mock import MockerFixture

from easyhla.easyhla import DATE_FORMAT, EXON_NAME, HLA_LOCI, EasyHLA
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

from .conftest import compare_ref_vs_test


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
    dummy_standards: list[HLAStandard] = [
        HLAStandard(
            allele=current_standard.allele,
            sequence=nuc2bin(current_standard.exon2 + current_standard.exon3),
        )
    ]
    dummy_frequencies: dict[HLAProteinPair, int] = {HLA_FREQUENCIES[locus]: 1}
    return EasyHLA(
        locus,
        hla_standards=dummy_standards,
        hla_frequencies=dummy_frequencies,
        last_modified=datetime(2025, 4, 8),
    )


@pytest.fixture(scope="module")
def easyhla(request: pytest.FixtureRequest):
    easyhla = EasyHLA(locus=request.param)
    return easyhla


@pytest.fixture
def hla_standard_file(tmp_path: Path):
    hla_standard = ["HELLO-WORLD", "AAAAC", "ARTY"]

    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_std.csv"
    p.write_text(",".join(hla_standard))

    return str(p)


@pytest.fixture
def hla_frequency_file(tmp_path: Path):
    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_freq.csv"
    p.write_text("2233,1423,5701,5703,4043,2529")

    return str(p)


@pytest.fixture(scope="session")
def timestamp() -> datetime:
    _dt = datetime.today()
    dt = _dt.replace(tzinfo=pytz.UTC)
    return dt


@pytest.fixture
def hla_last_modified_file(tmp_path: Path, timestamp: datetime) -> str:
    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_timestamp.txt"
    p.write_text(timestamp.strftime(DATE_FORMAT))

    return str(p)


class TestCombineStandards:
    @pytest.mark.parametrize(
        "sequence, matching_standards, thresholds, exp_result",
        [
            # Simple case
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                ],
                [0, 1, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 4, 2, 8]),
                        mismatch=2,
                    ),
                ],
                [0, 1, 2, 3, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 2, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 2,
                },
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_allmatch2",
                        sequence=np.array([1, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [0],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_allmatch2",
                        sequence=np.array([1, 4, 4, 8]),
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
            ),
            #
            (
                np.array([9, 6, 4, 6]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch2",
                        sequence=np.array([8, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [0, 1, 2],
                {
                    HLACombinedStandard(
                        standard_bin=(9, 6, 4, 12),
                        possible_allele_pairs=(("std_1mismatch2", "std_allmatch"),),
                    ): 1,
                },
            ),
            #
            (
                np.array([9, 6, 4, 6]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch2",
                        sequence=np.array([8, 4, 4, 8]),
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
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    )
                ],
                [0, 1, 2, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                    ): 1
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    )
                ],
                [0, 1, 3, 4, 5, 10],
                {
                    HLACombinedStandard(
                        standard_bin=(8, 4, 2, 1),
                        possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                    ): 4,
                },
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    ),
                ],
                [0],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
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
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
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
            ),
        ],
    )
    def test_combine_standards(
        self,
        sequence: list[int],
        matching_standards: list[HLAStandardMatch],
        thresholds: list[int],
        exp_result: dict[int, list[int]],
    ):
        for threshold in thresholds:
            result = EasyHLA.combine_standards(
                matching_stds=matching_standards,
                seq=sequence,
                mismatch_threshold=threshold,
            )
            assert result == exp_result


class TestPairExons:
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
        self,
        sr: SeqRecord,
        unmatched: dict[EXON_NAME, dict[str, Seq]],
        expected_id: str,
        expected_is_exon: bool,
        expected_matched: bool,
        expected_exon2: str,
        expected_exon3: str,
        expected_unmatched: dict[EXON_NAME, dict[str, Seq]],
    ):
        result: tuple[str, bool, bool, str, str] = EasyHLA.pair_exons_helper(
            sr, unmatched
        )
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
        self,
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


class TestGetMismatches:
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
                [1] * 170
                + [3]
                + [1] * 99
                + [11]
                + [4] * 99
                + [4] * 50
                + [1]
                + [4] * 125,
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
                [1] * 170
                + [3]
                + [1] * 99
                + [11]
                + [4] * 99
                + [4] * 50
                + [1]
                + [4] * 125,
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
        self,
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
        self,
        std_bin: Iterable[int],
        seq_bin: Iterable[int],
        expected_error: str,
    ):
        for locus in ["A", "B", "C"]:
            easyhla: EasyHLA = get_dummy_easyhla(locus)
            with pytest.raises(ValueError) as excinfo:
                easyhla.get_mismatches(tuple(std_bin), np.array(seq_bin))
            assert expected_error in str(excinfo.value)


class TestInterpret:
    @pytest.mark.parametrize(
        "sequence, locus, threshold, standards, expected_interpretation",
        [
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
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                    ),
                    HLAStandard(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                    ),
                    HLAStandard(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                    ),
                ],
                HLAInterpretation(
                    hla_sequence=HLASequence(
                        two=np.array([1, 2]),
                        intron=np.array([]),
                        three=np.array([4, 8]),
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
                                HLAMismatch(
                                    index=4, expected_base="K", observed_base="T"
                                ),
                            ],
                        ),
                        HLACombinedStandard(
                            standard_bin=(1, 2, 4, 4),
                            possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                        ): HLAMatchDetails(
                            mismatch_count=1,
                            mismatches=[
                                HLAMismatch(
                                    index=4, expected_base="G", observed_base="T"
                                ),
                            ],
                        ),
                        HLACombinedStandard(
                            standard_bin=(9, 6, 6, 9),
                            possible_allele_pairs=(
                                ("std_allmatch", "std_allmismatch"),
                            ),
                        ): HLAMatchDetails(
                            mismatch_count=4,
                            mismatches=[
                                HLAMismatch(
                                    index=1, expected_base="W", observed_base="A"
                                ),
                                HLAMismatch(
                                    index=2, expected_base="S", observed_base="C"
                                ),
                                HLAMismatch(
                                    index=3, expected_base="S", observed_base="G"
                                ),
                                HLAMismatch(
                                    index=4, expected_base="W", observed_base="T"
                                ),
                            ],
                        ),
                        HLACombinedStandard(
                            standard_bin=(9, 6, 6, 5),
                            possible_allele_pairs=(
                                ("std_1mismatch", "std_allmismatch"),
                            ),
                        ): HLAMatchDetails(
                            mismatch_count=4,
                            mismatches=[
                                HLAMismatch(
                                    index=1, expected_base="W", observed_base="A"
                                ),
                                HLAMismatch(
                                    index=2, expected_base="S", observed_base="C"
                                ),
                                HLAMismatch(
                                    index=3, expected_base="S", observed_base="G"
                                ),
                                HLAMismatch(
                                    index=4, expected_base="R", observed_base="T"
                                ),
                            ],
                        ),
                        HLACombinedStandard(
                            standard_bin=(8, 4, 2, 1),
                            possible_allele_pairs=(
                                ("std_allmismatch", "std_allmismatch"),
                            ),
                        ): HLAMatchDetails(
                            mismatch_count=4,
                            mismatches=[
                                HLAMismatch(
                                    index=1, expected_base="T", observed_base="A"
                                ),
                                HLAMismatch(
                                    index=2, expected_base="G", observed_base="C"
                                ),
                                HLAMismatch(
                                    index=3, expected_base="C", observed_base="G"
                                ),
                                HLAMismatch(
                                    index=4, expected_base="A", observed_base="T"
                                ),
                            ],
                        ),
                    },
                ),
                id="typical_case",
            ),
        ],
    )
    def test_interpret_good_cases(
        self,
        sequence: HLASequence,
        locus: HLA_LOCI,
        threshold: int,
        standards: list[HLAStandard],
        expected_interpretation: HLAInterpretation,
        mocker: MockerFixture,
    ):
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        # Replace the standards with the ones in the test.
        easyhla.hla_standards = standards

        # Spy on the internals to make sure they're called correctly.
        get_matching_standards_spy: mocker.MagicMock = mocker.spy(
            easyhla, "get_matching_standards"
        )
        combine_standards_spy: mocker.MagicMock = mocker.spy(
            easyhla, "combine_standards"
        )
        get_mismatches_spy: mocker.MagicMock = mocker.spy(easyhla, "get_mismatches")

        result: HLAInterpretation = easyhla.interpret(sequence, threshold=threshold)
        assert result == expected_interpretation

        get_matching_standards_spy.assert_called_once_with(
            sequence.sequence_for_interpretation,
            standards,
        )
        matching_standards: list[HLAStandardMatch] = (
            get_matching_standards_spy.spy_return
        )

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
        "sequence, locus, threshold, standards",
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
                        sequence=(2, 4, 8, 1, 10, 2, 8, 1, 5, 7, 11, 1),
                    ),
                    HLAStandard(
                        allele="std_2",
                        sequence=(8, 4, 2, 1, 10, 2, 4, 8, 10, 11, 4, 1),
                    ),
                    HLAStandard(
                        allele="std_3",
                        sequence=(1, 2, 4, 4, 5, 8, 8, 8, 5, 8, 11, 4),
                    ),
                ],
                id="no_matching_standards",
            ),
        ],
    )
    def test_interpret_error_cases(
        self,
        sequence: HLASequence,
        locus: HLA_LOCI,
        threshold: int,
        standards: list[HLAStandard],
        mocker: MockerFixture,
    ):
        easyhla: EasyHLA = get_dummy_easyhla(locus)
        # Replace the standards with the ones in the test.
        easyhla.hla_standards = standards

        # Spy on the internals to make sure they're called correctly.
        get_matching_standards_spy: mocker.MagicMock = mocker.spy(
            easyhla, "get_matching_standards"
        )
        combine_standards_spy: mocker.MagicMock = mocker.spy(
            easyhla, "combine_standards"
        )
        get_mismatches_spy: mocker.MagicMock = mocker.spy(easyhla, "get_mismatches")

        with pytest.raises(EasyHLA.NoMatchingStandards):
            easyhla.interpret(sequence, threshold=threshold)

        get_matching_standards_spy.assert_called_once_with(
            sequence.sequence_for_interpretation,
            standards,
        )
        combine_standards_spy.assert_not_called()
        get_mismatches_spy.assert_not_called()


class TestEasyHLAMisc:
    def test_unknown_hla_gene(self):
        """
        Assert we raise a value error if we put in an unknown HLA gene.
        """
        with pytest.raises(ValueError):
            _ = EasyHLA("D")  # type: ignore[arg-type]

    def test_known_hla_gene_lowercase(self):
        """
        Assert we raise a value error if we put in an HLA gene with wrong case.
        """
        with pytest.raises(ValueError):
            _ = EasyHLA("a")  # type: ignore[arg-type]

    def test_load_default_last_modified(
        self, hla_last_modified_file, timestamp, mocker
    ):
        """
        Assert we can load our mtime and that it is represented correctly.
        """
        mocker.patch.object(os.path, "join", return_value=hla_last_modified_file)

        # Annoyingly, while `strptime(%Z)` outputs a time in the correct timezone,
        # it doesn't keep that time in the object.
        expected_time = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )

        result = EasyHLA.load_default_last_modified()
        assert result == expected_time

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
    def test_check_bases(self, sequence: str, exp_good: bool):
        if exp_good:
            EasyHLA.check_bases(seq=sequence)
        else:
            with pytest.raises(ValueError):
                EasyHLA.check_bases(seq=sequence)

    @pytest.mark.parametrize(
        "sequence, standard, exp_result",
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
    def test_std_match(
        self,
        sequence: list[int],
        standard: list[int],
        exp_result: int,
    ):
        result = EasyHLA.std_match(std=np.array(standard), seq=np.array(sequence))
        print(result)
        assert result == exp_result

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
        self,
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
                [1] * 46
                + [1, 2, 4, 8]
                + [1] * 220
                + [2] * 150
                + [1, 2, 4, 8]
                + [1] * 122,
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
        self,
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
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([1, 2, 4, 8])
                    )
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    )
                ],
                id="one_standard_no_mismatches",
            ),
            pytest.param(
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([1, 2, 4, 4])
                    )
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    )
                ],
                id="one_standard_one_mismatch",
            ),
            pytest.param(
                np.array([1, 3, 4, 8]),
                [
                    HLAStandard(
                        allele="std_mixturematch", sequence=np.array([1, 2, 4, 8])
                    )
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_mixturematch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    )
                ],
                id="mixture_match",
            ),
            pytest.param(
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([8, 4, 2, 1])
                    )
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    )
                ],
                id="one_standard_all_mismatch",
            ),
            pytest.param(
                np.array([1, 2, 4, 8, 3, 5, 7, 9]),
                [
                    HLAStandard(
                        allele="std_mismatch_over_threshold",
                        sequence=np.array([1, 2, 8, 4, 4, 8, 8, 1]),
                    )
                ],
                5,
                [],
                id="one_standard_mismatch_above_threshold",
            ),
            pytest.param(
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(allele="std_allmatch", sequence=np.array([1, 2, 4, 8])),
                    HLAStandard(
                        allele="std_1mismatch", sequence=np.array([1, 2, 4, 4])
                    ),
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([8, 4, 2, 1])
                    ),
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    ),
                ],
                id="several_standards_below_threshold",
            ),
            pytest.param(
                np.array([1, 3, 4, 8, 2, 5, 4, 1]),
                [
                    HLAStandard(
                        allele="std_mixturematch",
                        sequence=np.array([1, 2, 4, 8, 2, 1, 4, 1]),
                    ),
                    HLAStandard(
                        allele="std_2mismatch",
                        sequence=np.array([1, 4, 4, 4, 2, 4, 4, 1]),
                    ),
                    HLAStandard(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1, 1, 8, 8, 8]),
                    ),
                    HLAStandard(
                        allele="std_4mismatch",
                        sequence=np.array([8, 4, 2, 1, 2, 1, 4, 1]),
                    ),
                ],
                5,
                [
                    HLAStandardMatch(
                        allele="std_mixturematch",
                        sequence=np.array([1, 2, 4, 8, 2, 1, 4, 1]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_2mismatch",
                        sequence=np.array([1, 4, 4, 4, 2, 4, 4, 1]),
                        mismatch=2,
                    ),
                    HLAStandardMatch(
                        allele="std_4mismatch",
                        sequence=np.array([8, 4, 2, 1, 2, 1, 4, 1]),
                        mismatch=4,
                    ),
                ],
                id="typical_case",
            ),
        ],
    )
    def test_get_matching_standards(
        self,
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


@pytest.mark.parametrize("easyhla", ["A"], indirect=True)
class TestEasyHLADiscreteHLALocusA:
    """
    Testing EasyHLA where tests require HLA-A.
    """

    # Tests of check_length:
    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * 1000, "myseq-a-bad", -1),
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-a00-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-a-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-a-maxbad", -1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-a-bad-exon2", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-a-bad-exon3", -1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-a00-good-exon2-short", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-a00-good-exon3-short", 1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-a01-good-exon2-short",
                1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-a01-good-exon3-short",
                1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-bad-exon3", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-bad-exon3", 0),
            # `
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-good-exon3-short", 1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_a(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_default_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="22",
                first_field_2="33",
                second_field_1="14",
                second_field_2="23",
            ): 1,
        }
        result = easyhla.load_default_hla_frequencies()
        assert result == exp_result


@pytest.mark.parametrize("easyhla", ["B"], indirect=True)
class TestEasyHLADiscreteHLALocusB:
    """
    Testing EasyHLA where tests require HLA-B.
    """

    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-b00-short", 1),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b01-short", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b02-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b-maxgood", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-b-good-exon2", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-b-good-exon3", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-b00-bad-exon2-short", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-b00-bad-exon3-short", -1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-b01-bad-exon2-short",
                -1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-b01-bad-exon3-short",
                -1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-bad-exon3", 0),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2-short", -1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3-short", -1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_b(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_default_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="57",
                first_field_2="01",
                second_field_1="57",
                second_field_2="03",
            ): 1,
        }
        result = easyhla.load_default_hla_frequencies()
        assert result == exp_result


@pytest.mark.parametrize("easyhla", ["C"], indirect=True)
class TestEasyHLADiscreteHLALocusC:
    """
    Testing EasyHLA where tests require HLA-C.
    """

    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-c00-short", 1),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c01-short", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c02-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c-maxgood", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-c-good-exon2", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-c-good-exon3", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-c00-bad-exon2-short", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-c00-bad-exon3-short", -1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-c01-bad-exon2-short",
                -1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-c01-bad-exon3-short",
                -1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-bad-exon3", 0),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2-short", -1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3-short", -1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_c(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_default_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="40",
                first_field_2="43",
                second_field_1="25",
                second_field_2="29",
            ): 1,
        }
        result = easyhla.load_default_hla_frequencies()
        assert result == exp_result

    # @pytest.mark.integration
    # def test_run(self, easyhla: EasyHLA):
    #     """
    #     Integration test, assert that pyEasyHLA produces an identical output to
    #     the original Ruby output.
    #     """
    #     input_file = os.path.dirname(__file__) + "/input/test.fasta"
    #     ref_output_file = os.path.dirname(__file__) + "/output/hla-c-output.csv"
    #     output_file = os.path.dirname(__file__) + "/output/test.csv"

    #     easyhla.run(
    #         input_file,
    #         output_file,
    #         0,
    #     )

    #     compare_ref_vs_test(
    #         easyhla=easyhla,
    #         reference_output_file=ref_output_file,
    #         output_file=output_file,
    #     )


@pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
class TestEasyHLA:
    def test_load_default_hla_standards(self, easyhla, hla_standard_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_standard_file)
        exp_result = [
            HLAStandard(
                allele="HELLO-WORLD", sequence=np.array([1, 1, 1, 1, 2, 1, 5, 8, 10])
            )
        ]

        result = easyhla.load_default_hla_standards()
        assert result == exp_result

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self, easyhla: EasyHLA):
        """
        Integration test, assert that pyEasyHLA produces an identical output to
        the original Ruby output.
        """

        input_file = (
            os.path.dirname(__file__) + f"/input/hla-{easyhla.locus.lower()}-seqs.fasta"
        )
        ref_output_file = (
            os.path.dirname(__file__)
            + f"/output/hla-{easyhla.locus.lower()}-output-ref.csv"
        )
        output_file = (
            os.path.dirname(__file__) + f"/output/hla-{easyhla.locus.lower()}-test.csv"
        )

        if not os.path.exists(input_file):
            pytest.skip("Input sequence does not exist!")
        if not os.path.exists(ref_output_file):
            pytest.skip("Reference output does not exist!")

        start_time = datetime.now()
        print(f"Test started at {start_time.isoformat()}")

        easyhla.run(
            input_file,
            output_file,
            0,
        )

        end_time = datetime.now()

        print(f"Interpretation ended at {end_time.isoformat()}")

        compare_ref_vs_test(
            easyhla=easyhla,
            reference_output_file=ref_output_file,
            output_file=output_file,
        )

        end_compare_time = datetime.now()

        print(f"Test ended at {end_compare_time.isoformat()}")

        print(f"Time elapsed: {(end_compare_time - start_time).total_seconds()}")
        print(
            f"Time elapsed for interpretation: {(end_time - start_time).total_seconds()}"
        )
        print(
            f"Time elapsed for output comparison: {(end_compare_time - end_time).total_seconds()}"
        )
