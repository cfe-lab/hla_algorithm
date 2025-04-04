import os
from collections.abc import Iterable
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import pytz
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord

from easyhla.easyhla import DATE_FORMAT, EXON_NAME, EasyHLA
from easyhla.models import (
    HLACombinedStandard,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)

from .conftest import compare_ref_vs_test


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
        "raw_sequence_records, raw_standards, expected_paired, expected_unmatched",
        [
            pytest.param(
                [("E1", "A" * 787)],
                [("A*01:01:01:01", "A" * 270, "AT" * 138)],
                [
                    HLASequence(
                        two="A" * 270,
                        intron="A" * 241,
                        three="AT" * 138,
                        sequence=np.array([1] * 546),
                        name="E1",
                        num_sequences_used=1,
                    ),
                ],
                {},
                id="single_full_sequence",
            ),
        ],
    )
    def test_pair_exons(
        self,
        raw_sequence_records: list[tuple[str, str]],
        raw_standards: list[tuple[str, str, str]],
        expected_paired: list[HLASequence],
        expected_unmatched: dict[EXON_NAME, dict[str, Seq]],
    ):
        dummy_standard_strings: list[str] = [
            f"{allele},{exon2},{exon3}" for allele, exon2, exon3 in raw_standards
        ]
        easyhla: EasyHLA = EasyHLA(
            "A", hla_standards=StringIO("\n".join(dummy_standard_strings) + "\n")
        )
        paired_seqs: list[HLASequence]
        unmatched: dict[EXON_NAME, dict[str, Seq]]

        srs: list[SeqRecord] = [
            SeqRecord(id=id, seq=Seq(sequence)) for id, sequence in raw_sequence_records
        ]
        paired_seqs, unmatched = easyhla.pair_exons(srs)
        assert paired_seqs == expected_paired
        assert unmatched == expected_unmatched


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

    def test_load_allele_definitions_last_modified_time(
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

        result = EasyHLA.load_allele_definitions_last_modified_time()
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
        "sequence_str, sequence_list",
        [
            ("ACGT", np.array([1, 2, 4, 8])),
            ("YANK", np.array([10, 1, 15, 12])),
            ("TARDY", np.array([8, 1, 5, 13, 10])),
            ("MRMAN", np.array([3, 5, 3, 1, 15])),
            ("MYSWARD", np.array([3, 10, 6, 9, 1, 5, 13])),
            # This is where I pull out scrabblewordfinder.org
            ("GANTRY", np.array([4, 1, 15, 8, 5, 10])),
            ("SKYWATCH", np.array([6, 12, 10, 9, 1, 8, 2, 11])),
            ("THWACK", np.array([8, 11, 9, 1, 2, 12])),
            ("VAN", np.array([7, 1, 15])),
            ("ABRA", np.array([1, 14, 5, 1])),
        ],
    )
    def test_nuc2bin_bin2nuc_good_cases(
        self, sequence_str: str, sequence_list: np.ndarray
    ):
        """
        Test that we can convert back and forth between a list of binary values
        and strings.
        """
        result_str = EasyHLA.bin2nuc(sequence_list)
        assert result_str == sequence_str
        result_list = EasyHLA.nuc2bin(sequence_str)
        assert np.array_equal(result_list, sequence_list)

    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("E", np.array([0])),
            ("a", np.array([0])),
            ("123", np.array([0, 0, 0])),
            ("AC_TT_G", np.array([1, 2, 0, 8, 8, 0, 4])),
        ],
    )
    def test_nuc2bin_bad_characters(self, sequence_str: str, sequence_list: np.ndarray):
        """
        Translating characters that aren't in the mapping turns them into 0s.
        """
        result_list = EasyHLA.nuc2bin(sequence_str)
        assert np.array_equal(result_list, sequence_list)

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
    def test_bin2nuc_bad_indices(self, sequence_list: np.ndarray, sequence_str: str):
        """
        Indices not in our mapping become underscores.
        """
        result_str = EasyHLA.bin2nuc(sequence_list)
        assert np.array_equal(result_str, sequence_str)

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
        std_bin: Iterable[int],
        seq_bin: Iterable[int],
        exon: Optional[EXON_NAME],
        exp_raw_result: Iterable[int],
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

    # FIXME: some mixtures here too? and maybe some ties?
    # FIXME continue from here
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
    def test_get_matching_stds(
        self,
        sequence: np.ndarray,
        hla_stds: Iterable[HLAStandard],
        mismatch_threshold: int,
        exp_result: Iterable[HLAStandardMatch],
    ):
        result = EasyHLA.get_matching_stds(
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

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="22",
                first_field_2="33",
                second_field_1="14",
                second_field_2="23",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
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

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="57",
                first_field_2="01",
                second_field_1="57",
                second_field_2="03",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
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

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="40",
                first_field_2="43",
                second_field_1="25",
                second_field_2="29",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
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
    def test_load_hla_stds(self, easyhla, hla_standard_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_standard_file)
        exp_result = [
            HLAStandard(
                allele="HELLO-WORLD", sequence=np.array([1, 1, 1, 1, 2, 1, 5, 8, 10])
            )
        ]

        result = easyhla.load_hla_stds()
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
