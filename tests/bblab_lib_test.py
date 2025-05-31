import pytest
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord

from easyhla.bblab_lib import (
    HLAInterpretationRow,
    HLAMismatchRow,
    pair_exons,
    pair_exons_helper,
)
from easyhla.models import (
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
)
from easyhla.utils import EXON_NAME, HLA_LOCUS, HLARawStandard, nuc2bin

from .clinical_hla_lib_test import DUMMY_FREQUENCIES, dummy_hla_sequence, dummy_matches
from .easyhla_test import HLA_STANDARDS


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
    result: tuple[str, bool, bool, str, str] = pair_exons_helper(sr, unmatched)
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
                    locus="A",
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
                    locus="B",
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
                    locus="B",
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
                    locus="B",
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
                    locus="B",
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
                    locus="B",
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
                    locus="C",
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
                    locus="C",
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
                    locus="C",
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
                    locus="B",
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
                    locus="B",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2[0:250] + "N" * 20),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E2",
                    locus="B",
                    num_sequences_used=2,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(1,) * 241,
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E4_full",
                    locus="B",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=nuc2bin(HLA_STANDARDS["B"].exon2),
                    intron=(),
                    three=nuc2bin(HLA_STANDARDS["B"].exon3),
                    name="E6",
                    locus="B",
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
    locus: HLA_LOCUS,
    expected_paired: list[HLASequence],
    expected_unmatched: dict[EXON_NAME, dict[str, Seq]],
):
    paired_seqs: list[HLASequence]
    unmatched: dict[EXON_NAME, dict[str, Seq]]

    current_standard: HLARawStandard = HLA_STANDARDS[locus]
    fake_standard: HLAStandard = HLAStandard(
        allele=current_standard.allele,
        two=nuc2bin(current_standard.exon2),
        three=nuc2bin(current_standard.exon3),
    )

    srs: list[SeqRecord] = [
        SeqRecord(id=id, seq=Seq(sequence)) for id, sequence in raw_sequence_records
    ]
    paired_seqs, unmatched = pair_exons(srs, locus, fake_standard)
    assert paired_seqs == expected_paired
    assert unmatched == expected_unmatched


@pytest.mark.parametrize(
    "hla_sequence, matches, allele_frequencies, expected_result",
    [
        pytest.param(
            dummy_hla_sequence("B"),
            dummy_matches("B"),
            DUMMY_FREQUENCIES,
            HLAInterpretationRow(
                enum="dummy_seq",
                alleles_clean="B*10:01 - B*20",
                alleles=(
                    "B*01:01:01 - B*02:02:02;B*10:01:10 - B*20:01;"
                    "B*10:01:10 - B*22:22:22;B*10:01:15 - B*20:02:03"
                ),
                ambiguous=1,
                homozygous=0,
                mismatch_count=1,
                mismatches="100:A->T;150:T->G",
                exon2="CCAC",
                intron="",
                exon3="AGGCT",
            ),
            id="no_colliding_mismatches",
        ),
        pytest.param(
            dummy_hla_sequence("A"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=150, observed_base="T", expected_base="G")
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                ): HLAMatchDetails(mismatch_count=3, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("A*10:01:10", "A*20:01"),
                        ("A*10:01:10", "A*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                    ],
                ),
            },
            DUMMY_FREQUENCIES,
            HLAInterpretationRow(
                enum="dummy_seq",
                alleles_clean="A*10:01 - A*20",
                alleles=(
                    "A*01:01:01 - A*02:02:02;A*10:01:10 - A*20:01;"
                    "A*10:01:10 - A*22:22:22;A*10:01:15 - A*20:02:03"
                ),
                ambiguous=1,
                homozygous=0,
                mismatch_count=1,
                mismatches="100:A->T;150:T->G",
                exon2="CCAC",
                intron="",
                exon3="AGGCT",
            ),
            id="one_colliding_identical_mismatch",
        ),
        pytest.param(
            dummy_hla_sequence("C"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("C*01:01:01", "C*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=150, observed_base="T", expected_base="R"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("C*10:01:15", "C*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("C*10:01:10", "C*20:22:20"),),
                ): HLAMatchDetails(mismatch_count=3, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("C*10:01:10", "C*20:01"),
                        ("C*10:01:10", "C*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                    ],
                ),
            },
            DUMMY_FREQUENCIES,
            HLAInterpretationRow(
                enum="dummy_seq",
                alleles_clean="C*10:01 - C*20",
                alleles=(
                    "C*01:01:01 - C*02:02:02;C*10:01:10 - C*20:01;"
                    "C*10:01:10 - C*22:22:22;C*10:01:15 - C*20:02:03"
                ),
                ambiguous=1,
                homozygous=0,
                mismatch_count=1,
                mismatches="100:A->T;150:T->G/R",
                exon2="CCAC",
                intron="",
                exon3="AGGCT",
            ),
            id="one_colliding_non_identical_mismatch",
        ),
        pytest.param(
            dummy_hla_sequence("B"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("B*01:01:01", "B*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:15", "B*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:10", "B*20:22:20"),),
                ): HLAMatchDetails(
                    mismatch_count=3,
                    mismatches=[
                        HLAMismatch(index=150, observed_base="T", expected_base="R")
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("B*10:01:10", "B*20:01"),
                        ("B*10:01:10", "B*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                    ],
                ),
            },
            DUMMY_FREQUENCIES,
            HLAInterpretationRow(
                enum="dummy_seq",
                alleles_clean="B*10:01 - B*20",
                alleles=(
                    "B*01:01:01 - B*02:02:02;B*10:01:10 - B*20:01;"
                    "B*10:01:10 - B*22:22:22;B*10:01:15 - B*20:02:03"
                ),
                ambiguous=1,
                homozygous=0,
                mismatch_count=1,
                mismatches="100:A->T;150:T->G",
                exon2="CCAC",
                intron="",
                exon3="AGGCT",
            ),
            id="collision_in_non_best_match_ignored",
        ),
        pytest.param(
            dummy_hla_sequence("B"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("B*01:01:01", "B*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:15", "B*20:02:03"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="C"),
                        HLAMismatch(index=175, observed_base="G", expected_base="T"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:10", "B*20:22:20"),),
                ): HLAMatchDetails(
                    mismatch_count=3,
                    mismatches=[
                        HLAMismatch(index=150, observed_base="T", expected_base="R"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("B*10:01:10", "B*20:01"),
                        ("B*10:01:10", "B*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                        HLAMismatch(index=175, observed_base="G", expected_base="Y"),
                    ],
                ),
            },
            DUMMY_FREQUENCIES,
            HLAInterpretationRow(
                enum="dummy_seq",
                alleles_clean="B*10:01 - B*20",
                alleles=(
                    "B*01:01:01 - B*02:02:02;B*10:01:10 - B*20:01;"
                    "B*10:01:10 - B*22:22:22;B*10:01:15 - B*20:02:03"
                ),
                ambiguous=1,
                homozygous=0,
                mismatch_count=1,
                mismatches="100:A->C/T;150:T->G;175:G->T/Y",
                exon2="CCAC",
                intron="",
                exon3="AGGCT",
            ),
            id="typical_case",
        ),
    ],
)
def test_hla_interpretation_row_summary_row_good_cases(
    hla_sequence: HLASequence,
    matches: dict[HLACombinedStandard, HLAMatchDetails],
    allele_frequencies: dict[HLAProteinPair, int],
    expected_result: HLAInterpretationRow,
):
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=hla_sequence,
        matches=matches,
        allele_frequencies=allele_frequencies,
    )
    result: HLAInterpretationRow = HLAInterpretationRow.summary_row(interp)
    assert result == expected_result


def test_hla_interpretation_row_summary_row_bad_case():
    matches: dict[HLACombinedStandard, HLAMatchDetails] = {
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 4),
            possible_allele_pairs=(("B*01:01:01", "B*02:02:02"),),
        ): HLAMatchDetails(
            mismatch_count=1,
            mismatches=[
                HLAMismatch(index=100, observed_base="A", expected_base="T"),
            ],
        ),
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 2),
            possible_allele_pairs=(("B*10:01:15", "B*20:02:03"),),
        ): HLAMatchDetails(
            mismatch_count=1,
            mismatches=[
                HLAMismatch(index=100, observed_base="C", expected_base="C"),
            ],
        ),
    }
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("B"),
        matches=matches,
        allele_frequencies=DUMMY_FREQUENCIES,
    )
    expected_error: str = (
        'In sequence "dummy_seq", different bases were reported in mismatches '
        "at position 100."
    )
    with pytest.raises(ValueError) as e:
        HLAInterpretationRow.summary_row(interp)
        assert expected_error in str(e.value)


@pytest.mark.parametrize(
    "hla_sequence, matches, raw_expected_result",
    [
        pytest.param(
            dummy_hla_sequence("A"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=0,
                    mismatches=[],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("A*10:01:10", "A*20:01"),
                        ("A*10:01:10", "A*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=0,
                    mismatches=[],
                ),
            },
            [
                ("A*01:01:01 - A*02:02:02", ""),
                ("A*10:01:10 - A*20:01|A*10:01:10 - A*22:22:22", ""),
                ("A*10:01:10 - A*20:22:20", ""),
                ("A*10:01:15 - A*20:02:03", ""),
            ],
            id="no_mismatches",
        ),
        pytest.param(
            dummy_hla_sequence("A"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("A*10:01:10", "A*20:01"),
                        ("A*10:01:10", "A*22:22:22"),
                    ),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
            },
            [
                ("A*10:01:10 - A*20:01|A*10:01:10 - A*22:22:22", ""),
                ("A*10:01:10 - A*20:22:20", ""),
                ("A*10:01:15 - A*20:02:03", ""),
                ("A*01:01:01 - A*02:02:02", "100:A->T"),
            ],
            id="one_combined_standard_with_single_mismatch",
        ),
        pytest.param(
            dummy_hla_sequence("A"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=0,
                    mismatches=[],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:15", "A*20:02:03"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("A*10:01:10", "A*20:22:20"),),
                ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("A*10:01:10", "A*20:01"),
                        ("A*10:01:10", "A*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=2,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                    ],
                ),
            },
            [
                ("A*01:01:01 - A*02:02:02", ""),
                ("A*10:01:10 - A*20:22:20", ""),
                ("A*10:01:15 - A*20:02:03", ""),
                ("A*10:01:10 - A*20:01|A*10:01:10 - A*22:22:22", "100:A->T;150:T->G"),
            ],
            id="one_combined_standard_with_two_mismatches",
        ),
        pytest.param(
            dummy_hla_sequence("B"),
            {
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 4),
                    possible_allele_pairs=(("B*01:01:01", "B*02:02:02"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(1, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:15", "B*20:02:03"),),
                ): HLAMatchDetails(
                    mismatch_count=2,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="C"),
                        HLAMismatch(index=175, observed_base="G", expected_base="T"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(2, 4, 9, 2),
                    possible_allele_pairs=(("B*10:01:10", "B*20:22:20"),),
                ): HLAMatchDetails(
                    mismatch_count=1,
                    mismatches=[
                        HLAMismatch(index=150, observed_base="T", expected_base="R"),
                    ],
                ),
                HLACombinedStandard(
                    standard_bin=(2, 4, 10, 2),
                    possible_allele_pairs=(
                        ("B*10:01:10", "B*20:01"),
                        ("B*10:01:10", "B*22:22:22"),
                    ),
                ): HLAMatchDetails(
                    mismatch_count=3,
                    mismatches=[
                        HLAMismatch(index=100, observed_base="A", expected_base="T"),
                        HLAMismatch(index=150, observed_base="T", expected_base="G"),
                        HLAMismatch(index=175, observed_base="G", expected_base="Y"),
                    ],
                ),
            },
            [
                ("B*01:01:01 - B*02:02:02", "100:A->T"),
                ("B*10:01:10 - B*20:22:20", "150:T->R"),
                ("B*10:01:15 - B*20:02:03", "100:A->C;175:G->T"),
                (
                    "B*10:01:10 - B*20:01|B*10:01:10 - B*22:22:22",
                    "100:A->T;150:T->G;175:G->Y",
                ),
            ],
            id="typical_case",
        ),
    ],
)
def test_hla_mismatch_row_mismatch_rows(
    hla_sequence: HLASequence,
    matches: dict[HLACombinedStandard, HLAMatchDetails],
    raw_expected_result: list[tuple[str, str]],
):
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=hla_sequence,
        matches=matches,
        allele_frequencies={},
    )
    result: list[HLAMismatchRow] = HLAMismatchRow.mismatch_rows(interp)
    expected_result: list[HLAMismatchRow] = [
        HLAMismatchRow(
            allele=allele,
            mismatches=mismatches,
            exon2="CCAC",
            intron="",
            exon3="AGGCT",
        )
        for allele, mismatches in raw_expected_result
    ]
    assert result == expected_result
