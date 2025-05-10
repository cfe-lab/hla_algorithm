import pytest
from datetime import datetime

from easyhla.clinical_hla_lib import HLASequenceA, sanitize_sequence
from easyhla.models import (
    HLASequence,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAProteinPair,
    HLAMismatch,
)


def test_hla_sequence_a_build_from_interpretation():
    hla_seq: HLASequence = HLASequence(
        two=(2, 2, 1, 2),  # "CCTC"
        intron=(),
        three=(1, 4, 4, 2, 8),  # "AGGCT"
        name="dummy_seq",
        locus="A",
        num_sequences_used=1,
    )
    matches: dict[HLACombinedStandard, HLAMatchDetails] = {
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 4),
            possible_allele_pairs=(("A*01:01:01", "A*02:02:02"),),
        ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 2),
            possible_allele_pairs=(("A*10:01:01", "A*20:02:03"),),
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
    }
    frequencies: dict[HLAProteinPair, int] = {
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
    }

    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=hla_seq,
        matches=matches,
        allele_frequencies=frequencies,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_a: HLASequenceA = HLASequenceA.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceA = HLASequenceA(
        enum="dummy_seq",
        alleles_clean="A*10:01 - A*20",
        alleles_all="A*01:01:01 - A*02:02:02;A*10:01:01 - A*20:02:03;A*10:01:10 - A*20:01;A*10:01:01 - A*22:22:22",  # FIXME this prob needs to be sorted?
        ambiguous="True",
        homozygous="False",
        mismatch_count=1,
        mismatches="(A*10:01:10 - A*20:01) 100:T->A;150:G->T",
        seq="CCTCAGGCT",
        enterdate=processing_datetime,
    )

    assert seq_a == expected_result
