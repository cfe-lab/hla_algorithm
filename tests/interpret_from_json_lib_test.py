from typing import Optional

import pytest

from easyhla.interpret_from_json_lib import HLAInput, HLAResult
from easyhla.models import (
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
)

from .clinical_hla_lib_test import (
    B5701_CASE_STANDARDS,
    DUMMY_FREQUENCIES,
    FREQUENCIES_FOR_B5701_CASES,
    MATCHES_FOR_B5701_CASES,
    dummy_hla_sequence,
    dummy_matches,
)


@pytest.mark.parametrize(
    "seq1, seq2, expected_result",
    [
        pytest.param(
            "A" * 787,
            None,
            [],
            id="no_errors",
        ),
        pytest.param(
            "A" * 270,
            "C" * 276,
            ["Wrong number of sequences (needs 1)"],
            id="wrong_number_of_sequences",
        ),
        pytest.param(
            "A" * 850,
            None,
            ["Sequence is the wrong size (should be 787)"],
            id="bad_length",
        ),
        pytest.param(
            "A" * 200 + "_" + "C" * 586,
            None,
            ["Sequence has invalid characters"],
            id="invalid_characters",
        ),
        pytest.param(
            "A" * 200 + "_" + "C" * 69,
            "C" * 276,
            [
                "Wrong number of sequences (needs 1)",
                "Sequence has invalid characters",
            ],
            id="wrong_number_of_sequences_and_invalid_characters",
        ),
        pytest.param(
            "A" * 500 + "_" + "C" * 100,
            None,
            [
                "Sequence is the wrong size (should be 787)",
                "Sequence has invalid characters",
            ],
            id="bad_length_and_invalid_characters",
        ),
    ],
)
def test_hla_input_check_sequences_locus_a(
    seq1: str,
    seq2: Optional[str],
    expected_result: list[str],
):
    hla_input: HLAInput = HLAInput(seq1=seq1, seq2=seq2, locus="A")
    result: list[str] = hla_input.check_sequences()
    assert result == expected_result


@pytest.mark.parametrize(
    "seq1, seq2, expected_result",
    [
        pytest.param(
            "A" * 270,
            "C" * 276,
            [],
            id="no_errors",
        ),
        pytest.param(
            "A" * 787,
            None,
            ["Wrong number of sequences (needs 2)"],
            id="wrong_number_of_sequences",
        ),
        pytest.param(
            "A" * 850,
            "C" * 276,
            ["Sequence 1 is the wrong size (should be 270)"],
            id="bad_length_seq1",
        ),
        pytest.param(
            "A" * 270,
            "C" * 300,
            ["Sequence 2 is the wrong size (should be 276)"],
            id="bad_length_seq2",
        ),
        pytest.param(
            "A" * 200,
            "C" * 277,
            [
                "Sequence 1 is the wrong size (should be 270)",
                "Sequence 2 is the wrong size (should be 276)",
            ],
            id="bad_length_both_sequences",
        ),
        pytest.param(
            "A" * 200 + "_" + "C" * 69,
            "C" * 276,
            ["Sequence has invalid characters"],
            id="invalid_characters_seq1",
        ),
        pytest.param(
            "A" * 270,
            "C" * 200 + "Q" + "C" * 75,
            ["Sequence has invalid characters"],
            id="invalid_characters_seq2",
        ),
        pytest.param(
            "A" * 200 + "_" + "C" * 69,
            "C" * 200 + "Q" + "C" * 75,
            ["Sequence has invalid characters"],
            id="invalid_characters_both_sequences",
        ),
        pytest.param(
            "A" * 200 + "_" + "C" * 69,
            None,
            [
                "Wrong number of sequences (needs 2)",
                "Sequence has invalid characters",
            ],
            id="wrong_number_of_sequences_and_invalid_characters",
        ),
        pytest.param(
            "A" * 500 + "_" + "G" * 200,
            "C" * 276,
            [
                "Sequence 1 is the wrong size (should be 270)",
                "Sequence has invalid characters",
            ],
            id="bad_length_seq1_and_invalid_characters",
        ),
        pytest.param(
            "A" * 270,
            "C" * 200 + "_" + "C" * 74,
            [
                "Sequence 2 is the wrong size (should be 276)",
                "Sequence has invalid characters",
            ],
            id="bad_length_seq2_and_invalid_characters",
        ),
        pytest.param(
            "A" * 200 + "_" + "A" * 68,
            "C" * 277,
            [
                "Sequence 1 is the wrong size (should be 270)",
                "Sequence 2 is the wrong size (should be 276)",
                "Sequence has invalid characters",
            ],
            id="bad_length_both_sequences_and_invalid_characters",
        ),
    ],
)
def test_hla_input_check_sequences_locus_bc(
    seq1: str,
    seq2: Optional[str],
    expected_result: list[str],
):
    for locus in ("B", "C"):
        hla_input: HLAInput = HLAInput(seq1=seq1, seq2=seq2, locus=locus)
        result: list[str] = hla_input.check_sequences()
        assert result == expected_result


def test_hla_input_hla_sequence_locus_a():
    hla_input: HLAInput = HLAInput(
        seq1="A" * 270 + "C" * 241 + "G" * 276,
        seq2=None,
        locus="A",
    )
    expected_result: HLASequence = HLASequence(
        two=(1,) * 270,
        intron=(2,) * 241,
        three=(4,) * 276,
        name="input_sequence",
        locus="A",
        num_sequences_used=1,
    )
    assert hla_input.hla_sequence() == expected_result


def test_hla_input_hla_sequence_locus_bc():
    for locus in ("B", "C"):
        hla_input: HLAInput = HLAInput(
            seq1="A" * 270,
            seq2="G" * 276,
            locus=locus,
        )
        expected_result: HLASequence = HLASequence(
            two=(1,) * 270,
            intron=(),
            three=(4,) * 276,
            name="input_sequence",
            locus=locus,
            num_sequences_used=2,
        )
        assert hla_input.hla_sequence() == expected_result


@pytest.mark.parametrize(
    "hla_sequence, matches, frequencies, b5701_standards, expected_result",
    [
        pytest.param(
            dummy_hla_sequence("A"),
            dummy_matches("A"),
            DUMMY_FREQUENCIES,
            None,
            HLAResult(
                seqs=["CCACAGGCT"],
                alleles_all=[
                    "A*01:01:01 - A*02:02:02",
                    "A*10:01:10 - A*20:01",
                    "A*10:01:10 - A*111:22:22",
                    "A*10:01:15 - A*20:02:03",
                ],
                alleles_clean="A*10:01 - A*20",
                alleles_for_mismatches="A*10:01:10 - A*20:01",
                mismatches=["100:A->T", "150:T->G"],
                ambiguous=True,
                homozygous=False,
                type="A",
                b5701=False,
                dist_b5701=None,
            ),
            id="a_typical_case",
        ),
        pytest.param(
            dummy_hla_sequence("B"),
            MATCHES_FOR_B5701_CASES,
            FREQUENCIES_FOR_B5701_CASES,
            B5701_CASE_STANDARDS,
            HLAResult(
                seqs=["CCAC", "AGGCT"],
                alleles_all=[
                    "B*57:01:01 - B*57:01:01",
                    "B*57:01:15 - B*57:01:03",
                    "B*57:02:10 - B*57:01:01:03N",
                    "B*57:04:10 - B*57:01:22",
                ],
                alleles_clean="B*57 - B*57:01",
                alleles_for_mismatches="B*57:01:01 - B*57:01:01",
                mismatches=["3:A->W"],
                ambiguous=False,
                homozygous=True,
                type="B",
                b5701=True,
                dist_b5701=1,
            ),
            id="b_typical_case",
        ),
        pytest.param(
            dummy_hla_sequence("C"),
            dummy_matches("C"),
            DUMMY_FREQUENCIES,
            None,
            HLAResult(
                seqs=["CCAC", "AGGCT"],
                alleles_all=[
                    "C*01:01:01 - C*02:02:02",
                    "C*10:01:10 - C*20:01",
                    "C*10:01:10 - C*111:22:22",
                    "C*10:01:15 - C*20:02:03",
                ],
                alleles_clean="C*10:01 - C*20",
                alleles_for_mismatches="C*10:01:10 - C*20:01",
                mismatches=["100:A->T", "150:T->G"],
                ambiguous=True,
                homozygous=False,
                type="C",
                b5701=False,
                dist_b5701=None,
            ),
            id="c_typical_case",
        ),
    ],
)
def test_hla_result_build_from_interpretation(
    hla_sequence: HLASequence,
    matches: dict[HLACombinedStandard, HLAMatchDetails],
    frequencies: dict[HLAProteinPair, int],
    b5701_standards: Optional[list[HLAStandard]],
    expected_result: HLAResult,
):
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=hla_sequence,
        matches=matches,
        allele_frequencies=frequencies,
        b5701_standards=b5701_standards,
    )
    result: HLAResult = HLAResult.build_from_interpretation(interp)
    assert result == expected_result
