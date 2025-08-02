from datetime import datetime
from pathlib import Path
from typing import Final

import pytest
from pytest_mock import MockerFixture

from easyhla.clinical_hla_lib import (
    HLASequenceA,
    HLASequenceB,
    HLASequenceC,
    identify_bc_sequence_files,
    read_a_sequences,
    read_bc_sequences,
    sanitize_sequence,
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
from easyhla.utils import (
    EXON_NAME,
    HLA_A_LENGTH,
    HLA_LOCUS,
    BadLengthException,
    InvalidBaseException,
)


def dummy_hla_sequence(locus: HLA_LOCUS) -> HLASequence:
    return HLASequence(
        two=(2, 2, 1, 2),  # "CCAC"
        intron=(),
        three=(1, 4, 4, 2, 8),  # "AGGCT"
        name="dummy_seq",
        locus=locus,
        num_sequences_used=1,
    )


def dummy_matches(locus: HLA_LOCUS) -> dict[HLACombinedStandard, HLAMatchDetails]:
    return {
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 4),
            possible_allele_pairs=((f"{locus}*01:01:01", f"{locus}*02:02:02"),),
        ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
        HLACombinedStandard(
            standard_bin=(1, 4, 9, 2),
            possible_allele_pairs=((f"{locus}*10:01:15", f"{locus}*20:02:03"),),
        ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
        HLACombinedStandard(
            standard_bin=(2, 4, 9, 2),
            possible_allele_pairs=((f"{locus}*10:01:10", f"{locus}*20:22:20"),),
        ): HLAMatchDetails(mismatch_count=3, mismatches=[]),
        HLACombinedStandard(
            standard_bin=(2, 4, 10, 2),
            possible_allele_pairs=(
                (f"{locus}*10:01:10", f"{locus}*20:01"),
                (f"{locus}*10:01:10", f"{locus}*111:22:22"),
            ),
        ): HLAMatchDetails(
            mismatch_count=1,
            mismatches=[
                HLAMismatch(index=100, observed_base="A", expected_base="T"),
                HLAMismatch(index=150, observed_base="T", expected_base="G"),
            ],
        ),
    }


def dummy_matches_no_mismatches(
    locus: HLA_LOCUS,
) -> dict[HLACombinedStandard, HLAMatchDetails]:
    return {
        HLACombinedStandard(
            standard_bin=(2, 2, 1, 2, 1, 4, 4, 2, 8),
            possible_allele_pairs=((f"{locus}*01:01:01", f"{locus}*02:02:02"),),
        ): HLAMatchDetails(mismatch_count=0, mismatches=[]),
    }


DUMMY_FREQUENCIES: Final[dict[HLAProteinPair, int]] = {
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


def test_hla_sequence_a_build_from_interpretation():
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("A"),
        matches=dummy_matches("A"),
        allele_frequencies=DUMMY_FREQUENCIES,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_a: HLASequenceA = HLASequenceA.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceA = HLASequenceA(
        enum="dummy_seq",
        alleles_clean="A*10:01 - A*20",
        alleles_all=(
            "A*01:01:01 - A*02:02:02;A*10:01:10 - A*20:01;"
            "A*10:01:10 - A*111:22:22;A*10:01:15 - A*20:02:03"
        ),
        ambiguous="True",
        homozygous="False",
        mismatch_count=1,
        mismatches="(A*10:01:10 - A*20:01) 100:A->T;150:T->G",
        seq="CCACAGGCT",
        enterdate=processing_datetime,
    )

    assert seq_a == expected_result


def test_hla_sequence_a_build_from_interpretation_no_mismatches():
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("A"),
        matches=dummy_matches_no_mismatches("A"),
        allele_frequencies=DUMMY_FREQUENCIES,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_a: HLASequenceA = HLASequenceA.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceA = HLASequenceA(
        enum="dummy_seq",
        alleles_clean="A*01:01:01 - A*02:02:02",
        alleles_all="A*01:01:01 - A*02:02:02",
        ambiguous="False",
        homozygous="False",
        mismatch_count=0,
        mismatches=None,
        seq="CCACAGGCT",
        enterdate=processing_datetime,
    )

    assert seq_a == expected_result


def test_hla_sequence_b_build_from_interpretation_non_b5701():
    b5701_standards: list[HLAStandard] = [
        # "Forgiving distance" from sequence: 8
        HLAStandard(
            allele="B*57:01:01",
            two=(8, 8, 8, 8),
            three=(8, 8, 8, 8, 8),
        ),
        # "Forgiving distance" from sequence: 7
        HLAStandard(
            allele="B*57:01:02:01N",
            two=(4, 8, 4, 8),
            three=(8, 4, 8, 4, 8),
        ),
        # "Forgiving distance" from sequence: 6, because 12 "contains" 8
        HLAStandard(
            allele="B*57:01:03",
            two=(4, 4, 8, 8),
            three=(8, 4, 4, 8, 12),
        ),
    ]

    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("B"),
        matches=dummy_matches("B"),
        allele_frequencies=DUMMY_FREQUENCIES,
        b5701_standards=b5701_standards,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_b: HLASequenceB = HLASequenceB.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceB = HLASequenceB(
        enum="dummy_seq",
        alleles_clean="B*10:01 - B*20",
        alleles_all=(
            "B*01:01:01 - B*02:02:02;B*10:01:10 - B*20:01;"
            "B*10:01:10 - B*111:22:22;B*10:01:15 - B*20:02:03"
        ),
        ambiguous="True",
        homozygous="False",
        mismatch_count=1,
        mismatches="(B*10:01:10 - B*20:01) 100:A->T;150:T->G",
        b5701="False",
        b5701_dist=6,
        seqa="CCAC",
        seqb="AGGCT",
        reso_status=None,
        enterdate=processing_datetime,
    )

    assert seq_b == expected_result


def test_hla_sequence_b_build_from_interpretation_no_mismatches():
    b5701_standards: list[HLAStandard] = [
        # "Forgiving distance" from sequence: 8
        HLAStandard(
            allele="B*57:01:01",
            two=(8, 8, 8, 8),
            three=(8, 8, 8, 8, 8),
        ),
        # "Forgiving distance" from sequence: 7
        HLAStandard(
            allele="B*57:01:02:01N",
            two=(4, 8, 4, 8),
            three=(8, 4, 8, 4, 8),
        ),
        # "Forgiving distance" from sequence: 6, because 12 "contains" 8
        HLAStandard(
            allele="B*57:01:03",
            two=(4, 4, 8, 8),
            three=(8, 4, 4, 8, 12),
        ),
    ]
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("B"),
        matches=dummy_matches_no_mismatches("B"),
        allele_frequencies=DUMMY_FREQUENCIES,
        b5701_standards=b5701_standards,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_b: HLASequenceB = HLASequenceB.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceB = HLASequenceB(
        enum="dummy_seq",
        alleles_clean="B*01:01:01 - B*02:02:02",
        alleles_all="B*01:01:01 - B*02:02:02",
        ambiguous="False",
        homozygous="False",
        mismatch_count=0,
        mismatches=None,
        b5701="False",
        b5701_dist=6,
        seqa="CCAC",
        seqb="AGGCT",
        reso_status=None,
        enterdate=processing_datetime,
    )

    assert seq_b == expected_result


MATCHES_FOR_B5701_CASES: dict[HLACombinedStandard, HLAMatchDetails] = {
    HLACombinedStandard(
        standard_bin=(1, 4, 9, 4),
        possible_allele_pairs=(("B*57:01:01", "B*57:01:01"),),
    ): HLAMatchDetails(
        mismatch_count=1,
        mismatches=[HLAMismatch(index=3, observed_base="A", expected_base="W")],
    ),
    HLACombinedStandard(
        standard_bin=(1, 4, 9, 2),
        possible_allele_pairs=(("B*57:01:15", "B*57:01:03"),),
    ): HLAMatchDetails(mismatch_count=1, mismatches=[]),
    HLACombinedStandard(
        standard_bin=(2, 4, 9, 2),
        possible_allele_pairs=(("B*57:02:33", "B*56:04:22"),),
    ): HLAMatchDetails(mismatch_count=3, mismatches=[]),
    HLACombinedStandard(
        standard_bin=(2, 4, 10, 2),
        possible_allele_pairs=(
            ("B*57:02:10", "B*57:01:01:03N"),
            ("B*57:04:10", "B*57:01:22"),
        ),
    ): HLAMatchDetails(
        mismatch_count=1,
        mismatches=[
            HLAMismatch(index=100, observed_base="A", expected_base="T"),
            HLAMismatch(index=150, observed_base="T", expected_base="G"),
        ],
    ),
}

FREQUENCIES_FOR_B5701_CASES: dict[HLAProteinPair, int] = {
    HLAProteinPair(
        first_field_1="01",
        first_field_2="01",
        second_field_1="02",
        second_field_2="02",
    ): 150,
    HLAProteinPair(
        first_field_1="57",
        first_field_2="01",
        second_field_1="57",
        second_field_2="01",
    ): 1500,
}

# The first of these has "forgiving distance" 1 to the sequence generated by
# dummy_hla_sequence("B").
B5701_CASE_STANDARDS: list[HLAStandard] = [
    HLAStandard(
        allele="B*57:01:01",
        two=(2, 4, 1, 2),
        three=(1, 4, 4, 2, 8),
    ),
    HLAStandard(
        allele="B*57:01:02",
        two=(2, 1, 1, 2),
        three=(1, 4, 8, 2, 8),
    ),
    HLAStandard(
        allele="B*57:01:03",
        two=(2, 5, 3, 2),
        three=(1, 4, 8, 10, 9),
    ),
]


def test_hla_sequence_b_build_from_interpretation_is_b5701():
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("B"),
        matches=MATCHES_FOR_B5701_CASES,
        allele_frequencies=FREQUENCIES_FOR_B5701_CASES,
        b5701_standards=B5701_CASE_STANDARDS,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_b: HLASequenceB = HLASequenceB.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceB = HLASequenceB(
        enum="dummy_seq",
        alleles_clean="B*57 - B*57:01",
        alleles_all=(
            "B*57:01:01 - B*57:01:01;B*57:01:15 - B*57:01:03;"
            "B*57:02:10 - B*57:01:01:03N;B*57:04:10 - B*57:01:22"
        ),
        ambiguous="False",
        homozygous="True",
        mismatch_count=1,
        mismatches="(B*57:01:01 - B*57:01:01) 3:A->W",
        b5701="True",
        b5701_dist=1,
        seqa="CCAC",
        seqb="AGGCT",
        reso_status="pending",
        enterdate=processing_datetime,
    )

    assert seq_b == expected_result


def test_hla_sequence_c_build_from_interpretation():
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("C"),
        matches=dummy_matches("C"),
        allele_frequencies=DUMMY_FREQUENCIES,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_c: HLASequenceC = HLASequenceC.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceC = HLASequenceC(
        enum="dummy_seq",
        alleles_clean="C*10:01 - C*20",
        alleles_all=(
            "C*01:01:01 - C*02:02:02;C*10:01:10 - C*20:01;"
            "C*10:01:10 - C*111:22:22;C*10:01:15 - C*20:02:03"
        ),
        ambiguous="True",
        homozygous="False",
        mismatch_count=1,
        mismatches="(C*10:01:10 - C*20:01) 100:A->T;150:T->G",
        seqa="CCAC",
        seqb="AGGCT",
        enterdate=processing_datetime,
    )

    assert seq_c == expected_result


def test_hla_sequence_c_build_from_interpretation_no_mismatches():
    interp: HLAInterpretation = HLAInterpretation(
        hla_sequence=dummy_hla_sequence("C"),
        matches=dummy_matches_no_mismatches("C"),
        allele_frequencies=DUMMY_FREQUENCIES,
    )
    processing_datetime: datetime = datetime(2025, 5, 9, 11, 0, 0)

    seq_c: HLASequenceC = HLASequenceC.build_from_interpretation(
        interp, processing_datetime
    )

    expected_result: HLASequenceC = HLASequenceC(
        enum="dummy_seq",
        alleles_clean="C*01:01:01 - C*02:02:02",
        alleles_all="C*01:01:01 - C*02:02:02",
        ambiguous="False",
        homozygous="False",
        mismatch_count=0,
        mismatches=None,
        seqa="CCAC",
        seqb="AGGCT",
        enterdate=processing_datetime,
    )

    assert seq_c == expected_result


@pytest.mark.parametrize(
    "raw_contents, locus, sample_name, expected_sanitized_contents",
    [
        pytest.param(
            "A" * HLA_A_LENGTH,
            "A",
            "E12345",
            "A" * HLA_A_LENGTH,
            id="no_sanitization_needed",
        ),
        pytest.param(
            ">sequence1\n" + "A" * HLA_A_LENGTH,
            "A",
            "E12345",
            "A" * HLA_A_LENGTH,
            id="fasta_header_removed",
        ),
        pytest.param(
            "A" * 50 + " " + "C" * 50 + "\t\t" + "A" * 100 + "\n" + "T" * 70,
            "B",
            "E12345_exon2",
            "A" * 50 + "C" * 50 + "A" * 100 + "T" * 70,
            id="whitespace_removed",
        ),
        pytest.param(
            (
                ">sequence123\n"
                + "A" * 50
                + " "
                + "C" * 50
                + "\t\t"
                + "A" * 100
                + "\n"
                + "T" * 76
            ),
            "C",
            "E12345_exon3",
            "A" * 50 + "C" * 50 + "A" * 100 + "T" * 76,
            id="fasta_header_and_whitespace_removed",
        ),
    ],
)
def test_sanitize_sequences_good(
    raw_contents: str,
    locus: HLA_LOCUS,
    sample_name: str,
    expected_sanitized_contents: str,
):
    result: str = sanitize_sequence(raw_contents, locus, sample_name)
    assert result == expected_sanitized_contents


@pytest.mark.parametrize(
    "raw_contents, locus, sample_name, expected_length_str, actual_length",
    [
        pytest.param(
            "A" * 300,
            "B",
            "E12345_exon2_short",
            "<270",
            300,
            id="bad_length",
        ),
        pytest.param(
            "A" * 100 + "_" + "C" * 796,
            "C",
            "E12345_full",
            "[787, 796]",
            897,
            id="bad_length_overrules_invalid_character",
        ),
    ],
)
def test_sanitize_sequences_bad_length(
    raw_contents: str,
    locus: HLA_LOCUS,
    sample_name: str,
    expected_length_str: str,
    actual_length: int,
):
    with pytest.raises(BadLengthException) as e:
        sanitize_sequence(raw_contents, locus, sample_name)
        assert e.expected_length == expected_length_str
        assert e.actual_length == actual_length


def test_sanitize_sequences_invalid_character():
    raw_contents: str = "A" * 100 + "_" + "C" * 150
    sample_name: str = "E12345_exon3_short"
    with pytest.raises(InvalidBaseException) as e:
        sanitize_sequence(raw_contents, "C", sample_name)


@pytest.mark.parametrize(
    "raw_sequences, expected_sequences, expected_logger_calls",
    [
        pytest.param(
            {"E1.A.txt": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_no_issues",
        ),
        pytest.param(
            {"E1_A.txt": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_underscore_delimiter_no_issues",
        ),
        pytest.param(
            {"E1-A.txt": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_dash_delimiter_no_issues",
        ),
        pytest.param(
            {"E1.a.txt": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_lower_case_a_no_issues",
        ),
        pytest.param(
            {"E1.A.TXT": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_upper_case_txt_no_issues",
        ),
        pytest.param(
            {"E1.a.TXT": "A" * 787},
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [],
            id="single_file_lower_case_a_upper_case_txt_no_issues",
        ),
        pytest.param(
            {"E1_A.txt": "A" * 700 + "_" + "C" * 86},
            [],
            [
                'Skipping HLA-A sequence file "E1_A.txt": it contains invalid characters.'
            ],
            id="single_file_invalid_characters",
        ),
        pytest.param(
            {"E1_A.txt": "A" * 700},
            [],
            [
                'Skipping HLA-A sequence file "E1_A.txt": expected 787 characters, found 700.'
            ],
            id="single_file_bad_length",
        ),
        pytest.param(
            {"E1.B.txt": "A" * 700 + "C" * 87},
            [],
            [],
            id="non_a_file_skipped",
        ),
        pytest.param(
            {
                "E1.A.TXT": "A" * 787,
                "E2.A.txt": "A" * 500 + "C" * 287,
                "G5_C.txt": "T" * 787,
                "E3_A.txt": "A" * 600 + "G" * 187,
                "E4-A.TXT": "A" * 700,
                "E5_A.TXT": "A" * 100 + "C" * 687,
                "E6-A.TXT": "A" * 600 + "_" + "C" * 186,
                "G7.B.TXT": "C" * 787,
            },
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 276,
                    name="E1",
                    locus="A",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(2,) * 276,
                    name="E2",
                    locus="A",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(1,) * 89 + (4,) * 187,
                    name="E3",
                    locus="A",
                    num_sequences_used=1,
                ),
                HLASequence(
                    two=(1,) * 100 + (2,) * 170,
                    intron=(),
                    three=(2,) * 276,
                    name="E5",
                    locus="A",
                    num_sequences_used=1,
                ),
            ],
            [
                (
                    'Skipping HLA-A sequence file "E4-A.TXT": expected 787 '
                    "characters, found 700."
                ),
                (
                    'Skipping HLA-A sequence file "E6-A.TXT": it contains '
                    "invalid characters."
                ),
            ],
            id="typical_case",
        ),
    ],
)
def test_read_a_sequences(
    raw_sequences: dict[str, str],
    expected_sequences: list[HLASequence],
    expected_logger_calls: list[str],
    tmp_path: Path,
    mocker: MockerFixture,
):
    for filename, file_contents in raw_sequences.items():
        dummy_path: Path = tmp_path / filename
        dummy_path.write_text(file_contents)

    mock_logger: mocker.MagicMock = mocker.MagicMock()

    result: list[HLASequence] = read_a_sequences(str(tmp_path), mock_logger)

    assert result == expected_sequences
    if len(expected_logger_calls) > 0:
        mock_logger.info.assert_has_calls(
            [mocker.call(x) for x in expected_logger_calls],
            any_order=False,
        )


@pytest.mark.parametrize(
    "filenames, locus, expected_result, expected_logger_calls",
    [
        pytest.param(
            ["E1.BA.txt"],
            "B",
            {"E1": {"exon2": "E1.BA.txt", "exon3": ""}},
            [],
            id="single_b_exon2_file",
        ),
        pytest.param(
            ["E1_BA.txt"],
            "B",
            {"E1": {"exon2": "E1_BA.txt", "exon3": ""}},
            [],
            id="single_b_exon2_file_underscore_delimiter",
        ),
        pytest.param(
            ["E1-BA.txt"],
            "B",
            {"E1": {"exon2": "E1-BA.txt", "exon3": ""}},
            [],
            id="single_b_exon2_file_dash_delimiter",
        ),
        pytest.param(
            ["E1.BA.TXT"],
            "B",
            {"E1": {"exon2": "E1.BA.TXT", "exon3": ""}},
            [],
            id="single_b_exon2_file_uppercase_extension",
        ),
        pytest.param(
            ["E1.Ba.txt"],
            "B",
            {"E1": {"exon2": "E1.Ba.txt", "exon3": ""}},
            [],
            id="single_b_exon2_file_lowercase_exon_character",
        ),
        pytest.param(
            ["E1.bA.txt"],
            "B",
            {"E1": {"exon2": "E1.bA.txt", "exon3": ""}},
            [],
            id="single_b_exon2_file_lowercase_locus",
        ),
        pytest.param(
            ["E1.BB.txt"],
            "B",
            {"E1": {"exon2": "", "exon3": "E1.BB.txt"}},
            [],
            id="single_b_exon3_file",
        ),
        pytest.param(
            ["E1-Bb.txt"],
            "B",
            {"E1": {"exon2": "", "exon3": "E1-Bb.txt"}},
            [],
            id="single_b_exon3_file_lowercase_exon_character",
        ),
        pytest.param(
            ["E1.CB.txt"],
            "B",
            {},
            ['Skipping file "E1.CB.txt".'],
            id="c_file_skipped_when_looking_for_b",
        ),
        pytest.param(
            ["E1.CA.txt"],
            "C",
            {"E1": {"exon2": "E1.CA.txt", "exon3": ""}},
            [],
            id="single_c_exon2_file",
        ),
        pytest.param(
            ["E1_cA.TXT"],
            "C",
            {"E1": {"exon2": "E1_cA.TXT", "exon3": ""}},
            [],
            id="single_c_exon2_file_lowercase_locus",
        ),
        pytest.param(
            ["E1-CB.txt"],
            "C",
            {"E1": {"exon2": "", "exon3": "E1-CB.txt"}},
            [],
            id="single_c_exon3_file",
        ),
        pytest.param(
            ["E1.BB.TXT"],
            "C",
            {},
            ['Skipping file "E1.BB.TXT".'],
            id="b_file_skipped_when_looking_for_c",
        ),
        pytest.param(
            ["E1.BA.txt", "E1.BB.txt"],
            "B",
            {"E1": {"exon2": "E1.BA.txt", "exon3": "E1.BB.txt"}},
            [],
            id="paired_b_files",
        ),
        pytest.param(
            ["E1.Ca.TXT", "E1_cB.txt"],
            "C",
            {"E1": {"exon2": "E1.Ca.TXT", "exon3": "E1_cB.txt"}},
            [],
            id="paired_c_files",
        ),
        pytest.param(
            [
                "E1.BA.txt",
                "E1.BB.txt",
                "E2.ba.TXT",
                "E2.bb.TXT",
                "G5_a.TXT",
                "totally_different_file.pl",
                "E3_Ba.txt",
                "G7_CA.TXT",
                "E3_Bb.txt",
            ],
            "B",
            {
                "E1": {"exon2": "E1.BA.txt", "exon3": "E1.BB.txt"},
                "E2": {"exon2": "E2.ba.TXT", "exon3": "E2.bb.TXT"},
                "E3": {"exon2": "E3_Ba.txt", "exon3": "E3_Bb.txt"},
            },
            [
                'Skipping file "G5_a.TXT".',
                'Skipping file "G7_CA.TXT".',
                'Skipping file "totally_different_file.pl".',
            ],
            id="typical_b_case",
        ),
        pytest.param(
            [
                "E1.CA.txt",
                "E1.CB.txt",
                "E2.ca.TXT",
                "E2.cb.TXT",
                "G5_a.TXT",
                "totally_different_file.pl",
                "E3_Ca.txt",
                "G7_bA.TXT",
                "E3_Cb.txt",
            ],
            "C",
            {
                "E1": {"exon2": "E1.CA.txt", "exon3": "E1.CB.txt"},
                "E2": {"exon2": "E2.ca.TXT", "exon3": "E2.cb.TXT"},
                "E3": {"exon2": "E3_Ca.txt", "exon3": "E3_Cb.txt"},
            },
            [
                'Skipping file "G5_a.TXT".',
                'Skipping file "G7_bA.TXT".',
                'Skipping file "totally_different_file.pl".',
            ],
            id="typical_c_case",
        ),
    ],
)
def test_identify_bc_sequences(
    filenames: list[str],
    locus: HLA_LOCUS,
    expected_result: dict[str, dict[EXON_NAME, str]],
    expected_logger_calls: list[str],
    tmp_path: Path,
    mocker: MockerFixture,
):
    for filename in filenames:
        dummy_path: Path = tmp_path / filename
        dummy_path.write_text("ACGT")

    mock_logger: mocker.MagicMock = mocker.MagicMock()

    result: dict[str, dict[EXON_NAME, str]] = identify_bc_sequence_files(
        str(tmp_path), locus, mock_logger
    )
    assert result == expected_result
    if len(expected_logger_calls) > 0:
        mock_logger.info.assert_has_calls(
            [mocker.call(x) for x in expected_logger_calls],
            any_order=False,
        )


@pytest.mark.parametrize(
    "raw_sequences, locus, expected_sequences, expected_logger_calls",
    [
        pytest.param(
            {"E1.BA.txt": "A" * 270},
            "B",
            [],
            [
                'Skipping HLA-B sequence "E1": could not find matching exon2 and exon3 sequences.'
            ],
            id="single_exon2_sequence_skipped",
        ),
        pytest.param(
            {"E1.cb.txt": "A" * 276},
            "C",
            [],
            [
                'Skipping HLA-C sequence "E1": could not find matching exon2 and exon3 sequences.'
            ],
            id="single_exon3_sequence_skipped",
        ),
        pytest.param(
            {
                "E1.BA.txt": "A" * 270,
                "E1.BB.txt": "C" * 200,
            },
            "B",
            [],
            [
                (
                    'Skipping HLA-B sequence "E1": expected 276 characters in '
                    'file "E1.BB.txt", found 200.'
                ),
            ],
            id="single_pair_bad_length",
        ),
        pytest.param(
            {
                "E1_ca.txt": "A" * 100 + "_" + "T" * 169,
                "E1_cb.txt": "C" * 276,
            },
            "C",
            [],
            [
                (
                    'Skipping HLA-C sequence "E1": file "E1_ca.txt" contains '
                    "invalid characters."
                ),
            ],
            id="single_pair_bad_length",
        ),
        pytest.param(
            {
                "E1-bA.TXT": "A" * 270,
                "E1-bB.TXT": "C" * 276,
            },
            "B",
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(2,) * 276,
                    name="E1",
                    locus="B",
                    num_sequences_used=2,
                ),
            ],
            [],
            id="single_good_pair",
        ),
        pytest.param(
            {
                "E1-cA.TXT": "A" * 270,
                "E1-cB.TXT": "C" * 276,
                "E2.CA.txt": "T" * 270,
                "E2.CB.txt": "G" * 276,
                "E3.ca.txt": "A" * 270,
                "E4.ca.txt": "A" * 270,
                "E4.cb.txt": "A" * 100 + "_" + "A" * 175,
                "E5-CA.txt": "A" * 270,
                "E5-CB.txt": "G" * 276,
                "E6-CA.txt": "A" * 100,
                "E6-CB.txt": "G" * 276,
            },
            "C",
            [
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(2,) * 276,
                    name="E1",
                    locus="C",
                    num_sequences_used=2,
                ),
                HLASequence(
                    two=(8,) * 270,
                    intron=(),
                    three=(4,) * 276,
                    name="E2",
                    locus="C",
                    num_sequences_used=2,
                ),
                HLASequence(
                    two=(1,) * 270,
                    intron=(),
                    three=(4,) * 276,
                    name="E5",
                    locus="C",
                    num_sequences_used=2,
                ),
            ],
            [
                (
                    'Skipping HLA-C sequence "E3": could not find matching '
                    "exon2 and exon3 sequences."
                ),
                (
                    'Skipping HLA-C sequence "E4": file "E4.cb.txt" contains '
                    "invalid characters."
                ),
                (
                    'Skipping HLA-C sequence "E6": expected 270 characters in '
                    'file "E6-CA.txt", found 100.'
                ),
            ],
            id="typical_case",
        ),
    ],
)
def test_read_bc_sequences(
    raw_sequences: dict[str, str],
    locus: HLA_LOCUS,
    expected_sequences: list[HLASequence],
    expected_logger_calls: list[str],
    tmp_path: Path,
    mocker: MockerFixture,
):
    for filename, file_contents in raw_sequences.items():
        dummy_path: Path = tmp_path / filename
        dummy_path.write_text(file_contents)

    mock_logger: mocker.MagicMock = mocker.MagicMock()

    result: list[HLASequence] = read_bc_sequences(str(tmp_path), locus, mock_logger)

    assert result == expected_sequences
    if len(expected_logger_calls) > 0:
        mock_logger.info.assert_has_calls(
            [mocker.call(x) for x in expected_logger_calls],
            any_order=False,
        )
