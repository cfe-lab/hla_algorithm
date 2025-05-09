import logging
import os
import re
from datetime import datetime
from typing import Final, Literal, Optional, TypedDict

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
)

from easyhla.models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLASequence,
)
from easyhla.utils import EXON_NAME, HLA_LOCUS, check_bases, check_length, nuc2bin


class HLASequenceCommonFields(TypedDict):
    enum: str
    alleles_clean: str
    alleles_all: str
    ambiguous: bool
    homozygous: bool
    mismatch_count: int
    mismatches: str
    enterdate: datetime


# As I understand it, this creates a "registry" for our application and is
# necessary (i.e. we can't just have all of our classes inherit from
# DeclarativeBase).
class HLADBBase(MappedAsDataclass, DeclarativeBase):
    @staticmethod
    def get_common_serialization_fields(
        interpretation: HLAInterpretation,
        processing_datetime: datetime,
    ) -> HLASequenceCommonFields:
        """
        Prepare an HLA interpretation for output.
        """
        ap: AllelePairs = interpretation.best_matching_allele_pairs()
        best_matches: set[HLACombinedStandard] = interpretation.best_matches()

        # For the mismatches, we arbitrarily choose one of the best matches
        # and get the mismatches from that.
        arbitrary_best: HLAMatchDetails = interpretation.matches[best_matches.pop()]

        return {
            "enum": interpretation.hla_sequence.name,
            "alleles_clean": interpretation.best_common_allele_pair_str(),
            "alleles_all": ap.stringify(),
            "ambiguous": ap.is_ambiguous(),
            "homozygous": ap.is_homozygous(),
            "mismatch_count": interpretation.lowest_mismatch_count(),
            "mismatches": ";".join(str(x) for x in arbitrary_best.mismatches),
            "enterdate": processing_datetime,
        }


class HLASequenceA(HLADBBase):
    __tablename__ = "hla_alleles_a"
    __table_args__ = {"schema": "specimen"}

    # Note that we explicitly do *not* include the length of the VARCHAR fields
    # that we're mapping; this is to make it impossible for us to attempt to
    # perform a "create table" operation.
    enum: Mapped[str] = mapped_column(String, primary_key=True)
    alleles_clean: Mapped[Optional[str]] = mapped_column(String)
    alleles_all: Mapped[Optional[str]] = mapped_column(String)
    ambiguous: Mapped[Optional[str]] = mapped_column(String)
    homozygous: Mapped[Optional[str]] = mapped_column(String)
    mismatch_count: Mapped[Optional[str]] = mapped_column(Integer)
    mismatches: Mapped[Optional[str]] = mapped_column(String)
    seq: Mapped[Optional[str]] = mapped_column(String)
    enterdate: Mapped[Optional[datetime]] = mapped_column(DateTime)
    # We omit the "comments" column as it isn't pertinent to us.

    CSV_HEADER: Final[tuple[str, str, str, str, str, str, str, str, str]] = (
        "enum",
        "alleles_clean",
        "alleles_all",
        "ambiguous",
        "homozygous",
        "mismatch_count",
        "mismatches",
        "seq",
        "enterdate",
    )

    @classmethod
    def build_from_interpretation(
        cls,
        interp: HLAInterpretation,
        processing_datetime: datetime,
    ) -> "HLASequenceA":
        """
        Build an instance from an HLAInterpretation object.
        """
        db_values_to_insert: HLASequenceCommonFields = (
            cls.get_common_serialization_fields(interp, processing_datetime)
        )
        hla_sequence: HLASequence = interp.hla_sequence
        return HLASequenceA(
            seq=hla_sequence.exon2_str + hla_sequence.exon3_str,
            **db_values_to_insert,
        )


class HLASequenceB(HLADBBase):
    __tablename__ = "hla_alleles_b"
    __table_args__ = {"schema": "specimen"}

    enum: Mapped[str] = mapped_column(String, primary_key=True)
    alleles_clean: Mapped[Optional[str]] = mapped_column(String)
    alleles_all: Mapped[Optional[str]] = mapped_column(String)
    ambiguous: Mapped[Optional[str]] = mapped_column(String)
    homozygous: Mapped[Optional[str]] = mapped_column(String)
    mismatch_count: Mapped[Optional[str]] = mapped_column(Integer)
    mismatches: Mapped[Optional[str]] = mapped_column(String)
    b5701: Mapped[Optional[str]] = mapped_column(String)
    b5701_dist: Mapped[Optional[int]] = mapped_column(Integer)
    seqa: Mapped[Optional[str]] = mapped_column(String)
    seqb: Mapped[Optional[str]] = mapped_column(String)
    reso_status: Mapped[Optional[str]] = mapped_column(String)
    enterdate: Mapped[Optional[datetime]] = mapped_column(DateTime)

    CSV_HEADER: Final[
        tuple[str, str, str, str, str, str, str, str, str, str, str, str, str]
    ] = (
        "enum",
        "alleles_clean",
        "alleles_all",
        "ambiguous",
        "homozygous",
        "mismatch_count",
        "mismatches",
        "b5701",
        "b5701_dist",
        "seqa",
        "seqb",
        "reso_status",
        "enterdate",
    )

    @classmethod
    def build_from_interpretation(
        cls,
        interp: HLAInterpretation,
        processing_datetime: datetime,
    ) -> "HLASequenceB":
        """
        Build an instance from an HLAInterpretation object.
        """
        db_values_to_insert: HLASequenceCommonFields = (
            cls.get_common_serialization_fields(interp, processing_datetime)
        )
        hla_sequence: HLASequence = interp.hla_sequence
        ap: AllelePairs = interp.best_matching_allele_pairs()
        reso_status: Optional[str] = "pending" if ap.contains_allele("B*57") else None
        return cls(
            b5701=interp.is_b5701(),
            b5701_dist=interp.distance_from_b7501(),
            seqa=hla_sequence.exon2_str,
            seqb=hla_sequence.exon3_str,
            reso_status=reso_status,
            **db_values_to_insert,
        )


class HLASequenceC(HLADBBase):
    __tablename__ = "hla_alleles_c"
    __table_args__ = {"schema": "specimen"}

    enum: Mapped[str] = mapped_column(String, primary_key=True)
    alleles_clean: Mapped[Optional[str]] = mapped_column(String)
    alleles_all: Mapped[Optional[str]] = mapped_column(String)
    ambiguous: Mapped[Optional[str]] = mapped_column(String)
    homozygous: Mapped[Optional[str]] = mapped_column(String)
    mismatch_count: Mapped[Optional[str]] = mapped_column(Integer)
    mismatches: Mapped[Optional[str]] = mapped_column(String)
    seqa: Mapped[Optional[str]] = mapped_column(String)
    seqb: Mapped[Optional[str]] = mapped_column(String)
    enterdate: Mapped[Optional[datetime]] = mapped_column(DateTime)

    CSV_HEADER: Final[tuple[str, str, str, str, str, str, str, str, str, str]] = (
        "enum",
        "alleles_clean",
        "alleles_all",
        "ambiguous",
        "homozygous",
        "mismatch_count",
        "mismatches",
        "seqa",
        "seqb",
        "enterdate",
    )

    @classmethod
    def build_from_interpretation(
        cls,
        interp: HLAInterpretation,
        processing_datetime: datetime,
    ) -> "HLASequenceC":
        """
        Build an instance from an HLAInterpretation object.
        """
        db_values_to_insert: HLASequenceCommonFields = (
            cls.get_common_serialization_fields(interp, processing_datetime)
        )
        hla_sequence: HLASequence = interp.hla_sequence
        return cls(
            seqa=hla_sequence.exon2_str,
            seqb=hla_sequence.exon3_str,
            **db_values_to_insert,
        )


def sanitize_sequence(
    raw_contents: str,
    locus: HLA_LOCUS,
    sample_name: str,
) -> str:
    """
    Sanitize "raw" sequence data to give the sequence without stray information.

    If this is in a FASTA format for some reason, do away with the header;
    then strip all whitespace, and perform some sanity checks.
    """
    # If this is in a FASTA format for some reason, do away with the header.
    # Then remove all whitespace.
    sanitized_contents: str = re.sub(r"^>.+\n", "", raw_contents.strip())
    sanitized_contents = re.sub(r"\s", "", sanitized_contents)

    try:
        check_length(locus, sanitized_contents, sample_name)
    except ValueError:
        error_message: str = (
            f"HLA-A sequence {sample_name} is the incorrect size; "
            f"expected 787 characters, found {len(sanitized_contents)}."
        )
        raise ValueError(error_message)

    try:
        check_bases(sanitized_contents)
    except ValueError:
        error_message: str = (
            f"HLA-A sequence {sample_name} contains invalid characters."
        )
        raise ValueError(error_message)

    return sanitized_contents


def read_a_sequences(input_directory: str, logger: logging.Logger) -> list[HLASequence]:
    """
    Read all HLA-A sequences from the input directory.

    The sequences will each be in their own file named "[sample name].A.txt",
    where the first period may be a "-" or a "_"; we read them all in and
    prepare HLASequence objects for further processing.
    """
    a_seq_file: re.Pattern = re.compile(r"^(.+)[_\-\.][aA].(?:txt|TXT)")
    all_files: list[str] = os.listdir(input_directory)
    sequence_file_matches: list[Optional[re.Match]] = [
        a_seq_file.match(x) for x in all_files
    ]
    sample_names_and_filenames: list[tuple[str, str]] = [
        (x.group(1), x.group(0)) for x in sequence_file_matches if x is not None
    ]

    sequences: list[HLASequence] = []

    for sample_name, filename in sample_names_and_filenames:
        contents: str = ""
        with open(filename) as f:
            contents = f.read()

        sanitized_contents: str = ""
        try:
            sanitized_contents = sanitize_sequence(contents, "A", sample_name)
        except ValueError as e:
            logger.info(e.msg)
            logger.info("Skipping sequence.")
            continue

        sequences.append(
            HLASequence(
                two=nuc2bin(sanitized_contents[:270]),
                intron=(),
                three=nuc2bin(sanitized_contents[512:]),
                name=sample_name,
                locus="A",
                num_sequences_used=1,
            )
        )

    return sequences


def identify_bc_sequence_files(
    input_directory: str, locus: Literal["B", "C"], logger: logging.Logger
) -> dict[str, dict[EXON_NAME, str]]:
    """
    Identify all HLA-B or -C sequences in the input directory.

    These sequences will come in *pairs* of files, e.g. "S12345.BA.txt" and
    "S12345.BB.txt"; the former will contain exon2 and the latter will contain
    exon3.
    """
    locus_pattern: str = f"[{locus.lower()}{locus}]"
    fn_pattern: re.Pattern = re.compile(
        f"^(.+)[_\\-\\.]{locus_pattern}([aAbB]).(?:txt|TXT)"
    )
    all_files: list[str] = os.listdir(input_directory)
    sample_matches: list[Optional[re.Match]] = [fn_pattern.match(x) for x in all_files]
    sample_names: list[str] = list(
        {x.group(1) for x in sample_matches if x is not None}
    )

    sample_files: dict[str, dict[EXON_NAME, str]] = {}
    for sample_name in sample_names:
        sample_files[sample_name] = {
            "exon2": "",
            "exon3": "",
        }

    for idx, filename in enumerate(all_files):
        sample_match: Optional[re.Match] = sample_matches[idx]
        if sample_match is None:
            logger.info(f"Skipping file {filename}.")
            continue
        sample_name: str = sample_match.group(1)
        sample_exon: EXON_NAME = (
            "exon2" if sample_match.group(2).upper() == "A" else "exon3"
        )
        sample_files[sample_name][sample_exon] = os.path.join(input_directory, filename)

    return sample_files


def read_bc_sequences(
    input_directory: str, locus: Literal["B", "C"], logger: logging.Logger
) -> list[HLASequence]:
    """
    Read all HLA-B or -C sequences from the input directory.

    These sequences will come in *pairs* of files, e.g. "S12345.BA.txt" and
    "S12345.BB.txt"; the former will contain exon2 and the latter will contain
    exon3.
    """
    sample_files: dict[str, dict[EXON_NAME, str]] = identify_bc_sequence_files(
        input_directory, locus, logger
    )

    sequences: list[HLASequence] = []
    for sample_name, files_for_sample in sample_files.items():
        if files_for_sample["exon2"] == "" or files_for_sample["exon3"] == "":
            logger.info(
                f"Skipping sample {sample_name}; could not find matching exon2 "
                "and exon3 sequences."
            )
            continue
        curr_sequences: dict[EXON_NAME, str] = {"exon2": "", "exon3": ""}

        skip_sample: bool = False
        for exon_name in ("exon2", "exon3"):
            raw_contents: str = ""
            with open(files_for_sample[exon_name]) as f:
                raw_contents = f.read()

            try:
                curr_sequences[exon_name] = sanitize_sequence(
                    raw_contents,
                    locus,
                    exon_name,
                )
            except ValueError as e:
                logger.info(
                    f"Skipping sequence {sample_name}; error encountered while "
                    f"processing {exon_name}:"
                )
                logger.info(e.msg)
                skip_sample = True
                break

        if skip_sample:
            continue

        sequences.append(
            HLASequence(
                two=nuc2bin(curr_sequences["exon2"]),
                intron=(),
                three=nuc2bin(curr_sequences["exon3"]),
                name=sample_name,
                locus=locus,
                num_sequences_used=2,
            )
        )

    return sequences
