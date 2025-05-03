#! /usr/bin/env python

import argparse
import logging
import os
import re
from datetime import datetime
from typing import Literal, Optional, Final

from sqlalchemy import DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..src.easyhla.easyhla import EasyHLA
from ..src.easyhla.models import HLAProteinPair, HLASequence, HLAStandard, HLAInterpretation
from ..src.easyhla.utils import EXON_NAME, HLA_LOCUS, check_bases, check_length, nuc2bin

logger: logging.Logger = logging.getLogger(__name__)

CFE_SCRIPTS_OUTPUT: Final[str] = os.environ.get("CFE_SCRIPTS_OUTPUT", "/output")
DEFAULT_INPUT_DIR: Final[str] = os.path.join(
    CFE_SCRIPTS_OUTPUT,
    "sanger_runs_dest",
)

# Database connection parameters:
HLA_DB_USER: Final[str] = os.environ.get("HLA_DB_USER")
HLA_DB_PASSWORD: Final[str] = os.environ.get("HLA_DB_PASSWORD")
HLA_DB_HOST: Final[str] = os.environ.get("HLA_DB_HOST", "192.168.67.7")
HLA_DB_PORT: Final[int] = os.environ.get("HLA_DB_PORT", 1521)
HLA_DB_SERVICE_NAME: Final[str] = os.environ.get("HLA_DB_SERVICE_NAME", "cfe")

# These are the "configuration files" that the algorithm uses; these are or may
# be updated, in which case you specify the path to the new version in the
# environment.
HLA_STANDARDS: Final[dict[HLA_LOCUS, Optional[str]]] = {
    "A": os.environ.get("HLA_STANDARDS_A"),
    "B": os.environ.get("HLA_STANDARDS_B"),
    "C": os.environ.get("HLA_STANDARDS_C"),
}
HLA_FREQUENCIES: Final[str] = os.environ.get("HLA_FREQUENCIES")


# As I understand it, this creates a "registry" for our application and is
# necessary (i.e. we can't just have all of our classes inherit from
# DeclarativeBase).
class Base(DeclarativeBase):
    pass


class HLASequenceA(Base):
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


class HLASequenceB(Base):
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


class HLASequenceC(Base):
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


def read_a_sequences(input_directory: str) -> list[HLASequence]:
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
                num_sequences_used=1,
            )
        )

    return sequences


def identify_bc_sequence_files(
    input_directory: str, locus: Literal["B", "C"]
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
        sample_files[sample_name][sample_exon] = filename

    return sample_files


def read_bc_sequences(
    input_directory: str, locus: Literal["B", "C"]
) -> list[HLASequence]:
    """
    Read all HLA-B or -C sequences from the input directory.

    These sequences will come in *pairs* of files, e.g. "S12345.BA.txt" and
    "S12345.BB.txt"; the former will contain exon2 and the latter will contain
    exon3.
    """
    sample_files: dict[str, dict[EXON_NAME, str]] = identify_bc_sequence_files(
        input_directory, locus
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
                num_sequences_used=2,
            )
        )

    return sequences


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate HLA interpretations as part of the CfE clinical pipeline",
    )
    parser.add_argument(
        "--input_dir",
        help="Directory to scan for input sequences",
        type=str,
        default=DEFAULT_INPUT_DIR,
    )
    for locus in ("A", "B", "C"):
        parser.add_argument(
            f"--hla_{locus.lower()}_standards",
            help=f"CSV file containing the (reduced) HLA-{locus} standards to use",
            type="str",
            default=None,
        )
    parser.add_argument(
        "--hla_frequencies",
        help=(
            "CSV file containing the HLA allele frequencies to reference when "
            "making interpretations"
        ),
        type="str",
        default=None,
    )
    # FIXME what to do about "last modified"?
    args: argparse.Namespace = parser.parse_args()

    sequences: dict[HLA_LOCUS, list[HLASequence]] = {
        "A": read_a_sequences(args.input_dir),
        "B": [],
        "C": [],
    }
    for locus in ("B", "C"):
        sequences[locus] = read_bc_sequences(args.input_dir, locus)

    standards_files: dict[HLA_LOCUS, Optional[str]] = {
        "A": args.hla_a_standards,
        "B": args.hla_b_standards,
        "C": args.hla_c_standards,
    }
    interpretations: dict[HLA_LOCUS], list[HLAInterpretation] = {
        "A": [],
        "B": [],
        "C": [],
    }
    for locus in ("A", "B", "C"):
        curr_standards: Optional[dict[str, HLAStandard]] = None
        curr_frequencies: Optional[dict[HLAProteinPair, int]] = None
        if args.hla_frequencies is not None:
            with open(args.hla_frequencies) as f:
                curr_frequencies = EasyHLA.read_hla_frequencies(locus, f)
        if standards_files[locus] is not None:
            with open(standards_files[locus]) as f:
                curr_standards = EasyHLA.read_hla_standards(f)
        easyhla: EasyHLA = EasyHLA(
            locus,
            hla_standards=curr_standards,
            hla_frequencies=curr_frequencies,
        )
        for sequence in sequences[locus]:
            try:
                interpretations[locus].append(easyhla.interpret(sequence))
            except EasyHLA.NoMatchingStandards:
                pass






if __name__ == "__main__":
    main()
