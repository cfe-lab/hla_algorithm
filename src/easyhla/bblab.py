import csv
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
from Bio.Seq import Seq
from Bio.SeqIO import parse

from .bblab_lib import (
    EXON_AND_OTHER_EXON,
    HLAInterpretationRow,
    HLAMismatchRow,
    pair_exons,
)
from .easyhla import DATE_FORMAT, EasyHLA
from .models import HLAInterpretation, HLASequence
from .utils import EXON_NAME

logger = logging.Logger(__name__, logging.ERROR)


class HLALocus(str, Enum):
    """Valid HLA genes."""

    A = "A"
    B = "B"
    C = "C"


def log_and_print(
    message: Any,
    log_level: int = logging.INFO,
    to_stdout: bool = False,
) -> None:
    """
    Output messages to logger, optionally prints to STDOUT.

    :param message: ...
    :type message: Any
    :param log_level: ..., defaults to logging.INFO
    :type log_level: int, optional
    :param to_stdout: Whether to print to STDOUT or not, defaults to False
    :type to_stdout: bool
    """
    logger.log(level=log_level, msg=message)
    if to_stdout:
        print(message)


def report_unmatched_sequences(
    unmatched: dict[EXON_NAME, dict[str, Seq]],
    to_stdout: bool = False,
) -> None:
    """
    Report exon sequences that did not have a matching exon.

    :param unmatched: unmatched exon sequences, grouped by which exon they represent
    :type unmatched: dict[EXON_NAME, dict[str, Seq]]
    :param to_stdout: ..., defaults to None
    :type to_stdout: Optional[bool], optional
    """
    for exon, other_exon in EXON_AND_OTHER_EXON:
        for sequence_id in unmatched[exon].keys():
            log_and_print(
                f"No matching {other_exon} for {sequence_id}",
                to_stdout=to_stdout,
            )


def process_from_file_to_files(
    hla_alg: EasyHLA,
    locus: HLALocus,
    filename: str,
    output_filename: str,
    mismatches_filename: str,
    threshold: Optional[int] = None,
    to_stdout: bool = False,
):
    if threshold and threshold < 0:
        raise RuntimeError("Threshold must be >=0 or None!")
    elif threshold is None:
        threshold = 0

    rows: list[HLAInterpretationRow] = []
    mismatch_rows: list[HLAMismatchRow] = []
    npats: int = 0
    nseqs: int = 0
    time_start: datetime = datetime.now()

    log_and_print(
        f"Run commencing {time_start.strftime(DATE_FORMAT)}. "
        f"Allele definitions last updated {hla_alg.last_updated}.",
        to_stdout=to_stdout,
    )

    matched_sequences: list[HLASequence]
    unmatched: dict[EXON_NAME, dict[str, Seq]]

    with open(filename, "r", encoding="utf-8") as f:
        matched_sequences, unmatched = pair_exons(
            parse(f, "fasta"),
            locus.value,
            list(hla_alg.hla_standards[locus.value].values())[0],
        )

    for hla_sequence in matched_sequences:
        try:
            result: HLAInterpretation = hla_alg.interpret(
                hla_sequence,
                threshold,
            )
        except EasyHLA.NoMatchingStandards:
            log_and_print(
                f"Sequence {hla_sequence.name} did not match any known alleles.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )
            log_and_print(
                "Please check the locus and the orientation.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )
            continue

        if result.lowest_mismatch_count() > threshold:
            log_and_print(
                "No matches found below specified threshold. "
                "Please check the locus, orientation, and/or increase "
                "the tolerated number of mismatches.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )

        row: HLAInterpretationRow = HLAInterpretationRow.summary_row(result)
        rows.append(row)

        mismatch_rows.extend(HLAMismatchRow.mismatch_rows(result))

        npats += 1
        nseqs += hla_sequence.num_sequences_used

    report_unmatched_sequences(unmatched, to_stdout=to_stdout)

    with open(output_filename, "w") as f:
        output_csv: csv.DictWriter = csv.DictWriter(
            f,
            (
                "enum",
                "alleles_clean",
                "alleles",
                "ambiguous",
                "homozygous",
                "mismatch_count",
                "mismatches",
                "exon2",
                "intron",
                "exon3",
            ),
        )
        output_csv.writeheader()
        output_csv.writerows([dict(row) for row in rows])

    with open(mismatches_filename, "w") as f:
        mismatch_csv: csv.DictWriter = csv.DictWriter(
            f,
            (
                "allele",
                "mismatches",
                "exon2",
                "intron",
                "exon3",
            ),
        )
        mismatch_csv.writeheader()
        mismatch_csv.writerows([dict(row) for row in mismatch_rows])

    log_and_print(
        f"{npats} patients, {nseqs} sequences processed.",
        log_level=logging.INFO,
        to_stdout=to_stdout,
    )


def main(
    locus: HLALocus = typer.Option(
        HLALocus.A.value, "--locus", "-l", help="", case_sensitive=False
    ),
    mismatch_threshold: int = typer.Option(
        0,
        "--threshold",
        "-t",
        help="Maximum allowed number of mismatches for a sequence to contain versus reference sequences.",
        min=0,
    ),
    log_level: int = typer.Option(
        0,
        "-v",
        count=True,
        help="Logging level from [Error, Warn, Info, Debug] default Error. Repeat -v's to receive more verbose output",
    ),
    print_to_stdout: bool = typer.Option(
        False,
        "--print",
        "-p",
        help="Print to stdout as sequences are interpreted",
    ),
    sequence_file: Path = typer.Argument(
        ...,
        help="Sequence file in fasta format to be classified.",
        dir_okay=False,
        file_okay=True,
        exists=True,
        readable=True,
        path_type=str,
    ),
    output_file: Path = typer.Argument(
        "output.csv",
        help="Output file in csv format.",
        dir_okay=False,
        file_okay=True,
        exists=False,
        writable=True,
        allow_dash=False,
        path_type=str,
    ),
    mismatch_file: Path = typer.Argument(
        "mismatches.csv",
        help="Mismatches file in csv format.",
        dir_okay=False,
        file_okay=True,
        exists=False,
        writable=True,
        allow_dash=False,
        path_type=str,
    ),
) -> None:
    min_log_level = max(min(40, (4 - log_level) * 10), 50)
    logger.setLevel(min_log_level)
    easyhla = EasyHLA()

    process_from_file_to_files(
        easyhla,
        locus,
        sequence_file.as_posix(),
        output_file.as_posix(),
        mismatch_file.as_posix(),
        threshold=mismatch_threshold,
        to_stdout=print_to_stdout,
    )


def run():
    typer.run(main)
