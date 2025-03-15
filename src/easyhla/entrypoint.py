import logging
from pathlib import Path
from enum import Enum

import typer

from easyhla import EasyHLA


class HLALocus(str, Enum):
    """Valid HLA genes."""

    A = "A"
    B = "B"
    C = "C"


def main(
    letter: HLALocus = typer.Option(
        HLALocus.A.value, "--letter", "-l", help="", case_sensitive=False
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
        help="Print to stdout as sequences are interpretted",
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
) -> None:
    min_log_level = max(min(40, (4 - log_level) * 10), 50)
    logger = logging.Logger(__name__, min_log_level)
    easyhla = EasyHLA(letter=letter.value, logger=logger)

    easyhla.run(
        easyhla.letter,
        sequence_file.as_posix(),
        output_file.as_posix(),
        threshold=mismatch_threshold,
        to_stdout=print_to_stdout,
    )


def run():
    typer.run(main)
