#! /usr/bin/env python

import argparse
import csv
import dataclasses
import logging
import os
from datetime import datetime
from typing import Final, Optional, TypedDict

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from easyhla.clinical_hla_lib import (
    HLADBBase,
    HLASequenceA,
    HLASequenceB,
    HLASequenceC,
    read_a_sequences,
    read_bc_sequences,
)
from easyhla.easyhla import EasyHLA
from easyhla.models import (
    HLAInterpretation,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
)
from easyhla.utils import HLA_LOCUS

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

HLA_ORACLE_LIB_PATH: Final[str] = os.environ.get("HLA_ORACLE_LIB_PATH")

# These are the "configuration files" that the algorithm uses; these are or may
# be updated, in which case you specify the path to the new version in the
# environment.
HLA_STANDARDS: Final[dict[HLA_LOCUS, Optional[str]]] = {
    "A": os.environ.get("HLA_STANDARDS_A"),
    "B": os.environ.get("HLA_STANDARDS_B"),
    "C": os.environ.get("HLA_STANDARDS_C"),
}
HLA_FREQUENCIES: Final[str] = os.environ.get("HLA_FREQUENCIES")


def interpret_sequences(
    sequences: list[HLASequence],
    locus: HLA_LOCUS,
    standards_file: Optional[str] = None,
    frequencies_file: Optional[str] = None,
) -> tuple[list[HLAInterpretation], dict[HLAProteinPair, int]]:
    curr_standards: Optional[dict[str, HLAStandard]] = None
    curr_frequencies: Optional[dict[HLAProteinPair, int]] = None
    if frequencies_file is not None:
        with open(frequencies_file) as f:
            curr_frequencies = EasyHLA.read_hla_frequencies(locus, f)
    if standards_file is not None:
        with open(standards_file) as f:
            curr_standards = EasyHLA.read_hla_standards(f)
    easyhla: EasyHLA = EasyHLA(
        locus,
        hla_standards=curr_standards,
        hla_frequencies=curr_frequencies,
    )
    interpretations: list[HLAInterpretation] = []
    for sequence in sequences:
        try:
            interpretations.append(easyhla.interpret(sequence))
        except EasyHLA.NoMatchingStandards:
            pass
    return interpretations, easyhla.hla_frequencies


def prepare_interpretation_for_serialization(
    interpretation: HLAInterpretation,
    locus: HLA_LOCUS,
    processing_datetime: datetime,
) -> HLASequenceA | HLASequenceB | HLASequenceC:
    """
    Prepare an HLA interpretation for output.
    """
    if locus == "A":
        return HLASequenceA.build_from_interpretation(
            interpretation, processing_datetime
        )
    elif locus == "B":
        return HLASequenceB.build_from_interpretation(
            interpretation, processing_datetime
        )
    return HLASequenceC.build_from_interpretation(
        interpretation, processing_datetime
    )

class SequencesByLocus(TypedDict):
    A: list[HLASequenceA]
    B: list[HLASequenceB]
    C: list[HLASequenceC]


def clinical_hla_driver(
    input_dir: str,
    db_engine: Optional[Engine] = None,
    hla_a_standards: Optional[str] = None,
    hla_a_results: Optional[str] = None,
    hla_b_standards: Optional[str] = None,
    hla_b_results: Optional[str] = None,
    hla_c_standards: Optional[str] = None,
    hla_c_results: Optional[str] = None,
    hla_frequencies: Optional[str] = None,
) -> None:
    # Read in the sequences:
    sequences: dict[HLA_LOCUS, list[HLASequence]] = {
        "A": read_a_sequences(input_dir, logger),
        "B": [],
        "C": [],
    }
    for locus in ("B", "C"):
        sequences[locus] = read_bc_sequences(input_dir, locus, logger)

    # Perform interpretations:
    standards_files: dict[HLA_LOCUS, Optional[str]] = {
        "A": hla_a_standards,
        "B": hla_b_standards,
        "C": hla_c_standards,
    }
    interpretations: dict[HLA_LOCUS, list[HLAInterpretation]] = {
        "A": [],
        "B": [],
        "C": [],
    }
    frequencies: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = {
        "A": {},
        "B": {},
        "C": {},
    }

    processing_datetime: datetime = datetime.now()

    for locus in ("A", "B", "C"):
        curr_interps: list[HLAInterpretation]
        curr_freqs: dict[HLAProteinPair, int]
        curr_interps, curr_freqs = interpret_sequences(
            sequences[locus],
            locus,
            standards_files[locus],
            hla_frequencies,
        )
        interpretations[locus] = curr_interps
        frequencies[locus] = curr_freqs

    # Prepare the interpretations for output:
    seqs_for_db: SequencesByLocus = {
        "A": [],
        "B": [],
        "C": [],
    }
    for locus in ("A", "B", "C"):
        # Each locus has a slightly different schema in the database, so we
        # customize for each one.
        for interp in interpretations[locus]:
            seqs_for_db[locus].append(
                prepare_interpretation_for_serialization(
                    interp,
                    locus,
                    processing_datetime,
                )
            )

    # First, write to the output files:
    output_files: dict[HLA_LOCUS, str] = {
        "A": hla_a_results,
        "B": hla_b_results,
        "C": hla_c_results,
    }
    csv_headers: dict[HLA_LOCUS, tuple[str, ...]] = {
        "A": HLASequenceA.CSV_HEADER,
        "B": HLASequenceB.CSV_HEADER,
        "C": HLASequenceC.CSV_HEADER,
    }
    for locus in ("A", "B", "C"):
        if len(seqs_for_db[locus]) > 0:
            with open(output_files[locus], "w") as f:
                result_csv: csv.DictWriter = csv.DictWriter(
                    f, fieldnames=csv_headers[locus], extrasaction="ignore"
                )
                result_csv.writeheader()
                result_csv.writerows(dataclasses.asdict(x) for x in seqs_for_db[locus])

    # Finally, write to the DB.
    if db_engine is not None:
        with Session(db_engine) as session:
            for locus in ("A", "B", "C"):
                session.add_all(seqs_for_db[locus])
            session.commit()


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
            type=str,
            default=None,
        )
        parser.add_argument(
            f"--hla_{locus.lower()}_results",
            help=f"CSV file containing the HLA-{locus} results",
            type=str,
            default=f"hla_{locus.lower()}_seq.csv",
        )
    parser.add_argument(
        "--hla_frequencies",
        help=(
            "CSV file containing the HLA allele frequencies to reference when "
            "making interpretations"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sqlite",
        help=(
            "File path of a SQLite database to use instead of connecting to "
            "Oracle.  The required database tables will be created if necessary."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--nodb",
        help="If set, skip connecting to the database entirely (overrules --sqlite)",
        action="store_true",
    )
    # FIXME what to do about "last modified"?
    args: argparse.Namespace = parser.parse_args()

    # Connect to the database:
    db_engine: Optional[Engine] = None
    if not args.nodb:
        if args.sqlite is not None:
            db_engine = create_engine(f"sqlite+pysqlite:///{args.sqlite}")

            @event.listens_for(db_engine, "connect")
            def schema_workaround(dbapi_connection, _):
                # dbapi_connection is a DBAPI connection, not a SQLAlchemy
                # Connection, so we get a cursor to execute a command.
                try:
                    cursor_obj = dbapi_connection.cursor()
                    cursor_obj.execute(
                        f"attach database {args.sqlite} as specimen;"
                    )
                finally:
                    cursor_obj.close()

            # Create the tables if necessary; this will do nothing if the tables
            # already exist.
            HLADBBase.metadata.create_all(db_engine, checkfirst=True)
        else:
            db_engine = create_engine(
                "oracle+oracledb://@",
                thick_mode={"lib_dir": HLA_ORACLE_LIB_PATH},
                connect_args={
                    "user": HLA_DB_USER,
                    "password": HLA_DB_PASSWORD,
                    "host": HLA_DB_HOST,
                    "port": HLA_DB_PORT,
                    "service_name": HLA_DB_SERVICE_NAME,
                },
            )

    clinical_hla_driver(
        args.input_dir,
        db_engine,
        args.hla_a_standards,
        args.hla_a_results,
        args.hla_b_standards,
        args.hla_b_results,
        args.hla_c_standards,
        args.hla_c_results,
        args.hla_frequencies,
    )


if __name__ == "__main__":
    main()
