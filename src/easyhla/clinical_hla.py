#! /usr/bin/env python

import argparse
import csv
import dataclasses
import logging
import os
from datetime import datetime
from typing import Final, Literal, Optional, TypedDict, cast

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .clinical_hla_lib import (
    HLADBBase,
    HLASequenceA,
    HLASequenceB,
    HLASequenceC,
    read_a_sequences,
    read_bc_sequences,
)
from .easyhla import EasyHLA
from .models import (
    HLAInterpretation,
    HLASequence,
)
from .utils import HLA_LOCUS

logger: logging.Logger = logging.getLogger(__name__)

CFE_SCRIPTS_OUTPUT: Final[str] = os.environ.get("CFE_SCRIPTS_OUTPUT", "/output")
DEFAULT_INPUT_DIR: Final[str] = os.path.join(
    CFE_SCRIPTS_OUTPUT,
    "sanger_runs_dest",
)

# Database connection parameters:
HLA_DB_USER: Final[Optional[str]] = os.environ.get("HLA_DB_USER")
HLA_DB_PASSWORD: Final[Optional[str]] = os.environ.get("HLA_DB_PASSWORD")
HLA_DB_HOST: Final[str] = os.environ.get("HLA_DB_HOST", "192.168.67.7")
HLA_DB_PORT: Final[int] = int(os.environ.get("HLA_DB_PORT", 1521))
HLA_DB_SERVICE_NAME: Final[str] = os.environ.get("HLA_DB_SERVICE_NAME", "cfe")

HLA_ORACLE_LIB_PATH: Final[str] = os.environ.get(
    "HLA_ORACLE_LIB_PATH", "/opt/oracle/instant_client"
)


class SequencesByLocus(TypedDict):
    A: list[HLASequenceA]
    B: list[HLASequenceB]
    C: list[HLASequenceC]


def interpret_sequences(
    hla_alg: EasyHLA,
    sequences: list[HLASequence],
) -> list[HLAInterpretation]:
    interpretations: list[HLAInterpretation] = []
    for sequence in sequences:
        try:
            interpretations.append(hla_alg.interpret(sequence))
        except EasyHLA.NoMatchingStandards:
            pass
    return interpretations


def clinical_hla_driver(
    input_dir: str,
    hla_a_results: str,
    hla_b_results: str,
    hla_c_results: str,
    db_engine: Optional[Engine] = None,
    standards_path: Optional[str] = None,
    frequencies_path: Optional[str] = None,
) -> None:
    # Read in the sequences:
    sequences: dict[HLA_LOCUS, list[HLASequence]] = {
        "A": read_a_sequences(input_dir, logger),
        "B": [],
        "C": [],
    }
    for locus in ("B", "C"):
        b_or_c: Literal["B", "C"] = cast(Literal["B", "C"], locus)
        sequences[b_or_c] = read_bc_sequences(input_dir, b_or_c, logger)

    # Perform interpretations:
    interpretations: dict[HLA_LOCUS, list[HLAInterpretation]] = {
        "A": [],
        "B": [],
        "C": [],
    }
    processing_datetime: datetime = datetime.now()
    easyhla: EasyHLA = EasyHLA.use_config(standards_path, frequencies_path)
    for locus in ("A", "B", "C"):
        interpretations[cast(HLA_LOCUS, locus)] = interpret_sequences(
            easyhla, sequences[cast(HLA_LOCUS, locus)]
        )

    # Prepare the interpretations for output:
    seqs_for_db: SequencesByLocus = {
        "A": [],
        "B": [],
        "C": [],
    }
    # This next bit looks repetitive but mypy didn't like my solution for doing
    # this in a loop (because each one is a different type).
    for interp in interpretations["A"]:
        seqs_for_db["A"].append(
            HLASequenceA.build_from_interpretation(interp, processing_datetime)
        )
    for interp in interpretations["B"]:
        seqs_for_db["B"].append(
            HLASequenceB.build_from_interpretation(interp, processing_datetime)
        )
    for interp in interpretations["C"]:
        seqs_for_db["C"].append(
            HLASequenceC.build_from_interpretation(interp, processing_datetime)
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
        if len(seqs_for_db[cast(HLA_LOCUS, locus)]) > 0:
            with open(output_files[cast(HLA_LOCUS, locus)], "w") as f:
                result_csv: csv.DictWriter = csv.DictWriter(
                    f,
                    fieldnames=csv_headers[cast(HLA_LOCUS, locus)],
                    extrasaction="ignore",
                )
                result_csv.writeheader()
                result_csv.writerows(
                    dataclasses.asdict(x) for x in seqs_for_db[cast(HLA_LOCUS, locus)]
                )

    # Finally, write to the DB.
    if db_engine is not None:
        with Session(db_engine) as session:
            for locus in ("A", "B", "C"):
                session.add_all(seqs_for_db[cast(HLA_LOCUS, locus)])
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
            f"--hla_{locus.lower()}_results",
            help=f"CSV file containing the HLA-{locus} results",
            type=str,
            default=f"hla_{locus.lower()}_seq.csv",
        )
    parser.add_argument(
        "--hla_standards",
        help="YAML file containing the HLA standards to use",
        type=str,
        default=None,
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
                    cursor_obj.execute(f"attach database '{args.sqlite}' as specimen;")
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
        args.hla_a_results,
        args.hla_b_results,
        args.hla_c_results,
        db_engine,
        args.hla_standards,
        args.hla_frequencies,
    )


if __name__ == "__main__":
    main()
