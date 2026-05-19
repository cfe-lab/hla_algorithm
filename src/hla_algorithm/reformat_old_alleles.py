#! /usr/bin/env python

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import yaml

from .utils import (
    HLA_LOCUS,
    GroupedAllele,
    HLARawStandard,
    StoredHLAStandards,
    group_identical_alleles,
)

logging.basicConfig()
logger: logging.Logger = logging.getLogger(__name__)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Update HLA alleles in the old (CSV-based) format to the new format.  "
        "The input allele lists may be reduced or unreduced."
    )
    parser.add_argument(
        "a_standards",
        help="CSV file containing all HLA-A alleles",
        type=Path,
    )
    parser.add_argument(
        "b_standards",
        help="CSV file containing all HLA-B alleles",
        type=Path,
    )
    parser.add_argument(
        "c_standards",
        help="CSV file containing all HLA-C alleles",
        type=Path,
    )
    parser.add_argument(
        "--output",
        help="filename to store the reformatted standards in YAML",
        type=Path,
        default="reformatted_hla_standards.yaml",
    )
    parser.add_argument(
        "--tag",
        help="human-readable name for this dataset",
        type=str,
        default="reformatted_hla_alleles",
    )
    parser.add_argument(
        "--last_updated",
        help=(
            "ISO-formatted datetime of time these alleles were updated "
            "(if blank, current time will be used.  "
            'A trailing "Z" denotes UTC.)'
        ),
        type=str,
        default="",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Output status messages (and debug messages if -vv is used)",
    )
    args = parser.parse_args()

    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    input_filenames_by_locus: dict[HLA_LOCUS, Path] = {
        "A": args.a_standards,
        "B": args.b_standards,
        "C": args.c_standards,
    }
    grouped_alleles: dict[HLA_LOCUS, list[GroupedAllele]] = {"A": [], "B": [], "C": []}
    for locus in ("A", "B", "C"):
        logger.info(f"Grouping HLA-{locus} alleles....")
        with input_filenames_by_locus[locus].open() as f:
            standards_csv: csv.DictReader = csv.DictReader(
                f,
                fieldnames=("allele", "exon2", "exon3"),
            )
            raw_standards: list[HLARawStandard] = [
                HLARawStandard(
                    allele=row["allele"],
                    exon2=row["exon2"],
                    exon3=row["exon3"],
                )
                for row in standards_csv
            ]
        grouped_alleles[cast(HLA_LOCUS, locus)] = group_identical_alleles(
            raw_standards,
            logger=logger,
        )

    last_updated: datetime = datetime.now()
    if args.last_updated != "":
        last_updated = datetime.fromisoformat(args.last_updated)

    standards_for_saving: StoredHLAStandards = StoredHLAStandards(
        tag=args.tag,
        commit_hash="n/a",
        last_updated=last_updated,
        standards=grouped_alleles,
    )

    logger.info(f"Writing HLA standards to {args.output}....")
    with args.output.open("w") as f:
        yaml.safe_dump(standards_for_saving.model_dump(), f)

    logger.info("Done.")


if __name__ == "__main__":
    main()
