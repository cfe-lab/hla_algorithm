#! /usr/bin/env python

import argparse
import csv
import logging
from datetime import datetime, timezone
from typing import cast

import yaml

from hla_algorithm.utils import (
    GroupedAllele,
    HLA_LOCUS,
    HLARawStandard,
    StoredHLAStandards,
    group_identical_alleles,
)

logging.basicConfig()
logger: logging.Logger = logging.getLogger(__name__)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Update the bblab HLA alleles to the new format."
    )
    parser.add_argument(
        "a_standards",
        help="CSV file containing all unreduced HLA-A alleles",
        type=str,
    )
    parser.add_argument(
        "b_standards",
        help="CSV file containing all unreduced HLA-B alleles",
        type=str,
    )
    parser.add_argument(
        "c_standards",
        help="CSV file containing all unreduced HLA-C alleles",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="filename to store the unreduced standards (YAML format)",
        type=str,
        default="bblab_hla_standards.yaml",
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

    input_filenames_by_locus: dict[HLA_LOCUS, str] = {
        "A": args.a_standards,
        "B": args.b_standards,
        "C": args.c_standards,
    }
    grouped_alleles: dict[HLA_LOCUS, list[GroupedAllele]] = {"A": [], "B": [], "C": []}
    for locus in ("A", "B", "C"):
        logger.info(f"Grouping HLA-{locus} alleles....")
        with open(input_filenames_by_locus[locus]) as f:
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

    standards_for_saving: StoredHLAStandards = StoredHLAStandards(
        tag="bblab_alleles",
        commit_hash="n/a",
        last_updated=datetime(2012, 10, 17, 8, 33, tzinfo=timezone.utc),
        standards=grouped_alleles,
    )

    # First, prepare the unreduced YAML output.
    logger.info(f"Writing HLA standards to {args.output}....")
    with open(args.output, "w") as f:
        yaml.safe_dump(standards_for_saving.model_dump(), f)

    logger.info("Done.")


if __name__ == "__main__":
    main()
