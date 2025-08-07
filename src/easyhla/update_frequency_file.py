#! /usr/bin/env python

import argparse
import csv
import logging
from collections import Counter

from .update_frequency_file_lib import (
    FREQUENCY_FIELDS,
    FrequencyRowDict,
    NewName,
    OldName,
    parse_nomenclature,
    update_old_frequencies,
)
from .utils import HLA_LOCUS

logging.basicConfig()
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Update the format of the old HLA frequencies file"
    )
    parser.add_argument(
        "old_frequency_file",
        help="Old frequency file path (6-column CSV without header)",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "name_mapping",
        help=(
            "File mapping old allele names to new allele names in the format "
            "kept by IMGT/HLA"
        ),
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--output",
        help="Output CSV file",
        type=argparse.FileType("w"),
        default="hla_frequencies.csv",
    )
    args: argparse.Namespace = parser.parse_args()

    logger.info("Reading in the update mapping....")
    with args.name_mapping:
        old_to_new: dict[OldName, NewName]
        deprecated: list[str]
        deprecated_maps_to_other: list[tuple[str, NewName]]
        mapping_overrides_deprecated: list[tuple[str, NewName]]
        (
            old_to_new,
            deprecated,
            deprecated_maps_to_other,
            mapping_overrides_deprecated,
        ) = parse_nomenclature(args.name_mapping.read())

    for old_name_str in deprecated:
        logger.info(f"Allele {old_name_str} is deprecated.")

    for old_name_str, existing_mapping in deprecated_maps_to_other:
        logger.info(
            f"Allele {old_name_str} is deprecated but another allele "
            "with the same first two coordinates maps to "
            f"{existing_mapping}; retaining that mapping."
        )

    for old_name_str, new_name_str in mapping_overrides_deprecated:
        logger.info(
            f"Allele {old_name_str} maps to {new_name_str} but another "
            f"allele with the same first two coordinates was "
            f"deprecated; updating the mapping."
        )

    logger.info("Updating old alleles....")
    with args.old_frequency_file:
        updated_frequencies: list[FrequencyRowDict]
        unmapped_alleles: Counter[tuple[HLA_LOCUS, str]]
        deprecated_alleles_seen: Counter[tuple[HLA_LOCUS, str]]
        updated_frequencies, unmapped_alleles, deprecated_alleles_seen = (
            update_old_frequencies(args.old_frequency_file, old_to_new)
        )

    if len(unmapped_alleles) > 1:
        logger.info(
            "Alleles present in the old frequencies that do not have a mapping "
            "in the new nomenclature, and their numbers of occurrences:"
        )
        for locus, name in unmapped_alleles:
            logger.info(
                f'HLA-{locus} allele "{name}": {unmapped_alleles[(locus, name)]}'
            )

    if len(deprecated_alleles_seen) > 1:
        logger.info(
            "Alleles present in the old frequencies that are deprecated "
            "in the new nomenclature, and their numbers of occurrences:"
        )
        for locus, name in deprecated_alleles_seen:
            logger.info(
                f'HLA-{locus} allele "{name}": {deprecated_alleles_seen[(locus, name)]}'
            )

    with args.output:
        frequencies_csv: csv.DictWriter = csv.DictWriter(args.output, FREQUENCY_FIELDS)
        frequencies_csv.writeheader()
        frequencies_csv.writerows(updated_frequencies)

    logger.info("... done.")


if __name__ == "__main__":
    main()
