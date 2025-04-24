#! /usr/bin/env python

import argparse
import csv
import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Final

import Bio
import requests

from easyhla.utils import (
    EXON_NAME,
    HLA_LOCUS,
    GroupedAllele,
    group_identical_alleles,
    parse_standards,
)

logger: logging.Logger = logging.getLogger(__name__)

# Exon sequences used for scoring/"aligning" the sequences in the source file.
EXON_REFERENCES: Final[dict[HLA_LOCUS, dict[EXON_NAME, str]]] = {
    "A": {
        "exon2": (
            "GCTCCCACTCCATGAGGTATTTCTTCACATCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCA"
            "TCGCCGTGGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGCCAGAAGA"
            "TGGAGCCGCGGGCGCCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCAGGAGACACGGAATA"
            "TGAAGGCCCACTCACAGACTGACCGAGCGAACCTGGGGACCCTGCGCGGCTACTACAACCAGAGCG"
            "AGGACG"
        ),
        "exon3": (
            "GTTCTCACACCATCCAGATAATGTATGGCTGCGACGTGGGGCCGGACGGGCGCTTCCTCCGCGGGT"
            "ACCGGCAGGACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCTTGGACCG"
            "CGGCGGACATGGCAGCTCAGATCACCAAGCGCAAGTGGGAGGCGGTCCATGCGGCGGAGCAGCGGA"
            "GAGTCTACCTGGAGGGCCGGTGCGTGGACGGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGC"
            "TGCAGCGCACGG"
        ),
    },
    "B": {
        "exon2": (
            "GCTCCCACTCCATGAGGTATTTCTACACCTCCGTGTCCCGGCCCGGCCGCGGGGAGCCCCGCTTCA"
            "TCTCAGTGGGCTACGTGGACGACACCCAGTTCGTGAGGTTCGACAGCGACGCCGCGAGTCCGAGAG"
            "AGGAGCCGCGGGCGCCGTGGATAGAGCAGGAGGGGCCGGAGTATTGGGACCGGAACACACAGATCT"
            "ACAAGGCCCAGGCACAGACTGACCGAGAGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCG"
            "AGGCCG"
        ),
        "exon3": (
            "GGTCTCACACCCTCCAGAGCATGTACGGCTGCGACGTGGGGCCGGACGGGCGCCTCCTCCGCGGGC"
            "ATGACCAGTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCG"
            "CCGCGGACACGGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGA"
            "GAGCCTACCTGGAGGGCGAGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGACAAGC"
            "TGGAGCGCGCTG"
        ),
    },
    "C": {
        "exon2": (
            "GCTCCCACTCCATGAAGTATTTCTTCACATCCGTGTCCCGGCCTGGCCGCGGAGAGCCCCGCTTCA"
            "TCTCAGTGGGCTACGTGGACGACACGCAGTTCGTGCGGTTCGACAGCGACGCCGCGAGTCCGAGAG"
            "GGGAGCCGCGGGCGCCGTGGGTGGAGCAGGAGGGGCCGGAGTATTGGGACCGGGAGACACAGAAGT"
            "ACAAGCGCCAGGCACAGACTGACCGAGTGAGCCTGCGGAACCTGCGCGGCTACTACAACCAGAGCG"
            "AGGCCG"
        ),
        "exon3": (
            "GGTCTCACACCCTCCAGTGGATGTGTGGCTGCGACCTGGGGCCCGACGGGCGCCTCCTCCGCGGGT"
            "ATGACCAGTACGCCTACGACGGCAAGGATTACATCGCCCTGAACGAGGACCTGCGCTCCTGGACCG"
            "CCGCGGACACCGCGGCTCAGATCACCCAGCGCAAGTGGGAGGCGGCCCGTGAGGCGGAGCAGCGGA"
            "GAGCCTACCTGGAGGGCACGTGCGTGGAGTGGCTCCGCAGATACCTGGAGAACGGGAAGGAGACGC"
            "TGCAGCGCGCGG"
        ),
    },
}


# Find all releases (and their corresponding tags) of the HLA data at
# https://github.com/ANHIG/IMGTHLA/releases
REPO_PATH: Final[str] = os.environ.get(
    "EASYHLA_REPO_PATH",
    "https://raw.githubusercontent.com/ANHIG/IMGTHLA",
)
HLA_ALLELES_FILENAME: Final[str] = os.environ.get(
    "EASYHLA_REPO_ALLELES_FILENAME",
    "hla_nuc.fasta",
)


class RetrieveAllelesError(Exception):
    pass


def get_alleles_file(
    tag: str,
    base_url: str = REPO_PATH,
    alleles_filename: str = HLA_ALLELES_FILENAME,
) -> str:
    """
    Retrieve the HLA alleles file from the specified tag.
    """
    url: str = f"{base_url}/{tag}/{alleles_filename}"
    response: requests.Response = requests.get(url)
    if response.status_code != requests.codes.ok:
        raise RetrieveAllelesError()
    return response.text


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Retrieve HLA alleles from IPD-IMGT/HLA."
    )
    parser.add_argument(
        "tag",
        help="Git tag to pull the data from",
        type=str,
    )

    for locus in ("A", "B", "C"):
        parser.add_argument(
            f"--output_{locus.lower()}",
            help="filename to store the HLA-{locus} data in",
            type=str,
            default=f"hla_{locus.lower()}_std.csv",
        )
        parser.add_argument(
            f"--output_{locus.lower()}_reduced",
            help="filename to store the reduced HLA-{locus} data in",
            type=str,
            default=f"hla_{locus.lower()}_std_reduced.csv",
        )

    parser.add_argument(
        "--checksum",
        help="filename to store the MD5 checksum of the retrieved data in",
        type=str,
        default="hla_nuc.fasta.checksum.txt",
    )
    parser.add_argument(
        "--mismatch_threshold",
        help="number of mismatches to tolerate when comparing sequences to standards",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--acceptable_match_search_threshold",
        help=(
            "number of mismatches to tolerate when searching for matches "
            "(note that this differs from --mismatch_threshold)"
        ),
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dump_full_fasta_to",
        help="if specified, the full original FASTA file is dumped to the specified path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--standard_report_interval",
        help=(
            "number of standards between status updates while parsing "
            "sequences; set to a number less than 1 to silence these updates"
        ),
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    logger.info(f"Retrieving alleles from tag {args.tag}....")
    alleles_str: str
    for i in range(5):
        try:
            alleles_str = get_alleles_file(args.tag)
        except RetrieveAllelesError:
            if i < 4:
                logger.info("Failed to retrieve alleles; retrying in 20 seconds....")
                time.sleep(20)
            else:
                raise
        else:
            break

    if args.dump_full_fasta_to != "":
        logger.info(f"Dumping the full FASTA file to {args.dump_full_fasta_to}.")
        with open(args.dump_full_fasta_to, "w") as f:
            f.write(alleles_str)

    # Compute the checksum.
    md5_calc = hashlib.md5()
    md5_calc.update(alleles_str.encode())
    with open(args.checksum, "w") as f:
        f.write(f"{md5_calc.hexdigest()} {HLA_ALLELES_FILENAME}\n")

    standards: dict[HLA_LOCUS, list[tuple[str, str, str]]] = parse_standards(
        Bio.SeqIO.parse(StringIO(alleles_str), "fasta"),
        EXON_REFERENCES,
        logger,
        args.mismatch_threshold,
        args.acceptable_match_search_threshold,
        args.standard_report_interval,
    )

    output_files: dict[HLA_LOCUS, str] = {
        "A": args.output_a,
        "B": args.output_b,
        "C": args.output_c,
    }
    for locus in ("A", "B", "C"):
        logger.info(f"Writing the HLA-{locus} sequences to {output_files[locus]}....")
        with open(output_files[locus], "w") as f:
            unreduced_output_csv: csv.writer = csv.writer(f)
            unreduced_output_csv.writerow(("allele", "exon2", "exon3"))
            unreduced_output_csv.writerows(standards[locus])

    reduced_output_files: dict[HLA_LOCUS, str] = {
        "A": args.output_a_reduced,
        "B": args.output_b_reduced,
        "C": args.output_c_reduced,
    }
    for locus in ("A", "B", "C"):
        logger.info(
            "Writing the reduced HLA-{locus} sequences to "
            f"{reduced_output_files[locus]}...."
        )
        reduced_standards: dict[str, GroupedAllele] = group_identical_alleles(
            standards[locus]
        )
        with open(reduced_output_files[locus], "w") as f:
            reduced_output_csv: csv.writer = csv.writer(f)
            reduced_output_csv.writerow(("reduced_allele", "exon2", "exon3"))
            for grouped_name, grouped_allele in reduced_standards.items():
                reduced_output_csv.writerow(
                    (
                        grouped_name,
                        grouped_allele.exon2,
                        grouped_allele.exon3,
                    )
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
