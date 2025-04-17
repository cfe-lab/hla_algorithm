#! /usr/bin/env python

import argparse
import csv
import hashlib
import time
from io import StringIO
from typing import Final

import Bio
import requests

from ..src.easyhla.easyhla import EXON_NAME, HLA_LOCUS
from ..src.easyhla.utils import get_acceptable_match

# Exon sequences used for scoring/"aligning" the sequences in the source file.
EXON_SEQUENCES: Final[dict[HLA_LOCUS, dict[EXON_NAME, str]]] = {
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


# Find all releases of the HLA data at
# https://github.com/ANHIG/IMGTHLA/releases
REPO_PATH: Final[str] = "https://raw.githubusercontent.com/ANHIG/IMGTHLA"
HLA_ALLELES_FILENAME: Final[str] = "hla_nuc.fasta"


class RetrieveAllelesError(Exception):
    pass


def get_alleles_file(
    release: str,
    base_url: str = REPO_PATH,
    alleles_filename: str = HLA_ALLELES_FILENAME,
) -> str:
    """
    Retrieve the HLA alleles file from the specified release.
    """
    url: str = f"{base_url}/{release}/{alleles_filename}"
    response: requests.Response = requests.get(url)
    if response.status_code != requests.codes.ok:
        raise RetrieveAllelesError()
    return response.text


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Retrieve HLA alleles from IPD-IMGT/HLA."
    )
    parser.add_argument(
        "release",
        help="release to pull the data from",
        type=str,
    )
    parser.add_argument(
        "--output_a",
        help="filename to store the HLA-A data in",
        type=str,
        default="hla_a_std.csv",
    )
    parser.add_argument(
        "--output_b",
        help="filename to store the HLA-B data in",
        type=str,
        default="hla_b_std.csv",
    )
    parser.add_argument(
        "--output_c",
        help="filename to store the HLA-C data in",
        type=str,
        default="hla_c_std.csv",
    )
    parser.add_argument(
        "--output_a_reduced",
        help="filename to store the reduced HLA-A data in",
        type=str,
        default="hla_a_std_reduced.csv",
    )
    parser.add_argument(
        "--output_b_reduced",
        help="filename to store the reduced HLA-B data in",
        type=str,
        default="hla_b_std_reduced.csv",
    )
    parser.add_argument(
        "--output_c_reduced",
        help="filename to store the reduced HLA-C data in",
        type=str,
        default="hla_c_std_reduced.csv",
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
    args = parser.parse_args

    print(f"Retrieving alleles from release {args.release}....")
    alleles_str: str
    for i in range(5):
        try:
            alleles_str = get_alleles_file(args.release)
        except RetrieveAllelesError:
            print("Failed to retrieve alleles; retrying in 20 seconds....")
            time.sleep(20)
        else:
            break

    # Compute the checksum.
    md5_calc = hashlib("md5")
    md5_calc.update(alleles_str.encode())
    with open(args.checksum, "w") as f:
        f.write(f"{md5_calc.hexdigest()}  {HLA_ALLELES_FILENAME}\n")

    standards: dict[HLA_LOCUS, list[tuple[str, str, str]]] = {
        "A": [],
        "B": [],
        "C": [],
    }
    for allele_sr in Bio.SeqIO.parse(StringIO(alleles_str)):
        # The FASTA headers look like:
        # >HLA:HLA00001 A*01:01:01:01 1098 bp
        allele_name: str = allele_sr.description.split(" ")[0]
        locus: HLA_LOCUS = allele_name[0]

        exon2_match: tuple[int, Optional[str]] = get_acceptable_match(
            str(allele_sr.seq),
            EXON_SEQUENCES[locus]["exon2"],
        )
        exon3_match: tuple[int, Optional[str]] = get_acceptable_match(
            str(allele_sr.seq),
            EXON_SEQUENCES[locus]["exon3"],
        )
        if (
            exon2_match[0] <= args.mismatch_threshold
            and exon3_match[0] <= args.mismatch_threshold
        ):
            standards[locus].append((allele_name, exon2_match[1], exon3_match[1]))

        # FIXME continue from here


if __name__ == "__main__":
    main()
