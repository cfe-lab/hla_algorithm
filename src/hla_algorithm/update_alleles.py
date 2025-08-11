#! /usr/bin/env python

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from io import StringIO
from typing import Final, Optional, TypedDict, cast

import Bio
import requests
import yaml

from .utils import (
    EXON_NAME,
    HLA_LOCUS,
    HLARawStandard,
    StoredHLAStandards,
    collate_standards,
    group_identical_alleles,
)

logging.basicConfig()
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
REPO_OWNER: Final[str] = os.environ.get(
    "HLA_ALGORITHM_ALLELES_REPO_OWNER",
    "ANHIG",
)
REPO_NAME: Final[str] = os.environ.get(
    "HLA_ALGORITHM_ALLELES_REPO_NAME",
    "IMGTHLA",
)
HLA_ALLELES_FILENAME: Final[str] = os.environ.get(
    "HLA_ALGORITHM_ALLELES_REPO_FILENAME",
    "hla_nuc.fasta",
)


class RetrieveAllelesError(Exception):
    pass


class RetrieveCommitHashError(Exception):
    pass


def get_alleles_file(
    tag: str,
    repo_owner: str = REPO_OWNER,
    repo_name: str = REPO_NAME,
    alleles_filename: str = HLA_ALLELES_FILENAME,
) -> str:
    """
    Retrieve the HLA alleles file from the specified tag.
    """
    url: str = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/"
        f"contents/{alleles_filename}?ref={tag}"
    )
    response: requests.Response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if response.status_code != requests.codes.ok:
        raise RetrieveAllelesError()
    return response.text


class CommitInfo(TypedDict):
    sha: str
    url: str


class TagInfo(TypedDict):
    name: str
    commit: CommitInfo
    zipball_url: str
    tarball_url: str
    node_id: str


def get_commit_hash(
    tag_name: str,
    repo_owner: str = REPO_OWNER,
    repo_name: str = REPO_NAME,
) -> Optional[str]:
    """
    Retrieve the commit hash of the specified tag.
    """
    url: str = f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags"
    response: requests.Response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if response.status_code != requests.codes.ok:
        raise RetrieveCommitHashError()

    tags: list[TagInfo] = json.loads(response.text)
    for tag in tags:
        if tag["name"] == tag_name:
            return tag["commit"]["sha"]

    return None


def get_from_git(tag: str) -> tuple[str, datetime, Optional[str]]:
    alleles_str: str
    retrieval_datetime: datetime
    for i in range(5):
        try:
            retrieval_datetime = datetime.now()
            alleles_str = get_alleles_file(tag)
        except RetrieveAllelesError:
            if i < 4:
                logger.info("Failed to retrieve alleles; retrying in 20 seconds....")
                time.sleep(20)
            else:
                raise
        else:
            break

    commit_hash: Optional[str]
    for i in range(5):
        try:
            commit_hash = get_commit_hash(tag)
        except RetrieveCommitHashError:
            if i < 4:
                logger.info(
                    "Failed to retrieve the commit hash; retrying in 20 seconds...."
                )
                time.sleep(20)
            else:
                raise
        else:
            break

    return alleles_str, retrieval_datetime, commit_hash


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Retrieve HLA alleles from IPD-IMGT/HLA."
    )
    parser.add_argument(
        "tag",
        help="Git tag to pull the data from",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="filename to store the unreduced standards (YAML format)",
        type=str,
        default="hla_standards.yaml",
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

    logger.info(f"Retrieving alleles from tag {args.tag}....")
    alleles_str: str
    retrieval_datetime: datetime
    commit_hash: Optional[str]
    alleles_str, retrieval_datetime, commit_hash = get_from_git(args.tag)
    logger.info(
        f"Alleles (version {args.tag}, commit hash {commit_hash}) retrieved at "
        f"{retrieval_datetime}."
    )

    if args.dump_full_fasta_to != "":
        logger.info(f"Dumping the full FASTA file to {args.dump_full_fasta_to}.")
        with open(args.dump_full_fasta_to, "w") as f:
            f.write(alleles_str)

    # Compute the checksum.
    md5_calc = hashlib.md5()
    md5_calc.update(alleles_str.encode())
    with open(args.checksum, "w") as f:
        f.write(f"{md5_calc.hexdigest()} {HLA_ALLELES_FILENAME}\n")

    raw_standards: dict[HLA_LOCUS, list[HLARawStandard]] = collate_standards(
        list(Bio.SeqIO.parse(StringIO(alleles_str), "fasta")),
        EXON_REFERENCES,
        logger,
        args.mismatch_threshold,
        args.acceptable_match_search_threshold,
        args.standard_report_interval,
    )

    logger.info("Identifying identical HLA alleles....")
    standards_for_saving: StoredHLAStandards = StoredHLAStandards(
        tag=args.tag,
        commit_hash=commit_hash or "",
        last_updated=retrieval_datetime,
        standards={
            cast(HLA_LOCUS, locus): group_identical_alleles(
                raw_standards[cast(HLA_LOCUS, locus)]
            )
            for locus in ("A", "B", "C")
        },
    )

    # First, prepare the unreduced YAML output.
    logger.info(f"Writing HLA standards to {args.output}....")
    with open(args.output, "w") as f:
        yaml.safe_dump(standards_for_saving.model_dump(), f)

    logger.info("Done.")


if __name__ == "__main__":
    main()
