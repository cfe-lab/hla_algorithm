import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Optional

import Bio
import numpy as np

# A lookup table of translations from ambiguous nucleotides to unambiguous
# nucleotides.
AMBIG: Final[dict[str, list[str]]] = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "S": ["C", "G"],
    "W": ["A", "T"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}

# Thanks to binary logic, we encode nucleotide positions as a 4 bit number
# the first position 000a represents 'A',
# the second position 00a0 represents 'C',
# the third position 0a00 represents 'G',
# the fourth position a000 represents 'T'
# We can then perform binary ORs, XORs, and ANDs, to check whether or not
# a mixture contains a specific nucleotide.
PURENUC2BIN: Final[dict[str, int]] = {nuc: 2**i for i, nuc in enumerate("ACGT")}

# Nucleotides converted to their binary representation
# LISTOFNUCS: list[str] = [
#     "A",  # => 0b0001,
#     "C",  # => 0b0010,
#     "G",  # => 0b0100,
#     "T",  # => 0b1000,
#     "M",  # => 0b0011,
#     "R",  # => 0b0101,
#     "W",  # => 0b1001,
#     "S",  # => 0b0110,
#     "Y",  # => 0b1010,
#     "K",  # => 0b1100,
#     "B",  # => 0b1110,
#     "D",  # => 0b1101,
#     "H",  # => 0b1011,
#     "V",  # => 0b0111,
#     "N",  # => 0b1111
# ]
NUC2BIN: Final[dict[str, int]] = {
    k: sum([{nuc: 2**i for i, nuc in enumerate("ACGT")}[nuc] for nuc in v])
    for k, v in AMBIG.items()
}
BIN2NUC: Final[dict[int, str]] = {v: k for k, v in NUC2BIN.items()}

HLA_LOCUS = Literal["A", "B", "C"]
EXON_NAME = Literal["exon2", "exon3"]


def nuc2bin(seq: str) -> tuple[int, ...]:
    """
    Convert a string sequence to a numpy array.

    Converts a string sequence to a numpy array containing binary
    equivalents of the strings.

    :param seq: ...
    :type seq: str
    :return: ...
    :rtype: tuple[int, ...]
    """
    return tuple(NUC2BIN.get(nuc, 0) for nuc in seq)


def bin2nuc(seq: Iterable[int]) -> str:
    """
    Convert an array of numbers to a string sequence.

    :param seq: ...
    :type seq: Iterable[int]
    :return: ...
    :rtype: str
    """
    return "".join([BIN2NUC.get(nuc, "_") for nuc in seq])


def count_strict_mismatches(
    sequence_1: Sequence[int], sequence_2: Sequence[int]
) -> int:
    """
    Compare two sequences in "binary" format.

    This will output the number of "strict mismatches" between the standard
    and the sequence, meaning positions where the bases have no overlapping
    possible resolutions.

    :param sequence_1: A sequence in "binary" format.
    :type sequence_1: Sequence[int]
    :param seq: A sequence in "binary" format.
    :type seq: Sequence[int]
    :return: Number of mismatches between the two sequences.
    :rtype: int
    """
    masked_array: np.ndarray = np.array(sequence_1) & np.array(sequence_2)
    return np.count_nonzero(masked_array == 0)


def count_forgiving_mismatches(
    sequence_1: Sequence[int], sequence_2: Sequence[int]
) -> int:
    """
    Compare two sequences in "binary" format, being more "forgiving" of mismatches.

    At each position, a match is defined as all possible bases in the
    sequence being possible bases in the standard, or vice versa.
    """
    if len(sequence_1) != len(sequence_2):
        raise ValueError("Sequences must be the same length")
    if len(sequence_1) == 0:
        raise ValueError("Sequences must be non-empty")
    seq1_np: np.ndarray = np.array(sequence_1)
    seq2_np: np.ndarray = np.array(sequence_2)

    overlaps: np.ndarray = seq1_np & seq2_np
    matches: np.ndarray = (overlaps == seq1_np) | (overlaps == seq2_np)
    return np.count_nonzero(matches == False)


def check_bases(seq: str) -> None:
    """
    Check a string sequence for invalid characters.

    If an invalid character is detected it will raise a ValueError.

    :param seq: ...
    :type seq: str
    :raises ValueError: Raised if a sequence contains letters we don't
    expect
    :return: True if our sequence only contains valid characters.
    :rtype: bool
    """
    if not re.match(r"^[ATGCRYKMSWNBDHV]+$", seq):
        raise ValueError("Sequence has invalid characters")


HLA_A_LENGTH: Final[int] = 787
MIN_HLA_BC_LENGTH: Final[int] = 787
MAX_HLA_BC_LENGTH: Final[int] = 796
EXON2_LENGTH: Final[int] = 270
EXON3_LENGTH: Final[int] = 276


def check_length(locus: HLA_LOCUS, seq: str, name: str) -> None:
    """
    Validates the length of a sequence. This asserts a sequence either
    exactly a certain size, or is within an allowed range.

    See the following values in utils:
        - HLA_A_LENGTH
        - EXON2_LENGTH
        - EXON3_LENGTH
        - MAX_HLA_BC_LENGTH
        - MIN_HLA_BC_LENGTH

    :param seq: Sequence to be validated.
    :type seq: str
    :param name: Name of sequence. This will commonly be the ID in the fasta
    file.
    :type name: str
    :raises ValueError: Raised if length of sequence is outside allowed
    parameters.
    :return: Returns true if sequence is within allowed parameters.
    :rtype: bool
    """
    error_condition: bool = False
    if name.lower().endswith("short"):
        if locus == "A":
            error_condition = len(seq) >= HLA_A_LENGTH
        elif "exon2" in name.lower():
            error_condition = len(seq) >= EXON2_LENGTH
        elif "exon3" in name.lower():
            error_condition = len(seq) >= EXON3_LENGTH
        else:
            error_condition = len(seq) >= MAX_HLA_BC_LENGTH
    elif locus == "A":
        error_condition = len(seq) != HLA_A_LENGTH
    elif "exon2" in name.lower():
        error_condition = len(seq) != EXON2_LENGTH
    elif "exon3" in name.lower():
        error_condition = len(seq) != EXON3_LENGTH
    else:
        error_condition = not (MIN_HLA_BC_LENGTH <= len(seq) <= MAX_HLA_BC_LENGTH)

    if error_condition:
        raise ValueError(
            f"Sequence {name} is the wrong length ({len(seq)}bp). Check the locus {locus}"
        )


def calc_padding(std: Sequence[int], seq: Sequence[int]) -> tuple[int, int]:
    """
    Calculate the number of units to pad a sequence.

    This will attempt to achieve the best pad value by minimizing the
    number of mismatches.

    :param std: ...
    :type std: Sequence[int]
    :param seq: ...
    :type seq: Sequence[int]
    :return: Returns the number of 'N's (b1111) needed to match the sequence
    to the standard.
    :rtype: tuple[int, int]
    """
    best = 10e10
    pad = len(std) - len(seq)
    left_pad = 0
    for i in range(pad + 1):  # 0, 1, ..., pad - 1, pad
        pseq = np.concatenate(
            (
                np.array(nuc2bin("N" * i), dtype="int8"),
                np.array(seq, dtype="int8"),
                np.array(nuc2bin("N" * (pad - i)), dtype="int8"),
            ),
        )
        mismatches = count_strict_mismatches(std, pseq)
        if mismatches < best:
            best = mismatches
            left_pad = i
    return left_pad, pad - left_pad


def get_acceptable_match(
    sequence: str, reference: str, mismatch_threshold: int = 20
) -> tuple[int, Optional[str]]:
    """
    Get an "acceptable match" between the sequence and reference.

    For every possible "shift" of the sequence (i.e. comparing it against the
    reference starting at every position between 0 and the difference in lengths
    of the sequence and reference), we count the number of mismatches.  If the
    score dips under the threshold, we return that score and sequence;
    otherwise, we return the best score and sequence observed.

    The sequence must be at least as long as the reference.
    """
    if len(sequence) < len(reference):
        raise ValueError("sequence must be at least as long as the reference")

    score: int = len(reference)
    best_match: Optional[str] = None

    ref_np: np.ndarray = np.array(list(reference))
    for shift in range(len(sequence) - len(reference) + 1):
        curr_sequence_window: str = sequence[shift : (shift + len(reference))]

        curr_seq_np: np.ndarray = np.array(list(curr_sequence_window))
        curr_score: int = np.count_nonzero(curr_seq_np != ref_np)
        if curr_score < score:
            score = curr_score
            best_match = curr_sequence_window

        if score < mismatch_threshold:
            break

    return (score, best_match)


def allele_coordinates(
    allele: str,
    digits_only: bool = False,
) -> list[str]:
    """
    Convert an allele string into a list of coordinates.

    For example, allele "A*01:23:45N" gets converted to
    ["A*01", "23", "45N"] or ["01", "23", "45] depending on the value of
    digits_only.
    """
    clean_allele_str: str = allele
    if digits_only:
        clean_allele_str = re.sub(r"[^\d:]", "", allele)
    return clean_allele_str.strip().split(":")


def allele_integer_coordinates(allele: str) -> tuple[int, ...]:
    """
    Convert an allele string into a list of integer coordinates.

    For example, allele "A*01:23:45N" gets converted to
    (1, 23, 45)
    """
    return tuple(int(coord) for coord in allele_coordinates(allele, True))


def collate_standards(
    allele_srs: Sequence[Bio.SeqIO.SeqRecord],
    exon_references: dict[HLA_LOCUS, dict[EXON_NAME, str]],
    logger: Optional[logging.Logger] = None,
    overall_mismatch_threshold: int = 32,
    acceptable_match_search_threshold: int = 20,
    report_interval: Optional[int] = 1000,
) -> dict[HLA_LOCUS, list[tuple[str, str, str]]]:
    """
    Collate and sort HLA-A, -B, and -C standards from the specified source.

    SequenceRecords are parsed to get their name and locus, and the sequence is
    checked to see if it has acceptable matches for both exon2 and exon3.
    """
    output_status_updates: bool = False
    if logger is not None and report_interval is not None and report_interval > 0:
        output_status_updates = True

    standards: dict[HLA_LOCUS, list[tuple[str, str, str]]] = {
        "A": [],
        "B": [],
        "C": [],
    }
    for idx, allele_sr in enumerate(allele_srs, start=1):
        if output_status_updates and idx % report_interval == 0:
            logger.info(f"Processing sequence {idx} of {len(allele_srs)}....")

        # The FASTA headers look like:
        # >HLA:HLA00001 A*01:01:01:01 1098 bp
        allele_name: str = allele_sr.description.split(" ")[1]
        locus: HLA_LOCUS = allele_name[0]

        if locus not in ("A", "B", "C"):
            continue

        exon2_match: tuple[int, Optional[str]] = get_acceptable_match(
            str(allele_sr.seq),
            exon_references[locus]["exon2"],
            mismatch_threshold=acceptable_match_search_threshold,
        )
        exon3_match: tuple[int, Optional[str]] = get_acceptable_match(
            str(allele_sr.seq),
            exon_references[locus]["exon3"],
            mismatch_threshold=acceptable_match_search_threshold,
        )
        if (
            exon2_match[0] <= overall_mismatch_threshold
            and exon3_match[0] <= overall_mismatch_threshold
        ):
            standards[locus].append((allele_name, exon2_match[1], exon3_match[1]))
        elif logger is not None:
            logger.info(
                f'Rejecting "{allele_name}": {exon2_match[0]} exon2 mismatches,'
                f" {exon3_match[0]} exon3 mismatches."
            )

    for locus in ("A", "B", "C"):
        standards[locus].sort(key=lambda x: allele_integer_coordinates(x[0]))

    return standards


@dataclass
class GroupedAllele:
    exon2: str
    exon3: str
    alleles: list[str]

    def get_group_name(self) -> str:
        """
        Get the "group name" of this grouped allele.

        From the "original allele", create the name of the grouped allele
        by taking (up to) the first 3 coordinates and adding a "G" at the
        end.  (This makes the most sense if these alleles are sorted.)
        """
        orig_allele: str = self.alleles[0]
        if len(self.alleles) == 1:
            return orig_allele

        coords: list[str] = allele_coordinates(orig_allele, digits_only=False)
        if len(coords) > 3:
            coords = coords[:3]
        return ":".join(coords) + "G"


def group_identical_alleles(
    allele_infos: list[tuple[str, str, str]],
    logger: Optional[logging.Logger] = None,
) -> dict[str, GroupedAllele]:
    """
    Collapse common alleles into single entries.
    """
    seq_to_name: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name, exon2, exon3 in allele_infos:
        seq_to_name[(exon2, exon3)].append(name)

    grouped_alleles: dict[str, GroupedAllele] = {}
    for exon2, exon3 in seq_to_name:
        alleles: list[str] = seq_to_name[(exon2, exon3)]
        grouped_allele: GroupedAllele = GroupedAllele(
            exon2,
            exon3,
            alleles,
        )
        grouped_name: str = grouped_allele.get_group_name()
        grouped_alleles[grouped_name] = grouped_allele
        if logger is not None and len(alleles) > 1:
            logger.info(f"[{', '.join(grouped_allele.alleles)}] -> {grouped_name}")

    return grouped_alleles
