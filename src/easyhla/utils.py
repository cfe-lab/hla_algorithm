import re
from collections.abc import Iterable, Sequence
from typing import Final, Optional

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

    The sequence must be at least as long as the alignment.
    """
    if len(sequence) < len(reference):
        raise ValueError("sequence must be at least as long as the reference")

    score: int = len(reference)
    best_match: Optional[str] = None

    ref_np: np.ndarray = np.array(list(reference))
    for shift in range(len(sequence) - len(reference)):
        curr_sequence_window: str = sequence[shift : (shift + len(reference))]

        curr_seq_np: np.ndarray = np.array(list(curr_sequence_window))
        curr_score: int = np.count_nonzero(curr_seq_np != ref_np)
        if curr_score < score:
            score = curr_score
            best_match = curr_sequence_window

        if score < mismatch_threshold:
            break

    return (score, best_match)
