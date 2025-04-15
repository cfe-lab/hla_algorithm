from collections.abc import Iterable, Sequence
from typing import Final

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


@staticmethod
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
