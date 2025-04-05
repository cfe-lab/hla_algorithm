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


def nuc2bin(seq: str) -> np.ndarray:
    """
    Convert a string sequence to a numpy array.

    Converts a string sequence to a numpy array containing binary
    equivalents of the strings.

    :param seq: ...
    :type seq: str
    :return: ...
    :rtype: np.ndarray
    """
    return np.array([NUC2BIN.get(seq[i], 0) for i in range(len(seq))], dtype="int8")


@staticmethod
def bin2nuc(seq: np.ndarray) -> str:
    """
    Convert an array of numbers to a string sequence.

    Converts an array of numbers back to a string sequence.

    :param seq: ...
    :type seq: np.ndarray
    :return: ...
    :rtype: str
    """
    return "".join([BIN2NUC.get(seq[i], "_") for i in range(len(seq))])
