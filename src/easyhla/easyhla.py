import logging
import os
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from io import TextIOBase
from operator import attrgetter
from typing import Any, Final, Literal, Optional

import Bio.SeqIO
import numpy as np
import pydantic_numpy.typing as pnd

from .models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAResult,
    HLAResultRow,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)

DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"

HLA_LOCI = Literal["A", "B", "C"]

EXON_NAME = Literal["exon2", "exon3"]
EXON_AND_OTHER_EXON: list[tuple[EXON_NAME, EXON_NAME]] = [
    ("exon2", "exon3"),
    ("exon3", "exon2"),
]


class EasyHLA:
    HLA_A_LENGTH: Final[int] = 787
    MIN_HLA_BC_LENGTH: Final[int] = 787
    MAX_HLA_BC_LENGTH: Final[int] = 796
    EXON2_LENGTH: Final[int] = 270
    EXON3_LENGTH: Final[int] = 276
    ALLELES_MAX_REPORTABLE_STRING: Final[int] = 3900

    ALLOWED_HLA_LOCI: Final[list[str]] = ["A", "B", "C"]

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
        "V": ["C", "G", "T"],
        "H": ["A", "G", "T"],
        "D": ["A", "C", "T"],
        "B": ["A", "C", "G"],
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
    #     "V",  # => 0b1110,
    #     "H",  # => 0b1101,
    #     "D",  # => 0b1011,
    #     "B",  # => 0b0111,
    #     "N",  # => 0b1111
    # ]
    NUC2BIN: Final[dict[str, int]] = {
        k: sum([{nuc: 2**i for i, nuc in enumerate("ACGT")}[nuc] for nuc in v])
        for k, v in AMBIG.items()
    }
    BIN2NUC: Final[dict[int, str]] = {v: k for k, v in NUC2BIN.items()}

    COLUMN_IDS: Final[dict[str, int]] = {"A": 0, "B": 2, "C": 4}

    def __init__(
        self,
        locus: HLA_LOCI,
        hla_standards: Optional[TextIOBase] = None,
        hla_frequencies: Optional[TextIOBase] = None,
        last_modified_time: Optional[datetime] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize an EasyHLA class.

        :param locus: HLA subtype that this object will be performing
        interpretation against.
        :type locus: "A", "B", or "C"
        :param logger: Python logger object, defaults to None
        :type logger: Optional[logging.Logger], optional
        :raises ValueError: Raised if locus != "A"/"B"/"C"
        """
        if locus not in ["A", "B", "C"]:
            raise ValueError("Invalid HLA locus specified; must be A, B, or C")
        self.locus: HLA_LOCI = locus
        self.hla_stds: list[HLAStandard] = self.load_hla_stds(
            hla_standards=hla_standards,
        )
        self.hla_freqs: dict[str, int] = self.load_hla_frequencies(
            hla_frequencies=hla_frequencies,
        )
        self.last_modified_time: datetime
        if last_modified_time is not None:
            self.last_modified_time = last_modified_time
        else:
            self.last_modified_time = self.load_allele_definitions_last_modified_time()
        self.log = logger or logging.Logger(__name__, logging.ERROR)

    def load_hla_frequencies(
        self,
        hla_frequencies: Optional[TextIOBase] = None,
    ) -> dict[str, int]:
        """
        Load HLA frequencies from reference file.

        This takes two columns AAAA,BBBB out of 6 (...FFFF), and then uses a
        subset of these two columns (AABB,CCDD) to use as the key, in this case
        "AA|BB,CC|DD", we then count the number of times this key appears in our
        columns.

        Implementation Note: This will eventually be consumed by a regex
        function! In the original ruby script we would output a hash similar to
        `{ ["AA|BB", "CC|DD"] => 0 }`, but Python does not support lists as keys
        in dict objects. If using this dict for regex you will need to perform a
        `.split(',')` on the key.

        :return: Lookup table of HLA frequencies.
        :rtype: dict[str, int]
        """
        hla_freqs: dict[str, int] = {}

        freqs_io: TextIOBase = hla_frequencies
        default_freqs_used: bool = False
        try:
            if hla_frequencies is None:
                freqs_io = open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_data",
                        "hla_frequencies.csv",
                    ),
                    "r",
                    encoding="utf-8",
                )
                default_freqs_used = True

            for line in freqs_io.readlines():
                column_id = EasyHLA.COLUMN_IDS[self.locus]
                line_array = line.strip().split(",")[column_id : column_id + 2]
                elements = ",".join([f"{a[:2]}|{a[-2:]}" for a in line_array])
                if hla_freqs.get(elements, None) is None:
                    hla_freqs[elements] = 0
                hla_freqs[elements] += 1
        finally:
            if default_freqs_used:
                freqs_io.close()

        return hla_freqs

    # In the future it may make sense to convert this to return a dictionary
    # keyed by the allele name, but in the current code it's only ever used
    # as a list.
    def load_hla_stds(
        self,
        hla_standards: Optional[TextIOBase] = None,
    ) -> list[HLAStandard]:
        """
        Load HLA Standards from reference file.

        :return: List of known HLA standards
        :rtype: list[HLAStandard]
        """
        hla_stds: list[HLAStandard] = []

        standards_io: TextIOBase = hla_standards
        default_standards_used: bool = False
        try:
            if hla_standards is None:
                standards_io = open(
                    os.path.join(
                        os.path.dirname(__file__),
                        "default_data",
                        f"hla_{self.locus.lower()}_std_reduced.csv",
                    ),
                    "r",
                    encoding="utf-8",
                )
                default_standards_used = True

            for line in standards_io.readlines():
                line_array = line.strip().split(",")
                seq = self.nuc2bin((line_array[1] + line_array[2]))
                hla_stds.append(HLAStandard(allele=line_array[0], sequence=seq))
        finally:
            if default_standards_used:
                standards_io.close()

        return hla_stds

    def load_allele_definitions_last_modified_time(self) -> datetime:
        """
        Load a datetime object describing when standard definitions were last updated.

        :return: Date representing time when references were last updated.
        :rtype: datetime
        """
        filename = os.path.join(
            os.path.dirname(__file__),
            "default_data",
            "hla_nuc.fasta.mtime",
        )
        with open(filename, "r", encoding="utf-8") as f:
            last_mod_date = "".join(f.readlines()).strip()
        return datetime.strptime(last_mod_date, DATE_FORMAT)

    def check_length(self, seq: str, name: str) -> None:
        """
        Validates the length of a sequence. This asserts a sequence either
        exactly a certain size, or is within an allowed range.

        See the following class values:
         - EasyHLA.HLA_A_LENGTH
         - EasyHLA.EXON2_LENGTH
         - EasyHLA.EXON3_LENGTH
         - EasyHLA.MAX_HLA_BC_LENGTH
         - EasyHLA.MIN_HLA_BC_LENGTH

        :param seq: Sequence to be validated.
        :type seq: str
        :param name: Name of sequence. This will commonly be the ID/descriptor
        in the fasta file.
        :type name: str
        :raises ValueError: Raised if length of sequence is outside allowed
        parameters.
        :return: Returns true if sequence is within allowed parameters.
        :rtype: bool
        """
        error_condition: bool = False
        if name.lower().endswith("short"):
            if self.locus.upper() == "A":
                error_condition = len(seq) >= EasyHLA.HLA_A_LENGTH
            elif "exon2" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON2_LENGTH
            elif "exon3" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON3_LENGTH
            else:
                error_condition = len(seq) >= EasyHLA.MAX_HLA_BC_LENGTH
        elif self.locus.upper() == "A":
            error_condition = len(seq) != EasyHLA.HLA_A_LENGTH
        elif "exon2" in name.lower():
            error_condition = len(seq) != EasyHLA.EXON2_LENGTH
        elif "exon3" in name.lower():
            error_condition = len(seq) != EasyHLA.EXON3_LENGTH
        else:
            error_condition = not (
                EasyHLA.MIN_HLA_BC_LENGTH <= len(seq) <= EasyHLA.MAX_HLA_BC_LENGTH
            )

        if error_condition:
            raise ValueError(
                f"Sequence {name} is the wrong length ({len(seq)} bp). Check the locus {self.locus}"
            )

    @staticmethod
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

    def nuc2bin(self, seq: str) -> np.ndarray:
        """
        Convert a string sequence to a numpy array.

        Converts a string sequence to a numpy array containing binary
        equivalents of the strings.

        :param seq: ...
        :type seq: str
        :return: ...
        :rtype: np.ndarray
        """
        return np.array(
            [EasyHLA.NUC2BIN.get(seq[i], 0) for i in range(len(seq))], dtype="int8"
        )

    def bin2nuc(self, seq: np.ndarray) -> str:
        """
        Convert an array of numbers to a string sequence.

        Converts an array of numbers back to a string sequence.

        :param seq: ...
        :type seq: np.ndarray
        :return: ...
        :rtype: str
        """
        return "".join([EasyHLA.BIN2NUC.get(seq[i], "_") for i in range(len(seq))])

    def calc_padding(self, std: np.ndarray, seq: np.ndarray) -> tuple[int, int]:
        """
        Calculate the number of units to pad a sequence.

        Calculate the number of units to pad a sequence if it doesn't match the
        length of the standard. This will attempt to achieve the best pad value
        by minimizing the number of mismatches.

        :param std: ...
        :type std: np.ndarray
        :param seq: ...
        :type seq: np.ndarray
        :return: Returns the number of 'N's (b1111) needed to match the sequence
        to the standard.
        :rtype: tuple[int, int]
        """
        best = 10e10
        pad = len(std) - len(seq)
        left_pad = 0
        for i in range(pad):
            pseq = np.concatenate(
                (self.nuc2bin("N" * i), seq, self.nuc2bin("N" * (pad - i)))
            )
            mismatches = self.std_match(std, pseq)
            if mismatches < best:
                best = mismatches
                left_pad = i

        return left_pad, pad - left_pad

    def pad_short(
        self,
        seq: np.ndarray,
        exon: Optional[EXON_NAME],
        hla_std: HLAStandard,
    ) -> np.ndarray:
        # hla_stds expects [ ["label0", [1,2,3,4]], ["label1", [2,3,4,5]] ]
        std = None
        has_intron = False
        if exon == "exon2":
            std = hla_std.sequence[: EasyHLA.EXON2_LENGTH]
        elif exon == "exon3":
            std = hla_std.sequence[EasyHLA.EXON2_LENGTH : EasyHLA.EXON3_LENGTH]
        else:
            has_intron = True
            std = hla_std.sequence

        if has_intron:
            left_pad, _ = self.calc_padding(
                std[: EasyHLA.EXON2_LENGTH], seq[: int(EasyHLA.EXON2_LENGTH / 2)]
            )
            _, right_pad = self.calc_padding(
                std[-int(EasyHLA.EXON3_LENGTH / 2) :],
                seq[-int(EasyHLA.EXON3_LENGTH / 2) :],
            )

        else:
            left_pad, right_pad = self.calc_padding(std, seq)

        short_padded_array = np.concatenate(
            (self.nuc2bin("N" * left_pad), seq, self.nuc2bin("N" * right_pad))
        )
        return short_padded_array

    def std_match(self, std: np.ndarray, seq: np.ndarray) -> int:
        """
        Compare an HLA standard against an incoming sequence.

        This will output the number of mismatches between the standard and the
        sequence.

        :param std: HLA standard sequence
        :type std: np.ndarray
        :param seq: Sequence being tested against
        :type seq: np.ndarray
        :return: Number of mismatches between the two sequences.
        :rtype: int
        """
        mismatches = 0
        masked_array: np.ndarray = std & seq
        mismatches = np.count_nonzero(masked_array == 0)
        return mismatches

    def get_matching_stds(
        self,
        seq: np.ndarray,
        hla_stds: list[HLAStandard],
        mismatch_threshold: int = 5,
    ) -> list[HLAStandardMatch]:
        # Returns [ ["std_name", [1,2,3,4], num_mismatches], ["std_name2", [2,3,4,5], num_mismatches2]]
        matching_stds: list[HLAStandardMatch] = []
        for std in hla_stds:
            mismatches = self.std_match(std.sequence, seq)
            if mismatches < mismatch_threshold:
                matching_stds.append(
                    HLAStandardMatch(
                        allele=std.allele, sequence=std.sequence, mismatch=mismatches
                    )
                )
        return matching_stds

    def combine_standards_helper(
        self,
        matching_stds: Sequence[HLAStandardMatch],
        seq: list[int],
        mismatch_threshold: Optional[int] = None,
    ) -> dict[str, tuple[int, list[tuple[str, str]]]]:
        """
        Helper to identify "good" combined standards for the specified sequence.

        Returns a mapping:
        sequence string -|-> (mismatch count, allele pair list)

        This mapping will contain "good" combined standards: it will definitely
        contain the best-matching combined standard(s), and if
        mismatch_threshold is an integer, it will also contain any combined
        standards which have fewer mismatches than the threshold.  It may also
        contain other combined standards, which will be winnowed out by the
        calling function.
        """
        combos: dict[str, tuple[int, list[tuple[str, str]]]] = {}

        if mismatch_threshold is None:
            # We only care about the best match, so we set this threshold
            # so that it never forces the rejection threshold to stay above
            # the current best match.
            mismatch_threshold = 0

        current_rejection_threshold: int = float("inf")
        for std_ai, std_a in enumerate(matching_stds):
            if std_a.mismatch > current_rejection_threshold:
                continue
            for std_bi, std_b in enumerate(matching_stds):
                if std_ai < std_bi:
                    break
                if std_b.mismatch > current_rejection_threshold:
                    continue

                # "Mush" the two standards together to produce something
                # that looks like what you get when you sequence HLA.
                std_bin = std_b.sequence | std_a.sequence
                seq_mask = np.full_like(std_bin, fill_value=15)
                mismatches: int = np.count_nonzero((std_bin ^ seq) & seq_mask != 0)

                if mismatches > current_rejection_threshold:
                    continue

                # This looks like 1-4-9-16-2-... where each number comes from
                # the binary representation of the base, i.e. from NUC2BIN.
                # There could be more than one combined standard with the
                # same sequence!
                combined_std_str = "-".join([str(s) for s in std_bin])

                if mismatches < current_rejection_threshold:
                    current_rejection_threshold = max(mismatches, mismatch_threshold)

                if combined_std_str not in combos:
                    combos[combined_std_str] = (mismatches, [])
                combos[combined_std_str][1].append(sorted((std_a.allele, std_b.allele)))
        return combos

    def combine_standards(
        self,
        matching_stds: Sequence[HLAStandardMatch],
        seq: list[int],
        mismatch_threshold: Optional[int] = None,
    ) -> dict[HLACombinedStandard, int]:
        """
        Find the combinations of standards that match the given sequence.

        Humans have two copies of their HLA genes, so when we use Sanger
        sequencing to sequence a person's HLA, we get a single sequence with
        potentially many mixtures.  That is, at any position that the two genes
        don't match, we see a nucleotide mixture consisting of the two
        corresponding bases.

        In order to find matches, we take allele sequences (reduced to ones that
        are already "decent" matches for our sequence, to reduce running time)
        and "mush" them together to produce potential matches for our sequence.

        PRECONDITION: matching_stds should contain no duplicates.

        Returns a dictionary mapping HLACombinedStandards to their mismatch
        counts.  If mismatch_threshold is None, then the result contains only
        the best-matching combined standard(s); otherwise, the result contains
        all combined standards with mismatch counts up to and including the
        threshold.
        """
        combos: dict[str, tuple[int, list[tuple[str, str]]]] = (
            self.combine_standards_helper(
                matching_stds,
                seq,
                mismatch_threshold,
            )
        )

        # Winnow out any extraneous combined standards that don't match our
        # criteria.
        result: dict[HLACombinedStandard, int] = {}

        fewest_mismatches: int = min([x[0] for x in combos.values()])
        cutoff: int = max(fewest_mismatches, mismatch_threshold)

        for combined_std_str, mismatch_count_and_pair_list in combos.items():
            mismatch_count: int
            pair_list: list[tuple[str, str]]
            mismatch_count, pair_list = mismatch_count_and_pair_list

            if mismatch_count <= cutoff:
                combined_std: HLACombinedStandard = HLACombinedStandard(
                    standard=combined_std_str,
                    discrete_allele_names=tuple(pair_list),
                )
                result[combined_std] = mismatch_count

        return result

    @staticmethod
    def get_all_allele_pairs(
        combined_standards: Iterable[HLACombinedStandard],
    ) -> AllelePairs:
        """
        Get all allele pairs in the specified combined standards.

        :param best_matches: ...
        :type best_matches: Iterable[HLACombinedStandard]
        :return: ...
        :rtype: AllelePairs
        """
        all_allele_pairs: list[tuple[str, str]] = []
        for combined_std in combined_standards:
            all_allele_pairs.extend(combined_std.discrete_allele_names)
        all_allele_pairs.sort()
        return AllelePairs(allele_pairs=all_allele_pairs)

    def pair_exons_helper(
        self,
        sequence_record: Bio.SeqIO.SeqRecord,
        unmatched: dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]],
    ) -> Optional[tuple[str, bool, bool, str, str]]:
        """
        Helper that attempts to match the given sequence with a "partner" exon.

        `sequence_record` represents a sequence that may be an exon2 or exon3
        sequence (or neither).  It determines which of these cases it is by
        examining its description string; then it either finds a partner for it
        from `unmatched`, or adds it to `unmatched`.

        Returns None if it cannot find a match; otherwise, it returns a tuple
        containing:
        - identifier
        - is exon?  (True/False)
        - did we find a match?  (True/False)
        - exon2 sequence
        - exon3 sequence
        """
        # The description field is expected to hold the sample name.
        samp: str = sequence_record.description
        is_exon: bool = False
        matched: bool = False
        exon2: str = ""
        exon3: str = ""
        identifier: str = samp

        # Check if the sequence is an exon2 or exon3. If so, try to match it
        # with an existing other exon.
        for exon, other_exon in EXON_AND_OTHER_EXON:
            if exon in samp.lower():
                is_exon = True
                identifier = samp.split("_")[0]
                for other_desc, other_sr in unmatched[other_exon].items():
                    if identifier.lower() in other_desc.lower():
                        matched = True
                        if exon == "exon2":
                            exon2 = str(sequence_record.seq)
                            exon3 = str(other_sr.seq)
                        else:
                            exon2 = str(other_sr.seq)
                            exon3 = str(sequence_record.seq)

                        unmatched[other_exon].pop(other_desc)
                        break
                # If we can't match the exon, put the entry in the list we
                # weren't looking in.
                if not matched:
                    unmatched[exon][samp] = sequence_record

        return (
            identifier,
            is_exon,
            matched,
            exon2,
            exon3,
        )

    def pair_exons(
        self,
        sequence_records: Iterable[Bio.SeqIO.SeqRecord],
    ) -> tuple[list[HLASequence], dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]]]:
        """
        Pair exons in the given input sequences.

        The section of HLA we sequence looks like
        exon2 - intron - exon3
        and is typically sequenced in two parts, one covering exon2 and exon3
        (the intron is not used in our testing).  We iterate through the
        sequences and attempt to match them up.
        """
        matched_sequences: list[HLASequence] = []
        unmatched: dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]] = {
            "exon2": {},
            "exon3": {},
        }

        for sr in sequence_records:
            # Skip over any sequences that aren't the right length or contain
            # bad bases.
            try:
                self.check_length(str(sr.seq), sr.description)
                self.check_bases(str(sr.seq))
            except ValueError:
                continue

            is_exon: bool = False
            matched: bool = False
            exon2: str = ""
            intron: str = ""
            exon3: str = ""
            identifier: str = ""

            identifier, is_exon, matched, exon2, exon3 = self.pair_exons_helper(
                sr,
                unmatched,
            )

            # If it was an exon2 or 3 but didn't have a pair, keep going.
            if is_exon and not matched:
                continue

            if is_exon:
                exon2_bin = self.pad_short(
                    self.nuc2bin(exon2), "exon2", hla_std=self.hla_stds[0]
                )
                exon3_bin = self.pad_short(
                    self.nuc2bin(exon3), "exon3", hla_std=self.hla_stds[0]
                )
                exon2 = self.bin2nuc(exon2_bin)
                exon3 = self.bin2nuc(exon3_bin)
                matched_sequences.append(
                    HLASequence(
                        two=exon2,
                        intron="",
                        three=exon3,
                        sequence=np.concatenate((exon2_bin, exon3_bin)),
                        name=identifier,
                        num_sequences_used=2,
                    )
                )
            else:
                seq = self.pad_short(
                    self.nuc2bin(sr.seq),  # type: ignore
                    None,
                    hla_std=self.hla_stds[0],
                )
                exon2 = self.bin2nuc(seq[: EasyHLA.EXON2_LENGTH])
                intron = self.bin2nuc(seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON3_LENGTH])
                exon3 = self.bin2nuc(seq[-EasyHLA.EXON3_LENGTH :])
                matched_sequences.append(
                    HLASequence(
                        two=exon2,
                        intron=intron,
                        three=exon3,
                        sequence=np.concatenate(
                            (seq[: EasyHLA.EXON2_LENGTH], seq[-EasyHLA.EXON3_LENGTH :])
                        ),
                        name=identifier,
                        num_sequences_used=1,
                    )
                )
        return matched_sequences, unmatched

    def get_mismatches(
        self,
        standard_string: str,
        seq: np.ndarray,
    ) -> list[HLAMismatch]:
        """
        Report mismatched bases and their location versus a standard.

        The standard should look like "1-4-9-14-2-...", where each position
        is represented as a 4-bit integer and positions are separated with
        hyphens.

        The output looks like "$LOC:$SEQ_BASE->$STANDARD_BASE", if multiple
        mismatches are present, they will be delimited with `;`'s.

        :param standard_string: the standard to compare the sequence to
        :type combined_standards: str
        :param seq: The sequence being interpreted.
        :type seq: np.ndarray
        :return: A list of HLAMismatches, sorted by their indices.
        :rtype: list[HLAMismatch]
        """
        correct_bases_at_pos: dict[int, list[int]] = {}

        std_bin_seq = np.array([int(nuc) for nuc in standard_string.split("-")])
        for idx in np.flatnonzero(std_bin_seq ^ seq):
            if idx not in correct_bases_at_pos:
                correct_bases_at_pos[idx] = []
            if std_bin_seq[idx] not in correct_bases_at_pos[idx]:
                correct_bases_at_pos[idx].append(std_bin_seq[idx])

        mislist: list[HLAMismatch] = []

        for index, correct_bases_bin in correct_bases_at_pos.items():
            if self.locus == "A" and index > 270:
                dex = index + 242
            else:
                dex = index + 1

            base = EasyHLA.BIN2NUC[seq[index]]
            correct_bases_str = [
                EasyHLA.BIN2NUC[correct_base_bin]
                for correct_base_bin in correct_bases_bin
            ]
            mislist.append(HLAMismatch(dex, base, correct_bases_str))

        mislist.sort(key=attrgetter("index"))
        return mislist

    class NoMatchingStandards(Exception):
        pass

    def interpret(
        self,
        hla_sequence: HLASequence,
        threshold: Optional[int] = None,
    ) -> HLAInterpretation:
        """
        Interpret sequence. The main function.

        :param hla_sequence: the sequence to perform interpretation on
        :type hla_sequence: HLASequence
        :param threshold: _description_, defaults to None
        :type threshold: Optional[int], optional
        :return: _description_
        :rtype: HLAInterpretation
        """
        seq: pnd.NpNDArray = hla_sequence.sequence

        matching_stds = self.get_matching_stds(seq, self.hla_stds)
        if len(matching_stds) == 0:
            raise EasyHLA.NoMatchingStandards()

        # Now, combine all the stds (pick up that can citizen!)
        # DR 2023-02-24: To whomever made this comment, great shoutout!
        all_combos: dict[HLACombinedStandard, int] = self.combine_standards(
            matching_stds,
            seq,
            mismatch_threshold=threshold,
        )

        return HLAInterpretation(
            hla_sequence=hla_sequence,
            matches={
                combined_std: HLAMatchDetails(
                    mismatch_count=mismatch_count,
                    mismatches=self.get_mismatches(combined_std.standard, seq),
                )
                for combined_std, mismatch_count in all_combos.items()
            },
        )

        # lowest_mismatch_count: int = min(all_combos.values())
        # best_matches: set[HLACombinedStandard] = {
        #     combined_std
        #     for combined_std, mismatch_count in all_combos.items()
        #     if mismatch_count == lowest_mismatch_count
        # }

        # # FIXME: currently we include all mismatches from all possible best
        # # matches; we should perhaps pick a "very best" of these matches and
        # # only include those mismatches.
        # best_match_mismatches: list[str] = []
        # for best_match in best_matches:
        #     best_match_mismatches.extend([str(x) for x in all_mismatches[best_match]])

        # name: str = hla_sequence.name
        # alleles = self.get_all_allele_pairs(best_matches)
        # ambig = alleles.is_ambiguous()
        # homozygous = alleles.is_homozygous()
        # alleles_all_str = alleles.stringify()
        # clean_allele_str = alleles.best_common_allele_pair_str(self.hla_freqs)

        # row = HLAResultRow(
        #     samp=name,
        #     clean_allele_str=clean_allele_str,
        #     alleles_all_str=alleles_all_str,
        #     ambig=int(ambig),
        #     homozygous=int(homozygous),
        #     mismatch_count=lowest_mismatch_count,
        #     mismatches=";".join(best_match_mismatches),
        #     exon2=hla_sequence.two.upper(),
        #     intron=hla_sequence.intron.upper(),
        #     exon3=hla_sequence.three.upper(),
        # )
        # return HLAResult(
        #     result_row=row,
        #     all_mismatches=all_mismatches,
        #     num_seqs=hla_sequence.num_sequences_used,
        # )

    def print(
        self,
        message: Any,
        log_level: int = logging.INFO,
        to_stdout: bool = False,
    ) -> None:
        """
        Output messages to logger, optionally prints to STDOUT.

        :param message: ...
        :type message: Any
        :param log_level: ..., defaults to logging.INFO
        :type log_level: int, optional
        :param to_stdout: Whether to print to STDOUT or not, defaults to False
        :type to_stdout: bool
        """
        self.log.log(level=log_level, msg=message)
        if to_stdout:
            print(message)

    def report_unmatched_sequences(
        self,
        unmatched: dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]],
        to_stdout: bool = False,
    ) -> None:
        """
        Report exon sequences that did not have a matching exon.

        :param unmatched: unmatched exon sequences, grouped by which exon they represent
        :type unmatched: dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]]
        :param to_stdout: ..., defaults to None
        :type to_stdout: Optional[bool], optional
        """
        for exon, other_exon in EXON_AND_OTHER_EXON:
            for entry in unmatched[exon]:
                self.print(
                    f"No matching {other_exon} for {entry.description}",
                    to_stdout=to_stdout,
                )

    def run(
        self,
        filename: str,
        output_filename: str,
        mismatches_filename: str,
        threshold: Optional[int] = None,
        to_stdout: bool = False,
    ):
        if threshold and threshold < 0:
            raise RuntimeError("Threshold must be >=0 or None!")

        rows: list[str] = []
        mismatch_rows: list[str] = []
        npats: int = 0
        nseqs: int = 0
        time_start: datetime = datetime.now()
        csv_header: str = (
            "ENUM,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,"
            "MISMATCHES,EXON2,INTRON,EXON3"
        )

        start_message: str = (
            f"Run commencing {time_start.strftime(DATE_FORMAT)}. "
            f"Allele definitions last updated {self.last_modified_time}."
        )

        self.print(start_message, to_stdout=to_stdout)
        self.print(csv_header, to_stdout=to_stdout)

        matched_sequences: list[HLASequence]
        unmatched: dict[EXON_NAME, dict[str, Bio.SeqIO.SeqRecord]]

        with open(filename, "r", encoding="utf-8") as f:
            matched_sequences, unmatched = self.pair_exons(Bio.SeqIO.parse(f, "fasta"))

        for hla_sequence in matched_sequences:
            try:
                result: HLAInterpretation = self.interpret(
                    hla_sequence,
                    threshold,
                )
            except EasyHLA.NoMatchingStandards:
                self.print(
                    f"Sequence {hla_sequence.name} did not match any known alleles.",
                    log_level=logging.WARN,
                    to_stdout=to_stdout,
                )
                self.print(
                    "Please check the locus and the orientation.",
                    log_level=logging.WARN,
                    to_stdout=to_stdout,
                )
                continue

            if result.lowest_mismatch_count() > threshold:
                self.print(
                    "No matches found below specified threshold. "
                    "Please check the locus, orientation, and/or increase "
                    "the tolerated number of mismatches.",
                    log_level=logging.WARN,
                    to_stdout=to_stdout,
                )

            row_str: str = result.summary_row.get_result_as_str()
            rows.append(row_str)
            self.print(row_str, to_stdout=to_stdout)

            mismatch_rows.extend(result.mismatch_rows())

            npats += 1
            nseqs += hla_sequence.num_seqs

        self.report_unmatched_sequences(unmatched, to_stdout=to_stdout)

        counts_str: str = f"{npats} patients, {nseqs} sequences processed."
        self.print(counts_str, to_stdout=to_stdout)
        self.log.info(counts_str)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"{start_message}\n")
            f.write(f"{csv_header}\n")
            for _r in rows:
                f.write(_r + "\n")

        with open(mismatches_filename, "w", encoding="utf-8") as f:
            f.write("ALLELE,MISMATCHES,EXON2,INTRON,EXON3\n")
            for mismatch_row in mismatch_rows:
                f.write(mismatch_row + "\n")
