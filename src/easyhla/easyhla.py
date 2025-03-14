# ruff: noqa: C901
# TODO: Remove this noqa line and refactor/reduce code-complexity

import logging
import os
import re
from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from io import TextIOBase
from typing import Any, Dict, Final, List, Literal, Optional, Tuple

import Bio.SeqIO
import numpy as np
import pydantic_numpy.typing as pnd

from .models import (
    Alleles,
    HLACombinedStandard,
    HLAResult,
    HLAResultRow,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)


class HLALocus(str, Enum):
    """Valid HLA genes."""

    A = "A"
    B = "B"
    C = "C"


DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"

HLA_LOCI = Literal["A", "B", "C"]

EXON_NAME = Literal["exon2", "exon3"]


class EasyHLA:
    HLA_A_LENGTH: Final[int] = 787
    MIN_HLA_BC_LENGTH: Final[int] = 787
    MAX_HLA_BC_LENGTH: Final[int] = 796
    EXON2_LENGTH: Final[int] = 270
    EXON3_LENGTH: Final[int] = 276
    ALLELES_MAX_REPORTABLE_STRING: Final[int] = 3900

    ALLOWED_HLA_LOCI: Final[List[str]] = ["A", "B", "C"]

    # A lookup table of translations from ambiguous nucleotides to unambiguous
    # nucleotides.
    AMBIG: Final[Dict[str, List[str]]] = {
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
    PURENUC2BIN: Final[Dict[str, int]] = {nuc: 2**i for i, nuc in enumerate("ACGT")}

    # Nucleotides converted to their binary representation
    # LISTOFNUCS: List[str] = [
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
    NUC2BIN: Final[Dict[str, int]] = {
        k: sum([{nuc: 2**i for i, nuc in enumerate("ACGT")}[nuc] for nuc in v])
        for k, v in AMBIG.items()
    }
    BIN2NUC: Final[Dict[int, str]] = {v: k for k, v in NUC2BIN.items()}

    COLUMN_IDS: Final[Dict[str, int]] = {"A": 0, "B": 2, "C": 4}

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
            raise ValueError("Invalid HLA Type!")
        self.locus: HLA_LOCI = locus
        self.hla_stds: List[HLAStandard] = self.load_hla_stds(
            locus=self.locus,
            hla_standards=hla_standards,
        )
        self.hla_freqs: Dict[str, int] = self.load_hla_frequencies(
            locus=self.locus,
            hla_frequencies=hla_frequencies,
        )
        self.last_modified_time: datetime = last_modified_time
        if last_modified_time is None:
            self.last_modified_time = self.load_allele_definitions_last_modified_time()
        self.log = logger or logging.Logger(__name__, logging.ERROR)

    def load_hla_frequencies(
        self,
        locus: HLA_LOCI,
        hla_frequencies: Optional[TextIOBase] = None,
    ) -> Dict[str, int]:
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

        :param locus: ...
        :type locus: HLA_LOCI
        :return: Lookup table of HLA frequencies.
        :rtype: Dict[str, int]
        """
        hla_freqs: Dict[str, int] = {}

        freqs_io: TextIOBase = hla_frequencies
        default_freqs_used: bool = False
        try:
            if hla_frequencies is None:
                freqs_io = open(
                    os.path.join(os.path.dirname(__file__), "hla_frequencies.csv"),
                    "r",
                    encoding="utf-8",
                )
                default_freqs_used = True

            for line in freqs_io.readlines():
                column_id = EasyHLA.COLUMN_IDS[locus]
                line_array = line.strip().split(",")[column_id : column_id + 2]
                elements = ",".join([f"{a[:2]}|{a[-2:]}" for a in line_array])
                if hla_freqs.get(elements, None) is None:
                    hla_freqs[elements] = 0
                hla_freqs[elements] += 1
        finally:
            if default_freqs_used:
                freqs_io.close()

        return hla_freqs

    # TODO: Convert this to a dictionary instead of a object that looks like:
    # [ [allele_name, [1,2,3,4,5]], [allele_name2, [2,5,2,5,4]] ]
    # TODO: allow changes to the source files; perhaps allow TextIO there as well
    # for, e.g., test purposes
    def load_hla_stds(
        self,
        locus: HLA_LOCI,
        hla_standards: Optional[TextIOBase],
    ) -> List[HLAStandard]:
        """
        Load HLA Standards from reference file.

        :param locus: ...
        :type locus: HLA_LOCI
        :return: List of known HLA standards
        :rtype: List[HLAStandard]
        """
        hla_stds: List[HLAStandard] = []

        standards_io: TextIOBase = hla_standards
        default_standards_used: bool = False
        try:
            if hla_standards is None:
                standards_io = open(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"hla_{locus.lower()}_std_reduced.csv",
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
        filename = os.path.join(os.path.dirname(__file__), "hla_nuc.fasta.mtime")
        with open(filename, "r", encoding="utf-8") as f:
            last_mod_date = "".join(f.readlines()).strip()
        return datetime.strptime(last_mod_date, DATE_FORMAT)

    def check_length(self, locus: HLA_LOCI, seq: str, name: str) -> None:
        """
        Validates the length of a sequence. This asserts a sequence either
        exactly a certain size, or is within an allowed range.

        See the following class values:
         - EasyHLA.HLA_A_LENGTH
         - EasyHLA.EXON2_LENGTH
         - EasyHLA.EXON3_LENGTH
         - EasyHLA.MAX_HLA_BC_LENGTH
         - EasyHLA.MIN_HLA_BC_LENGTH

        :param locus: HLA gene
        :type locus: HLA_LOCI
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
            if locus.upper() == "A":
                error_condition = len(seq) >= EasyHLA.HLA_A_LENGTH
            elif "exon2" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON2_LENGTH
            elif "exon3" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON3_LENGTH
            else:
                error_condition = len(seq) >= EasyHLA.MAX_HLA_BC_LENGTH
        elif locus.upper() == "A":
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
                f"Sequence {name} is the wrong length ({len(seq)} bp). Check the locus {locus}"
            )

    def print(
        self,
        message: Any,
        log_level: int = logging.INFO,
        to_stdout: Optional[bool] = None,
    ) -> None:
        """
        Output messages to logger, optionally prints to STDOUT.

        :param message: ...
        :type message: Any
        :param log_level: ..., defaults to logging.INFO
        :type log_level: int, optional
        :param to_stdout: Whether to print to STDOUT or not, defaults to None
        :type to_stdout: Optional[bool]
        """
        self.log.log(level=log_level, msg=message)
        if to_stdout:
            print(message)

    def check_bases(self, seq: str, name: str) -> None:
        """
        Check a string sequence for invalid characters.

        If an invalid character is detected it will raise a ValueError.

        :param seq: ...
        :type seq: str
        :param name: Name of sequence. This will commonly be the ID/descriptor
        in the fasta file.
        :type name: str
        :raises ValueError: Raised if a sequence contains letters we don't
        expect
        :return: True if our sequence only contains valid characters.
        :rtype: bool
        """
        if not re.match(r"^[ATGCRYKMSWNBDHV]+$", seq):
            raise ValueError(f"Sequence {name} has invalid characters")

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

    def calc_padding(self, std: np.ndarray, seq: np.ndarray) -> Tuple[int, int]:
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
        :rtype: Tuple[int, int]
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
        self, seq: np.ndarray, hla_stds: List[HLAStandard]
    ) -> List[HLAStandardMatch]:
        # Returns [ ["std_name", [1,2,3,4], num_mismatches], ["std_name2", [2,3,4,5], num_mismatches2]]
        matching_stds: List[HLAStandardMatch] = []
        for std in hla_stds:
            mismatches = self.std_match(std.sequence, seq)
            if mismatches < 5:
                matching_stds.append(
                    HLAStandardMatch(
                        allele=std.allele, sequence=std.sequence, mismatch=mismatches
                    )
                )

        return matching_stds

    def combine_stds(
        self,
        matching_stds: list[HLAStandardMatch],
        seq: list[int],
    ) -> dict[int, list[HLACombinedStandard]]:
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
        """
        combos: dict[int, dict[str, list[list[str]]]] = {}

        for std_ai, std_a in enumerate(matching_stds):
            for std_bi, std_b in enumerate(matching_stds):
                if std_ai < std_bi:
                    break

                # "Mush" the two standards together to produce something
                # that looks like what you get when you sequence HLA.
                std = std_b.sequence | std_a.sequence
                seq_mask = np.full_like(std, fill_value=15)
                mismatches: int = np.count_nonzero((std ^ seq) & seq_mask != 0)

                # This looks like 1-4-9-16-2-... where each number comes from
                # the binary representation of the base, i.e. from NUC2BIN.
                combined_std_name = "-".join([str(s) for s in std])
                if mismatches not in combos:
                    combos[mismatches] = {}
                if combined_std_name not in combos[mismatches]:
                    combos[mismatches][combined_std_name] = []
                stds = [std_a.allele, std_b.allele]
                stds.sort()
                combos[mismatches][combined_std_name].append(stds)

        result: dict[int, list[HLACombinedStandard]] = {}
        for mismatch, matching_combos_dict in combos.items():
            cur_combo: list[HLACombinedStandard] = []
            for std, allele_list in matching_combos_dict.items():
                cur_combo.append(
                    HLACombinedStandard(
                        standard=std, discrete_allele_names=allele_list
                    )
                )
            result[mismatch] = cur_combo

        return result

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
            # The description field is expected to hold the sample name.
            samp: str = sr.description

            is_exon: bool = False
            matched = False
            exon2 = ""
            intron = ""
            exon3 = ""

            # Check if the sequence is an exon2 or exon3. If so, try to match it
            # with an existing other exon.
            exon_and_other_exon: list[tuple[EXON_NAME, EXON_NAME]] = [
                ("exon2", "exon3"),
                ("exon3", "exon2"),
            ]
            for exon, other_exon in exon_and_other_exon:
                if exon in samp.lower():
                    is_exon = True
                    identifier = samp.split("_")[0]
                    for other_desc, other_sr in unmatched[other_exon].items():
                        if identifier.lower() in other_desc.lower():
                            matched = True
                            intron = ""
                            if exon == "exon2":
                                exon2 = str(sr.seq)
                                exon3 = str(other_sr.seq)
                            else:
                                exon2 = str(other_sr.seq)
                                exon3 = str(sr.seq)

                            unmatched[other_exon].pop(other_desc)
                            samp = identifier
                            break
                    # If we can't match the exon, put the entry in the list we
                    # weren't looking in.
                    if not matched:
                        unmatched[exon][samp] = sr

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
                        three=exon3,
                        seq=np.concatenate((exon2_bin, exon3_bin)),
                        name=samp,
                        num_sequences_used=2,
                    )
                )
            else:
                seq = self.pad_short(
                    self.nuc2bin(entry.seq),  # type: ignore
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
                        seq=np.concatenate(
                            (seq[: EasyHLA.EXON2_LENGTH], seq[-EasyHLA.EXON3_LENGTH :])
                        ),
                        name=samp,
                        num_sequences_used=1,
                    )
                )
        return matched_sequences, unmatched

    # FIXME move the length and base check to the calling function
    def interpret(
        self,
        hla_sequence: HLASequence,
        threshold: Optional[int] = None,
        to_stdout: Optional[bool] = None,
    ) -> Optional[HLAResult]:
        """
        Interpret sequence. The main function.

        :param hla_sequence: the sequence to perform interpretation on
        :type hla_sequence: HLASequence
        :param threshold: _description_, defaults to None
        :type threshold: Optional[int], optional
        :param to_stdout: _description_, defaults to None
        :type to_stdout: Optional[bool], optional
        :return: _description_
        :rtype: Optional[HLAResult]
        """
        # try:
        #     self.check_length(locus, str(entry.seq), samp)
        #     self.check_bases(str(entry.seq), samp)
        # except ValueError:
        #     return None

        seq: pnd.NpNDArray = hla_sequence.sequence
        name: str = hla_sequence.name

        matching_stds = self.get_matching_stds(seq, self.hla_stds)
        if len(matching_stds) == 0:
            self.print(
                f"Sequence {name} did not match any known alleles.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )
            self.print(
                "Please check the locus and the orientation.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )
            return None

        # Now, combine all the stds (pick up that can citizen!)
        # DR 2023-02-24: To whomever made this comment, great shoutout!
        all_combos = self.combine_stds(matching_stds, seq)

        if threshold is not None:
            self.report_mismatches(
                all_combos=all_combos,
                seq=seq,
                threshold=threshold,
                exon2=hla_sequence.two,
                intron=hla_sequence.intron,
                exon3=hla_sequence.three,
                to_stdout=to_stdout,
            )

        best_matches: List[HLACombinedStandard] = min(all_combos.items())[1]
        mismatch_count: int = min(all_combos.items())[0]

        # Clean the alleles

        mismatches = self.get_mismatches(self.locus, best_matches=best_matches, seq=seq)

        alleles = self.get_all_allele_pairs(best_matches=best_matches)
        ambig = alleles.is_ambiguous()
        homozygous = alleles.is_homozygous()
        alleles_all_str = alleles.stringify()
        clean_allele_str = alleles.best_common_allele_pair_str(self.hla_freqs)

        row = HLAResultRow(
            samp=name,
            clean_allele_str=clean_allele_str,
            alleles_all_str=alleles_all_str,
            ambig=int(ambig),
            homozygous=int(homozygous),
            mismatch_count=mismatch_count,
            mismatches=f"{mismatches}",
            exon2=hla_sequence.two.upper(),
            intron=hla_sequence.intron.upper(),
            exon3=hla_sequence.three.upper(),
        )
        return HLAResult(
            result=row,
            num_seqs=hla_sequence.num_sequences_used,
        )

    def report_unmatched_sequences(
        self,
        unmatched: List[List[Bio.SeqIO.SeqRecord]],
        to_stdout: Optional[bool] = None,
    ) -> None:
        """
        Report exon sequences that did not have a matching exon.

        :param unmatched: ...
        :type unmatched: List[List[Bio.SeqIO.SeqRecord]]
        :param to_stdout: ..., defaults to None
        :type to_stdout: Optional[bool], optional
        """
        for exon in [2, 3]:
            for entry in unmatched[exon % 2]:
                self.print(
                    f"No matching exon{3 - exon % 2} for {entry.description}",
                    to_stdout=to_stdout,
                )

    def run(
        self,
        locus: HLA_LOCI,
        filename: str,
        output_filename: str,
        threshold: Optional[int] = None,
        to_stdout: Optional[bool] = None,
    ):
        if threshold and threshold < 0:
            raise RuntimeError("Threshold must be >=0 or None!")

        rows = []
        npats = 0
        nseqs = 0
        time_start = datetime.now()
        unmatched: List[List[Bio.SeqIO.SeqRecord]] = [[], []]
        self.print(
            f"Run commencing {time_start.strftime(DATE_FORMAT)}. Allele definitions last updated {self.load_allele_definitions_last_modified_time().strftime(DATE_FORMAT)}.",
            to_stdout=to_stdout,
        )
        self.print(
            "ENUM,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3",
            to_stdout=to_stdout,
        )
        with open(filename, "r", encoding="utf-8") as f:
            fasta = Bio.SeqIO.parse(f, "fasta")
            for i, entry in enumerate(fasta):
                result = self.interpret(
                    locus,
                    entry,
                    unmatched=unmatched,
                    threshold=threshold,
                )
                if not result:
                    continue
                else:
                    rows.append(result.result.get_result_as_str())
                    self.print(result.result.get_result_as_str(), to_stdout=to_stdout)
                    npats += 1
                    nseqs += result.num_seqs

        self.report_unmatched_sequences(unmatched, to_stdout=to_stdout)
        self.print(
            f"{npats} patients, {nseqs} sequences processed.", to_stdout=to_stdout
        )

        self.log.info("% patients, % sequences processed.", npats, nseqs)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(
                f"Run commencing {time_start.strftime(DATE_FORMAT)}. Allele definitions last updated {self.load_allele_definitions_last_modified_time().strftime(DATE_FORMAT)}.\n"
            )
            f.write(
                "ENUM,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3\n"
            )
            for _r in rows:
                f.write(_r + "\n")

    def get_all_allele_pairs(self, best_matches: list[HLACombinedStandard]) -> AllelePairs:
        """
        Get all allele pairs that best match our sequence.

        :param best_matches: ...
        :type best_matches: List[HLACombinedStandard]
        :return: ...
        :rtype: Alleles
        """
        alleles_all: list[tuple[str, str]] = []
        for combined_std in best_matches:
            for _allele in combined_std.discrete_allele_names:
                alleles_all.append((_allele[0], _allele[1]))

        alleles_all.sort()

        return AllelePairs(allele_pairs=alleles_all)

    @staticmethod
    def sort_allele_tuple(
        allele_pair: str,
        frequency: int,
    ) -> tuple[int, int, int, int, int]:
        """
        Produce a tuple from an allele pair and frequency to allow sorting.
        """
        first_allele: str
        second_allele: str
        first_allele, second_allele = allele_pair.split(",")

        first_group: int
        first_protein: int
        first_group, first_protein = (int(x) for x in first_allele.split("|"))

        second_group: int
        second_protein: int
        second_group, second_protein = (int(x) for x in second_allele.split("|"))
        return (
            frequency,
            first_group,
            first_protein,
            second_group,
            second_protein,
        )

    def get_mismatches(
        self,
        best_matches: list[HLACombinedStandard],
        seq: np.ndarray,
    ) -> str:
        """
        Report mismatched bases and their location versus a standard reference.

        The output looks like "$LOC:$SEQ_BASE->$STANDARD_BASE", if multiple
        mismatches are present, they will be delimited with `;`'s.

        :param best_matches: List of the "best matched" standards to the sequence.
        :type best_matches: list[HLACombinedStandard]
        :param seq: The sequence being interpreted.
        :type seq: np.ndarray
        :return: A string-concatenated list of locations containing mismatches.
        :rtype: str
        """
        correct_bases_at_pos: dict[int, list[int]] = {}

        for combined_std in best_matches:
            std_bin_seq = np.array([int(nuc) for nuc in combined_std.standard.split("-")])
            for idx in np.flatnonzero(std_bin_seq ^ seq):
                if idx not in correct_bases_at_pos:
                    correct_bases_at_pos[idx] = []
                if std_bin_seq[idx] not in correct_bases_at_pos[idx]:
                    correct_bases_at_pos[idx].append(std_bin_seq[idx])

        mislist: list[str] = []

        for index, correct_bases in correct_bases_at_pos.items():
            if self.locus == "A" and index > 270:
                dex = index + 242
            else:
                dex = index + 1

            base = EasyHLA.BIN2NUC[seq[index]]
            _correct_bases = "/".join(
                [EasyHLA.BIN2NUC[correct_bin] for correct_bin in correct_bases]
            )
            mislist.append(f"{dex}:{base}->{_correct_bases}")

        mislist.sort(key=lambda item: item.split(":")[0])
        mismatches = ";".join(mislist)

        return mismatches

    def report_mismatches(
        self,
        all_combos: Dict[int, List[HLACombinedStandard]],
        seq: np.ndarray,
        threshold: int,
        exon2: str,
        intron: str,
        exon3: str,
        to_stdout: Optional[bool] = None,
    ) -> None:
        """
        Report mismatches to log/stdout (if applicable).

        :param all_combos: All possible combos
        :type all_combos: Dict[int, List[HLACombinedStandard]]
        :param seq: Sequence currently being interpretted.
        :type seq: np.ndarray
        :param threshold: Maximum allowed mismatches in a sequence compared to a standard, must be non-negative
        :type threshold: int
        :param exon2: exon2 of the sequence in question
        :type exon2: str
        :param intron: intron of the sequence in question
        :type intron: str
        :param exon3: exon3 of the sequence in question
        :type exon3: str
        :param to_stdout: Print to STDOUT if true, defaults to None
        :type to_stdout: Optional[bool], optional
        :raises RuntimeError: Raised if threshold is < 0
        """
        if threshold < 0:
            raise RuntimeError("threshold must be 0 or greater")

        any_matches_found: bool = False
        for i, combos in sorted(all_combos.items()):
            if i > threshold:
                break

            # We can reuse get_mismatches here, as instead of the "best" match, we have "a match"
            mismatches = self.get_mismatches(
                locus=locus, best_matches=combos, seq=seq
            )
            self.print(
                f"{mismatches},{exon2},{intron},{exon3}",
                log_level=logging.INFO,
                to_stdout=to_stdout,
            )

        if not any_matches_found:
            self.print(
                "No matches found below specified threshold. "
                "Please check the locus, orientation, and/or increase "
                "number of mismatches.",
                log_level=logging.WARN,
                to_stdout=to_stdout,
            )
