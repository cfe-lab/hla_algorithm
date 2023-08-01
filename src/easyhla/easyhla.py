import logging
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple

import Bio.SeqIO
import numpy as np
import typer

from .models import (
    Alleles,
    Exon,
    HLACombinedStandardResult,
    HLAResult,
    HLAResultRow,
    HLAStandard,
    HLAStandardMatch,
)


class HLAType(str, Enum):
    """Valid HLA subtypes."""

    A = "A"
    B = "B"
    C = "C"


DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"

HLA_TYPES = Literal["A", "B", "C"]


class EasyHLA:
    HLA_A_LENGTH: Final[int] = 787
    MIN_HLA_BC_LENGTH: Final[int] = 787
    MAX_HLA_BC_LENGTH: Final[int] = 796
    EXON2_LENGTH: Final[int] = 270
    EXON3_LENGTH: Final[int] = 276
    ALLELES_MAX_REPORTABLE_STRING: Final[int] = 3900

    ALLOWED_HLA_TYPES: Final[List[str]] = ["A", "B", "C"]

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

    def __init__(self, letter: HLA_TYPES, logger: Optional[logging.Logger] = None):
        """
        Initialize an EasyHLA class.

        :param letter: HLA subtype that this object will be performing
        interpretation against.
        :type letter: "A", "B", or "C"
        :param logger: Python logger object, defaults to None
        :type logger: Optional[logging.Logger], optional
        :raises ValueError: Raised if letter != "A"/"B"/"C"
        """
        if letter not in ["A", "B", "C"]:
            raise ValueError("Invalid HLA Type!")
        self.letter: HLA_TYPES = letter
        self.hla_stds: List[HLAStandard] = self.load_hla_stds(letter=self.letter)
        self.hla_freqs: Dict[str, int] = self.load_hla_frequencies(letter=self.letter)
        self.log = logger or logging.Logger(__name__, logging.ERROR)

    def check_length(self, letter: HLA_TYPES, seq: str, name: str) -> bool:
        """
        Validates the length of a sequence. This asserts a sequence either
        exactly a certain size, or is within an allowed range.

        See the following class values:
         - EasyHLA.HLA_A_LENGTH
         - EasyHLA.EXON2_LENGTH
         - EasyHLA.EXON3_LENGTH
         - EasyHLA.MAX_HLA_BC_LENGTH
         - EasyHLA.MIN_HLA_BC_LENGTH

        :param letter: HLA Subtype
        :type letter: HLA_TYPES
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
            if letter.upper() == "A":
                error_condition = len(seq) >= EasyHLA.HLA_A_LENGTH
            elif "exon2" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON2_LENGTH
            elif "exon3" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON3_LENGTH
            else:
                error_condition = len(seq) >= EasyHLA.MAX_HLA_BC_LENGTH
        elif letter.upper() == "A":
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
                f"Sequence {name} is the wrong length ({len(seq)} bp). Check the locus {letter}"
            )
        return True

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

    def check_bases(self, seq: str, name: str) -> bool:
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
        return True

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
        name: str,
        hla_std: HLAStandard,
    ) -> np.ndarray:
        # hla_stds expects [ ["label0", [1,2,3,4]], ["label1", [2,3,4,5]] ]
        std = None
        has_intron = False
        if "exon2" in name.lower():
            std = hla_std.sequence[: EasyHLA.EXON2_LENGTH]
        elif "exon3" in name.lower():
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
        matching_stds: List[HLAStandardMatch],
        seq: List[int],
        max_mismatch_threshold: Optional[int],
    ) -> Dict[int, List[HLACombinedStandardResult]]:
        length = len(matching_stds[0].sequence)

        default_min = 9999
        if max_mismatch_threshold is None:
            tmp_max_mismatch_threshold = 0
        else:
            tmp_max_mismatch_threshold = max_mismatch_threshold

        combos: Dict[int, Dict[str, List[List[str]]]] = {}

        # NOTE: This was a max() comparison in the original code
        computed_minimum_mismatches = max(default_min, tmp_max_mismatch_threshold)

        for std_ai, std_a in enumerate(matching_stds):
            # matching_stds = [ [name, sequence[], threshold] ]
            if std_a.mismatch > computed_minimum_mismatches:
                continue
            for std_bi, std_b in enumerate(matching_stds):
                if std_ai < std_bi:
                    break
                if std_b.mismatch > computed_minimum_mismatches:
                    continue

                mismatches = 0
                std = std_b.sequence | std_a.sequence
                seq_mask = np.full_like(std, fill_value=15)
                mismatches = np.count_nonzero((std ^ seq) & seq_mask != 0)

                if mismatches <= computed_minimum_mismatches:
                    combined_std_name = "-".join([str(s) for s in std])
                    if mismatches < computed_minimum_mismatches:
                        computed_minimum_mismatches = max(
                            mismatches, tmp_max_mismatch_threshold
                        )
                    if not mismatches in combos:
                        combos[mismatches] = {}
                    if not combined_std_name in combos[mismatches]:
                        combos[mismatches][combined_std_name] = []
                    stds = [std_a.allele, std_b.allele]
                    stds.sort()
                    combos[mismatches][combined_std_name].append(stds)

        result: Dict[int, List[HLACombinedStandardResult]] = {}
        for mismatch, standard in combos.items():
            cur_combo: List[HLACombinedStandardResult] = []
            for std, allele_list in standard.items():
                cur_combo.append(
                    HLACombinedStandardResult(
                        standard=std, discrete_allele_names=allele_list
                    )
                )
            result[mismatch] = cur_combo

        return result

    def load_hla_frequencies(self, letter: HLA_TYPES) -> Dict[str, int]:
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

        :param letter: ...
        :type letter: HLA_TYPES
        :return: Lookup table of HLA frequencies.
        :rtype: Dict[str, int]
        """
        hla_freqs: Dict[str, int] = {}
        filepath = os.path.join(os.path.dirname(__file__), "hla_frequencies.csv")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                column_id = EasyHLA.COLUMN_IDS[letter]
                l = line.strip().split(",")[column_id : column_id + 2]
                _l = ",".join([f"{a[:2]}|{a[-2:]}" for a in l])
                if hla_freqs.get(_l, None) is None:
                    hla_freqs[_l] = 0
                hla_freqs[_l] += 1
        return hla_freqs

    # TODO: Convert this to a dictionary instead of a object that looks like:
    # [ [allele_name, [1,2,3,4,5]], [allele_name2, [2,5,2,5,4]] ]
    def load_hla_stds(self, letter: HLA_TYPES) -> List[HLAStandard]:
        """
        Load HLA Standards from reference file.

        :param letter: ...
        :type letter: HLA_TYPES
        :return: List of known HLA standards
        :rtype: List[HLAStandard]
        """
        hla_stds: List[HLAStandard] = []

        filepath = os.path.join(
            os.path.dirname(__file__), f"hla_{letter.lower()}_std_reduced.csv"
        )

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                l = line.strip().split(",")
                seq = self.nuc2bin((l[1] + l[2]))
                hla_stds.append(HLAStandard(allele=l[0], sequence=seq))
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

    def interpret(
        self,
        letter: HLA_TYPES,
        entry: Bio.SeqIO.SeqRecord,
        unmatched: List[List[Bio.SeqIO.SeqRecord]],
        threshold: Optional[int] = None,
        to_stdout: Optional[bool] = None,
    ) -> Optional[HLAResult]:
        """
        Interpret sequence. The main function.

        :param letter: _description_
        :type letter: HLA_TYPES
        :param entry: _description_
        :type entry: Bio.SeqIO.SeqRecord
        :param unmatched: _description_
        :type unmatched: List[List[Bio.SeqIO.SeqRecord]]
        :param threshold: _description_, defaults to None
        :type threshold: Optional[int], optional
        :param to_stdout: _description_, defaults to None
        :type to_stdout: Optional[bool], optional
        :return: _description_
        :rtype: Optional[HLAResult]
        """
        samp = entry.description

        try:
            if not self.check_length(letter, str(entry.seq), samp):
                return None
            if not self.check_bases(str(entry.seq), samp):
                return None
        except ValueError as e:
            return None

        is_exon = False
        matched = False
        exon2 = ""
        intron = ""
        exon3 = ""

        # Check if the sequence is an exon2 or exon3. If so, try to match it with an
        # existing other exon.
        if "exon" in samp.lower():
            for exon in [2, 3]:
                if f"exon{exon}" in samp.lower():
                    is_exon = True
                    _samp = samp.split("_")[0]
                    for i, other in enumerate(unmatched[3 - exon]):
                        if _samp.lower() in other.description.lower():
                            matched = True
                            intron = ""
                            if exon == 2:
                                exon2 = str(entry.seq)
                                exon3 = str(other.seq)
                            else:
                                exon2 = str(other.seq)
                                exon3 = str(entry.seq)

                            unmatched[3 - exon].pop(i)
                            samp = _samp
                            break
                    # If we can't match the exon, put the entry in the list we weren't looking in
                    # Ex: exon2 looks in mismatches[1] ( [[],[] <- this bucket] )
                    # If it can't find its pair in mismatches[1], then it puts itself into
                    # mismatches[0]
                    if not matched:
                        unmatched[exon % 2].append(entry)

        # If it was an exon2 or 3 but didn't have a pair, keep going.
        if is_exon and not matched:
            return None

        if is_exon:
            exon2_bin = self.pad_short(
                self.nuc2bin(exon2), "exon2", hla_std=self.hla_stds[0]
            )
            exon3_bin = self.pad_short(
                self.nuc2bin(exon3), "exon3", hla_std=self.hla_stds[0]
            )
            exon2 = self.bin2nuc(exon2_bin)
            exon3 = self.bin2nuc(exon3_bin)
            seq_parts = Exon(two=exon2, three=exon3)
            seq = np.concatenate((exon2_bin, exon3_bin))
        else:
            seq = self.pad_short(
                self.nuc2bin(entry.seq), samp, hla_std=self.hla_stds[0]
            )
            exon2 = self.bin2nuc(seq[: EasyHLA.EXON2_LENGTH])
            intron = self.bin2nuc(seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON3_LENGTH])
            exon3 = self.bin2nuc(seq[-EasyHLA.EXON3_LENGTH :])
            seq_parts = Exon(two=exon2, intron=intron, three=exon3)
            seq = np.concatenate(
                (seq[: EasyHLA.EXON2_LENGTH], seq[-EasyHLA.EXON3_LENGTH :])
            )

        matching_stds = self.get_matching_stds(seq, self.hla_stds)
        if len(matching_stds) == 0:
            self.print(
                f"Sequence {samp} did not match any known alleles.",
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
        all_combos = self.combine_stds(matching_stds, seq, threshold)

        self.report_mismatches(
            letter=self.letter,
            all_combos=all_combos,
            seq=seq,
            threshold=threshold,
            sequence_components=seq_parts,
            to_stdout=to_stdout,
        )

        best_matches: List[HLACombinedStandardResult] = min(all_combos.items())[1]
        mismatch_count: int = min(all_combos.items())[0]

        # Clean the alleles

        mismatches = self.get_mismatches(
            self.letter, best_matches=best_matches, seq=seq
        )

        alleles = self.get_all_alleles(best_matches=best_matches)
        ambig = alleles.is_ambiguous()
        homozygous = alleles.is_homozygous()
        alleles_all_str = alleles.stringify()
        if alleles.is_ambiguous():
            alleles.alleles = self.filter_reportable_alleles(
                letter=self.letter, alleles=alleles
            )
        clean_allele_str = alleles.stringify_clean()

        # if this is an exon, then nseqs = 2
        nseqs = 1 + int(is_exon)

        row = HLAResultRow(
            samp=samp,
            clean_allele_str=clean_allele_str,
            alleles_all_str=alleles_all_str,
            ambig=int(ambig),
            homozygous=int(homozygous),
            mismatch_count=f"{mismatch_count}",
            mismatches=f"{mismatches}",
            exon2=exon2.upper(),
            intron=intron.upper(),
            exon3=exon3.upper(),
        )
        # print(row)
        return HLAResult(result=row, num_pats=1, num_seqs=nseqs)

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
        letter: HLA_TYPES,
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
                    letter,
                    entry,
                    unmatched=unmatched,
                    threshold=threshold,
                )
                if not result:
                    continue
                else:
                    rows.append(result.result.get_result_as_str())
                    self.print(result.result.get_result_as_str(), to_stdout=to_stdout)
                    npats += result.num_pats
                    nseqs += result.num_seqs

        self.report_unmatched_sequences(unmatched, to_stdout=to_stdout)
        self.print(
            f"{npats} patients, {nseqs} sequences processed.", to_stdout=to_stdout
        )

        self.log.info(f"% patients, % sequences processed.", npats, nseqs)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(
                f"Run commencing {time_start.strftime(DATE_FORMAT)}. Allele definitions last updated {self.load_allele_definitions_last_modified_time().strftime(DATE_FORMAT)}.\n"
            )
            f.write(
                "ENUM,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3\n"
            )
            for _r in rows:
                f.write(_r + "\n")

    def get_all_alleles(self, best_matches: List[HLACombinedStandardResult]) -> Alleles:
        """
        Get all alleles that best match our sequence.

        :param best_matches: ...
        :type best_matches: List[HLACombinedStandardResult]
        :return: ...
        :rtype: Alleles
        """
        alleles_all: List[Tuple[str, str]] = []
        # Lets say if we detect two of the same mixtures, its heterozygous
        for a in best_matches:
            for _allele in a.discrete_allele_names:
                alleles_all.append((_allele[0], _allele[1]))

        alleles_all.sort()

        return Alleles(alleles=alleles_all)

    def filter_reportable_alleles(
        self, letter: HLA_TYPES, alleles: Alleles
    ) -> List[Tuple[str, str]]:
        """
        In case we have an ambiguous set of alleles, remove ambiguous alleles
        using HLA Freq standards.

        :param letter: ...
        :type letter: HLA_TYPES
        :param alleles: ...
        :type alleles: Alleles
        :return: List of alleles filtered by HLA frequency.
        :rtype: List[Tuple[str,str]]
        """

        collection_ambig = alleles.get_ambiguous_collection()
        for k in collection_ambig:
            for freq in self.hla_freqs:
                if freq.startswith(k):
                    collection_ambig[k] = self.hla_freqs.get(freq, 0)

        def sort_allele(item: Tuple[str, int]):
            """
            Produce a tuple that the sort function will use to determine the maximum allele.
            """
            # Try to find the allele occuring the maximum number of times. If it's a tie,
            # just pick the alphabetically first one.
            # max_allele = collection_ambig.max do |a,b|
            #     if(a[2] != b[2]) #Go by frequency
            #         a[2] <=> b[2]
            #     elsif(b[0][0].to_i != a[0][0].to_i) #Then lowest first allele
            #         b[0][0].to_i <=> a[0][0].to_i
            #     elsif(b[0][1].to_i != a[0][1].to_i)
            #         b[0][1].to_i <=> a[0][1].to_i
            #     elsif(b[1][0].to_i != a[1][0].to_i) #Then lowest second allele
            #         b[1][0].to_i <=> a[1][0].to_i
            #     else
            #         b[1][1].to_i <=> a[1][1].to_i
            #     end
            # end
            allele_pair0 = item[0].split(",")[0]
            allele_pair1 = item[0].split(",")[1]
            return (
                item[1],
                int(allele_pair0.split("|")[0]),
                int(allele_pair0.split("|")[1]),
                int(allele_pair1.split("|")[0]),
                int(allele_pair1.split("|")[1]),
            )

        max_allele = sorted(collection_ambig.items(), key=sort_allele, reverse=True)

        a1 = max_allele[0][0].split(",")[0]
        a2 = max_allele[0][0].split(",")[1]
        _alleles = alleles.alleles
        for i, a in enumerate(_alleles.copy()):
            regex_str_a1 = f"^{letter}\\*({a1}):([^\\s])+"
            if not re.match(regex_str_a1, a[0]) and a in _alleles:
                _alleles.remove(a)
            regex_str_a2 = f"^{letter}\\*({a2}):([^\\s])+"
            if not re.match(regex_str_a2, a[1]) and a in _alleles:
                _alleles.remove(a)

        return _alleles

    def get_mismatches(
        self,
        letter: HLA_TYPES,
        best_matches: List[HLACombinedStandardResult],
        seq: np.ndarray,
    ) -> str:
        """
        Report mismatched bases and their location versus a standard reference.

        The output looks like "$LOC:$SEQ_BASE->$STANDARD_BASE", if multiple
        mismatches are present, this will be delimited with `;`'s.

        :param letter: ...
        :type letter: HLA_TYPES
        :param best_matches: List of the "best matched" standards to the sequence.
        :type best_matches: List[HLACombinedStandardResult]
        :param seq: The sequence being interpretted.
        :type seq: np.ndarray
        :return: A string-concatentated list of locations containing mismatches.
        :rtype: str
        """
        correct_bases_at_pos: Dict[int, List[int]] = {}

        for hla_csr in best_matches:
            _seq = np.array([int(nuc) for nuc in hla_csr.standard.split("-")])
            # TODO: replace with https://stackoverflow.com/questions/16094563/numpy-get-index-where-value-is-true
            for idx in np.flatnonzero(_seq ^ seq):
                if not idx in correct_bases_at_pos:
                    correct_bases_at_pos[idx] = []
                if not _seq[idx] in correct_bases_at_pos[idx]:
                    correct_bases_at_pos[idx].append(_seq[idx])

        mislist: List[str] = []

        for index, correct_bases in correct_bases_at_pos.items():
            if letter == "A" and index > 270:
                dex = index + 241
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
        letter: HLA_TYPES,
        all_combos: Dict[int, List[HLACombinedStandardResult]],
        seq: np.ndarray,
        threshold: Optional[int] = None,
        sequence_components: Optional[Exon] = None,
        to_stdout: Optional[bool] = None,
    ) -> None:
        """
        Report mismatches to log/stdout (if applicable).

        :param letter: ...
        :type letter: HLA_TYPES
        :param all_combos: All possible combos
        :type all_combos: Dict[int, List[HLACombinedStandardResult]]
        :param seq: Sequence currently being interpretted.
        :type seq: np.ndarray
        :param threshold: Maximum allowed mismatches in a sequence compared to a standard, must be non-negative or None, defaults to None
        :type threshold: Optional[int], optional
        :param sequence_components: Components of a sequence, ex: Exon2, Intron, Exon3, defaults to None
        :type sequence_components: Optional[Exon], optional
        :param to_stdout: Print to STDOUT if true, defaults to None
        :type to_stdout: Optional[bool], optional
        :raises RuntimeError: Raised if threshold is < 0
        """
        if threshold:
            if threshold < 0:
                raise RuntimeError("Threshold must be >=0 or None!")
            for i, combos in sorted(all_combos.items()):
                if i > threshold:
                    if i == 0:
                        self.print(
                            "No matches found below specified threshold. "
                            "Please check the locus, orientation, and/or increase "
                            "number of mismatches.",
                            log_level=logging.WARN,
                            to_stdout=to_stdout,
                        )
                    break
                # We can reuse get_mismatches here, as instead of the "best" match, we have "a match"
                mismatches = self.get_mismatches(
                    letter=letter, best_matches=combos, seq=seq
                )
                _seq_str = ""
                if sequence_components:
                    _seq_str = f"{sequence_components.two},{sequence_components.intron},{sequence_components.three}"
                self.print(
                    f"{mismatches},{_seq_str}",
                    log_level=logging.INFO,
                    to_stdout=to_stdout,
                )
