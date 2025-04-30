import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from io import TextIOBase
from operator import attrgetter
from typing import Final, Literal, Optional

import numpy as np
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord

from .models import (
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLAMismatch,
    HLAProteinPair,
    HLASequence,
    HLAStandard,
    HLAStandardMatch,
)
from .utils import (
    BIN2NUC,
    EXON2_LENGTH,
    EXON3_LENGTH,
    EXON_NAME,
    HLA_A_LENGTH,
    HLA_LOCUS,
    MAX_HLA_BC_LENGTH,
    MIN_HLA_BC_LENGTH,
    calc_padding,
    check_bases,
    check_length,
    count_strict_mismatches,
    nuc2bin,
)

EXON_AND_OTHER_EXON: list[tuple[EXON_NAME, EXON_NAME]] = [
    ("exon2", "exon3"),
    ("exon3", "exon2"),
]

DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"


class EasyHLA:
    # HLA_A_LENGTH: Final[int] = 787
    # MIN_HLA_BC_LENGTH: Final[int] = 787
    # MAX_HLA_BC_LENGTH: Final[int] = 796
    # EXON2_LENGTH: Final[int] = 270
    # EXON3_LENGTH: Final[int] = 276

    # For HLA-B interpretations, these alleles are the ones we use to determine
    # how close a sequence is to "B*57:01".
    B5701_ALLELES: Final[list[str]] = [
        "B*57:01:01G",
        "B*57:01:02",
        "B*57:01:03",
    ]

    COLUMN_IDS: Final[dict[str, int]] = {"A": 0, "B": 2, "C": 4}

    def __init__(
        self,
        locus: HLA_LOCUS,
        hla_standards: Optional[dict[str, HLAStandard]] = None,
        hla_frequencies: Optional[dict[HLAProteinPair, int]] = None,
        last_modified: Optional[datetime] = None,
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
        self.locus: HLA_LOCUS = locus

        self.hla_standards: dict[str, HLAStandard]
        if hla_standards is not None:
            self.hla_standards = hla_standards
        else:
            self.hla_standards = self.load_default_hla_standards()

        self.hla_frequencies: dict[HLAProteinPair, int]
        if hla_frequencies is not None:
            self.hla_frequencies = hla_frequencies
        else:
            self.hla_frequencies = self.load_default_hla_frequencies()

        self.last_modified: datetime
        if last_modified is not None:
            self.last_modified = last_modified
        else:
            self.last_modified = self.load_default_last_modified()

    @staticmethod
    def read_hla_standards(standards_io: TextIOBase) -> dict[str, HLAStandard]:
        """
        Read HLA standards from a specified file-like object.

        :return: Dictionary of known HLA standards keyed by their name
        :rtype: dict[str, HLAStandard]
        """
        hla_stds: dict[str, HLAStandard] = {}
        for line in standards_io.readlines():
            line_array = line.strip().split(",")
            allele: str = line_array[0]
            hla_stds[allele] = HLAStandard(
                allele=line_array[0],
                two=nuc2bin(line_array[1]),
                three=nuc2bin(line_array[2]),
            )
        return hla_stds

    def load_default_hla_standards(self) -> dict[str, HLAStandard]:
        """
        Load HLA Standards from reference file.

        :return: List of known HLA standards
        :rtype: list[HLAStandard]
        """
        standards_filename: str = os.path.join(
            os.path.dirname(__file__),
            "default_data",
            f"hla_{self.locus.lower()}_std_reduced.csv",
        )
        with open(standards_filename) as standards_file:
            return self.read_hla_standards(standards_file)

    @staticmethod
    def read_hla_frequencies(
        locus: HLA_LOCUS,
        frequencies_io: TextIOBase,
    ) -> dict[HLAProteinPair, int]:
        """
        Load HLA frequencies from a specified file-like object.

        This takes two columns AAAA,BBBB out of 6 (...FFFF), and then uses a
        subset of these two columns (AABB,CCDD) to use as the key, in this case
        "AA|BB,CC|DD", we then count the number of times this key appears in our
        columns.

        :return: Lookup table of HLA frequencies.
        :rtype: dict[HLAProteinPair, int]
        """
        hla_freqs: dict[HLAProteinPair, int] = {}
        for line in frequencies_io.readlines():
            column_id = EasyHLA.COLUMN_IDS[locus]
            line_array = line.strip().split(",")[column_id : column_id + 2]

            protein_pair: HLAProteinPair = HLAProteinPair(
                first_field_1=line_array[0][0:2],
                first_field_2=line_array[0][2:4],
                second_field_1=line_array[1][0:2],
                second_field_2=line_array[1][2:4],
            )
            if hla_freqs.get(protein_pair, None) is None:
                hla_freqs[protein_pair] = 0
            hla_freqs[protein_pair] += 1
        return hla_freqs

    def load_default_hla_frequencies(self) -> dict[HLAProteinPair, int]:
        """
        Load HLA frequencies from reference file.

        This takes two columns AAAA,BBBB out of 6 (...FFFF), and then uses a
        subset of these two columns (AABB,CCDD) to use as the key, in this case
        "AA|BB,CC|DD", we then count the number of times this key appears in our
        columns.

        :return: Lookup table of HLA frequencies.
        :rtype: dict[HLAProteinPair, int]
        """
        hla_freqs: dict[HLAProteinPair, int]
        default_frequencies_filename: str = os.path.join(
            os.path.dirname(__file__),
            "default_data",
            "hla_frequencies.csv",
        )
        with open(default_frequencies_filename, "r") as f:
            hla_freqs = self.read_hla_frequencies(self.locus, f)
        return hla_freqs

    @staticmethod
    def load_default_last_modified() -> datetime:
        """
        Load a datetime object describing when standard definitions were last updated.

        :return: Date and time representing when references were last updated.
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

    @staticmethod
    def pad_short(
        std_bin: Sequence[int],
        seq_bin: Sequence[int],
        exon: Optional[EXON_NAME],
    ) -> np.ndarray:
        left_pad: int
        right_pad: int
        exon2_std_bin: np.ndarray = np.array(std_bin[: EasyHLA.EXON2_LENGTH])
        exon3_std_bin: np.ndarray = np.array(std_bin[-EasyHLA.EXON3_LENGTH :])
        if exon == "exon2":
            left_pad, right_pad = calc_padding(
                exon2_std_bin,
                seq_bin,
            )
        elif exon == "exon3":
            left_pad, right_pad = calc_padding(
                exon3_std_bin,
                seq_bin,
            )
        else:  # i.e. this is a full sequence possibly with intron
            left_pad, _ = calc_padding(
                exon2_std_bin,
                seq_bin[: int(EasyHLA.EXON2_LENGTH / 2)],
            )
            _, right_pad = calc_padding(
                exon3_std_bin,
                seq_bin[-int(EasyHLA.EXON3_LENGTH / 2) :],
            )
        return np.concatenate(
            (
                nuc2bin("N" * left_pad),
                seq_bin,
                nuc2bin("N" * right_pad),
            )
        )

    @staticmethod
    def get_matching_standards(
        seq: Sequence[int],
        hla_stds: Iterable[HLAStandard],
        mismatch_threshold: int = 5,
    ) -> list[HLAStandardMatch]:
        # Returns [ ["std_name", [1,2,3,4], num_mismatches], ["std_name2", [2,3,4,5], num_mismatches2]]
        matching_stds: list[HLAStandardMatch] = []
        for std in hla_stds:
            mismatches = count_strict_mismatches(std.sequence, seq)
            if mismatches < mismatch_threshold:
                matching_stds.append(
                    HLAStandardMatch(
                        allele=std.allele,
                        two=std.two,
                        three=std.three,
                        mismatch=mismatches,
                    )
                )
        return matching_stds

    @staticmethod
    def combine_standards_helper(
        matching_stds: Sequence[HLAStandardMatch],
        seq: Sequence[int],
        mismatch_threshold: int = 0,
    ) -> dict[tuple[int, ...], tuple[int, list[tuple[str, str]]]]:
        """
        Helper to identify "good" combined standards for the specified sequence.

        Returns a mapping:
        binary sequence tuple -|-> (mismatch count, allele pair list)

        This mapping will contain "good" combined standards.  It will always
        contain the best-matching combined standard(s).  If mismatch_threshold
        is 0, then we only care about the best match; if mismatch_threshold is a
        positive integer, it will also contain any combined standards which have
        fewer mismatches than the threshold.

        The result may also contain other combined standards, which will be
        winnowed out by the calling function.
        """
        combos: dict[tuple[int, ...], tuple[int, list[tuple[str, str]]]] = {}

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
                std_bin = np.array(std_b.sequence) | np.array(std_a.sequence)
                seq_mask = np.full_like(std_bin, fill_value=15)
                # Note that seq is implicitly cast to a NumPy array:
                mismatches: int = np.count_nonzero((std_bin ^ seq) & seq_mask != 0)

                if mismatches > current_rejection_threshold:
                    continue

                # There could be more than one combined standard with the
                # same sequence, so keep track of all the possible combinations.
                combined_std_bin: tuple[int, ...] = tuple(int(s) for s in std_bin)
                if combined_std_bin not in combos:
                    combos[combined_std_bin] = (mismatches, [])
                combos[combined_std_bin][1].append(sorted((std_a.allele, std_b.allele)))

                if mismatches < current_rejection_threshold:
                    current_rejection_threshold = max(mismatches, mismatch_threshold)
        return combos

    @staticmethod
    def combine_standards(
        matching_stds: Sequence[HLAStandardMatch],
        seq: Sequence[int],
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
        threshold.  All of the HLACombinedStandards have their
        `possible_allele_pairs` value sorted.
        """
        if mismatch_threshold is None:
            # We only care about the best match, so we set this threshold
            # so that it never forces the rejection threshold to stay above
            # the current best match.
            mismatch_threshold = 0

        combos: dict[tuple[int, ...], tuple[int, list[tuple[str, str]]]] = (
            EasyHLA.combine_standards_helper(
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

        for combined_std_bin, mismatch_count_and_pair_list in combos.items():
            mismatch_count: int
            pair_list: list[tuple[str, str]]
            mismatch_count, pair_list = mismatch_count_and_pair_list

            if mismatch_count <= cutoff:
                combined_std: HLACombinedStandard = HLACombinedStandard(
                    standard_bin=combined_std_bin,
                    possible_allele_pairs=tuple(sorted(pair_list)),
                )
                result[combined_std] = mismatch_count

        return result

    @staticmethod
    def pair_exons_helper(
        sequence_record: SeqRecord,
        unmatched: dict[EXON_NAME, dict[str, Seq]],
    ) -> tuple[str, bool, bool, str, str]:
        """
        Helper that attempts to match the given sequence with a "partner" exon.

        `sequence_record` represents a sequence that may be an exon2 or exon3
        sequence (or neither).  It determines which of these cases it is by
        examining its `id` string; then it either finds a partner for it
        from `unmatched`, or adds it to `unmatched`.

        Returns None if it cannot find a match; otherwise, it returns a tuple
        containing:
        - identifier
        - is exon?  (True/False)
        - did we find a match?  (True/False)
        - exon2 sequence
        - exon3 sequence
        """
        # The `id`` field is expected to hold the sample name.
        samp: str = sequence_record.id
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
                for other_id, other_seq in unmatched[other_exon].items():
                    if identifier.lower() in other_id.lower():
                        matched = True
                        if exon == "exon2":
                            exon2 = str(sequence_record.seq)
                            exon3 = str(other_seq)
                        else:
                            exon2 = str(other_seq)
                            exon3 = str(sequence_record.seq)

                        unmatched[other_exon].pop(other_id)
                        break
                # If we can't match the exon, put the entry in the list we
                # weren't looking in.
                if not matched:
                    unmatched[exon][samp] = sequence_record.seq

        return (
            identifier,
            is_exon,
            matched,
            exon2,
            exon3,
        )

    def pair_exons(
        self,
        sequence_records: Iterable[SeqRecord],
    ) -> tuple[list[HLASequence], dict[EXON_NAME, dict[str, Seq]]]:
        """
        Pair exons in the given input sequences.

        The section of HLA we sequence looks like
        exon2 - intron - exon3
        and is typically sequenced in two parts, one covering exon2 and exon3
        (the intron is not used in our testing).  We iterate through the
        sequences and attempt to match them up.
        """
        matched_sequences: list[HLASequence] = []
        unmatched: dict[EXON_NAME, dict[str, Seq]] = {
            "exon2": {},
            "exon3": {},
        }

        example_standard: HLAStandard = list(self.hla_standards.values())[0]

        for sr in sequence_records:
            # Skip over any sequences that aren't the right length or contain
            # bad bases.
            try:
                check_length(self.locus, str(sr.seq), sr.id)
                check_bases(str(sr.seq))
            except ValueError:
                continue

            is_exon: bool = False
            matched: bool = False
            exon2: str = ""
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
                    example_standard.sequence, nuc2bin(exon2), "exon2"
                )
                exon3_bin = self.pad_short(
                    example_standard.sequence, nuc2bin(exon3), "exon3"
                )
                matched_sequences.append(
                    HLASequence(
                        two=(int(x) for x in exon2_bin),
                        intron=(),
                        three=(int(x) for x in exon3_bin),
                        name=identifier,
                        num_sequences_used=2,
                    )
                )
            else:
                seq_numpy: np.array = self.pad_short(
                    example_standard.sequence,
                    nuc2bin(sr.seq),  # type: ignore
                    None,
                )
                seq: tuple[int] = tuple(int(x) for x in seq_numpy)
                matched_sequences.append(
                    HLASequence(
                        two=seq[: EasyHLA.EXON2_LENGTH],
                        intron=seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON3_LENGTH],
                        three=seq[-EasyHLA.EXON3_LENGTH :],
                        name=identifier,
                        num_sequences_used=1,
                    )
                )
        return matched_sequences, unmatched

    def get_mismatches(
        self,
        standard_bin: Sequence[int],
        sequence_bin: Sequence[int],
    ) -> list[HLAMismatch]:
        """
        Report mismatched bases and their location versus a standard.

        standard_bin should look like (1, 4, 9, 14, 2, ...) where each position
        is represented as a 4-bit integer.

        :param standard_bin: the standard to compare the sequence to
        :type standard_bin: Sequence[int]
        :param sequence_bin: The sequence being interpreted.
        :type sequence_bin: Sequence[int]
        :return: A list of HLAMismatches, sorted by their indices.
        :rtype: list[HLAMismatch]
        """
        if len(standard_bin) == 0:
            raise ValueError("standard must be non-trivial")
        if len(standard_bin) != len(sequence_bin):
            raise ValueError("standard and sequence must be the same length")

        correct_base_at_pos: dict[int, int] = {}

        std_bin_seq = np.array(standard_bin)
        for idx in np.flatnonzero(std_bin_seq ^ sequence_bin):  # 0-based indices
            correct_base_at_pos[idx] = std_bin_seq[idx]

        mislist: list[HLAMismatch] = []

        for index, correct_base_bin in correct_base_at_pos.items():
            if self.locus == "A" and index > 269:  # i.e. > 270 in 1-based indices
                # This is 241 + 1, where the 1 converts from 0-based to 1-based
                # indices.
                dex = index + 242
            else:
                dex = index + 1

            mislist.append(
                HLAMismatch(
                    index=dex,
                    observed_base=BIN2NUC[sequence_bin[index]],
                    expected_base=BIN2NUC[correct_base_bin],
                )
            )

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
        :param threshold: number of mismatches between the sequence and possible
        matching standards; defaults to None
        :type threshold: Optional[int], optional
        :return: the HLA interpretation, which holds all matching standards and
        other details on each
        :rtype: HLAInterpretation
        """
        seq: tuple[int, ...] = hla_sequence.sequence_for_interpretation

        matching_stds = self.get_matching_standards(seq, self.hla_standards.values())
        if len(matching_stds) == 0:
            raise EasyHLA.NoMatchingStandards()

        # Now, combine all the stds (pick up that can citizen!)
        # DR 2023-02-24: To whomever made this comment, great shoutout!
        all_combos: dict[HLACombinedStandard, int] = self.combine_standards(
            matching_stds,
            seq,
            mismatch_threshold=threshold,
        )

        b5701_standards: Optional[list[HLAStandard]] = None
        if self.locus == "B":
            b5701_standards = [
                self.hla_standards[allele] for allele in self.B5701_ALLELES
            ]

        return HLAInterpretation(
            hla_sequence=hla_sequence,
            matches={
                combined_std: HLAMatchDetails(
                    mismatch_count=mismatch_count,
                    mismatches=self.get_mismatches(
                        combined_std.standard_bin,
                        seq,
                    ),
                )
                for combined_std, mismatch_count in all_combos.items()
            },
            b5701_standards=b5701_standards,
        )
