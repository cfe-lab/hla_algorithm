import csv
import os
from collections.abc import Generator, Iterable, Sequence
from datetime import datetime
from io import TextIOBase
from operator import attrgetter
from typing import Final, Optional, TypedDict, cast

import numpy as np
import yaml

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
    HLA_LOCUS,
    StoredHLAStandards,
    count_strict_mismatches,
    nuc2bin,
)

DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"


class LoadedStandards(TypedDict):
    tag: str
    last_updated: datetime
    standards: dict[HLA_LOCUS, dict[str, HLAStandard]]


class HLAAlgorithm:
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
        loaded_standards: Optional[LoadedStandards] = None,
        hla_frequencies: Optional[dict[HLA_LOCUS, dict[HLAProteinPair, int]]] = None,
    ):
        """
        Initialize an HLAAlgorithm class.

        :param logger: Python logger object, defaults to None
        :type logger: Optional[logging.Logger], optional
        """
        if loaded_standards is None:
            loaded_standards = self.load_default_hla_standards()

        self.hla_standards: dict[HLA_LOCUS, dict[str, HLAStandard]] = loaded_standards[
            "standards"
        ]
        self.last_updated: datetime = loaded_standards["last_updated"]
        self.tag: str = loaded_standards["tag"]

        self.hla_frequencies: dict[HLA_LOCUS, dict[HLAProteinPair, int]]
        if hla_frequencies is not None:
            self.hla_frequencies = hla_frequencies
        else:
            self.hla_frequencies = self.load_default_hla_frequencies()

    @classmethod
    def use_config(
        cls,
        standards_path: Optional[str] = None,
        frequencies_path: Optional[str] = None,
    ) -> "HLAAlgorithm":
        """
        An alternate constructor that accepts file paths for the configuration.
        """
        processed_stds: Optional[LoadedStandards] = None
        frequencies: Optional[dict[HLA_LOCUS, dict[HLAProteinPair, int]]] = None

        if standards_path is not None:
            with open(standards_path) as f:
                processed_stds = cls.read_hla_standards(f)

        if frequencies_path is not None:
            with open(frequencies_path) as f:
                frequencies = cls.read_hla_frequencies(f)

        return cls(processed_stds, frequencies)

    @staticmethod
    def read_hla_standards(standards_io: TextIOBase) -> LoadedStandards:
        """
        Read HLA standards from a specified file-like object.

        :return: Dictionary of known HLA standards keyed by their name
        :rtype: dict[str, HLAStandard]
        """
        stored_stds: StoredHLAStandards = StoredHLAStandards.model_validate(
            yaml.safe_load(standards_io)
        )

        hla_stds: dict[HLA_LOCUS, dict[str, HLAStandard]] = {
            "A": {},
            "B": {},
            "C": {},
        }
        for locus in ("A", "B", "C"):
            for grouped_allele in stored_stds.standards[locus]:
                hla_stds[locus][grouped_allele.name] = HLAStandard(
                    allele=grouped_allele.name,
                    two=nuc2bin(grouped_allele.exon2),
                    three=nuc2bin(grouped_allele.exon3),
                )

        return {
            "tag": stored_stds.tag,
            "last_updated": stored_stds.last_updated,
            "standards": hla_stds,
        }

    @staticmethod
    def load_default_hla_standards() -> LoadedStandards:
        """
        Load HLA Standards from reference file.

        :return: List of known HLA standards
        :rtype: list[HLAStandard]
        """
        standards_filename: str = HLAAlgorithm._path_join_shim(
            os.path.dirname(__file__),
            "default_data",
            "hla_standards.yaml",
        )
        with open(standards_filename) as standards_file:
            return HLAAlgorithm.read_hla_standards(standards_file)

    FREQUENCY_LOCUS_COLUMNS: dict[HLA_LOCUS, tuple[str, str]] = {
        "A": ("a_first", "a_second"),
        "B": ("b_first", "b_second"),
        "C": ("c_first", "c_second"),
    }

    @staticmethod
    def read_hla_frequencies(
        frequencies_io: TextIOBase,
    ) -> dict[HLA_LOCUS, dict[HLAProteinPair, int]]:
        """
        Load HLA frequencies from a specified file-like object.

        This takes each two columns AAAA,BBBB out of 6 (...FFFF), and then uses a
        subset of these two columns (AABB,CCDD) to use as the key, in this case
        "AA|BB,CC|DD", we then count the number of times this key appears in our
        columns.

        :return: Lookup table of locus and HLA frequencies.
        :rtype: dict[HLA_LOCUS, dict[HLAProteinPair, int]]
        """
        hla_freqs: dict[HLA_LOCUS, dict[HLAProteinPair, int]] = {
            "A": {},
            "B": {},
            "C": {},
        }

        with frequencies_io:
            frequencies_csv: csv.DictReader = csv.DictReader(frequencies_io)

            for row in frequencies_csv:
                for locus in ("A", "B", "C"):
                    curr_col1: str
                    curr_col2: str
                    curr_col1, curr_col2 = HLAAlgorithm.FREQUENCY_LOCUS_COLUMNS[locus]

                    try:
                        protein_pair: HLAProteinPair = (
                            HLAProteinPair.from_frequency_entry(
                                row[curr_col1], row[curr_col2]
                            )
                        )
                    except HLAProteinPair.NonAlleleException:
                        continue

                    if hla_freqs[locus].get(protein_pair, None) is None:
                        hla_freqs[locus][protein_pair] = 0
                    hla_freqs[locus][protein_pair] += 1
        return hla_freqs

    @staticmethod
    def _path_join_shim(*args) -> str:
        """
        A shim for os.path.join which allows us to mock out the method easily in testing.
        """
        return os.path.join(*args)

    @staticmethod
    def load_default_hla_frequencies() -> dict[HLA_LOCUS, dict[HLAProteinPair, int]]:
        """
        Load HLA frequencies from the default reference file.

        :return: Lookup table of HLA frequencies.
        :rtype: dict[HLA_LOCUS, dict[HLAProteinPair, int]]
        """
        hla_freqs: dict[HLA_LOCUS, dict[HLAProteinPair, int]]
        default_frequencies_filename: str = HLAAlgorithm._path_join_shim(
            os.path.dirname(__file__),
            "default_data",
            "hla_frequencies.csv",
        )
        with open(default_frequencies_filename, "r") as f:
            hla_freqs = HLAAlgorithm.read_hla_frequencies(f)
        return hla_freqs

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
    def combine_standards_stepper(
        matching_stds: Sequence[HLAStandardMatch],
        seq: Sequence[int],
        mismatch_threshold: int = 0,
    ) -> Generator[tuple[tuple[int, ...], int, tuple[str, str]], None, None]:
        """
        Identifies "good" combined standards for the specified sequence.

        On each iteration, it continues checking combined standards until it
        finds a "match", and yields a tuple containing the details of that
        match:
        - the combined standard, as a tuple of integers 0-15;
        - the number of mismatches identified; and
        - the allele pair (i.e. names of the two alleles in the combination).

        A "match" is defined by the number of mismatches between the combined
        standard and the sequence:
        - this is the best-matching combined standard found so far (may
          be above our mismatch threshold) or as good as the best-matching one
          found so far; or
        - this is below our mismatch threshold.
        If the mismatch threshold is 0, then we will only ever get the former.
        """
        # Keep track of matches we've already found:
        combos: dict[tuple[int, ...], int] = {}

        current_rejection_threshold: int | float = float("inf")
        for std_ai, std_a in enumerate(matching_stds):
            if std_a.mismatch > current_rejection_threshold:
                continue
            for std_b in matching_stds[: (std_ai + 1)]:
                if std_b.mismatch > current_rejection_threshold:
                    continue

                # "Mush" the two standards together to produce something
                # that looks like what you get when you sequence HLA.
                std_bin = np.array(std_b.sequence) | np.array(std_a.sequence)
                allele_pair: tuple[str, str] = cast(
                    tuple[str, str], tuple(sorted((std_a.allele, std_b.allele)))
                )

                # There could be more than one combined standard with the
                # same sequence, so check if this one's already been found.
                combined_std_bin: tuple[int, ...] = tuple(int(s) for s in std_bin)

                mismatches: int = -1
                if combined_std_bin in combos:
                    mismatches = combos[combined_std_bin]

                else:
                    # Note that seq is implicitly cast to a NumPy array:
                    mismatches = np.count_nonzero(std_bin ^ seq != 0)
                    combos[combined_std_bin] = mismatches  # cache this value

                if mismatches > current_rejection_threshold:
                    continue
                elif mismatches < current_rejection_threshold:
                    current_rejection_threshold = max(mismatches, mismatch_threshold)

                yield (combined_std_bin, mismatches, allele_pair)

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
        counts.  If mismatch_threshold is None or 0, then the result contains
        only the best-matching combined standard(s); otherwise, the result
        contains all combined standards with mismatch counts up to and including
        the threshold.  All of the HLACombinedStandards have their
        `possible_allele_pairs` value sorted.
        """
        if mismatch_threshold is None:
            # We only care about the best match, so we set this threshold
            # so that it never forces the rejection threshold to stay above
            # the current best match.
            mismatch_threshold = 0

        combos: dict[tuple[int, ...], tuple[int, list[tuple[str, str]]]] = {}

        fewest_mismatches: int | float = float("inf")
        for (
            combined_std_bin,
            mismatches,
            allele_pair,
        ) in HLAAlgorithm.combine_standards_stepper(
            matching_stds, seq, mismatch_threshold
        ):
            if combined_std_bin not in combos:
                combos[combined_std_bin] = (mismatches, [])
            combos[combined_std_bin][1].append(allele_pair)
            if mismatches < fewest_mismatches:
                fewest_mismatches = mismatches

        # Winnow out any extraneous combined standards that don't match our
        # criteria.
        result: dict[HLACombinedStandard, int] = {}

        cutoff: int | float = max(fewest_mismatches, mismatch_threshold)
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
    def get_mismatches(
        standard_bin: Sequence[int],
        sequence_bin: Sequence[int],
        locus: HLA_LOCUS,
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
            if locus == "A" and index > 269:  # i.e. > 270 in 1-based indices
                # This is 241 + 1, where the 1 converts from 0-based to 1-based
                # indices.
                dex = index + 242
            else:
                dex = index + 1

            mislist.append(
                HLAMismatch(
                    index=dex,
                    sequence_base=BIN2NUC[sequence_bin[index]],
                    standard_base=BIN2NUC[correct_base_bin],
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
        locus: HLA_LOCUS = hla_sequence.locus

        matching_stds = self.get_matching_standards(
            seq, self.hla_standards[locus].values()
        )
        if len(matching_stds) == 0:
            raise HLAAlgorithm.NoMatchingStandards()

        # Now, combine all the stds (pick up that can citizen!)
        # DR 2023-02-24: To whomever made this comment, great shoutout!
        all_combos: dict[HLACombinedStandard, int] = self.combine_standards(
            matching_stds,
            seq,
            mismatch_threshold=threshold,
        )

        b5701_standards: Optional[list[HLAStandard]] = None
        if locus == "B":
            b5701_standards = [
                self.hla_standards[locus][allele] for allele in self.B5701_ALLELES
            ]

        return HLAInterpretation(
            hla_sequence=hla_sequence,
            matches={
                combined_std: HLAMatchDetails(
                    mismatches=self.get_mismatches(
                        combined_std.standard_bin,
                        seq,
                        locus,
                    ),
                )
                for combined_std in all_combos
            },
            allele_frequencies=self.hla_frequencies[locus],
            b5701_standards=b5701_standards,
        )
