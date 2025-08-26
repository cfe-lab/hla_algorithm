import re
from collections.abc import Iterable
from operator import itemgetter
from typing import Final, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from .utils import (
    HLA_LOCUS,
    HLARawStandard,
    allele_coordinates,
    bin2nuc,
    count_forgiving_mismatches,
    nuc2bin,
    sort_allele_pairs,
)


class HLASequence(BaseModel):
    two: tuple[int, ...]
    intron: tuple[int, ...]
    three: tuple[int, ...]
    name: str
    locus: HLA_LOCUS
    num_sequences_used: int = 1

    @property
    def sequence_for_interpretation(self) -> tuple[int, ...]:
        """
        Returns the "binary" sequence for interpretation purposes.

        This is exon2 concatenated with exon3 (no intron).
        """
        return self.two + self.three

    @property
    def exon2_str(self) -> str:
        return bin2nuc(self.two)

    @property
    def exon3_str(self) -> str:
        return bin2nuc(self.three)

    @property
    def intron_str(self) -> str:
        return bin2nuc(self.intron)


class HLAStandard(BaseModel):
    allele: str
    two: tuple[int, ...]
    three: tuple[int, ...]

    @property
    def sequence(self) -> tuple[int, ...]:
        return self.two + self.three

    @property
    def sequence_np(self) -> np.ndarray:
        return np.array(self.sequence)

    @classmethod
    def from_raw_standard(cls, raw_standard: HLARawStandard) -> "HLAStandard":
        return cls(
            allele=raw_standard.allele,
            two=nuc2bin(raw_standard.exon2),
            three=nuc2bin(raw_standard.exon3),
        )


class HLAStandardMatch(HLAStandard):
    mismatch: int


class HLACombinedStandard(BaseModel):
    """
    Represents a combined HLA standard and all of its possible combinations.

    standard_bin is the sequence in "binary" form, as defined by the methods
    "bin2nuc" and "nuc2bin" (see `utils`).

    possible_allele_pairs represents the possible allele pairs that produce this
    combined standard (as there may be more than one).
    """

    model_config = ConfigDict(frozen=True)

    standard_bin: tuple[int, ...]
    possible_allele_pairs: tuple[tuple[str, str], ...]

    def get_allele_pair_str(self):
        pair_strings: list[str] = [
            " - ".join(pair) for pair in self.possible_allele_pairs
        ]
        return "|".join(pair_strings)


class HLAMismatch(BaseModel):
    index: int
    sequence_base: str
    standard_base: str

    def __str__(self):
        return f"{self.index}:{self.sequence_base}->{self.standard_base}"


class HLAMatchDetails(BaseModel):
    mismatches: list[HLAMismatch]

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)


class HLAProteinPair(BaseModel):
    # Allows this to be hashable:
    model_config = ConfigDict(frozen=True)

    first_field_1: str
    first_field_2: str
    second_field_1: str
    second_field_2: str

    def __lt__(self, other: "HLAProteinPair") -> bool:
        me_tuple: tuple[int, int, int, int] = (
            int(self.first_field_1),
            int(self.first_field_2),
            int(self.second_field_1),
            int(self.second_field_2),
        )
        other_tuple: tuple[int, int, int, int] = (
            int(other.first_field_1),
            int(other.first_field_2),
            int(other.second_field_1),
            int(other.second_field_2),
        )
        return me_tuple < other_tuple

    # Note: originally these were annotated as ClassVar[Final[str]] but this
    # isn't supported in versions of Python prior to 3.13.
    UNMAPPED: Final[str] = "unmapped"
    DEPRECATED: Final[str] = "deprecated"

    class NonAlleleException(Exception):
        def __init__(
            self,
            first_unmapped: bool = False,
            first_deprecated: bool = False,
            second_unmapped: bool = False,
            second_deprecated: bool = False,
        ):
            self.first_unmapped = first_unmapped
            self.first_deprecated = first_deprecated
            self.second_unmapped = second_unmapped
            self.second_deprecated = second_deprecated

        @classmethod
        def from_frequency_entry(
            cls, raw_first_allele: str, raw_second_allele: str
        ) -> Optional["HLAProteinPair.NonAlleleException"]:
            first_unmapped: bool = False
            first_deprecated: bool = False
            second_unmapped: bool = False
            second_deprecated: bool = False

            if raw_first_allele == HLAProteinPair.UNMAPPED:
                first_unmapped = True
            elif raw_first_allele == HLAProteinPair.DEPRECATED:
                first_deprecated = True

            if raw_second_allele == HLAProteinPair.UNMAPPED:
                second_unmapped = True
            elif raw_second_allele == HLAProteinPair.DEPRECATED:
                second_deprecated = True

            if any(
                (first_unmapped, first_deprecated, second_unmapped, second_deprecated)
            ):
                return cls(
                    first_unmapped, first_deprecated, second_unmapped, second_deprecated
                )
            return None

    @classmethod
    def from_frequency_entry(
        cls,
        raw_first_allele: str,
        raw_second_allele: str,
    ) -> "HLAProteinPair":
        any_problems: Optional[HLAProteinPair.NonAlleleException] = (
            HLAProteinPair.NonAlleleException.from_frequency_entry(
                raw_first_allele, raw_second_allele
            )
        )
        if any_problems is not None:
            raise any_problems

        first_field_1: str
        first_field_2: str
        second_field_1: str
        second_field_2: str

        first_field_1, first_field_2 = raw_first_allele.split(":")
        second_field_1, second_field_2 = raw_second_allele.split(":")

        return cls(
            first_field_1=first_field_1,
            first_field_2=first_field_2,
            second_field_1=second_field_1,
            second_field_2=second_field_2,
        )


class AllelePairs(BaseModel):
    allele_pairs: list[tuple[str, str]]

    def is_homozygous(self) -> bool:
        """
        Determine the homozygousness of these allele pairs.

        A pair is homozygous if both elements match, e.g. C*07:22 - C*07:22.
        If *any* pair of alleles matches, then we declare the whole set to be
        homozygous.

        :return: ...
        :rtype: bool
        """
        return any(a1 == a2 for a1, a2 in self.allele_pairs)

    def is_ambiguous(self) -> bool:
        """
        Determine whether our collection of allele pairs is ambiguous or not.

        This is determined by checking whether every allele pair in the
        collection belongs to the same allele groups (i.e. every first allele
        has the same first field in its "coordinates", and likewise for the
        second).

        :return: ...
        :rtype: bool
        """
        paired_allele_groups: set[tuple[str, str]] = {
            (first[0], second[0])
            for first, second in self.get_paired_gene_coordinates(True)
        }
        return len(paired_allele_groups) != 1

    def get_paired_gene_coordinates(
        self, digits_only: bool = False
    ) -> list[tuple[list[str], list[str]]]:
        """
        Retrieve paired lists of gene coordinates for all allele pairs in the collection.

        For example, if the allele were "A*01:23:45 - A*98:76", this
        would be converted to (["A*01", "23", "45"], ["A*98", "76N"]).
        If digits_only is True, then "A*01" would simply be "01",
        "A*98" would be "98", and "76N" would be "76".
        """
        return [
            (
                allele_coordinates(allele_pair[0], digits_only),
                allele_coordinates(allele_pair[1], digits_only),
            )
            for allele_pair in self.allele_pairs
        ]

    def get_protein_pairs(self) -> set[HLAProteinPair]:
        """
        Gets the allele groups and proteins of each pair of alleles.

        For example, the allele pair "A*01:23:45 - A*98:76" would be
        represented by 01:23 and 98:76.

        :return: _description_
        :rtype: set[str, int]
        """
        return {
            HLAProteinPair(
                first_field_1=e[0][0],
                first_field_2=e[0][1],
                second_field_1=e[1][0],
                second_field_2=e[1][1],
            )
            for e in self.get_paired_gene_coordinates(True)
        }

    def get_unambiguous_allele_pairs(
        self,
        frequencies: dict[HLAProteinPair, int],
    ) -> list[tuple[str, str]]:
        """
        Filter the allele pairs to an unambiguous set.

        The "top" allele pair is determined by the following criteria, in
        descending order of precedence:

        1) its frequency, according to the specified dict;
        2) lowest "first coordinate" of the first allele;
        3) lowest "second coordinate" of the first allele;
        4) lowest "first coordinate" of the second allele; and
        5) lowest "second coordinate" of the second allele.

        The allele pairs that belong to the same allele groups (i.e. the first
        allele has the same first "coordinate" as the top first allele, and
        likewise for the second) are retained.

        :param frequencies: a table mapping protein pairs to their frequency
        :type frequencies: dict[HLAProteinPair, int]
        :return: List of alleles filtered by HLA frequency.
        :rtype: list[tuple[str,str]]
        """

        protein_pairs_and_frequencies: list[tuple[HLAProteinPair, int]] = []
        for protein_pair in self.get_protein_pairs():
            protein_pairs_and_frequencies.append(
                (protein_pair, frequencies.get(protein_pair, 0))
            )
        # First, sort by protein pair, ascending:
        protein_pairs_and_frequencies.sort(key=itemgetter(0))
        # Then, sort by frequency, descending:
        protein_pairs_and_frequencies.sort(key=itemgetter(1), reverse=True)
        best_pair: HLAProteinPair = protein_pairs_and_frequencies[0][0]

        regex_str_a1 = f"^[ABCabc]\\*({best_pair.first_field_1}):([^\\s])+"
        regex_str_a2 = f"^[ABCabc]\\*({best_pair.second_field_1}):([^\\s])+"
        reduced_set: list[tuple[str, str]] = []
        for allele_1, allele_2 in self.allele_pairs:
            if re.match(regex_str_a1, allele_1) and re.match(regex_str_a2, allele_2):
                reduced_set.append((allele_1, allele_2))

        return reduced_set

    def best_common_allele_pair_str(
        self,
        frequencies: dict[HLAProteinPair, int],
    ) -> tuple[str, set[tuple[str, str]]]:
        """
        Produce a string representation of the "best common allele pair".

        The allele pairs are filtered to an unambiguous set (using the specified
        frequencies to determine which ones to retain).  Then, the "best common
        coordinates" for all the remaining allele allele pairs are used to build
        a string representation of the set.

        Example: if, after filtering, the allele pairs remaining are:
        ```
        [   [A*11:02:01, A*12:01],
            [A*11:02:02, A*12:02],
            [A*11:02:03, A*12:03]   ]
        ```
        we expect to get `A*11:02 - A*12`.

        :return: A string representing the best common allele pair, and the
        unambiguous set this string represents.
        :rtype: tuple[str, set[tuple[str, str]]]
        """
        # Starting with an unambiguous set assures that we will definitely get
        # a result.
        unambiguous_aps: AllelePairs = AllelePairs(
            allele_pairs=self.get_unambiguous_allele_pairs(frequencies)
        )
        paired_gene_coordinates: list[tuple[list[str], list[str]]] = (
            unambiguous_aps.get_paired_gene_coordinates()
        )

        clean_allele: list[str] = []
        for n in [0, 1]:
            for i in [4, 3, 2, 1]:
                all_leading_coordinates = {
                    ":".join(a[n][0:i]) for a in paired_gene_coordinates
                }
                if len(all_leading_coordinates) == 1:
                    best_common_coords = all_leading_coordinates.pop()
                    clean_allele.append(
                        re.sub(
                            r"[A-Z]$",
                            "",
                            best_common_coords,
                        )
                    )
                    if i > 1:
                        # This branch is unnecessary but it gets us 100% code
                        # coverage ¯\_(ツ)_/¯
                        break

        clean_allele_pair_str: str = " - ".join(clean_allele)
        return (clean_allele_pair_str, set(unambiguous_aps.allele_pairs))

    def stringify(self, sorted=True, max_length: int = 3900) -> str:
        """
        Produce a final outputtable string.

        If the string exceeds max_length, it will be
        truncated.

        :return: ...
        :rtype: str
        """
        allele_pairs: list[tuple[str, str]] = self.allele_pairs
        if sorted:
            allele_pairs = sort_allele_pairs(self.allele_pairs)
        summary_str: str = ";".join([f"{_a[0]} - {_a[1]}" for _a in allele_pairs])
        if len(summary_str) > max_length:
            summary_str = re.sub(
                r";[^;]+$",
                ";...TRUNCATED",
                summary_str[: max_length + 20],
            )
        return summary_str

    @classmethod
    def get_allele_pairs(
        cls,
        combined_standards: Iterable[HLACombinedStandard],
    ) -> "AllelePairs":
        """
        Get all allele pairs in the specified combined standards.

        :param best_matches: ...
        :type best_matches: Iterable[HLACombinedStandard]
        :return: ...
        :rtype: AllelePairs
        """
        all_allele_pairs: list[tuple[str, str]] = []
        for combined_std in combined_standards:
            all_allele_pairs.extend(combined_std.possible_allele_pairs)
        all_allele_pairs = sort_allele_pairs(all_allele_pairs)
        return cls(allele_pairs=all_allele_pairs)

    def contains_allele(self, allele_name: str) -> bool:
        """
        Returns True if allele_name is among the alleles in the pairs.
        """
        all_individual_alleles: list[str] = list(sum(self.allele_pairs, ()))
        return any(x.startswith(allele_name) for x in all_individual_alleles)


class HLAInterpretation(BaseModel):
    hla_sequence: HLASequence
    matches: dict[HLACombinedStandard, HLAMatchDetails]
    allele_frequencies: dict[HLAProteinPair, int]
    b5701_standards: Optional[list[HLAStandard]] = None

    @property
    def locus(self) -> HLA_LOCUS:
        return self.hla_sequence.locus

    def lowest_mismatch_count(self) -> int:
        return min([x.mismatch_count for x in self.matches.values()])

    def best_matches(self) -> set[HLACombinedStandard]:
        best_mismatch_count: int = self.lowest_mismatch_count()
        return {
            combined_std
            for combined_std, match_details in self.matches.items()
            if match_details.mismatch_count == best_mismatch_count
        }

    def best_matching_allele_pairs(self) -> AllelePairs:
        return AllelePairs.get_allele_pairs(self.best_matches())

    def best_common_allele_pair(
        self,
    ) -> tuple[tuple[str, str], str, HLACombinedStandard]:
        """
        Find the (or *a*) "best common allele pair" for this interpretation.
        """
        # First, make a lookup table mapping allele pairs to the combined
        # standards they come from.
        best_matches: set[HLACombinedStandard] = self.best_matches()
        ap_to_cs: dict[tuple[str, str], HLACombinedStandard] = {}
        for cs in best_matches:
            for ap in cs.possible_allele_pairs:
                ap_to_cs[ap] = cs

        # Get an unambiguous set of allele pairs from the best matches:
        best_aps: AllelePairs = AllelePairs.get_allele_pairs(best_matches)
        clean_ap_str: str
        best_unambiguous: set[tuple[str, str]]
        clean_ap_str, best_unambiguous = best_aps.best_common_allele_pair_str(
            self.allele_frequencies
        )

        best_representative: tuple[str, str] = sorted(best_unambiguous)[0]
        return (
            best_representative,
            clean_ap_str,
            ap_to_cs[best_representative],
        )

    def distance_from_b7501(self) -> Optional[int]:
        """
        Return the Hamming distance from this sequence to B*57:01.

        The HLA sequence in question is compared to the specified B*57:01
        sequences (typically B*57:01:01G, B*57:01:02, and B*57:01:03) and
        computes the minimum distance between this sequence and those three.

        If no B*57:01 references are specified (i.e. this is not an HLA-B
        sequence), return None.
        """
        if self.b5701_standards is None:
            return None
        distances: list[int] = [
            count_forgiving_mismatches(
                self.hla_sequence.sequence_for_interpretation, standard.sequence
            )
            for standard in self.b5701_standards
        ]
        return min(distances)

    def is_b5701(self) -> bool:
        """
        Returns True if this sequence is a B*57:01 sequence, and False otherwise.
        """
        all_allele_pairs: AllelePairs = self.best_matching_allele_pairs()
        return all_allele_pairs.contains_allele("B*57:01")
