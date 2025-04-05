import re
from collections.abc import Iterable
from operator import itemgetter

import numpy as np
import pydantic_numpy.typing as pnd
from pydantic import BaseModel, ConfigDict
from pydantic_numpy.model import NumpyModel

from .utils import bin2nuc


class HLASequence(NumpyModel):
    two: pnd.NpNDArray
    intron: pnd.NpNDArray = np.array([])
    three: pnd.NpNDArray
    name: str
    num_sequences_used: int = 1

    @property
    def sequence_for_interpretation(self) -> np.ndarray:
        """
        Returns the "binary" sequence for interpretation purposes.

        This is exon2 concatenated with exon3 (no intron)..
        """
        return np.concatenate((self.two, self.three))

    @property
    def exon2_str(self) -> str:
        return bin2nuc(self.two)

    @property
    def exon3_str(self) -> str:
        return bin2nuc(self.three)

    @property
    def intron_str(self) -> str:
        return bin2nuc(self.intron)


class HLAStandard(NumpyModel):
    allele: str
    sequence: pnd.NpNDArray

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all(
            [self.allele == other.allele, np.array_equal(self.sequence, other.sequence)]
        )


class HLAStandardMatch(HLAStandard):
    mismatch: int

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all(
            [
                self.allele == other.allele,
                np.array_equal(self.sequence, other.sequence),
                self.mismatch == other.mismatch,
            ]
        )


class HLACombinedStandard(BaseModel):
    """
    Represents a combined HLA standard and all of its possible combinations.

    standard_bin is the sequence in "binary" form, as defined by the EasyHLA
    methods "bin2nuc" and "nuc2bin".

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
    observed_base: str
    expected_bases: list[str]

    def __str__(self):
        return f"{self.index}:{self.observed_base}->{'/'.join(self.expected_bases)}"


class HLAMatchDetails(BaseModel):
    mismatch_count: int
    mismatches: list[HLAMismatch]


class HLAProteinPair(BaseModel):
    # Allows this to be hashable:
    model_config = ConfigDict(frozen=True)

    first_field_1: str
    first_field_2: str
    second_field_1: str
    second_field_2: str

    def __lt__(self, other: "HLAProteinPair") -> bool:
        me_tuple: tuple[int, int, int, int] = (
            self.first_field_1,
            self.first_field_2,
            self.second_field_1,
            self.second_field_2,
        )
        other_tuple: tuple[int, int, int, int] = (
            other.first_field_1,
            other.first_field_2,
            other.second_field_1,
            other.second_field_2,
        )
        return me_tuple < other_tuple


class AllelePairs(BaseModel):
    allele_pairs: list[tuple[str, str]]

    def is_homozygous(self) -> bool:
        """
        Determine the homozygousness of alleles.

        Homozygousity meaning a pair is matching on both sides, ex:
        `Cw*0722 - Cw*0722`

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

    @staticmethod
    def _allele_coordinates(
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
                self._allele_coordinates(allele_pair[0], digits_only),
                self._allele_coordinates(allele_pair[1], digits_only),
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

        :param locus: ...
        :type locus: HLA_LOCI
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
    ) -> str:
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

        :return: A string representing the best common allele pair.
        :rtype: str
        """
        # Starting with an unambiguous set assures that we will definitely get
        # a result.
        unambiguous_set: AllelePairs = AllelePairs(
            allele_pairs=self.get_unambiguous_allele_pairs(frequencies)
        )

        clean_allele: list[str] = []
        for n in [0, 1]:
            for i in [4, 3, 2, 1]:
                all_leading_coordinates = {
                    ":".join(a[n][0:i])
                    for a in unambiguous_set.get_paired_gene_coordinates()
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
                    break

        clean_allele_pair_str: str = " - ".join(clean_allele)
        return clean_allele_pair_str

    def stringify(self, max_length: int = 3900) -> str:
        """
        Produce a final outputtable string.

        If the string exceeds max_length, it will be
        truncated.

        :return: ...
        :rtype: str
        """
        summary_str: str = ";".join([f"{_a[0]} - {_a[1]}" for _a in self.allele_pairs])
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
        all_allele_pairs.sort()
        return cls(allele_pairs=all_allele_pairs)


class HLAInterpretation(BaseModel):
    hla_sequence: HLASequence
    matches: dict[HLACombinedStandard, HLAMatchDetails]

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
