import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pydantic_numpy.typing as pnd
from pydantic import BaseModel
from pydantic_numpy.model import NumpyModel

ALLELES_MAX_REPORTABLE_STRING: int = 3900


class Exon(BaseModel):
    two: str
    intron: str = ""
    three: str


class Alleles(BaseModel):
    alleles: List[Tuple[str, str]]

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
        return any(_a[0] == _a[1] for _a in self.alleles)

    def is_ambiguous(self) -> bool:
        """
        Determine whether our collection of alleles is ambiguous or not.

        This is determined by checking whether every allele in the collection
        belongs to the same allele group (i.e. has the first field in its
        "coordinates").

        :return: ...
        :rtype: bool
        """
        if len(self.get_allele_groups()) != 1:
            return True
        return False

    @staticmethod
    def _allele_coordinates(
        allele: str,
        remove_subtype: bool = False,
    ) -> List[str]:
        """
        Convert an allele string into a list of coordinates.

        For example, allele "A*01:23:45" gets converted to
        ["A*01", "23", "45"] or ["01", "23", "45] depending on the value of
        remove_subtype.
        """
        clean_allele_str: str = allele
        if remove_subtype:
            clean_allele_str = re.sub("r[^\d:]", "", allele)
        return clean_allele_str.strip().split(":")

    def get_gene_coordinates(
        self, remove_subtype: bool = False
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Retrieve a list of gene coordinates for all alleles in the collection.

        For example, if the allele were "A*01:23:45 - A*98:76", this
        would be converted to (["A*01", "23", "45"], ["A*98", "76"]).
        If remove_subtype is True, then "A*01" would simply be "01",
        and "A*98" would be "98".
        """
        return [
            (
                self._allele_coordinates(allele_pair[0], remove_subtype),
                self._allele_coordinates(allele_pair[1], remove_subtype),
            )
            for allele_pair in self.alleles
        ]

    def get_allele_groups(self) -> Set[str]:
        return {
            f"{e[0][0]}, {e[1][0]}"
            for e in self.get_gene_coordinates(remove_subtype=True)
        }

    def get_proteins_as_strings(self) -> Set[str]:
        """
        Gets the allele groups and proteins of each pair of alleles.

        For example, the allele pair "A*01:23:45 - A*98:76" becomes
        "01|23,98|76".

        :return: _description_
        :rtype: Dict[str, int]
        """
        return {
            f"{'|'.join(e[0][0:2])},{'|'.join(e[1][0:2])}": 0
            for e in self.get_gene_coordinates(remove_subtype=True)
        }

    def stringify_clean(self) -> str:
        """
        Get most common allele in all identified alleles.

        Example:
        ```
        [   [A*11:02:01, A*12:01],
            [A*11:02:02, A*12:02],
            [A*11:02:03, A*12:03]   ]
        ```

        We expect to get `A*11:02 - A*12`

        **Implementation Note:** This should be run *after*
        `EasyHLA.filter_reportable_alleles` if `self.is_ambiguous()` is true:

        ```
        alleles_all_str = alleles.stringify()
        if alleles.is_ambiguous():
            alleles.alleles = self.filter_reportable_alleles(
                letter=self.letter, alleles=alleles
            )
        clean_allele_str = alleles.stringify_clean()
        ```

        :param all_alleles: ...
        :type all_alleles: List[List[str]]
        :return: A string representing the most common allele pair.
        :rtype: str
        """
        clean_allele: List[str] = []
        for n in [0, 1]:
            for i in [4, 3, 2, 1]:
                if len({":".join(a[n][0:i]) for a in self.get_gene_coordinates()}) == 1:
                    clean_allele.append(
                        re.sub(
                            r"[A-Z]$",
                            "",
                            ":".join(self.get_gene_coordinates()[0][n][0:i]),
                        )
                    )
                    break

        clean_allele_str: str = " - ".join(clean_allele)
        return clean_allele_str

    def stringify(self) -> str:
        """
        Produce a final outputtable string.

        If the string exceeds EasyHLA.ALLELES_MAX_REPORTABLE_STRING, it will be
        truncated.

        :return: ...
        :rtype: str
        """
        alleles_all_str = ";".join([f"{_a[0]} - {_a[1]}" for _a in self.alleles])

        if len(alleles_all_str) > ALLELES_MAX_REPORTABLE_STRING:
            alleles_all_str = re.sub(
                r";[^;]+$",
                ";...TRUNCATED",
                alleles_all_str[: ALLELES_MAX_REPORTABLE_STRING + 20],
            )

        return alleles_all_str


class HLASequence(NumpyModel):
    exon: Exon
    sequence: pnd.NpNDArray


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


class HLACombinedStandardResult(BaseModel):
    standard: str
    discrete_allele_names: List[List[str]]


class HLAResultRow(BaseModel):
    samp: str = ""
    clean_allele_str: str = ""
    alleles_all_str: str = ""
    ambig: int = 0
    homozygous: int = 0
    mismatch_count: int = 0
    mismatches: str = ""
    exon2: str = ""
    intron: str = ""
    exon3: str = ""

    def get_result(self) -> List[str]:
        return [
            self.samp,
            self.clean_allele_str,
            self.alleles_all_str,
            f"{self.ambig}",
            f"{self.homozygous}",
            f"{self.mismatch_count}",
            self.mismatches,
            self.exon2.upper(),
            self.intron.upper(),
            self.exon3.upper(),
        ]

    def get_result_as_str(self) -> str:
        return ",".join(el for el in self.get_result())


class HLAResult(BaseModel):
    result: HLAResultRow
    num_seqs: int = 1
    num_pats: int = 1
