import numpy as np
import re
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from pydantic_numpy.ndarray import NDArray


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

        :return: ...
        :rtype: bool
        """
        return any([_a[0] == _a[1] for _a in self.alleles])

    def is_ambiguous(self) -> bool:
        """
        Determine whether our collection of alleles is ambiguous or not.

        :return: ...
        :rtype: bool
        """
        if len(self.get_unique_collection()) != 1:
            return True
        return False

    def get_collection(
        self, remove_subtype: bool = False
    ) -> List[Tuple[List[str], List[str]]]:
        if remove_subtype:
            return [
                (
                    re.sub(r"[^\d:]", "", a[0]).split(":"),
                    re.sub(r"[^\d:]", "", a[1]).split(":"),
                )
                for a in self.alleles
            ]
        return [
            (a[0].strip().split(":"), a[1].strip().split(":")) for a in self.alleles
        ]

    def get_unique_collection(self) -> set[str]:
        return {
            f"{e[0][0]}, {e[1][0]}" for e in self.get_collection(remove_subtype=True)
        }

    def get_ambiguous_collection(self) -> Dict[str, int]:
        """
        Gets an ambiguous collection of alleles as a dict of frequencies.

        This will be filled by

        :return: _description_
        :rtype: Dict[str, int]
        """
        return {
            f"{'|'.join(e[0][0:2])},{'|'.join(e[1][0:2])}": 0
            for e in self.get_collection(remove_subtype=True)
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
                if len(set([":".join(a[n][0:i]) for a in self.get_collection()])) == 1:
                    clean_allele.append(
                        re.sub(
                            r"[A-Z]$", "", ":".join(self.get_collection()[0][n][0:i])
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


class HLASequence(BaseModel):
    exon: Exon
    sequence: NDArray


class HLAStandard(BaseModel):
    allele: str
    sequence: NDArray

    def __eq__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all(
            [self.allele == other.allele, np.array_equal(self.sequence, other.sequence)]
        )


class HLAStandardMatch(HLAStandard):
    mismatch: int

    def __eq__(self, other):
        if type(other) != type(self):
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
        return ",".join([el for el in self.get_result()])


class HLAResult(BaseModel):
    result: HLAResultRow
    num_seqs: int = 1
    num_pats: int = 1
