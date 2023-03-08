from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple


class Exon(BaseModel):
    two: str
    intron: str = ""
    three: str

    def __eq__(self, other):
        if type(other) != type(self):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all(
            [
                self.two == other.two,
                self.intron == other.intron,
                self.three == other.three,
            ]
        )


class HLASequence(BaseModel):
    exon: Exon
    sequence: List[int]

    def __eq__(self, other):
        if type(other) != type(self):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all([self.exon == other.exon, self.sequence == other.sequence])


class HLAStandard(BaseModel):
    allele: str
    sequence: List[int]

    def __eq__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all([self.allele == other.allele, self.sequence == other.sequence])

    def __lt__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.allele < other.allele

    def __le__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.allele <= other.allele

    def __gt__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.allele > other.allele

    def __ge__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.allele >= other.allele


class HLAStandardMatch(HLAStandard):
    mismatch: int

    def __eq__(self, other):
        if type(other) != type(self):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all(
            [
                self.allele == other.allele,
                self.sequence == other.sequence,
                self.mismatch == other.mismatch,
            ]
        )


class HLACombinedStandardResult(BaseModel):
    standard: str
    discrete_allele_names: List[List[str]]

    def __lt__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.standard < other.standard

    def __le__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.standard <= other.standard

    def __gt__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.standard > other.standard

    def __ge__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Cannot compare against {type(other)}")
        return self.standard >= other.standard


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
