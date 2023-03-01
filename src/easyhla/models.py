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
        if type(other) != type(self):
            raise TypeError(f"Cannot compare against {type(other)}")
        return all([self.allele == other.allele, self.sequence == other.sequence])


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
