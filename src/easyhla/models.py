from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple


class Exon(BaseModel):
    two: str
    intron: str = ""
    three: str


class HLASequence(BaseModel):
    exon: Exon
    sequence: List[int]
