from typing import Optional

from pydantic import BaseModel, Field

from ._version import __version__
from .models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMatchDetails,
    HLASequence,
)
from .utils import (
    HLA_LOCUS,
    BadLengthException,
    InvalidBaseException,
    check_bases,
    check_length,
    nuc2bin,
)


class HLAInput(BaseModel):
    seq1: str
    seq2: Optional[str]
    locus: HLA_LOCUS
    hla_std_path: Optional[str] = None
    hla_freq_path: Optional[str] = None

    def check_sequences(self) -> list[str]:
        errors: list[str] = []
        if self.locus == "A":
            if self.seq2 is not None:
                errors.append("Wrong number of sequences (needs 1)")
            else:
                try:
                    check_length("A", self.seq1, "")
                except BadLengthException:
                    errors.append("Sequence is the wrong size (should be 787)")
        else:
            if self.seq2 is None:
                errors.append("Wrong number of sequences (needs 2)")
            else:
                try:
                    check_length(self.locus, self.seq1, "exon2")
                except BadLengthException:
                    errors.append("Sequence 1 is the wrong size (should be 270)")
                try:
                    check_length(self.locus, self.seq2, "exon3")
                except BadLengthException:
                    errors.append("Sequence 2 is the wrong size (should be 276)")
        try:
            check_bases(self.seq1)
            if self.seq2 is not None:
                check_bases(self.seq2)
        except InvalidBaseException:
            errors.append("Sequence has invalid characters")
        return errors

    def hla_sequence(self) -> HLASequence:
        exon2_str: str = ""
        exon3_str: str = ""
        intron_str: str = ""

        if self.locus == "A":
            exon2_str = self.seq1[:270]
            intron_str = self.seq1[270:-276]
            exon3_str = self.seq1[-276:]
        else:
            exon2_str = self.seq1
            exon3_str = self.seq2 or ""

        num_sequences_used: int = 1 if self.locus == "A" else 2
        return HLASequence(
            two=nuc2bin(exon2_str),
            intron=nuc2bin(intron_str),
            three=nuc2bin(exon3_str),
            name="input_sequence",
            locus=self.locus,
            num_sequences_used=num_sequences_used,
        )


class HLAResult(BaseModel):
    seqs: list[str] = Field(default_factory=list)
    alleles_all: list[str] = Field(default_factory=list)
    alleles_clean: str = ""
    alleles_for_mismatches: str = ""
    mismatches: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    homozygous: bool = False
    locus: HLA_LOCUS = "B"
    alg_version: str = __version__
    alleles_version: str = ""
    b5701: bool = False
    dist_b5701: Optional[int] = None
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def build_from_interpretation(
        cls, interp: HLAInterpretation, alleles_version: str
    ) -> "HLAResult":
        aps: AllelePairs = interp.best_matching_allele_pairs()

        # Pick one of the combined standards represented by what goes into
        # "alleles_clean" and report the mismatches coming from that.
        rep_ap: tuple[str, str]
        alleles_clean: str
        rep_cs: HLACombinedStandard
        rep_ap, alleles_clean, rep_cs = interp.best_common_allele_pair()

        match_details: HLAMatchDetails = interp.matches[rep_cs]
        hla_seq: HLASequence = interp.hla_sequence
        seqs: list[str] = []
        if interp.locus == "A":
            seqs.append(hla_seq.exon2_str + hla_seq.intron_str + hla_seq.exon3_str)
        else:
            seqs.append(hla_seq.exon2_str)
            seqs.append(hla_seq.exon3_str)

        return HLAResult(
            seqs=seqs,
            alleles_all=[f"{x[0]} - {x[1]}" for x in aps.sort_pairs()],
            alleles_clean=alleles_clean,
            alleles_for_mismatches=f"{rep_ap[0]} - {rep_ap[1]}",
            mismatches=[str(x) for x in match_details.mismatches],
            ambiguous=aps.is_ambiguous(),
            homozygous=aps.is_homozygous(),
            locus=interp.locus,
            alleles_version=alleles_version,
            b5701=interp.is_b5701(),
            dist_b5701=interp.distance_from_b7501(),
        )
