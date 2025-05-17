#! /usr/bin/env python

import fileinput
import json
import os
from typing import Final, Optional

from pydantic import BaseModel, Field

from easyhla.easyhla import EasyHLA
from easyhla.models import (
    AllelePairs,
    HLACombinedStandard,
    HLAMatchDetails,
    HLAInterpretation,
    HLASequence,
)
from easyhla.utils import (
    HLA_LOCUS,
    check_bases,
    check_length,
    nuc2bin,
    BadLengthException,
    InvalidBaseException,
)


# These are the "configuration files" that the algorithm uses; these are or may
# be updated, in which case you specify the path to the new version in the
# environment.
HLA_STANDARDS: Final[dict[HLA_LOCUS, Optional[str]]] = {
    "A": os.environ.get("HLA_STANDARDS_A"),
    "B": os.environ.get("HLA_STANDARDS_B"),
    "C": os.environ.get("HLA_STANDARDS_C"),
}
HLA_FREQUENCIES: Final[str] = os.environ.get("HLA_FREQUENCIES")


class HLAInput(BaseModel):
    seq1: str
    seq2: str = ""
    locus: HLA_LOCUS

    def check_sequences(self) -> list[str]:
        errors: list[str] = []
        if self.locus == "A":
            if self.seq2 != "":
                errors.append("Wrong number of sequences (needs 1)")
            else:
                try:
                    check_length("A", self.seq1, "")
                except BadLengthException:
                    errors.append("Sequence is the wrong size (should be 787)")
        else:
            if self.seq2 == "":
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
            exon3_str = self.seq2

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
    allele_for_mismatches: str = ""
    mismatches: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    homozygous: bool = False
    type: HLA_LOCUS = "B"
    alg_version: str = "foo"
    b5701: bool = False
    dist_b5701: int = 0
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def build_from_interpretation(cls, interp: HLAInterpretation) -> "HLAResult":
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

        # FIXME consider if we should change this format, and unify it with
        # the clinical driver
        return HLAResult(
            seqs=seqs,
            alleles_all=aps.stringify(),
            alleles_clean=alleles_clean,
            allele_for_mismatches=f"{rep_ap[0]} - {rep_ap[1]}",
            mismatches=[str(x) for x in match_details.mismatches],
            ambiguous=aps.is_ambiguous(),
            homozygous=aps.is_homozygous(),
            type=interp.locus,
            alg_version="FOO FIXME",
            b5701=interp.is_b5701(),
            dist_b5701=interp.distance_from_b7501(),
            errors=[],
        )



def main():
    hla_input_str: str = ""
    with fileinput.input() as f:
        hla_input_str = f.read()

    hla_input: HLAInput = HLAInput(json.loads(hla_input_str))

    errors: list[str] = hla_input.check_sequences()
    if len(errors) > 0:
        error_result: HLAResult = HLAResult(errors=errors)
        print(error_result.model_dump_json())
    else:
        interp: HLAInterpretation = EasyHLA.interpret(
            hla_input.hla_sequence()
        )
        print(HLAResult.build_from_interpretation(interp).model_dump_json())


if __name__ == "__main__":
    main()
