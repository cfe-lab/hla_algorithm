"""
This implements the main interpret loop of EasyHLA.
"""

import os
import re
import typer
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Union
from operator import itemgetter, attrgetter

from .easyhla import EasyHLA
from .models import Exon, HLASequence

import Bio.SeqIO


class EasyHLAInterpreter(EasyHLA):
    def __init__(self, letter: str):
        self.letter = letter
        self.unmatched_sequences: List[List[Bio.SeqIO.SeqRecord]] = [[], []]
        self.nseqs: int = 0
        self.npats: int = 0

    def interpret(
        self, entry: Bio.SeqIO.SeqRecord, threshold: Optional[int] = None
    ) -> Optional[Any]:
        pass

    def match_exons(
        self, samp: str, entry: Bio.SeqIO.SeqRecord
    ) -> Optional[Tuple[str, str]]:
        is_exon: bool = False
        matched: bool = False
        exon2: str = ""
        intron: str = ""
        exon3: str = ""

        # Check if the sequence is an exon2 or exon3. If so, try to match it with an
        # existing other exon.
        for exon in [2, 3]:
            if f"exon{exon}" in samp.lower():
                is_exon = True
                samp = samp.split("_")[0]
                for other in self.unmatched_sequences[3 - exon]:
                    if other.description.lower().startswith(samp):
                        matched = True
                        intron = ""
                        if exon == 2:
                            exon2 = entry.seq
                            exon3 = other.seq
                        else:
                            exon2 = other.seq
                            exon3 = entry.seq
                        self.unmatched_sequences[3 - exon].remove(other)
                        break
                if not matched:
                    self.unmatched_sequences[exon % 2].append(entry)

        # If it was an exon2 or 3 but didn't have a pair, keep going.
        if is_exon and not matched:
            return None

        return Exon(two=exon2, three=exon3, intron=intron)

    def get_sequence(
        self,
        entry: Bio.SeqIO.SeqRecord,
        exon: Optional[Exon] = None,
    ) -> HLASequence:
        seq: List[int] = []

        if exon:
            exon2_bin = self.pad_short(
                self.letter, self.nuc2bin(exon.two), "exon2", hla_stds=self.hla_stds
            )
            exon3_bin = self.pad_short(
                self.letter, self.nuc2bin(exon.three), "exon3", hla_stds=self.hla_stds
            )
            exon2 = self.bin2nuc(exon2_bin)
            exon3 = self.bin2nuc(exon3_bin)
            seq = exon2_bin + exon3_bin
            return HLASequence(exon=exon, seq=seq)

        seq = self.pad_short(self.letter, self.nuc2bin(entry.seq), "", self.hla_stds)
        exon2 = self.bin2nuc(seq[: EasyHLA.EXON2_LENGTH])
        intron = self.bin2nuc(seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON2_LENGTH])
        exon3 = self.bin2nuc(seq[-EasyHLA.EXON3_LENGTH :])
        seq = seq[: EasyHLA.EXON2_LENGTH] + seq[-EasyHLA.EXON3_LENGTH :]

        return HLASequence(exon=Exon(two=exon2, intron=intron, three=exon3), seq=seq)

    def find_below_threshold(
        self, threshold: int, seq: HLASequence, all_combos: List[Any]
    ):
        if threshold:
            for i, combos in enumerate(all_combos):
                if combos[0] > threshold:
                    if i == 0:
                        print("No matches found below specified threshold.")
                        print("Please heck the locus, orientation, and/or increase")
                        print("number of mismatches.")
                    return None
                for cons in combos[1]:
                    for pair in cons[1]:
                        print(" - ".join(pair))
                        misstrings = []
                        for n in range(len(cons[0])):
                            base = EasyHLA.BIN2NUC[seq.seq[n]]
                            if cons[0][n] ^ seq.seq[i] != 0:
                                correct_base = EasyHLA.BIN2NUC[cons[0][n]]
                                if self.letter == "A" and n > 270:
                                    dex = n + 242
                                else:
                                    dex = n + 1
                                misstrings.append(f"{dex}:{base}->{correct_base}")
                        print(";".join(misstrings) + ",")
                        print(f"{seq.exon.two},{seq.exon.intron},{seq.exon.three}")

    def assemble_mishash(
        self, seq: HLASequence, best_matches: List[List[int]]
    ) -> Dict[int, List[int]]:
        mishash: Dict[int, List[Any]] = {}
        for cons in best_matches:
            for i in range(len(cons)):
                base = EasyHLA.BIN2NUC[seq.sequence[i]]
                if cons[0][i] ^ seq[i] != 0:
                    correct_base = EasyHLA.BIN2NUC[cons[0][i]]
                    if self.letter == "A" and i > 270:
                        dex = i + 242
                    else:
                        dex = i + 1
                    if mishash[i] is None:
                        mishash[i] = []
                    if not cons[0][i] in mishash[i]:
                        mishash[i].append(cons[0][i])

        return mishash

    def find_mislist(self, seq: HLASequence, mishash: Dict[int, List[int]]):
        mislist: List[str] = []

        for m in mishash.values():
            if self.letter == "A" and m[0] > 270:
                dex = m[0] + 241
            else:
                dex = m[0] + 1

            base = EasyHLA.BIN2NUC[seq[m[0]]]
            correct_bases = ""
            for correct_bin in m[1]:
                if not correct_bases:
                    correct_bases = EasyHLA.BIN2NUC[correct_bin]
                else:
                    correct_bases += "/" + EasyHLA.BIN2NUC[correct_bin]
            mislist.append(f"{dex}:{base}->{correct_bases}")

        return mislist
