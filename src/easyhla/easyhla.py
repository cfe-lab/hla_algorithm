import os
import re
import typer
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Final
from operator import itemgetter, attrgetter

import Bio.SeqIO

from .models import (
    HLAStandard,
    HLAStandardMatch,
    HLACombinedStandardResult,
    HLAResult,
    HLAResultRow,
)

DATE_FORMAT = "%a %b %d %H:%M:%S %Z %Y"


class EasyHLA:
    HLA_A_LENGTH: Final[int] = 787
    MIN_HLA_BC_LENGTH: Final[int] = 787
    MAX_HLA_BC_LENGTH: Final[int] = 796
    EXON2_LENGTH: Final[int] = 270
    EXON3_LENGTH: Final[int] = 276

    ALLOWED_HLA_TYPES: Final[List[str]] = ["A", "B", "C"]

    # A lookup table of translations from ambiguous nucleotides to unambiguous
    # nucleotides.
    AMBIG: Final[Dict[str, List[str]]] = {
        "A": ["A"],
        "C": ["C"],
        "G": ["G"],
        "T": ["T"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "K": ["G", "T"],
        "M": ["A", "C"],
        "S": ["C", "G"],
        "W": ["A", "T"],
        "V": ["C", "G", "T"],
        "H": ["A", "G", "T"],
        "D": ["A", "C", "T"],
        "B": ["A", "C", "G"],
        "N": ["A", "C", "G", "T"],
    }

    # Thanks to binary logic, we encode nucleotide positions as a 4 bit number
    # the first position 000a represents 'A',
    # the second position 00a0 represents 'C',
    # the third position 0a00 represents 'G',
    # the fourth position a000 represents 'T'
    # We can then perform binary ORs, XORs, and ANDs, to check whether or not
    # a mixture contains a specific nucleotide.
    PURENUC2BIN: Final[Dict[str, int]] = {nuc: 2**i for i, nuc in enumerate("ACGT")}

    # Nucleotides converted to their binary representation
    # LISTOFNUCS: List[str] = [
    #     "A",  # => 0b0001,
    #     "C",  # => 0b0010,
    #     "G",  # => 0b0100,
    #     "T",  # => 0b1000,
    #     "M",  # => 0b0011,
    #     "R",  # => 0b0101,
    #     "W",  # => 0b1001,
    #     "S",  # => 0b0110,
    #     "Y",  # => 0b1010,
    #     "K",  # => 0b1100,
    #     "V",  # => 0b1110,
    #     "H",  # => 0b1101,
    #     "D",  # => 0b1011,
    #     "B",  # => 0b0111,
    #     "N",  # => 0b1111
    # ]
    NUC2BIN: Final[Dict[str, int]] = {
        k: sum([{nuc: 2**i for i, nuc in enumerate("ACGT")}[nuc] for nuc in v])
        for k, v in AMBIG.items()
    }
    BIN2NUC: Final[Dict[int, str]] = {v: k for k, v in NUC2BIN.items()}

    COLUMN_IDS: Final[Dict[str, int]] = {"A": 0, "B": 2, "C": 4}

    def __init__(self, letter: str):
        if letter.upper() not in ["A", "B", "C"]:
            raise ValueError("Invalid HLA Type!")
        self.letter: str = letter.upper()
        self.hla_stds: List[HLAStandard] = self.load_hla_stds(letter=self.letter)
        self.hla_freqs: Dict[str, int] = self.load_hla_frequencies(letter=self.letter)

    def check_length(self, letter: str, seq: str, name: str) -> bool:
        error_condition: bool = False
        if name.lower().endswith("short"):
            if letter.upper() == "A":
                error_condition = len(seq) >= EasyHLA.HLA_A_LENGTH
            elif "exon2" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON2_LENGTH
            elif "exon3" in name.lower():
                error_condition = len(seq) >= EasyHLA.EXON3_LENGTH
            else:
                error_condition = len(seq) >= EasyHLA.MAX_HLA_BC_LENGTH
        elif letter.upper() == "A":
            error_condition = len(seq) != EasyHLA.HLA_A_LENGTH
        elif "exon2" in name.lower():
            error_condition = len(seq) != EasyHLA.EXON2_LENGTH
        elif "exon3" in name.lower():
            error_condition = len(seq) != EasyHLA.EXON3_LENGTH
        else:
            error_condition = not (
                EasyHLA.MIN_HLA_BC_LENGTH <= len(seq) <= EasyHLA.MAX_HLA_BC_LENGTH
            )

        if error_condition:
            raise ValueError(
                f"Sequence {name} is the wrong length ({len(seq)} bp). Check the locus {letter}"
            )
        return True

    def check_bases(self, seq: str, name: str) -> bool:
        if not re.match(r"^[ATGCRYKMSWNBDHV]+$", seq):
            raise ValueError(f"Sequence {name} has invalid characters")
        return True

    def nuc2bin(self, seq: str) -> List[int]:
        return [EasyHLA.NUC2BIN.get(seq[i], 0) for i in range(len(seq))]

    def bin2nuc(self, seq: List[int]) -> str:
        return "".join([EasyHLA.BIN2NUC.get(seq[i], "_") for i in range(len(seq))])

    def calc_padding(self, std: List[int], seq: List[int]) -> Tuple[int, int]:
        best = 10e10
        pad = len(std) - len(seq)
        left_pad = 0
        for i in range(pad):
            pseq = self.nuc2bin("N" * i) + seq + self.nuc2bin("N" * (pad - i))
            mismatches = self.std_match(std, pseq)
            if mismatches < best:
                best = mismatches
                left_pad = i

        return left_pad, pad - left_pad

    def pad_short(
        self,
        seq: List[int],
        name: str,
        hla_std: HLAStandard,
    ) -> List[int]:
        # hla_stds expects [ ["label0", [1,2,3,4]], ["label1", [2,3,4,5]] ]
        std = None
        has_intron = False
        if "exon2" in name.lower():
            std = hla_std.sequence[: EasyHLA.EXON2_LENGTH]
        elif "exon3" in name.lower():
            std = hla_std.sequence[EasyHLA.EXON2_LENGTH : EasyHLA.EXON3_LENGTH]
        else:
            has_intron = True
            std = hla_std.sequence

        if has_intron:
            left_pad, _ = self.calc_padding(
                std[: EasyHLA.EXON2_LENGTH], seq[: int(EasyHLA.EXON2_LENGTH / 2)]
            )
            _, right_pad = self.calc_padding(
                std[-int(EasyHLA.EXON3_LENGTH / 2) :],
                seq[-int(EasyHLA.EXON3_LENGTH / 2) :],
            )

        else:
            left_pad, right_pad = self.calc_padding(std, seq)

        return self.nuc2bin("N" * left_pad) + seq + self.nuc2bin("N" * right_pad)

    def std_match(self, std: List[int], seq: List[int]) -> int:
        mismatches = 0
        delta = 0
        for i in range(len(std)):
            if std[i] & seq[i] == 0:
                mismatches += 1
        return mismatches

    def get_matching_stds(
        self, seq: List[int], hla_stds: List[HLAStandard]
    ) -> List[HLAStandardMatch]:
        # Returns [ ["std_name", [1,2,3,4], num_mismatches], ["std_name2", [2,3,4,5], num_mismatches2]]
        matching_stds: List[HLAStandardMatch] = []
        for std in hla_stds:
            mismatches = self.std_match(std.sequence, seq)
            if mismatches < 5:
                matching_stds.append(
                    HLAStandardMatch(
                        allele=std.allele, sequence=std.sequence, mismatch=mismatches
                    )
                )

        return matching_stds

    def combine_stds(
        self,
        matching_stds: List[HLAStandardMatch],
        seq: List[int],
        max_mismatch_threshold: Optional[int],
    ) -> Dict[int, List[HLACombinedStandardResult]]:
        length = len(matching_stds[0].sequence)

        default_min = 9999
        if max_mismatch_threshold is None:
            tmp_max_mismatch_threshold = 0
        else:
            tmp_max_mismatch_threshold = max_mismatch_threshold

        combos: Dict[Any, Any] = {}

        # NOTE: This was a max() comparison in the original code
        computed_minimum_mismatches = max(default_min, tmp_max_mismatch_threshold)

        for std_ai, std_a in enumerate(matching_stds):
            # matching_stds = [ [name, sequence[], threshold] ]
            if std_a.mismatch > computed_minimum_mismatches:
                continue
            for std_bi, std_b in enumerate(matching_stds):
                if std_ai < std_bi:
                    break
                if std_b.mismatch > computed_minimum_mismatches:
                    continue

                std = []
                # Ruby allows lists as keys in hashes, python doesn't

                mismatches = 0
                for i in range(length):
                    if std_b.sequence[i] == std_a.sequence[i]:
                        std.append(std_b.sequence[i])
                    else:
                        std.append(std_b.sequence[i] | std_a.sequence[i])

                    if (std[i] ^ seq[i]) & 15 != 0:
                        mismatches += 1
                    if mismatches > computed_minimum_mismatches:
                        break

                if mismatches <= computed_minimum_mismatches:
                    combined_std_name = "-".join([str(s) for s in std])
                    if mismatches < computed_minimum_mismatches:
                        computed_minimum_mismatches = max(
                            mismatches, tmp_max_mismatch_threshold
                        )
                    if not mismatches in combos:
                        combos[mismatches] = {}
                    if not combined_std_name in combos[mismatches]:
                        combos[mismatches][combined_std_name] = []
                    stds = [std_a.allele, std_b.allele]
                    stds.sort()
                    combos[mismatches][combined_std_name].append(stds)

        result: Dict[int, List[HLACombinedStandardResult]] = {}
        for mismatch, standard in combos.items():
            cur_combo: List[HLACombinedStandardResult] = []
            for std, allele_list in standard.items():
                cur_combo.append(
                    HLACombinedStandardResult(
                        standard=std, discrete_allele_names=allele_list
                    )
                )
            result[mismatch] = cur_combo

        return result

    def load_hla_frequencies(self, letter: str) -> Dict[str, int]:
        hla_freqs: Dict[str, int] = {}
        filepath = os.path.join(os.path.dirname(__file__), "hla_frequencies.csv")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                column_id = EasyHLA.COLUMN_IDS[letter]
                l = line.strip().split(",")[column_id : column_id + 2]
                _l = ",".join([f"{a[:2]}|{a[-2:]}" for a in l])
                if hla_freqs.get(_l, None) is None:
                    hla_freqs[_l] = 0
                hla_freqs[_l] += 1
        return hla_freqs

    # TODO: Convert this to a dictionary instead of a object that looks like:
    # [ [allele_name, [1,2,3,4,5]], [allele_name2, [2,5,2,5,4]] ]
    def load_hla_stds(self, letter: str) -> List[HLAStandard]:
        hla_stds: List[HLAStandard] = []

        filepath = os.path.join(
            os.path.dirname(__file__), f"hla_{letter.lower()}_std_reduced.csv"
        )

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f.readlines():
                l = line.strip().split(",")
                allele = l[0]
                seq = self.nuc2bin((l[1] + l[2]))
                hla_stds.append(HLAStandard(allele=l[0], sequence=seq))
        return hla_stds

    def load_allele_definitions_last_modified_time(self) -> datetime:
        filename = os.path.join(os.path.dirname(__file__), "hla_nuc.fasta.mtime")
        with open(filename, "r", encoding="utf-8") as f:
            last_mod_date = "".join(f.readlines()).strip()
        return datetime.strptime(last_mod_date, DATE_FORMAT)

    def interpret(
        self,
        letter: str,
        entry: Bio.SeqIO.SeqRecord,
        unmatched: List[List[Bio.SeqIO.SeqRecord]],
        threshold: Optional[int] = None,
    ) -> Optional[HLAResult]:
        samp = entry.description

        if not self.check_length(letter, str(entry.seq), samp):
            return None
        if not self.check_bases(str(entry.seq), samp):
            return None

        is_exon = False
        matched = False
        exon2 = ""
        intron = ""
        exon3 = ""

        # Check if the sequence is an exon2 or exon3. If so, try to match it with an
        # existing other exon.
        if "exon" in samp.lower():
            for exon in [2, 3]:
                if f"exon{exon}" in samp.lower():
                    is_exon = True
                    _samp = samp.split("_")[0]
                    for other in unmatched[3 - exon]:
                        if other.description.lower().startswith(_samp):
                            matched = True
                            intron = ""
                            if exon == 2:
                                exon2 = str(entry.seq)
                                exon3 = str(other.seq)
                            else:
                                exon2 = str(other.seq)
                                exon3 = str(entry.seq)

                            unmatched[3 - exon].remove(other)
                            samp = _samp
                            break

                    if not matched:
                        unmatched[exon % 2].append(entry)

        # If it was an exon2 or 3 but didn't have a pair, keep going.
        if is_exon and not matched:
            return None

        if is_exon:
            exon2_bin = self.pad_short(
                self.nuc2bin(exon2), "exon2", hla_std=self.hla_stds[0]
            )
            exon3_bin = self.pad_short(
                self.nuc2bin(exon3), "exon3", hla_std=self.hla_stds[0]
            )
            exon2 = self.bin2nuc(exon2_bin)
            exon3 = self.bin2nuc(exon3_bin)
            seq = exon2_bin + exon3_bin
        else:
            seq = self.pad_short(
                self.nuc2bin(entry.seq), samp, hla_std=self.hla_stds[0]
            )
            exon2 = self.bin2nuc(seq[: EasyHLA.EXON2_LENGTH])
            intron = self.bin2nuc(seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON3_LENGTH])
            exon3 = self.bin2nuc(seq[-EasyHLA.EXON3_LENGTH :])
            seq = seq[: EasyHLA.EXON2_LENGTH] + seq[-EasyHLA.EXON3_LENGTH :]

        matching_stds = self.get_matching_stds(seq, self.hla_stds)
        if len(matching_stds) == 0:
            print(f"Sequence {samp} did not match any known alleles.")
            print("Please check the locus and the orientation.")
            return None

        # Now, combine all the stds (pick up that can citizen!)
        # DR 2023-02-24: To whomever made this comment, great shoutout!
        all_combos = self.combine_stds(matching_stds, seq, threshold)
        all_combos_sorted: List[Tuple[int, List[HLACombinedStandardResult]]] = sorted(
            all_combos.items()
        )
        if threshold:
            for i, combos in all_combos_sorted:
                if i > threshold:
                    if i == 0:
                        print("No matches found below specified threshold.")
                        print("Please heck the locus, orientation, and/or increase")
                        print("number of mismatches.")
                    break
                for cons in combos:
                    for pair in cons.discrete_allele_names:
                        # print(" - ".join(pair))
                        misstrings = []
                        _seq = [int(nuc) for nuc in cons.standard.split("-")]
                        for n in range(len(_seq)):
                            base = EasyHLA.BIN2NUC[seq[n]]
                            if _seq[n] ^ seq[i] != 0:
                                correct_base = EasyHLA.BIN2NUC[_seq[n]]
                                if letter == "A" and n > 270:
                                    dex = n + 242
                                else:
                                    dex = n + 1
                                misstrings.append(f"{dex}:{base}->{correct_base}")
                        # print(";".join(misstrings) + ",")
                        # print(f"{exon2},{intron},{exon3}")

        best_matches = all_combos_sorted[0][1]
        mismatch_count = all_combos_sorted[0][0]

        mishash: Dict[int, List[int]] = {}

        for cons in best_matches:
            _seq = [int(nuc) for nuc in cons.standard.split("-")]
            for i in range(len(_seq)):
                base = EasyHLA.BIN2NUC[seq[i]]
                if _seq[i] ^ seq[i] != 0:
                    correct_base = EasyHLA.BIN2NUC[_seq[i]]
                    if letter == "A" and i > 270:
                        dex = i + 242
                    else:
                        dex = i + 1
                    if not i in mishash:
                        mishash[i] = []
                    if not _seq[i] in mishash[i]:
                        mishash[i].append(_seq[i])

        mislist: List[str] = []

        for m, mlist in mishash.items():
            if letter == "A" and m > 270:
                dex = m + 241
            else:
                dex = m + 1

            base = EasyHLA.BIN2NUC[seq[m]]
            correct_bases = ""
            for correct_bin in mlist:
                if not correct_bases:
                    correct_bases = EasyHLA.BIN2NUC[correct_bin]
                else:
                    correct_bases += "/" + EasyHLA.BIN2NUC[correct_bin]
            mislist.append(f"{dex}:{base}->{correct_bases}")

        # mislist = mislist.sort_by{|b| b.split(":")[0].to_i}
        # mismatches = mislist.join(";")
        mislist.sort(key=lambda item: item.split(":")[0])
        mismatches = ";".join(mislist)

        # Clean the alleles

        fcnt = EasyHLA.COLUMN_IDS[letter]

        ambig, alleles = self.get_alleles(letter=letter, best_matches=best_matches)

        clean_allele_str = self.get_clean_alleles(all_alleles=alleles)

        homozygous, alleles_all_str = self.get_all_alleles(best_matches=best_matches)

        # if this is an exon, then nseqs = 2
        nseqs = 1 + int(is_exon)

        row = HLAResultRow(
            samp=samp,
            clean_allele_str=clean_allele_str,
            alleles_all_str=alleles_all_str,
            ambig=ambig,
            homozygous=homozygous,
            mismatch_count=f"{mismatch_count}",
            mismatches=f"{mismatches}",
            exon2=exon2.upper(),
            intron=intron.upper(),
            exon3=exon3.upper(),
        )
        # print(row)
        return HLAResult(result=row, num_pats=1, num_seqs=nseqs)

    def run(
        self,
        letter: str,
        filename: str,
        output_filename: str,
        threshold: Optional[int] = None,
    ):
        rows = []
        npats = 0
        nseqs = 0
        time_start = datetime.now()
        with open(filename, "r", encoding="utf-8") as f:
            fasta = Bio.SeqIO.parse(f, "fasta")
            unmatched: List[List[Bio.SeqIO.SeqRecord]] = [[], []]
            for i, entry in enumerate(fasta):
                result = self.interpret(
                    letter,
                    entry,
                    unmatched=unmatched,
                    threshold=threshold,
                )
                if not result:
                    continue
                else:
                    rows.append(result.result.get_result_as_str())
                    npats += result.num_pats
                    nseqs += result.num_seqs

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(
                f"Run commencing {time_start.strftime(DATE_FORMAT)}. Allele definitions last updated {self.load_allele_definitions_last_modified_time().strftime(DATE_FORMAT)}.\n"
            )
            f.write(
                "ENUM,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3\n"
            )
            for _r in rows:
                f.write(_r + "\n")

    def get_all_alleles(
        self, best_matches: List[HLACombinedStandardResult]
    ) -> Tuple[bool, str]:
        # OK, now we must find homozygousity.	IE: Cw*0722 - Cw*0722
        homozygous = False
        alleles_all = []
        # Lets say if we detect two of the same mixtures, its heterozygous
        for a in best_matches:
            for _allele in a.discrete_allele_names:
                alleles_all.append(f"{_allele[0]} - {_allele[1]}")
                if _allele[0] == _allele[1]:
                    homozygous = True

        alleles_all.sort()
        alleles_all_str = ";".join(alleles_all)

        if len(alleles_all_str) > 3900:
            alleles_all_str = re.sub(
                r";[^;]+$", ";...TRUNCATED", alleles_all_str[:3920]
            )

        return homozygous, alleles_all_str

    def get_clean_alleles(self, all_alleles: List[List[str]]) -> str:
        # non ambiguous now, do the easy way

        collection = [
            [a[0].strip().split(":"), a[1].strip().split(":")] for a in all_alleles
        ]

        print(all_alleles)

        clean_allele: List[str] = []
        for n in [0, 1]:
            for i in [4, 3, 2, 1]:
                if len(set([":".join(a[n][0:i]) for a in collection])) == 1:
                    clean_allele.append(
                        re.sub(r"[A-Z]$", "", ":".join(collection[0][n][0:i]))
                    )
                    break

        clean_allele_str: str = " - ".join(clean_allele)
        print(clean_allele_str)
        return clean_allele_str

    def get_alleles(
        self, letter: str, best_matches: List[HLACombinedStandardResult]
    ) -> Tuple[bool, List[List[str]]]:
        alleles: List[List[str]] = []
        ambig = False
        for match in best_matches:
            for list_a_name in match.discrete_allele_names:
                alleles.append(list_a_name)

        # Strip leading "A:" , "B:", or "C:" from each allele.
        collection: List[List[List[str]]] = []
        for a in alleles:
            arr = [
                re.sub(r"[^\d:]", "", a[0]).split(":"),
                re.sub(r"[^\d:]", "", a[1]).split(":"),
            ]
            collection.append(arr)

        uniq_collection = set([f"{e[0][0]}, {e[1][0]}" for e in collection])
        # ambig_collection = {[e[0][0:1], e[1][0:1]] for e in collection}

        if len(uniq_collection) != 1:
            ambig = True
            collection_ambig = {
                f"{'|'.join(e[0][0:2])},{'|'.join(e[1][0:2])}": 0 for e in collection
            }

            for k in collection_ambig:
                for freq in self.hla_freqs:
                    if freq.startswith(k):
                        collection_ambig[k] = self.hla_freqs.get(freq, 0)
                # if not freq:
                #     freq = 0
                # a.append(freq)

            # TODO: Implement like the following commented ruby
            # Easier if we made things a model.
            def sort_allele(item: Tuple[str, int]):
                """
                Produces a tuple that the sort function will use to determine
                the maximum allele
                """
                allele_pair0 = item[0].split(",")[0]
                allele_pair1 = item[0].split(",")[1]
                return (
                    item[1],
                    int(allele_pair0.split("|")[0]),
                    int(allele_pair0.split("|")[1]),
                    int(allele_pair1.split("|")[0]),
                    int(allele_pair1.split("|")[1]),
                )

            max_allele = sorted(collection_ambig.items(), key=sort_allele, reverse=True)

            # Try to find the allele occuring the maximum number of times. If it's a tie,
            # just pick the alphabetically first one.
            # max_allele = collection_ambig.max do |a,b|
            #     if(a[2] != b[2]) #Go by frequency
            #         a[2] <=> b[2]
            #     elsif(b[0][0].to_i != a[0][0].to_i) #Then lowest first allele
            #         b[0][0].to_i <=> a[0][0].to_i
            #     elsif(b[0][1].to_i != a[0][1].to_i)
            #         b[0][1].to_i <=> a[0][1].to_i
            #     elsif(b[1][0].to_i != a[1][0].to_i) #Then lowest second allele
            #         b[1][0].to_i <=> a[1][0].to_i
            #     else
            #         b[1][1].to_i <=> a[1][1].to_i
            #     end
            # end

            a1 = max_allele[0][0].split(",")[0]
            a2 = max_allele[0][0].split(",")[1]
            for i, a in enumerate(alleles.copy()):
                regex_str_a1 = f"^{letter}\\*({a1}):([^\\s])+"
                if not re.match(regex_str_a1, a[0]) and a in alleles:
                    alleles.remove(a)
                regex_str_a2 = f"^{letter}\\*({a2}):([^\\s])+"
                if not re.match(regex_str_a2, a[1]) and a in alleles:
                    alleles.remove(a)

        return ambig, alleles


if __name__ == "__main__":
    input_file = "tests/input/test.fasta"
    output_file = "tests/output/test.csv"

    easyhla = EasyHLA("A")

    easyhla.run(
        easyhla.letter,
        input_file,
        output_file,
        0,
    )
