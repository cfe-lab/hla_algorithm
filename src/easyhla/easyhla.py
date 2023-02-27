import os
import re
import typer
import logging
from datetime import datetime
from typing import List, Optional, Dict, Literal, Tuple, Any, Union
from operator import itemgetter, attrgetter

import Bio.SeqIO


class EasyHLA:
    HLA_A_LENGTH: int = 787
    MIN_HLA_BC_LENGTH: int = 787
    MAX_HLA_BC_LENGTH: int = 796
    EXON2_LENGTH: int = 270
    EXON3_LENGTH: int = 276

    ALLOWED_HLA_TYPES: List[str] = ["A", "B", "C"]

    # A lookup table of translations from ambiguous nucleotides to unambiguous
    # nucleotides.
    AMBIG: Dict[str, List[str]] = {
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
    PURENUC2BIN: Dict[str, int] = {nuc: 2**i for i, nuc in enumerate("ACGT")}

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
    NUC2BIN: Dict[str, int] = {
        k: sum([{nuc: 2**i for i, nuc in enumerate("ACGT")}[nuc] for nuc in v])
        for k, v in AMBIG.items()
    }
    BIN2NUC: Dict[int, str] = {v: k for k, v in NUC2BIN.items()}

    COLUMN_IDS: Dict[str, int] = {"A": 0, "B": 2, "C": 4}

    def __init__(self, letter: Literal["A", "B", "C"]):
        self.hla_stds = self.load_hla_stds(letter=letter)
        self.hla_freqs = self.load_hla_frequencies(letter=letter)

    def check_length(self, letter: Literal["A", "B", "C"], seq: str, name: str) -> bool:
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
        if re.match(r"(?i)^[atgcrykmswnbdhv]+$", seq):
            raise ValueError(f"Sequence {name} has invalid characters")
        return True

    def nuc2bin(self, seq: str) -> List[int]:
        return [EasyHLA.NUC2BIN[seq[i]] for i in range(len(seq))]

    def bin2nuc(self, seq: List[int]) -> str:
        return "".join([EasyHLA.BIN2NUC[seq[i]] for i in range(len(seq))])

    def calc_padding(self, std: List[int], seq: List[int]) -> Tuple[int, int]:
        best = 10e10
        pad = len(std) - len(seq)
        left_pad = 0
        for i in range(pad):
            pseq = self.nuc2bin("n" * i) + seq + self.nuc2bin("n" * (pad - i))
            mismatches = self.std_match(std, pseq)
            if mismatches < best:
                best = mismatches
                left_pad = i

        return left_pad, pad - left_pad

    def pad_short(
        self,
        letter: Literal["A", "B", "C"],
        seq: List[int],
        name: str,
        hla_stds: List[Any],
    ) -> List[int]:
        std = None
        has_intron = False
        if "exon2" in name.lower():
            std = hla_stds[0][1][: EasyHLA.EXON2_LENGTH]
        elif "exon3" in name.lower():
            std = hla_stds[0][1][EasyHLA.EXON2_LENGTH : EasyHLA.EXON3_LENGTH]
        else:
            has_intron = True
            std = hla_stds[0][1]

        if has_intron:
            left_pad, _ = self.calc_padding(
                std[: EasyHLA.EXON2_LENGTH], seq[: int(EasyHLA.EXON2_LENGTH / 2)]
            )
            _, right_pad = self.calc_padding(
                std[-EasyHLA.EXON3_LENGTH :], seq[int(-EasyHLA.EXON3_LENGTH / 2) :]
            )

        else:
            left_pad, right_pad = self.calc_padding(std, seq)

        return self.nuc2bin("n" * left_pad) + seq + self.nuc2bin("n" * right_pad)

    def std_match(self, std: List[int], seq: List[int]) -> int:
        mismatches = 0
        delta = 0
        for i in range(len(std)):
            if std[i] & seq[i] == 0:
                mismatches += 1
        return mismatches

    def get_matching_stds(
        self, seq: List[int], hla_stds: List[List[Any]]
    ) -> List[List[Union[str, int]]]:
        matching_stds: List[List[Union[str, int]]] = []
        for std in hla_stds:
            allele, std_seq = std
            mismatches = self.std_match(std_seq, seq)
            if mismatches < 5:
                matching_stds.append([allele, std_seq, mismatches])

        return matching_stds

    def combine_stds(
        self, matching_stds: List[List[Any]], seq: List[int], threshold: Optional[int]
    ) -> Any:
        alleles_hash = {}
        length = len(matching_stds[0][1])

        min = 9999
        if threshold is None:
            tmp_threshold = 0
        else:
            tmp_threshold = threshold

        combos: Dict[Any, Any] = {}

        for std_ai, std_a in enumerate(matching_stds):
            if std_a[2] > max(min, tmp_threshold):
                continue
            for std_bi, std_b in enumerate(matching_stds):
                if std_ai < std_bi:
                    break
                if std_b[2] > max(min, tmp_threshold):
                    continue

                std = []

                mismatches = 0
                for i in range(length):
                    if std_b[1][i] == std_a[1][i]:
                        std.append(std_b[1][i])
                    else:
                        std.append(std_b[1][i] | std_a[1][i])

                    if std[i] ^ seq[i] & 15 != 0:
                        mismatches += 1
                    if mismatches > max(min, tmp_threshold):
                        break

                if mismatches <= max(min, tmp_threshold):
                    if mismatches < min:
                        min = mismatches
                    if combos.get(mismatches, None) is None:
                        combos[mismatches] = {}
                    if combos[mismatches].get(std, None) is None:
                        combos[mismatches][std] = []
                    stds = [std_a[0], std_b[0]].sort()
                    combos[mismatches][std].append(stds)

        result = []
        for c in combos:
            cur_combo = []
            for std, allele_list in c[1]:
                cur_combo.append([std, allele_list])
            result.append(cur_combo)

        return result.sort()

    def load_hla_frequencies(self, letter: Literal["A", "B", "C"]) -> Dict[str, int]:
        hla_freqs: Dict[str, int] = {}
        filepath = os.path.join(
            os.path.dirname(__file__), f"hla_{letter.lower()}_std_reduced.csv"
        )

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
    def load_hla_stds(self, letter: Literal["A", "B", "C"]) -> List[List[Any]]:
        hla_stds = []

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
                hla_stds.append([l[0], seq])
        return hla_stds

    def load_allele_definitions_last_modified_time(self) -> datetime:
        return datetime.fromtimestamp(os.path.getmtime("hla_nuc.fasta.mtime"))

    def interpret(
        self,
        letter: Literal["A", "B", "C"],
        entry: Bio.SeqIO.SeqRecord,
        threshold: Optional[int] = None,
    ) -> Optional[Tuple[str, int, int]]:
        samp = entry.description

        if not self.check_length(letter, entry.seq, samp):
            return None
        if not self.check_length(letter, entry.seq, samp):
            return None

        is_exon = False
        matched = False
        exon2 = ""
        intron = ""
        exon3 = ""

        unmatched: List[List[Bio.SeqIO.SeqRecord]] = [[], []]
        # Check if the sequence is an exon2 or exon3. If so, try to match it with an
        # existing other exon.
        for exon in [2, 3]:
            if f"exon{exon}" in samp.lower():
                is_exon = True
                samp = samp.split("_")[0]
                for other in unmatched[3 - exon]:
                    if other.description.lower().startswith(samp):
                        matched = True
                        intron = ""
                        if exon == 2:
                            exon2 = entry.seq
                            exon3 = other.seq
                        else:
                            exon2 = other.seq
                            exon3 = entry.seq
                        unmatched[3 - exon].remove(other)
                        break
                if not matched:
                    unmatched[exon % 2].append(entry)

        # If it was an exon2 or 3 but didn't have a pair, keep going.
        if is_exon and not matched:
            return None

        if is_exon:
            exon2_bin = self.pad_short(
                letter, self.nuc2bin(exon2), "exon2", hla_stds=self.hla_stds
            )
            exon3_bin = self.pad_short(
                letter, self.nuc2bin(exon3), "exon3", hla_stds=self.hla_stds
            )
            exon2 = self.bin2nuc(exon2_bin)
            exon3 = self.bin2nuc(exon3_bin)
            seq = exon2_bin + exon3_bin
        else:
            seq = self.pad_short(letter, self.nuc2bin(entry.seq), "", self.hla_stds)
            exon2 = self.bin2nuc(seq[: EasyHLA.EXON2_LENGTH])
            intron = self.bin2nuc(seq[EasyHLA.EXON2_LENGTH : -EasyHLA.EXON2_LENGTH])
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

        if threshold:
            for i, combos in enumerate(all_combos):
                if combos[i] > threshold:
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
                            base = EasyHLA.BIN2NUC[seq[n]]
                            if cons[0][n] ^ seq[i] != 0:
                                correct_base = EasyHLA.BIN2NUC[cons[0][n]]
                                if letter == "A" and n > 270:
                                    dex = n + 242
                                else:
                                    dex = n + 1
                                misstrings.append(f"{dex}:{base}->{correct_base}")
                        print(";".join(misstrings) + ",")
                        print(f"{exon2},{intron},{exon3}")

        best_matches = all_combos[0][1]
        mismatch_count = all_combos[0][0]

        mishash: Dict[int, List[Any]] = {}

        for cons in best_matches:
            for i in range(len(cons)):
                base = EasyHLA.BIN2NUC[seq[i]]
                if cons[0][n] ^ seq[i] != 0:
                    correct_base = EasyHLA.BIN2NUC[cons[0][n]]
                    if letter == "A" and n > 270:
                        dex = n + 242
                    else:
                        dex = n + 1
                    if mishash[i] is None:
                        mishash[i] = []
                    if not cons[0][i] in mishash[i]:
                        mishash[i].append(cons[0][i])

        mislist: List[str] = []

        for m in mishash.values():
            if letter == "A" and m[0] > 270:
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

        # mislist = mislist.sort_by{|b| b.split(":")[0].to_i}
        # mismatches = mislist.join(";")
        mislist.sort()
        [int(b.split(":")[0]) for b in mislist]
        mismatches = ";".join(mislist)

        # Clean the alleles

        fcnt = EasyHLA.COLUMN_IDS[letter]

        clean_allele = ""
        alleles = []
        ambig = "0"

        for match in best_matches:
            alleles.append(match[1])

        # Strip leading "A:" , "B:", or "C:" from each allele.
        collection = []
        for a in alleles:
            arr = [
                re.sub(r"[^\d:]", "", a[0]).split(":"),
                re.sub(r"[^\d:]", "", a[1]).split(":"),
            ]
            collection.append(arr)

        if len({[e[0][0], e[1][0]] for e in collection}) != 1:
            ambig = "1"
            collection_ambig = {[e[0][0:1], e[1][0:1]] for e in collection}

            for a in collection_ambig:
                freq = self.hla_freqs.get(a, None)
                if not freq:
                    freq = 0
                a.append(freq)

            # TODO: Implement like the following commented ruby
            # Easier if we made things a model.
            max_allele = sorted(collection_ambig, key=itemgetter(2))

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

            a1 = max_allele[0][0]
            a2 = max_allele[1][0]
            for i, a in enumerate(alleles):
                if not re.match(f"^#{letter}\\*#{a1}:([^\\s])+", a[0]):
                    alleles.pop(i)
                if not re.match(f"^#{letter}\\*#{a2}:([^\\s])+", a[1]):
                    alleles.pop(i)

        # non ambiguous now, do the easy way

        collection = [
            [a[0].strip().split(":"), a[1].strip().split(":")] for a in alleles
        ]

        for n in [0, 1]:
            for i in range(4, 1, -1):
                if len({[e[n][0:i]] for e in collection}) != 1:
                    clean_allele += (
                        re.sub(r"[A-Z]$", "", ":".join(collection[0][n][0:i])) + " - "
                    )
                    break

        # OK, now we must find homozygousity.	IE: Cw*0722 - Cw*0722
        homozygous = "0"
        # Lets say if we detect two of the same mixtures, its heterozygous
        for a in best_matches:
            for allele in a[1]:
                if allele[0] == allele[1]:
                    homozygous = "1"

        alleles_all = []

        for _, alleles in best_matches:
            for a in alleles:
                alleles_all.append(f"{a} - {a}")

        alleles_all_str = ";".join(alleles_all)

        if len(alleles_all_str) > 3900:
            alleles_all_str = re.sub(
                r";[^;]+$", ";...TRUNCATED", alleles_all_str[:3920]
            )

        nseqs = 1
        if is_exon:
            nseqs += 2

        row = [
            samp,
            clean_allele,
            alleles_all_str,
            ambig,
            homozygous,
            mismatch_count,
            mismatches,
            exon2.upper(),
            intron.upper(),
            exon3.upper(),
        ]
        return ",".join(row), 1, nseqs

    def run(
        self,
        letter: Literal["A", "B", "C"],
        filename: str,
        threshold: Optional[int] = None,
    ):
        rows = []
        npats = 0
        nseqs = 0
        with open(filename, "r", encoding="utf-8") as f:
            fasta = Bio.SeqIO.parse(f, "fasta")
            for entry in fasta:
                result = self.interpret(
                    letter,
                    entry,
                    threshold=threshold,
                )
                if not result:
                    continue
                else:
                    rows.append(result[0])
                    npats += result[1]
                    nseqs += result[2]
