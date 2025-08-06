from collections.abc import Iterable
from operator import attrgetter, itemgetter
from typing import TypedDict

import numpy as np
from Bio.Seq import MutableSeq, Seq
from Bio.SeqIO import SeqRecord
from pydantic import BaseModel

from .models import (
    AllelePairs,
    HLACombinedStandard,
    HLAInterpretation,
    HLAMismatch,
    HLASequence,
    HLAStandard,
)
from .utils import (
    EXON2_LENGTH,
    EXON3_LENGTH,
    EXON_NAME,
    HLA_LOCUS,
    BadLengthException,
    InvalidBaseException,
    check_bases,
    check_length,
    nuc2bin,
    pad_short,
)

EXON_AND_OTHER_EXON: list[tuple[EXON_NAME, EXON_NAME]] = [
    ("exon2", "exon3"),
    ("exon3", "exon2"),
]


def pair_exons_helper(
    sequence_record: SeqRecord,
    unmatched: dict[EXON_NAME, dict[str, Seq | MutableSeq | None]],
) -> tuple[str, bool, bool, str, str]:
    """
    Helper that attempts to match the given sequence with a "partner" exon.

    `sequence_record` represents a sequence that may be an exon2 or exon3
    sequence (or neither).  It determines which of these cases it is by
    examining its `id` string; then it either finds a partner for it
    from `unmatched`, or adds it to `unmatched`.

    Returns None if it cannot find a match; otherwise, it returns a tuple
    containing:
    - identifier
    - is exon?  (True/False)
    - did we find a match?  (True/False)
    - exon2 sequence
    - exon3 sequence
    """
    # The `id`` field is expected to hold the sample name.
    samp: str = sequence_record.id or ""
    is_exon: bool = False
    matched: bool = False
    exon2: str = ""
    exon3: str = ""
    identifier: str = samp

    # Check if the sequence is an exon2 or exon3. If so, try to match it
    # with an existing other exon.
    for exon, other_exon in EXON_AND_OTHER_EXON:
        if exon in samp.lower():
            is_exon = True
            identifier = samp.split("_")[0]
            for other_id, other_seq in unmatched[other_exon].items():
                if identifier.lower() in other_id.lower():
                    matched = True
                    if exon == "exon2":
                        exon2 = str(sequence_record.seq)
                        exon3 = str(other_seq)
                    else:
                        exon2 = str(other_seq)
                        exon3 = str(sequence_record.seq)

                    unmatched[other_exon].pop(other_id)
                    break
            # If we can't match the exon, put the entry in the list we
            # weren't looking in.
            if not matched:
                unmatched[exon][samp] = sequence_record.seq

    return (
        identifier,
        is_exon,
        matched,
        exon2,
        exon3,
    )


def pair_exons(
    sequence_records: Iterable[SeqRecord],
    locus: HLA_LOCUS,
    example_standard: HLAStandard,
) -> tuple[list[HLASequence], dict[EXON_NAME, dict[str, Seq | MutableSeq | None]]]:
    """
    Pair exons in the given input sequences.

    The section of HLA we sequence looks like
    exon2 - intron - exon3
    and is typically sequenced in two parts, one covering exon2 and exon3
    (the intron is not used in our testing).  We iterate through the
    sequences and attempt to match them up.
    """
    matched_sequences: list[HLASequence] = []
    unmatched: dict[EXON_NAME, dict[str, Seq | MutableSeq | None]] = {
        "exon2": {},
        "exon3": {},
    }

    for sr in sequence_records:
        # Skip over any sequences that aren't the right length or contain
        # bad bases.
        try:
            check_length(locus, str(sr.seq), sr.id or "")
        except BadLengthException:
            continue

        try:
            check_bases(str(sr.seq))
        except InvalidBaseException:
            continue

        is_exon: bool = False
        matched: bool = False
        exon2: str = ""
        exon3: str = ""
        identifier: str = ""

        identifier, is_exon, matched, exon2, exon3 = pair_exons_helper(
            sr,
            unmatched,
        )

        # If it was an exon2 or 3 but didn't have a pair, keep going.
        if is_exon and not matched:
            continue

        if is_exon:
            exon2_bin = pad_short(example_standard.sequence, nuc2bin(exon2), "exon2")
            exon3_bin = pad_short(example_standard.sequence, nuc2bin(exon3), "exon3")
            matched_sequences.append(
                HLASequence(
                    two=tuple(int(x) for x in exon2_bin),
                    intron=(),
                    three=tuple(int(x) for x in exon3_bin),
                    name=identifier,
                    locus=locus,
                    num_sequences_used=2,
                )
            )
        else:
            seq_numpy: np.ndarray = pad_short(
                example_standard.sequence,
                nuc2bin(sr.seq),  # type: ignore
                None,
            )
            seq: tuple[int, ...] = tuple(int(x) for x in seq_numpy)
            matched_sequences.append(
                HLASequence(
                    two=seq[:EXON2_LENGTH],
                    intron=seq[EXON2_LENGTH:-EXON3_LENGTH],
                    three=seq[-EXON3_LENGTH:],
                    name=identifier,
                    locus=locus,
                    num_sequences_used=1,
                )
            )
    return matched_sequences, unmatched


class CombinedMismatch(TypedDict):
    observed_base: str
    expected_bases: set[str]


class HLAInterpretationRow(BaseModel):
    """
    Represents the table row summarizing an HLAInterpretation.
    """

    enum: str = ""
    alleles_clean: str = ""
    alleles: str = ""
    ambiguous: int = 0
    homozygous: int = 0
    mismatch_count: int = 0
    mismatches: str = ""
    exon2: str = ""
    intron: str = ""
    exon3: str = ""

    @classmethod
    def summary_row(cls, interpretation: HLAInterpretation) -> "HLAInterpretationRow":
        combined_mismatches: dict[int, CombinedMismatch] = {}
        for best_match in interpretation.best_matches():
            for mismatch in interpretation.matches[best_match].mismatches:
                if mismatch.index not in combined_mismatches:
                    combined_mismatches[mismatch.index] = {
                        "observed_base": mismatch.observed_base,
                        "expected_bases": {mismatch.expected_base},
                    }
                else:
                    if (
                        mismatch.observed_base
                        != combined_mismatches[mismatch.index]["observed_base"]
                    ):
                        error_msg: str = (
                            f'In sequence "{interpretation.hla_sequence.name}", '
                            "different bases were reported in mismatches at "
                            f"position {mismatch.index}."
                        )
                        raise ValueError(error_msg)
                    combined_mismatches[mismatch.index]["expected_bases"].add(
                        mismatch.expected_base
                    )

        best_match_mismatches: list[str] = []
        for mismatch_idx in sorted(combined_mismatches.keys()):
            cm: CombinedMismatch = combined_mismatches[mismatch_idx]
            expected_bases_str: str = "/".join(sorted(cm["expected_bases"]))
            best_match_mismatches.append(
                f"{mismatch_idx}:{cm['observed_base']}->{expected_bases_str}"
            )

        allele_pairs: AllelePairs = interpretation.best_matching_allele_pairs()
        alleles_all_str = allele_pairs.stringify(sorted=True)

        alleles_clean: str
        _, alleles_clean, __ = interpretation.best_common_allele_pair()

        return cls(
            enum=interpretation.hla_sequence.name,
            alleles_clean=alleles_clean,
            alleles=alleles_all_str,
            ambiguous=int(allele_pairs.is_ambiguous()),
            homozygous=int(allele_pairs.is_homozygous()),
            mismatch_count=interpretation.lowest_mismatch_count(),
            mismatches=";".join(best_match_mismatches),
            exon2=interpretation.hla_sequence.exon2_str.upper(),
            intron=interpretation.hla_sequence.intron_str.upper(),
            exon3=interpretation.hla_sequence.exon3_str.upper(),
        )


class HLAMismatchRow(BaseModel):
    """
    Represents the row for an HLA mismatch in a table.
    """

    allele: str
    mismatches: str
    exon2: str
    intron: str
    exon3: str

    @classmethod
    def mismatch_rows(cls, interpretation: HLAInterpretation) -> list["HLAMismatchRow"]:
        # First, sort by the combined standard:
        all_mismatches: list[tuple[int, HLACombinedStandard, list[HLAMismatch]]] = (
            sorted(
                [
                    (details.mismatch_count, cs, details.mismatches)
                    for cs, details in interpretation.matches.items()
                ],
                key=lambda x: x[1].get_allele_pair_str(),
            )
        )
        # Then, sort by the number of mismatches:
        all_mismatches.sort(key=itemgetter(0))

        rows: list["HLAMismatchRow"] = []
        for _, combined_std, mismatches in all_mismatches:
            sorted_mismatches: list[HLAMismatch] = sorted(
                mismatches, key=attrgetter("index")
            )
            curr_row: "HLAMismatchRow" = cls(
                allele=combined_std.get_allele_pair_str(),
                mismatches=";".join([str(x) for x in sorted_mismatches]),
                exon2=interpretation.hla_sequence.exon2_str,
                intron=interpretation.hla_sequence.intron_str,
                exon3=interpretation.hla_sequence.exon3_str,
            )
            rows.append(curr_row)

        return rows
