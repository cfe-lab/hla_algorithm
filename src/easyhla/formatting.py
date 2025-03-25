from operator import itemgetter

from pydantic import BaseModel

from .models import AllelePairs, HLACombinedStandard, HLAInterpretation, HLAMismatch


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

    # FIXME: currently we include all mismatches from all possible best
    # matches; we should perhaps pick a "very best" of these matches and
    # only include those mismatches.
    @classmethod
    def summary_row(
        cls, interpretation: HLAInterpretation, hla_frequencies: dict[str, int]
    ) -> "HLAInterpretationRow":
        best_match_mismatches: list[str] = []
        for best_match in interpretation.best_matches():
            best_match_mismatches.extend(
                [str(x) for x in interpretation.matches[best_match].mismatches]
            )

        allele_pairs: AllelePairs = interpretation.best_matching_allele_pairs()
        alleles_all_str = allele_pairs.stringify()
        clean_allele_str = allele_pairs.best_common_allele_pair_str(hla_frequencies)

        return cls(
            enum=interpretation.hla_sequence.name,
            alleles_clean=clean_allele_str,
            alleles=alleles_all_str,
            ambig=int(allele_pairs.is_ambiguous()),
            homozygous=int(allele_pairs.is_homozygous()),
            mismatch_count=interpretation.lowest_mismatch_count(),
            mismatches=";".join(best_match_mismatches),
            exon2=interpretation.hla_sequence.two.upper(),
            intron=interpretation.hla_sequence.intron.upper(),
            exon3=interpretation.hla_sequence.three.upper(),
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
        matches_by_count: list[tuple[int, HLACombinedStandard, list[HLAMismatch]]] = (
            sorted(
                [
                    (details.mismatch_count, cs, details.mismatches)
                    for cs, details in interpretation.matches.items()
                ],
                key=itemgetter(0),
            )
        )

        rows: list[cls] = []
        for _, combined_std, mismatches in matches_by_count:
            curr_row: cls = cls(
                allele=combined_std.get_allele_pair_str(),
                mismatches=";".join([str(x) for x in mismatches]),
                exon2=interpretation.hla_sequence.two.upper(),
                intron=interpretation.hla_sequence.intron.upper(),
                exon3=interpretation.hla_sequence.three.upper(),
            )
            rows.append(",".join(curr_row))

        return rows
