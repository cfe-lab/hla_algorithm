#! /usr/bin/env python

import argparse
import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass, fields
from io import TextIOBase
from typing import ClassVar, Final, Optional, Self, TypedDict

from easyhla.models import HLAProteinPair
from easyhla.utils import HLA_LOCUS

logging.basicConfig()
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OtherLocusException(Exception):
    pass


@dataclass(frozen=True)
class OldName:
    locus: HLA_LOCUS
    field_1: str
    field_2: str

    @classmethod
    def from_string(cls, old_name_str: str) -> Self:
        # The old name looks like "A*010507N", or "Cw*010203N".
        locus: str = old_name_str[0]
        if locus not in ("A", "B", "C"):
            raise OtherLocusException()

        raw_coordinates: str = old_name_str[2:]
        if locus == "C":
            raw_coordinates = old_name_str[3:]

        field_1: str = raw_coordinates[:2]
        field_2: str = raw_coordinates[2:4]
        return cls(locus, field_1, field_2)

    @classmethod
    def from_old_frequency_format(cls, locus: HLA_LOCUS, four_digit_code: str) -> Self:
        # The entries in the old frequency file look like "5701".
        return cls(
            locus,
            four_digit_code[:2],
            four_digit_code[2:],
        )


@dataclass
class NewName:
    locus: Optional[HLA_LOCUS]
    field_1: str
    field_2: str

    @classmethod
    def from_string(cls, new_name_str: str) -> Self:
        if new_name_str == "None":
            return cls(None, "", "")

        # The new name looks like "A*01:05:07N".
        fields: list[str] = new_name_str.split(":")
        locus: str = fields[0][0]
        if locus not in ("A", "B", "C"):
            raise OtherLocusException()
        field_1: str = fields[0][2:]
        field_2_match: Optional[re.Match] = re.match(r"(\d+)[a-zA-Z]*", fields[1])
        if field_2_match is None:
            raise ValueError(
                f"Could not parse {new_name_str} into a proper allele name"
            )
        field_2: str = field_2_match.group(1)
        return cls(locus, field_1, field_2)

    def to_frequency_format(self) -> str:
        if self.locus is None:
            return HLAProteinPair.DEPRECATED
        return f"{self.field_1}:{self.field_2}"


def parse_nomenclature(remapping_str: str) -> dict[OldName, NewName]:
    # Split by newline and skip the first two lines, which are a header.
    # Also skip the last line, which is blank.
    remapping_lines: list[str] = remapping_str.split("\n")[2:-1]
    remapping_lines = remapping_lines[2:]

    remapping: dict[OldName, NewName] = {}
    for remapping_line in remapping_lines:
        old_name_str: str
        new_name_str: str
        old_name_str, new_name_str = remapping_line.split()

        try:
            old_name: OldName = OldName.from_string(old_name_str)
        except OtherLocusException:
            continue
        new_name: NewName = NewName.from_string(new_name_str)
        if new_name.locus is None:
            logger.info(f"Allele {old_name_str} is deprecated.")
        if old_name in remapping:
            if new_name.locus is None:
                logger.info(
                    f"Allele {old_name_str} maps to {new_name_str} but already "
                    f"maps to {remapping[old_name]}; retaining "
                    f"{remapping[old_name]}."
                )

            elif remapping[old_name] is None:
                logger.info(
                    f"Allele {old_name_str} maps to {new_name_str} but already "
                    f"maps to {remapping[old_name]}; updating the mapping."
                )
                remapping[old_name] = new_name
        else:
            remapping[old_name] = new_name

    return remapping


class FrequencyRowDict(TypedDict):
    a_first: Optional[str]
    a_second: Optional[str]
    b_first: Optional[str]
    b_second: Optional[str]
    c_first: Optional[str]
    c_second: Optional[str]


@dataclass
class FrequencyRow:
    a_first: Optional[NewName]
    a_second: Optional[NewName]
    b_first: Optional[NewName]
    b_second: Optional[NewName]
    c_first: Optional[NewName]
    c_second: Optional[NewName]

    HEADER_ROW: ClassVar[Final[tuple[str, str, str, str, str, str]]] = (
        "a_first",
        "a_second",
        "b_first",
        "b_second",
        "c_first",
        "c_second",
    )

    def to_row_dict(self) -> FrequencyRowDict:
        row_dict: FrequencyRowDict = {
            "a_first": None,
            "a_second": None,
            "b_first": None,
            "b_second": None,
            "c_first": None,
            "c_second": None,
        }
        for field in fields(self):
            curr_new_name: Optional[NewName] = getattr(self, field.name)
            if curr_new_name is None:
                row_dict[field.name] = HLAProteinPair.UNKNOWN
            else:
                row_dict[field.name] = curr_new_name.to_frequency_format()
        return row_dict


def update_old_frequencies(
    old_frequencies_file: TextIOBase, old_to_new: dict[OldName, NewName]
) -> list[FrequencyRow]:
    old_frequencies_csv: csv.reader = csv.reader(old_frequencies_file)

    # Report to the user any frequencies that are either unmapped or
    # deprecated.
    unmapped_alleles: Counter[tuple[HLA_LOCUS, str]] = Counter()
    deprecated_alleles_seen: Counter[tuple[HLA_LOCUS, str]] = Counter()

    updated_frequencies: list[FrequencyRow] = []
    for row in old_frequencies_csv:
        loci: tuple[HLA_LOCUS, HLA_LOCUS, HLA_LOCUS] = ("A", "B", "C")

        updated: list[Optional[NewName]] = []
        for idx in range(6):
            locus: HLA_LOCUS = loci[int(idx / 2)]
            old_name: OldName = OldName.from_old_frequency_format(locus, row[idx])
            new_name: Optional[NewName] = old_to_new.get(old_name)

            if new_name is None:
                unmapped_alleles[(locus, row[idx])] += 1
            elif new_name.locus is None:
                deprecated_alleles_seen[(locus, row[idx])] += 1

            updated.append(new_name)

        updated_frequencies.append(FrequencyRow(*updated))

    if len(unmapped_alleles) > 1:
        logger.info(
            "Alleles present in the old frequencies that do not have a mapping "
            "in the new nomenclature, and their numbers of occurrences:"
        )
        for locus, name in unmapped_alleles:
            logger.info(
                f'HLA-{locus} allele "{name}": {unmapped_alleles[(locus, name)]}'
            )

    if len(deprecated_alleles_seen) > 1:
        logger.info(
            "Alleles present in the old frequencies that are deprecated "
            "in the new nomenclature, and their numbers of occurrences:"
        )
        for locus, name in deprecated_alleles_seen:
            logger.info(
                f'HLA-{locus} allele "{name}": {deprecated_alleles_seen[(locus, name)]}'
            )

    return updated_frequencies


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Update the format of the old HLA frequencies file"
    )
    parser.add_argument(
        "old_frequency_file",
        help="Old frequency file path (6-column CSV without header)",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "name_mapping",
        help=(
            "File mapping old allele names to new allele names in the format "
            "kept by IMGT/HLA"
        ),
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--output",
        help="Output CSV file",
        type=argparse.FileType("w"),
        default="hla_frequencies.csv",
    )
    args: argparse.Namespace = parser.parse_args()

    logger.info("Reading in the update mapping....")
    with args.name_mapping:
        old_to_new: dict[OldName, NewName] = parse_nomenclature(
            args.name_mapping.read()
        )

    logger.info("Updating old alleles....")
    with args.old_frequency_file:
        updated_frequencies: list[FrequencyRow] = update_old_frequencies(
            args.old_frequency_file, old_to_new
        )

    with args.output:
        frequencies_csv: csv.DictWriter = csv.DictWriter(
            args.output, FrequencyRow.HEADER_ROW
        )
        frequencies_csv.writeheader()
        frequencies_csv.writerows([x.to_row_dict() for x in updated_frequencies])

    logger.info("... done.")


if __name__ == "__main__":
    main()
