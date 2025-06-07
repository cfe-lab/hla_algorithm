import csv
import re
from collections import Counter
from dataclasses import dataclass
from io import TextIOBase
from typing import Final, Optional, Self, TypedDict

from .easyhla import EasyHLA
from .models import HLAProteinPair
from .utils import HLA_LOCUS


class OtherLocusException(Exception):
    pass


@dataclass(frozen=True)
class OldName:
    """
    Representation of an "old name" entry in the nomenclature mapping.
    """

    locus: HLA_LOCUS
    field_1: str
    field_2: str

    @classmethod
    def from_string(cls, old_name_str: str) -> Self:
        """
        Build an instance directly from an entry in the nomenclature mapping.

        The old names look like "A*010507N" for loci other than C; HLA-C old
        names look like "Cw*010203N".
        """
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
        """
        Build an instance from an entry in the old frequency file format.

        These entries are four-digit codes like "5701", with no locus
        information; the locus must be provided separately.
        """
        return cls(
            locus,
            four_digit_code[:2],
            four_digit_code[2:],
        )


@dataclass
class NewName:
    """
    Representation of a "new name" entry in the nomenclature mapping.
    """

    locus: Optional[HLA_LOCUS]
    field_1: str
    field_2: str

    @classmethod
    def from_string(cls, new_name_str: str) -> Self:
        """
        Build an instance directly from an entry in the nomenclature mapping.

        The new names look like "A*01:05:07N".
        """
        if new_name_str == "None":
            return cls(None, "", "")

        coords: list[str] = new_name_str.split(":")
        locus: str = coords[0][0]
        if locus not in ("A", "B", "C"):
            raise OtherLocusException()
        field_1: str = coords[0][2:]
        field_2_match: Optional[re.Match] = re.match(r"(\d+)[a-zA-Z]*", coords[1])
        if field_2_match is None:
            raise ValueError(
                f"Could not parse {new_name_str} into a proper allele name"
            )
        field_2: str = field_2_match.group(1)
        return cls(locus, field_1, field_2)

    def to_frequency_format(self) -> str:
        """
        Render this to the format our new frequencies file will use.

        This will be the first two "coordinates", separated by a colon; this
        will thus allow coordinates with more than two digits if they exist.
        Similarly to the old format, no locus information will be included in
        the entries themselves (the locus information will come from the column
        they belong to).
        """
        if self.locus is None:
            return HLAProteinPair.DEPRECATED
        return f"{self.field_1}:{self.field_2}"


def parse_nomenclature(
    remapping_str: str,
) -> tuple[
    dict[OldName, NewName],
    list[str],
    list[tuple[str, NewName]],
    list[tuple[str, NewName]],
]:
    """
    Parse the mapping from old names to their post-2010 updated names.

    This mapping is currently provided by IMGT/HLA in their Github repo, in a
    file named "Nomenclature_2009.txt".  The first two lines are a header.  The
    lines in between are the mappings, and they are of the form
        [old name] [new name]
    where there is an unspecified amount of whitespace between the columns.

    If the new name is "None" then we call this a "deprecated" allele.
    """
    remapping_lines: list[str] = remapping_str.split("\n")[2:-1]

    deprecated: list[str] = []
    deprecated_maps_to_other: list[tuple[str, str]] = []
    mapping_overrides_deprecated: list[tuple[str, str]] = []

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
            deprecated.append(old_name_str)

        if old_name in remapping:
            if new_name.locus is None:
                deprecated_maps_to_other.append((old_name_str, remapping[old_name]))

            elif remapping[old_name].locus is None:
                mapping_overrides_deprecated.append((old_name_str, new_name))
                remapping[old_name] = new_name
        else:
            remapping[old_name] = new_name

    return (
        remapping,
        deprecated,
        deprecated_maps_to_other,
        mapping_overrides_deprecated,
    )


class FrequencyRowDict(TypedDict):
    a_first: str
    a_second: str
    b_first: str
    b_second: str
    c_first: str
    c_second: str


FREQUENCY_FIELDS: Final[tuple[str, str, str, str, str, str]] = sum(
    EasyHLA.FREQUENCY_LOCUS_COLUMNS.values(), ()
)


def update_old_frequencies(
    old_frequencies_file: TextIOBase, old_to_new: dict[OldName, NewName]
) -> tuple[
    list[FrequencyRowDict],
    Counter[tuple[HLA_LOCUS, str]],
    Counter[tuple[HLA_LOCUS, str]],
]:
    """
    Update the old frequencies file to the new format.

    Returns a list of dictionaries that can be written using a DictWriter,
    and Counters that represent the alleles that are unmapped in the new
    naming scheme and the alleles that are deprecated in the new naming scheme.
    """
    old_frequencies_csv: csv.reader = csv.reader(old_frequencies_file)

    # Report to the user any frequencies that are either unmapped or
    # deprecated.
    unmapped_alleles: Counter[tuple[HLA_LOCUS, str]] = Counter()
    deprecated_alleles_seen: Counter[tuple[HLA_LOCUS, str]] = Counter()

    updated_frequencies: list[FrequencyRowDict] = []
    for row in old_frequencies_csv:
        loci: tuple[HLA_LOCUS, HLA_LOCUS, HLA_LOCUS] = ("A", "B", "C")

        updated: FrequencyRowDict = {x: None for x in FREQUENCY_FIELDS}
        for idx in range(6):
            locus: HLA_LOCUS = loci[int(idx / 2)]
            column_name: str = FREQUENCY_FIELDS[idx]
            old_name: OldName = OldName.from_old_frequency_format(locus, row[idx])
            new_name: Optional[NewName] = old_to_new.get(old_name)

            new_name_str: str = HLAProteinPair.UNMAPPED
            if new_name is None:
                unmapped_alleles[(locus, row[idx])] += 1
            else:
                new_name_str = new_name.to_frequency_format()
                if new_name.locus is None:
                    deprecated_alleles_seen[(locus, row[idx])] += 1
            updated[column_name] = new_name_str

        updated_frequencies.append(updated)

    return updated_frequencies, unmapped_alleles, deprecated_alleles_seen
