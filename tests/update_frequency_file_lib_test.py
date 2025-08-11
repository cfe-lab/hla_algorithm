from collections import Counter
from io import StringIO

import pytest

from hla_algorithm.models import HLAProteinPair
from hla_algorithm.update_frequency_file_lib import (
    FrequencyRowDict,
    NewName,
    OldName,
    OtherLocusException,
    parse_nomenclature,
    update_old_frequencies,
)
from hla_algorithm.utils import HLA_LOCUS


@pytest.mark.parametrize(
    "old_name_str, expected_result",
    [
        pytest.param(
            "A*010507N",
            OldName("A", "01", "05"),
            id="a_more_than_two_coordinates",
        ),
        pytest.param(
            "A*1522",
            OldName("A", "15", "22"),
            id="a_exactly_two_coordinates",
        ),
        pytest.param(
            "B*57010305N",
            OldName("B", "57", "01"),
            id="b_more_than_two_coordinates",
        ),
        pytest.param(
            "B*5703",
            OldName("B", "57", "03"),
            id="b_exactly_two_coordinates",
        ),
        pytest.param(
            "Cw*22334455N",
            OldName("C", "22", "33"),
            id="c_more_than_two_coordinates",
        ),
        pytest.param(
            "Cw*2233",
            OldName("C", "22", "33"),
            id="c_exactly_two_coordinates",
        ),
    ],
)
def test_old_name_from_string_good_cases(
    old_name_str: str,
    expected_result: OldName,
):
    result: OldName = OldName.from_string(old_name_str)
    assert result == expected_result


@pytest.mark.parametrize(
    "old_name_str",
    [
        pytest.param(
            "DPB1*110101",
            id="locus_d",
        ),
        pytest.param(
            "G*01010201",
            id="locus_g",
        ),
    ],
)
def test_old_name_from_string_exception_cases(old_name_str: str):
    with pytest.raises(OtherLocusException):
        OldName.from_string(old_name_str)


@pytest.mark.parametrize(
    "locus, four_digit_code, expected_result",
    [
        pytest.param("A", "7400", OldName("A", "74", "00"), id="typical_a_case"),
        pytest.param("B", "5703", OldName("B", "57", "03"), id="typical_b_case"),
        pytest.param("C", "2551", OldName("C", "25", "51"), id="typical_c_case"),
    ],
)
def test_old_name_from_old_frequency_format(
    locus: HLA_LOCUS, four_digit_code: str, expected_result: OldName
):
    result: OldName = OldName.from_old_frequency_format(locus, four_digit_code)
    assert result == expected_result


@pytest.mark.parametrize(
    "new_name_str, expected_result",
    [
        pytest.param(
            "A*01:05:07N",
            NewName("A", "01", "05"),
            id="a_more_than_two_coordinates",
        ),
        pytest.param(
            "A*15:22",
            NewName("A", "15", "22"),
            id="a_exactly_two_coordinates",
        ),
        pytest.param(
            "B*57:01:03:05N",
            NewName("B", "57", "01"),
            id="b_more_than_two_coordinates",
        ),
        pytest.param(
            "B*57:03",
            NewName("B", "57", "03"),
            id="b_exactly_two_coordinates",
        ),
        pytest.param(
            "C*22:33:44:55N",
            NewName("C", "22", "33"),
            id="c_more_than_two_coordinates",
        ),
        pytest.param(
            "C*22:33",
            NewName("C", "22", "33"),
            id="c_exactly_two_coordinates",
        ),
    ],
)
def test_new_name_from_string_good_cases(
    new_name_str: str,
    expected_result: OldName,
):
    result: NewName = NewName.from_string(new_name_str)
    assert result == expected_result


@pytest.mark.parametrize(
    "new_name_str",
    [
        pytest.param(
            "DPB1*11:01:01",
            id="locus_d",
        ),
        pytest.param(
            "G*01:01:02:01",
            id="locus_g",
        ),
    ],
)
def test_new_name_from_string_bad_locus_exception_cases(new_name_str: str):
    with pytest.raises(OtherLocusException):
        NewName.from_string(new_name_str)


def test_new_name_from_string_cannot_parse_exception():
    error_msg: str = 'Could not parse "A*01:fdsa" into a proper allele name'
    with pytest.raises(ValueError) as e:
        NewName.from_string("A*01:fdsa")
        assert error_msg in str(e.value)


@pytest.mark.parametrize(
    "locus, field_1, field_2, expected_result",
    [
        pytest.param("B", "57", "01", "57:01", id="typical_case"),
        pytest.param(None, "", "", HLAProteinPair.DEPRECATED, id="deprecated"),
    ],
)
def test_new_name_to_frequency_format(
    locus: HLA_LOCUS, field_1: str, field_2: str, expected_result: str
):
    new_name: NewName = NewName(locus, field_1, field_2)
    assert new_name.to_frequency_format() == expected_result


@pytest.mark.parametrize(
    (
        "rows, expected_remapping, expected_deprecated, "
        "expected_deprecated_maps_to_other, "
        "expected_mapping_overrides_deprecated"
    ),
    [
        pytest.param(
            [("B*570101", "B*57:01:01")],
            {OldName("B", "57", "01"): NewName("B", "57", "01")},
            [],
            [],
            [],
            id="one_trivial_mapping",
        ),
        pytest.param(
            [("Cw*223344", "C*22:122")],
            {OldName("C", "22", "33"): NewName("C", "22", "122")},
            [],
            [],
            [],
            id="one_nontrivial_mapping",
        ),
        pytest.param(
            [("A*0105N", "None")],
            {OldName("A", "01", "05"): NewName(None, "", "")},
            ["A*0105N"],
            [],
            [],
            id="one_deprecated_mapping",
        ),
        pytest.param(
            [
                ("A*020119", "A*02:01:19"),
                ("A*020120", "None"),
            ],
            {OldName("A", "02", "01"): NewName("A", "02", "01")},
            ["A*020120"],
            [("A*020120", NewName("A", "02", "01"))],
            [],
            id="one_deprecated_maps_to_other",
        ),
        pytest.param(
            [
                ("B*505001", "None"),
                ("B*505002", "B*49:32:11"),
            ],
            {OldName("B", "50", "50"): NewName("B", "49", "32")},
            ["B*505001"],
            [],
            [("B*505002", NewName("B", "49", "32"))],
            id="one_mapping_overrides_deprecated",
        ),
        pytest.param(
            [
                ("DPB1*020102", "DPB1*02:01:02"),
            ],
            {},
            [],
            [],
            [],
            id="one_skipped_mapping",
        ),
        pytest.param(
            [
                ("B*505001", "B*50:50:01"),
                ("B*505002", "B*50:50:02"),
            ],
            {OldName("B", "50", "50"): NewName("B", "50", "50")},
            [],
            [],
            [],
            id="two_compatible_mappings_overrides_deprecated",
        ),
        pytest.param(
            [
                ("A*020119", "A*02:01:19"),
                ("A*020120", "None"),
                ("A*493352N", "A*48:122"),
                ("B*150103", "B*15:01:03"),
                ("B*505001", "None"),
                ("B*505002", "B*49:32:11"),
                ("B*570101", "B*57:01:01"),
                ("B*570111", "B*57:01:11"),
                ("Cw*223344", "C*22:122"),
                ("Cw*223445", "None"),
                ("DPB1*020102", "DPB1*02:01:02"),
            ],
            {
                OldName("A", "02", "01"): NewName("A", "02", "01"),
                OldName("A", "49", "33"): NewName("A", "48", "122"),
                OldName("B", "15", "01"): NewName("B", "15", "01"),
                OldName("B", "50", "50"): NewName("B", "49", "32"),
                OldName("B", "57", "01"): NewName("B", "57", "01"),
                OldName("C", "22", "33"): NewName("C", "22", "122"),
                OldName("C", "22", "34"): NewName(None, "", ""),
            },
            [
                "A*020120",
                "B*505001",
                "Cw*223445",
            ],
            [("A*020120", NewName("A", "02", "01"))],
            [("B*505002", NewName("B", "49", "32"))],
            id="typical_case",
        ),
    ],
)
def test_parse_nomenclature(
    rows: list[tuple[str, str]],
    expected_remapping: dict[OldName, NewName],
    expected_deprecated: list[str],
    expected_deprecated_maps_to_other: list[tuple[str, NewName]],
    expected_mapping_overrides_deprecated: list[tuple[str, str]],
):
    for num_spaces in (1, 2, 5, 10, 20):
        fake_text_input: str = "ignored line 1\nignored line 2\n"
        for row in rows:
            fake_text_input += f"{row[0]}{' ' * num_spaces}{row[1]}\n"
        result: tuple[
            dict[OldName, NewName],
            list[str],
            list[tuple[str, NewName]],
            list[tuple[str, NewName]],
        ] = parse_nomenclature(fake_text_input)

        assert result[0] == expected_remapping
        assert result[1] == expected_deprecated
        assert result[2] == expected_deprecated_maps_to_other
        assert result[3] == expected_mapping_overrides_deprecated


@pytest.mark.parametrize(
    (
        "old_frequency_lines, remapping, expected_updated_frequencies, "
        "expected_unmapped_alleles, expected_deprecated_alleles_seen"
    ),
    [
        pytest.param(
            ["1234,5678,5701,5603,2233,4455"],
            {
                OldName("A", "12", "34"): NewName("A", "12", "34"),
                OldName("A", "56", "78"): NewName("A", "56", "78"),
                OldName("B", "57", "01"): NewName("B", "57", "01"),
                OldName("B", "56", "03"): NewName("B", "56", "03"),
                OldName("C", "22", "33"): NewName("C", "22", "33"),
                OldName("C", "44", "55"): NewName("C", "44", "55"),
            },
            [
                {
                    "a_first": "12:34",
                    "a_second": "56:78",
                    "b_first": "57:01",
                    "b_second": "56:03",
                    "c_first": "22:33",
                    "c_second": "44:55",
                },
            ],
            Counter(),
            Counter(),
            id="one_row_all_trivial",
        ),
        pytest.param(
            ["1234,5678,5701,5603,2233,4455"],
            {
                OldName("A", "12", "34"): NewName("A", "12", "340"),
                OldName("A", "56", "78"): NewName("A", "56", "110"),
                OldName("B", "57", "01"): NewName("B", "55", "02"),
                OldName("B", "56", "03"): NewName("B", "53", "04"),
                OldName("C", "22", "33"): NewName("C", "22", "115"),
                OldName("C", "44", "55"): NewName("C", "43", "02"),
            },
            [
                {
                    "a_first": "12:340",
                    "a_second": "56:110",
                    "b_first": "55:02",
                    "b_second": "53:04",
                    "c_first": "22:115",
                    "c_second": "43:02",
                },
            ],
            Counter(),
            Counter(),
            id="one_row_all_nontrivial",
        ),
        pytest.param(
            ["1234,5678,5701,5603,2233,4455"],
            {},
            [
                {
                    "a_first": "unmapped",
                    "a_second": "unmapped",
                    "b_first": "unmapped",
                    "b_second": "unmapped",
                    "c_first": "unmapped",
                    "c_second": "unmapped",
                },
            ],
            Counter(
                {
                    ("A", "1234"): 1,
                    ("A", "5678"): 1,
                    ("B", "5701"): 1,
                    ("B", "5603"): 1,
                    ("C", "2233"): 1,
                    ("C", "4455"): 1,
                }
            ),
            Counter(),
            id="one_row_all_unmapped_no_mappings",
        ),
        pytest.param(
            ["1234,5678,5701,5603,2233,4455"],
            {
                OldName("A", "55", "34"): NewName("A", "12", "340"),
                OldName("A", "77", "78"): NewName("A", "56", "110"),
                OldName("B", "10", "01"): NewName("B", "55", "02"),
                OldName("B", "22", "03"): NewName("B", "53", "04"),
                OldName("C", "85", "33"): NewName("C", "22", "115"),
                OldName("C", "12", "55"): NewName("C", "43", "02"),
            },
            [
                {
                    "a_first": "unmapped",
                    "a_second": "unmapped",
                    "b_first": "unmapped",
                    "b_second": "unmapped",
                    "c_first": "unmapped",
                    "c_second": "unmapped",
                },
            ],
            Counter(
                {
                    ("A", "1234"): 1,
                    ("A", "5678"): 1,
                    ("B", "5701"): 1,
                    ("B", "5603"): 1,
                    ("C", "2233"): 1,
                    ("C", "4455"): 1,
                }
            ),
            Counter(),
            id="one_row_all_unmapped_no_mappings_used",
        ),
        pytest.param(
            ["1234,5678,5701,5603,2233,4455"],
            {
                OldName("A", "12", "34"): NewName(None, "", ""),
                OldName("A", "56", "78"): NewName(None, "", ""),
                OldName("B", "57", "01"): NewName(None, "", ""),
                OldName("B", "56", "03"): NewName(None, "", ""),
                OldName("C", "22", "33"): NewName(None, "", ""),
                OldName("C", "44", "55"): NewName(None, "", ""),
            },
            [
                {
                    "a_first": "deprecated",
                    "a_second": "deprecated",
                    "b_first": "deprecated",
                    "b_second": "deprecated",
                    "c_first": "deprecated",
                    "c_second": "deprecated",
                },
            ],
            Counter(),
            Counter(
                {
                    ("A", "1234"): 1,
                    ("A", "5678"): 1,
                    ("B", "5701"): 1,
                    ("B", "5603"): 1,
                    ("C", "2233"): 1,
                    ("C", "4455"): 1,
                }
            ),
            id="one_row_all_deprecated",
        ),
        pytest.param(
            [
                "1234,5678,5701,5603,2233,4455",
                "1234,5678,6602,6303,2233,5471",
            ],
            {
                OldName("A", "12", "34"): NewName("A", "12", "34"),
                OldName("A", "56", "78"): NewName(None, "", ""),
                OldName("B", "57", "01"): NewName("B", "57", "01"),
                OldName("B", "56", "03"): NewName("B", "56", "03"),
                OldName("C", "44", "55"): NewName("C", "44", "55"),
                OldName("B", "66", "02"): NewName("B", "64", "11"),
                OldName("B", "63", "03"): NewName("B", "63", "03"),
                OldName("C", "54", "71"): NewName("C", "53", "110"),
            },
            [
                {
                    "a_first": "12:34",
                    "a_second": "deprecated",
                    "b_first": "57:01",
                    "b_second": "56:03",
                    "c_first": "unmapped",
                    "c_second": "44:55",
                },
                {
                    "a_first": "12:34",
                    "a_second": "deprecated",
                    "b_first": "64:11",
                    "b_second": "63:03",
                    "c_first": "unmapped",
                    "c_second": "53:110",
                },
            ],
            Counter({("C", "2233"): 2}),
            Counter({("A", "5678"): 2}),
            id="two_rows_multiple_deprecated_and_unmapped",
        ),
        pytest.param(
            [
                "1234,5678,5701,5603,2233,4455",
                "5501,7400,5523,5823,1500,1503",
                "1111,2222,3333,4444,5555,6666",
                "1234,7400,4444,5823,1501,1507",
            ],
            {
                OldName("A", "12", "34"): NewName("A", "12", "34"),
                OldName("A", "56", "78"): NewName("A", "56", "110"),
                OldName("B", "57", "01"): NewName("B", "57", "01"),
                OldName("B", "56", "03"): NewName("B", "55", "114"),
                OldName("C", "22", "33"): NewName("C", "22", "33"),
                OldName("C", "44", "55"): NewName("C", "44", "55"),
                OldName("A", "55", "01"): NewName("A", "55", "01"),
                OldName("B", "55", "23"): NewName("B", "55", "23"),
                OldName("B", "58", "23"): NewName("B", "58", "23"),
                OldName("C", "15", "03"): NewName(None, "", ""),
                OldName("A", "11", "11"): NewName("A", "10", "223"),
                OldName("A", "22", "34"): NewName("A", "22", "35"),  # unused
                OldName("A", "22", "22"): NewName("A", "19", "190"),
                OldName("B", "33", "33"): NewName("B", "33", "33"),
                OldName("B", "44", "44"): NewName(None, "", ""),
                OldName("C", "55", "55"): NewName("C", "55", "55"),
                OldName("C", "66", "66"): NewName("C", "62", "114"),
                OldName("C", "15", "01"): NewName("C", "15", "01"),
                OldName("C", "15", "07"): NewName("C", "15", "07"),
            },
            [
                {
                    "a_first": "12:34",
                    "a_second": "56:110",
                    "b_first": "57:01",
                    "b_second": "55:114",
                    "c_first": "22:33",
                    "c_second": "44:55",
                },
                {
                    "a_first": "55:01",
                    "a_second": "unmapped",
                    "b_first": "55:23",
                    "b_second": "58:23",
                    "c_first": "unmapped",
                    "c_second": "deprecated",
                },
                {
                    "a_first": "10:223",
                    "a_second": "19:190",
                    "b_first": "33:33",
                    "b_second": "deprecated",
                    "c_first": "55:55",
                    "c_second": "62:114",
                },
                {
                    "a_first": "12:34",
                    "a_second": "unmapped",
                    "b_first": "deprecated",
                    "b_second": "58:23",
                    "c_first": "15:01",
                    "c_second": "15:07",
                },
            ],
            Counter(
                {
                    ("A", "7400"): 2,
                    ("C", "1500"): 1,
                }
            ),
            Counter(
                {
                    ("C", "1503"): 1,
                    ("B", "4444"): 2,
                }
            ),
            id="typical_case",
        ),
    ],
)
def test_update_old_frequencies(
    old_frequency_lines: list[str],
    remapping: dict[OldName, NewName],
    expected_updated_frequencies: list[FrequencyRowDict],
    expected_unmapped_alleles: Counter[tuple[HLA_LOCUS, str]],
    expected_deprecated_alleles_seen: Counter[tuple[HLA_LOCUS, str]],
):
    fake_old_frequency_file: StringIO = StringIO("\n".join(old_frequency_lines))
    updated_frequencies: list[FrequencyRowDict]
    unmapped_alleles: Counter[tuple[HLA_LOCUS, str]]
    deprecated_alleles_seen: Counter[tuple[HLA_LOCUS, str]]

    updated_frequencies, unmapped_alleles, deprecated_alleles_seen = (
        update_old_frequencies(fake_old_frequency_file, remapping)
    )

    assert updated_frequencies == expected_updated_frequencies
    assert unmapped_alleles == expected_unmapped_alleles
    assert deprecated_alleles_seen == expected_deprecated_alleles_seen
