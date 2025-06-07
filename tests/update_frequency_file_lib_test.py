import pytest

from easyhla.models import HLAProteinPair
from easyhla.update_frequency_file_lib import (
    NewName,
    OldName,
    OtherLocusException,
    parse_nomenclature,
)
from easyhla.utils import HLA_LOCUS


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
def test_new_name_from_string_exception_cases(new_name_str: str):
    with pytest.raises(OtherLocusException):
        NewName.from_string(new_name_str)


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


#    dict[OldName, NewName], list[str], list[tuple[str, str]], list[tuple[str, str]]


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
                ("A*020119", "A*02:01:19"),
                ("A*020120", "None"),
                ("A*493352N", "A*48:122"),
                ("B*150103", "B*15:01:03"),
                ("B*505001", "None"),
                ("B*505002", "B*49:32:11"),
                ("B*570101", "B*57:01:01"),
                ("Cw*223344", "C*22:122"),
                ("DPB1*020102", "DPB1*02:01:02"),
            ],
            {
                OldName("A", "02", "01"): NewName("A", "02", "01"),
                OldName("A", "49", "33"): NewName("A", "48", "122"),
                OldName("B", "15", "01"): NewName("B", "15", "01"),
                OldName("B", "50", "50"): NewName("B", "49", "32"),
                OldName("B", "57", "01"): NewName("B", "57", "01"),
                OldName("C", "22", "33"): NewName("C", "22", "122"),
            },
            [
                "A*020120",
                "B*505001",
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
            list[tuple[str, str]],
            list[tuple[str, str]],
        ] = parse_nomenclature(fake_text_input)

        assert result[0] == expected_remapping
        assert result[1] == expected_deprecated
        assert result[2] == expected_deprecated_maps_to_other
        assert result[3] == expected_mapping_overrides_deprecated
