import pytest
import json
import os
import pytz
from datetime import datetime
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from src.easyhla.easyhla import EasyHLA, DATE_FORMAT
from src.easyhla.models import HLAStandard, HLAStandardMatch, HLACombinedStandardResult
from typing import List, Optional, Dict, Tuple, Any

from .conftest import make_comparison, compare_ref_vs_test


@pytest.fixture(scope="module")
def easyhla(request: pytest.FixtureRequest):
    easyhla = EasyHLA(letter=request.param)
    return easyhla


@pytest.fixture
def hla_standard_file(tmp_path: Path):
    hla_standard = ["HELLO-WORLD", "AAAAC", "ARTY"]

    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_std.csv"
    p.write_text(",".join(hla_standard))

    return str(p)


@pytest.fixture
def hla_frequency_file(tmp_path: Path):
    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_freq.csv"
    p.write_text("ABCD,1234,ABCD,1234,ABCD,1234")

    return str(p)


@pytest.fixture(scope="session")
def timestamp() -> datetime:
    _dt = datetime.today()
    dt = _dt.replace(tzinfo=pytz.UTC)
    return dt


@pytest.fixture
def hla_last_modified_file(tmp_path: Path, timestamp: datetime) -> str:
    d = tmp_path / "hla_std"
    d.mkdir()
    p = d / "hla_timestamp.txt"
    p.write_text(timestamp.strftime(DATE_FORMAT))

    return str(p)


class TestEasyHLAMisc:
    def test_unknown_hla_type(self):
        """
        Assert we raise a value error if we put in an unknown HLA type.
        """
        with pytest.raises(ValueError):
            easyhla = EasyHLA("D")

    def test_known_hla_type_lowercase(self):
        """
        Assert no error is raised if we put in an HLA type with wrong case.
        """
        with does_not_raise():
            easyhla = EasyHLA("a")

    @pytest.mark.parametrize("easyhla", ["A"], indirect=True)
    def test_load_allele_definitions_last_modified_time(
        self, easyhla, hla_last_modified_file, timestamp, mocker
    ):
        """
        Assert we can load our mtime and that it is represented correctly.
        """
        mocker.patch.object(os.path, "join", return_value=hla_last_modified_file)

        # Annoyingly, while `strptime(%Z)` outputs a time in the correct timezone,
        # it doesn't keep that time in the object.
        expected_time = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )

        result = easyhla.load_allele_definitions_last_modified_time()
        assert result == expected_time


@pytest.mark.parametrize("easyhla", ["A"], indirect=True)
class TestEasyHLADiscreteHLATypeA:
    """
    Testing EasyHLA where tests require HLA type A.
    """

    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * 1000, "myseq-a-bad", -1),
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-a00-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-a-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-a-maxbad", -1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-a-bad-exon2", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-a-bad-exon3", -1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-a00-good-exon2-short", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-a00-good-exon3-short", 1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-a01-good-exon2-short",
                1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-a01-good-exon3-short",
                1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-bad-exon3", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-bad-exon3", 0),
            # `
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-good-exon3-short", 1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_a(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            result = easyhla.check_length(
                letter=easyhla.letter, seq=sequence, name=name
            )

            assert bool(exp_return) == result
        else:
            with pytest.raises(ValueError):
                result = easyhla.check_length(
                    letter=easyhla.letter, seq=sequence, name=name
                )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self, easyhla: EasyHLA):
        """
        Integration test, assert that pyEasyHLA produces an identical output to
        the original Ruby output.
        """
        input_file = os.path.dirname(__file__) + "/input/hla-a-seqs.fasta"
        ref_output_file = os.path.dirname(__file__) + "/output/hla-a-output.csv"
        output_file = os.path.dirname(__file__) + "/output/hla-a-test.csv"

        easyhla.run(
            easyhla.letter,
            input_file,
            output_file,
            0,
        )

        compare_ref_vs_test(
            easyhla=easyhla,
            reference_output_file=ref_output_file,
            output_file=output_file,
        )

    @pytest.mark.parametrize(
        "best_matches, exp_ambig, exp_alleles",
        [
            (
                [
                    HLACombinedStandardResult(
                        standard="",
                        discrete_allele_names=[
                            ["A*02:01:01G", "A*03:01:01G"],
                            ["A*02:01:52", "A*03:01:03"],
                            ["A*02:01:02", "A*03:01:12"],
                            ["A*02:01:36", "A*03:01:38"],
                            ["A*02:237", "A*03:05:01"],
                            ["A*02:26", "A*03:07"],
                            ["A*02:34", "A*03:08"],
                            ["A*02:90", "A*03:09"],
                            ["A*02:24:01", "A*03:17:01"],
                            ["A*02:195", "A*03:23:01"],
                            ["A*02:338", "A*03:95"],
                            ["A*02:35:01", "A*03:108"],
                            ["A*02:86", "A*03:123"],
                            ["A*02:20:01", "A*03:157"],
                        ],
                    )
                ],
                False,
                [
                    ["A*02:01:01G", "A*03:01:01G"],
                    ["A*02:01:52", "A*03:01:03"],
                    ["A*02:01:02", "A*03:01:12"],
                    ["A*02:01:36", "A*03:01:38"],
                    ["A*02:237", "A*03:05:01"],
                    ["A*02:26", "A*03:07"],
                    ["A*02:34", "A*03:08"],
                    ["A*02:90", "A*03:09"],
                    ["A*02:24:01", "A*03:17:01"],
                    ["A*02:195", "A*03:23:01"],
                    ["A*02:338", "A*03:95"],
                    ["A*02:35:01", "A*03:108"],
                    ["A*02:86", "A*03:123"],
                    ["A*02:20:01", "A*03:157"],
                ],
            ),
            (
                [
                    HLACombinedStandardResult(
                        standard="",
                        discrete_allele_names=[
                            ["A*11:01:01G", "A*26:01:01G"],
                            ["A*11:01:07", "A*26:01:17"],
                            ["A*11:19", "A*26:13"],
                            ["A*11:40", "A*66:01G"],
                        ],
                    )
                ],
                True,
                [
                    ["A*11:01:01G", "A*26:01:01G"],
                    ["A*11:01:07", "A*26:01:17"],
                    ["A*11:19", "A*26:13"],
                ],
            ),
        ],
    )
    def test_get_alleles(
        self,
        easyhla: EasyHLA,
        best_matches: List[HLACombinedStandardResult],
        exp_ambig: bool,
        exp_alleles: List[List[str]],
    ):
        result_ambig, result_alleles = easyhla.get_alleles(
            easyhla.letter, best_matches=best_matches
        )

        assert result_ambig == exp_ambig
        assert result_alleles == exp_alleles


@pytest.mark.parametrize("easyhla", ["B"], indirect=True)
class TestEasyHLADiscreteHLATypeB:
    """
    Testing EasyHLA where tests require HLA type B.
    """

    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-b00-short", 1),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b01-short", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b02-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b-maxgood", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-b-good-exon2", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-b-good-exon3", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-b00-bad-exon2-short", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-b00-bad-exon3-short", -1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-b01-bad-exon2-short",
                -1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-b01-bad-exon3-short",
                -1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-bad-exon3", 0),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2-short", -1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3-short", -1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_b(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            result = easyhla.check_length(
                letter=easyhla.letter, seq=sequence, name=name
            )

            assert bool(exp_return) == result
        else:
            with pytest.raises(ValueError):
                result = easyhla.check_length(
                    letter=easyhla.letter, seq=sequence, name=name
                )


@pytest.mark.parametrize("easyhla", ["C"], indirect=True)
class TestEasyHLADiscreteHLATypeC:
    """
    Testing EasyHLA where tests require HLA type C.
    """

    @pytest.mark.parametrize(
        "sequence, name, exp_return",
        [
            ("A" * EasyHLA.HLA_A_LENGTH, "myseq-c00-short", 1),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c01-short", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c02-short", 0),
            #
            ("A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c-mingood", 1),
            ("A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c-maxgood", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-c-good-exon2", 1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-c-good-exon3", 1),
            #
            ("A" * EasyHLA.EXON2_LENGTH, "myseq-c00-bad-exon2-short", -1),
            ("A" * EasyHLA.EXON3_LENGTH, "myseq-c00-bad-exon3-short", -1),
            #
            (
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-c01-bad-exon2-short",
                -1,
            ),
            (
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-c01-bad-exon3-short",
                -1,
            ),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c02-good-exon3-short", 1),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3", 0),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-bad-exon2", 0),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-bad-exon3", 0),
            #
            ("A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2-short", -1),
            ("A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3-short", -1),
            ("A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-good-exon3-short", 1),
        ],
    )
    def test_check_length_hla_type_c(
        self, easyhla: EasyHLA, sequence: str, name: str, exp_return: int
    ):
        if exp_return > 0:
            result = easyhla.check_length(
                letter=easyhla.letter, seq=sequence, name=name
            )

            assert bool(exp_return) == result
        else:
            with pytest.raises(ValueError):
                result = easyhla.check_length(
                    letter=easyhla.letter, seq=sequence, name=name
                )

    @pytest.mark.integration
    def test_run(self, easyhla: EasyHLA):
        """
        Integration test, assert that pyEasyHLA produces an identical output to
        the original Ruby output.
        """
        input_file = os.path.dirname(__file__) + "/input/test.fasta"
        ref_output_file = os.path.dirname(__file__) + "/output/hla-c-output.csv"
        output_file = os.path.dirname(__file__) + "/output/test.csv"

        easyhla.run(
            easyhla.letter,
            input_file,
            output_file,
            0,
        )

        compare_ref_vs_test(
            easyhla=easyhla,
            reference_output_file=ref_output_file,
            output_file=output_file,
        )


@pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
class TestEasyHLA:
    @pytest.mark.parametrize(
        "sequence, exp_result",
        [
            ("A", 1),
            ("C", 1),
            ("G", 1),
            ("T", 1),
            ("R", 1),
            ("Y", 1),
            ("K", 1),
            ("M", 1),
            ("S", 1),
            ("W", 1),
            ("V", 1),
            ("H", 1),
            ("D", 1),
            ("B", 1),
            ("N", 1),
            ("Z", -1),
            ("AZ", -1),
            ("aZ", -1),
            ("CZ", -1),
            ("cZ", -1),
            ("GZ", -1),
            ("gZ", -1),
            ("TZ", -1),
            ("tZ", -1),
        ],
    )
    def test_check_bases(self, easyhla: EasyHLA, sequence: str, exp_result: int):
        if exp_result > 0:
            result = easyhla.check_bases(seq=sequence, name=sequence)
            assert exp_result == result
        else:
            with pytest.raises(ValueError):
                result = easyhla.check_bases(seq=sequence, name=sequence)

    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("ACGT", [1, 2, 4, 8]),
            ("YANK", [10, 1, 15, 12]),
            ("TARDY", [8, 1, 5, 11, 10]),
            ("MRMAN", [3, 5, 3, 1, 15]),
            ("MYSWARD", [3, 10, 6, 9, 1, 5, 11]),
            # This is where I pull out scrabblewordfinder.org
            ("GANTRY", [4, 1, 15, 8, 5, 10]),
            ("SKYWATCH", [6, 12, 10, 9, 1, 8, 2, 13]),
            ("THWACK", [8, 13, 9, 1, 2, 12]),
        ],
    )
    def test_nuc2bin_bin2nuc(
        self, easyhla: EasyHLA, sequence_str: str, sequence_list: List[int]
    ):
        """
        Test that we can convert back and forth between a list of binary values
        and strings
        """
        result_str = easyhla.bin2nuc(sequence_list)
        assert result_str == sequence_str
        result_list = easyhla.nuc2bin(sequence_str)
        assert result_list == sequence_list

    @pytest.mark.parametrize(
        "HLAStandard, sequence, exp_left_pad, exp_right_pad",
        [
            # NOTE: It breaks my mind less if I hand it letters
            ("ACGT", "ACGT", 0, 0),
            ("ACGTTTT", "ACGT", 0, 3),
            ("TTACGTTTT", "ACGT", 2, 3),
            ("TTACGTTGT", "ATCGT", 1, 3),
            ("TTACGTTTG", "ATTCGT", 0, 3),
        ],
    )
    def test_calc_padding(
        self,
        easyhla: EasyHLA,
        HLAStandard: str,
        sequence: str,
        exp_left_pad: int,
        exp_right_pad: int,
    ):
        std = easyhla.nuc2bin(HLAStandard)
        seq = easyhla.nuc2bin(sequence)
        left_pad, right_pad = easyhla.calc_padding(std, seq)
        print(left_pad, right_pad)
        assert left_pad == exp_left_pad
        assert right_pad == exp_right_pad

    @pytest.mark.parametrize(
        "sequence, standard, exp_result",
        [
            #
            ([1, 2, 4, 8], [1, 2, 4, 8], 0),
            ([1, 2, 4, 8], [1, 2, 4, 4], 1),
            ([1, 2, 4, 8], [8, 4, 2, 1], 4),
        ],
    )
    def test_std_match(
        self,
        easyhla: EasyHLA,
        sequence: List[int],
        standard: List[int],
        exp_result: int,
    ):
        result = easyhla.std_match(std=standard, seq=sequence)
        print(result)
        assert result == exp_result

    @pytest.mark.parametrize(
        "sequence, hla_stds, exp_result",
        [
            #
            (
                [1, 2, 4, 8],
                [HLAStandard(allele="std_allmismatch", sequence=[1, 2, 4, 8])],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[1, 2, 4, 8], mismatch=0
                    )
                ],
            ),
            (
                [1, 2, 4, 8],
                [HLAStandard(allele="std_allmismatch", sequence=[1, 2, 4, 4])],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[1, 2, 4, 4], mismatch=1
                    )
                ],
            ),
            (
                [1, 2, 4, 8],
                [HLAStandard(allele="std_allmismatch", sequence=[8, 4, 2, 1])],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[8, 4, 2, 1], mismatch=4
                    )
                ],
            ),
            #
            (
                [1, 2, 4, 8],
                [
                    HLAStandard(allele="std_allmatch", sequence=[1, 2, 4, 8]),
                    HLAStandard(allele="std_1mismatch", sequence=[1, 2, 4, 4]),
                    HLAStandard(allele="std_allmismatch", sequence=[8, 4, 2, 1]),
                ],
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 8], mismatch=0
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch", sequence=[1, 2, 4, 4], mismatch=1
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[8, 4, 2, 1], mismatch=4
                    ),
                ],
            ),
        ],
        ids=[
            "std_allmatch",
            "std_1mismatch",
            "std_allmismatch",
            "multi_stdmatchmismatch",
        ],
    )
    def test_get_matching_stds(
        self,
        easyhla: EasyHLA,
        sequence: List[int],
        hla_stds: List[HLAStandard],
        exp_result: List[HLAStandardMatch],
    ):
        result = easyhla.get_matching_stds(seq=sequence, hla_stds=hla_stds)
        print(result)
        assert result == exp_result

    def test_load_hla_stds(self, easyhla, hla_standard_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_standard_file)
        exp_result = [
            HLAStandard(allele="HELLO-WORLD", sequence=[1, 1, 1, 1, 2, 1, 5, 8, 10])
        ]

        result = easyhla.load_hla_stds(easyhla.letter)
        assert result == exp_result

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {"AB|CD,12|34": 1}

        result = easyhla.load_hla_frequencies(easyhla.letter)
        assert result == exp_result

    @pytest.mark.parametrize(
        "sequence, name, hla_std, exp_result",
        [
            #
            (
                "ACGT",
                "ACGT",
                HLAStandard(allele="std", sequence=[1, 2, 4, 8]),
                [1, 2, 4, 8],
            ),
            (
                "ACGT",
                "ACGT-exon2",
                HLAStandard(allele="std", sequence=[1, 2, 4, 8]),
                [1, 2, 4, 8],
            ),
            (
                "ACGT",
                "ACGT-exon3",
                HLAStandard(allele="std", sequence=[1, 2, 4, 8]),
                [1, 2, 4, 8],
            ),
            # This is going to be an absolute nightmare to test
            # Full test with intron
            (
                "A" * EasyHLA.EXON2_LENGTH + "ACGT" + "A" * EasyHLA.EXON3_LENGTH,
                "ACGT",
                HLAStandard(
                    allele="std",
                    sequence=[1, 2, 4, 8],
                ),
                [
                    *([1] * EasyHLA.EXON2_LENGTH),
                    1,
                    2,
                    4,
                    8,
                    *([1] * EasyHLA.EXON3_LENGTH),
                ],
            ),
            # Full test with exon2
            (
                "A" * (EasyHLA.EXON2_LENGTH - 4) + "ACGT",
                "ACGT-full-exon2",
                HLAStandard(
                    allele="std",
                    sequence=[
                        *([1] * EasyHLA.EXON2_LENGTH),
                        *([1] * EasyHLA.EXON3_LENGTH),
                    ],
                ),
                [
                    *([1] * int(EasyHLA.EXON2_LENGTH - 4)),
                    1,
                    2,
                    4,
                    8,
                ],
            ),
            # Full test with exon3
            (
                "ACGT" + "A" * (EasyHLA.EXON3_LENGTH - 4),
                "ACGT-full-exon3",
                HLAStandard(
                    allele="std",
                    sequence=[
                        *([1] * EasyHLA.EXON2_LENGTH),
                        *([1] * EasyHLA.EXON3_LENGTH),
                    ],
                ),
                [
                    1,
                    2,
                    4,
                    8,
                    *([1] * int(EasyHLA.EXON3_LENGTH - 4)),
                ],
            ),
            # Full test two possible choices, should select the best match which
            # is the second string of 5s
            (
                "A" * (EasyHLA.EXON2_LENGTH)
                + "RRRRRR"
                + "A" * (EasyHLA.EXON3_LENGTH - 6),
                "RRRRRR-two-options-choose-last",
                HLAStandard(
                    allele="std",
                    sequence=[
                        *([4] * (int(EasyHLA.EXON2_LENGTH / 2) - 2)),
                        5,
                        4,
                        5,
                        5,
                        *([4] * (int(EasyHLA.EXON2_LENGTH / 2) - 2)),
                        5,
                        5,
                        5,
                        5,
                        5,
                        *([4] * (EasyHLA.EXON3_LENGTH - 5)),
                    ],
                ),
                [
                    *([15] * 2),
                    *([1] * int(EasyHLA.EXON2_LENGTH)),
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    1,
                    *([1] * int(EasyHLA.EXON3_LENGTH - 7)),
                ],
            ),
        ],
    )
    def test_pad_short(
        self,
        easyhla: EasyHLA,
        sequence: str,
        name: str,
        hla_std: HLAStandard,
        exp_result: List[int],
    ):
        bin_list = easyhla.nuc2bin(sequence)
        result = easyhla.pad_short(seq=bin_list, name=name, hla_std=hla_std)
        # Debug code for future users
        # print(
        #     result,
        #     sum([1 for a in result if a == 1]),
        #     sum([1 for a in result if a == 15]),
        #     len(result),
        # )
        # print(
        #     exp_result,
        #     sum([1 for a in exp_result if a == 1]),
        #     sum([1 for a in exp_result if a == 15]),
        #     len(exp_result),
        # )
        assert result == exp_result

    @pytest.mark.parametrize(
        "sequence, threshold, matching_standards, exp_result",
        [
            # Simple case
            (
                [1, 2, 4, 8],
                1,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 8], mismatch=0
                    ),
                ],
                {
                    0: [
                        HLACombinedStandardResult(
                            standard="1-2-4-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ]
                },
            ),
            # No threshold defined
            (
                [1, 2, 4, 8],
                None,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 8], mismatch=0
                    ),
                ],
                {
                    0: [
                        HLACombinedStandardResult(
                            standard="1-2-4-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ]
                },
            ),
            # Same case but HLAStandardMatch.mismatch is above threshold
            (
                [1, 2, 4, 8],
                1,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 4, 2, 8], mismatch=2
                    ),
                ],
                {
                    2: [
                        HLACombinedStandardResult(
                            standard="1-4-2-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ]
                },
            ),
            #
            (
                [1, 2, 4, 8],
                1,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 8], mismatch=0
                    ),
                    HLAStandardMatch(
                        allele="std_allmatch2", sequence=[1, 4, 4, 8], mismatch=1
                    ),
                ],
                {
                    0: [
                        HLACombinedStandardResult(
                            standard="1-2-4-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ],
                    1: [
                        HLACombinedStandardResult(
                            standard="1-6-4-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch2"]],
                        ),
                        HLACombinedStandardResult(
                            standard="1-4-4-8",
                            discrete_allele_names=[["std_allmatch2", "std_allmatch2"]],
                        ),
                    ],
                },
            ),
            #
            (
                [9, 6, 4, 6],
                1,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 4], mismatch=0
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch2", sequence=[8, 4, 4, 8], mismatch=1
                    ),
                ],
                {
                    1: [
                        HLACombinedStandardResult(
                            standard="9-6-4-12",
                            discrete_allele_names=[["std_1mismatch2", "std_allmatch"]],
                        )
                    ],
                    3: [
                        HLACombinedStandardResult(
                            standard="1-2-4-4",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ],
                },
            ),
            #
            (
                [1, 2, 4, 8],
                0,
                [
                    HLAStandardMatch(
                        allele="std_1mismatch", sequence=[1, 2, 4, 4], mismatch=1
                    )
                ],
                {
                    1: [
                        HLACombinedStandardResult(
                            standard="1-2-4-4",
                            discrete_allele_names=[["std_1mismatch", "std_1mismatch"]],
                        )
                    ]
                },
            ),
            (
                [1, 2, 4, 8],
                0,
                [
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[8, 4, 2, 1], mismatch=4
                    )
                ],
                {
                    4: [
                        HLACombinedStandardResult(
                            standard="8-4-2-1",
                            discrete_allele_names=[
                                ["std_allmismatch", "std_allmismatch"]
                            ],
                        )
                    ]
                },
            ),
            #
            (
                [1, 2, 4, 8],
                0,
                [
                    HLAStandardMatch(
                        allele="std_allmatch", sequence=[1, 2, 4, 8], mismatch=0
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch", sequence=[1, 2, 4, 4], mismatch=1
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch", sequence=[8, 4, 2, 1], mismatch=4
                    ),
                ],
                {
                    0: [
                        HLACombinedStandardResult(
                            standard="1-2-4-8",
                            discrete_allele_names=[["std_allmatch", "std_allmatch"]],
                        )
                    ]
                },
            ),
        ],
    )
    def test_combine_stds(
        self,
        easyhla: EasyHLA,
        sequence: List[int],
        threshold: int,
        matching_standards: List[HLAStandardMatch],
        exp_result: List[int],
    ):
        result = easyhla.combine_stds(
            matching_stds=matching_standards,
            seq=sequence,
            max_mismatch_threshold=threshold,
        )
        assert sorted(result.items()) == [(k, v) for k, v in exp_result.items()]

    @pytest.mark.parametrize(
        "best_matches, exp_homozygous, exp_alleles",
        [
            (
                [
                    HLACombinedStandardResult(
                        standard="",
                        discrete_allele_names=[
                            ["A*02:01:01G", "A*03:01:01G"],
                            ["A*02:01:52", "A*03:01:03"],
                            ["A*02:01:02", "A*03:01:12"],
                            ["A*02:01:36", "A*03:01:38"],
                            ["A*02:237", "A*03:05:01"],
                            ["A*02:26", "A*03:07"],
                            ["A*02:34", "A*03:08"],
                            ["A*02:90", "A*03:09"],
                            ["A*02:24:01", "A*03:17:01"],
                            ["A*02:195", "A*03:23:01"],
                            ["A*02:338", "A*03:95"],
                            ["A*02:35:01", "A*03:108"],
                            ["A*02:86", "A*03:123"],
                            ["A*02:20:01", "A*03:157"],
                        ],
                    )
                ],
                False,
                # NOTE: This is one string concatenated together
                (
                    "A*02:01:01G - A*03:01:01G;A*02:01:02 - A*03:01:12;"
                    "A*02:01:36 - A*03:01:38;A*02:01:52 - A*03:01:03;"
                    "A*02:195 - A*03:23:01;A*02:20:01 - A*03:157;"
                    "A*02:237 - A*03:05:01;A*02:24:01 - A*03:17:01;"
                    "A*02:26 - A*03:07;A*02:338 - A*03:95;"
                    "A*02:34 - A*03:08;A*02:35:01 - A*03:108;"
                    "A*02:86 - A*03:123;A*02:90 - A*03:09"
                ),
            ),
            (
                [
                    HLACombinedStandardResult(
                        standard="",
                        discrete_allele_names=[
                            ["A*11:01:01G", "A*26:01:01G"],
                            ["A*11:01:07", "A*26:01:17"],
                            ["A*11:19", "A*26:13"],
                            ["A*11:40", "A*66:01G"],
                        ],
                    )
                ],
                False,
                (
                    "A*11:01:01G - A*26:01:01G;"
                    "A*11:01:07 - A*26:01:17;"
                    "A*11:19 - A*26:13;"
                    "A*11:40 - A*66:01G"
                ),
            ),
        ],
    )
    def test_get_all_alleles(
        self,
        easyhla: EasyHLA,
        best_matches: List[HLACombinedStandardResult],
        exp_homozygous: bool,
        exp_alleles: List[List[str]],
    ):
        result_homozygous, result_alleles = easyhla.get_all_alleles(
            best_matches=best_matches
        )

        assert result_homozygous == exp_homozygous
        assert result_alleles == exp_alleles

    @pytest.mark.parametrize(
        "alleles, exp_result",
        [
            (
                [
                    ["A*02:01:01G", "A*03:01:01G"],
                    ["A*02:01:52", "A*03:01:03"],
                    ["A*02:01:02", "A*03:01:12"],
                    ["A*02:01:36", "A*03:01:38"],
                    ["A*02:237", "A*03:05:01"],
                    ["A*02:26", "A*03:07"],
                    ["A*02:34", "A*03:08"],
                    ["A*02:90", "A*03:09"],
                    ["A*02:24:01", "A*03:17:01"],
                    ["A*02:195", "A*03:23:01"],
                    ["A*02:338", "A*03:95"],
                    ["A*02:35:01", "A*03:108"],
                    ["A*02:86", "A*03:123"],
                    ["A*02:20:01", "A*03:157"],
                ],
                "A*02 - A*03",
            ),
            (
                [
                    ["A*11:01:01G", "A*26:01:01G"],
                    ["A*11:01:07", "A*26:01:17"],
                    ["A*11:19", "A*26:13"],
                ],
                "A*11 - A*26",
            ),
            (
                [
                    ["A*11:01:07", "A*26:01:17"],
                    ["A*11:40", "A*26:01G"],
                ],
                "A*11 - A*26",
            ),
        ],
    )
    def test_alleles_clean(
        self, easyhla: EasyHLA, alleles: List[str], exp_result: List[str]
    ):
        result = easyhla.get_clean_alleles(all_alleles=alleles)

        assert result == exp_result
