import pytest
import json
import os
from src.easyhla.easyhla import EasyHLA
from typing import List, Optional, Dict, Literal, Tuple, Any, Union


@pytest.fixture(scope="session")
def easyhla(request: pytest.FixtureRequest):
    easyhla = EasyHLA(letter=request.param)
    return easyhla


class TestEasyHLA:
    """
    Testing EasyHLA
    """

    @pytest.mark.parametrize(
        "easyhla, sequence, name, exp_return",
        [
            ("A", "A" * 1000, "myseq-a-bad", -1),
            ("A", "A" * EasyHLA.HLA_A_LENGTH, "myseq-a00-short", 0),
            ("B", "A" * EasyHLA.HLA_A_LENGTH, "myseq-b00-short", 1),
            ("C", "A" * EasyHLA.HLA_A_LENGTH, "myseq-c00-short", 1),
            #
            ("B", "A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b01-short", 1),
            ("C", "A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c01-short", 1),
            ("B", "A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b02-short", 0),
            ("C", "A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c02-short", 0),
            #
            ("A", "A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-a-mingood", 1),
            ("B", "A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-b-mingood", 1),
            ("C", "A" * EasyHLA.MIN_HLA_BC_LENGTH, "myseq-c-mingood", 1),
            ("A", "A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-a-maxbad", -1),
            ("B", "A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-b-maxgood", 1),
            ("C", "A" * EasyHLA.MAX_HLA_BC_LENGTH, "myseq-c-maxgood", 1),
            #
            ("A", "A" * EasyHLA.EXON2_LENGTH, "myseq-a-bad-exon2", -1),
            ("B", "A" * EasyHLA.EXON2_LENGTH, "myseq-b-good-exon2", 1),
            ("C", "A" * EasyHLA.EXON2_LENGTH, "myseq-c-good-exon2", 1),
            ("A", "A" * EasyHLA.EXON3_LENGTH, "myseq-a-bad-exon3", -1),
            ("B", "A" * EasyHLA.EXON3_LENGTH, "myseq-b-good-exon3", 1),
            ("C", "A" * EasyHLA.EXON3_LENGTH, "myseq-c-good-exon3", 1),
            #
            ("A", "A" * EasyHLA.EXON2_LENGTH, "myseq-a00-good-exon2-short", 1),
            ("B", "A" * EasyHLA.EXON2_LENGTH, "myseq-b00-bad-exon2-short", -1),
            ("C", "A" * EasyHLA.EXON2_LENGTH, "myseq-c00-bad-exon2-short", -1),
            ("A", "A" * EasyHLA.EXON3_LENGTH, "myseq-a00-good-exon3-short", 1),
            ("B", "A" * EasyHLA.EXON3_LENGTH, "myseq-b00-bad-exon3-short", -1),
            ("C", "A" * EasyHLA.EXON3_LENGTH, "myseq-c00-bad-exon3-short", -1),
            #
            (
                "A",
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-a01-good-exon2-short",
                1,
            ),
            (
                "B",
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-b01-bad-exon2-short",
                -1,
            ),
            (
                "C",
                "A" * (EasyHLA.EXON2_LENGTH + 1),
                "myseq-c01-bad-exon2-short",
                -1,
            ),
            (
                "A",
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-a01-good-exon3-short",
                1,
            ),
            (
                "B",
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-b01-bad-exon3-short",
                -1,
            ),
            (
                "C",
                "A" * (EasyHLA.EXON3_LENGTH + 1),
                "myseq-c01-bad-exon3-short",
                -1,
            ),
            ("A", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("B", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("C", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-good-exon3-short", 1),
            ("B", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b02-good-exon3-short", 1),
            ("C", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c02-good-exon3-short", 1),
            #
            ("A", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-bad-exon2", 0),
            ("B", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2", 0),
            ("C", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2", 0),
            ("A", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-bad-exon3", 0),
            ("B", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3", 0),
            ("C", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3", 0),
            ("B", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-bad-exon2", 0),
            ("C", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-bad-exon2", 0),
            ("A", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a02-bad-exon3", 0),
            ("A", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-bad-exon3", 0),
            ("B", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-bad-exon3", 0),
            ("C", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-bad-exon3", 0),
            #
            ("A", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-a00-good-exon2-short", 1),
            ("B", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-b00-bad-exon2-short", -1),
            ("C", "A" * (EasyHLA.EXON2_LENGTH + 1), "myseq-c00-bad-exon2-short", -1),
            ("A", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-a01-good-exon3-short", 1),
            ("B", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-b01-bad-exon3-short", -1),
            ("C", "A" * (EasyHLA.EXON3_LENGTH + 1), "myseq-c01-bad-exon3-short", -1),
            ("A", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-a02-good-exon2-short", 1),
            ("B", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-b02-good-exon2-short", 1),
            ("C", "A" * (EasyHLA.EXON2_LENGTH - 1), "myseq-c02-good-exon2-short", 1),
            ("A", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-a03-good-exon3-short", 1),
            ("B", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-b03-good-exon3-short", 1),
            ("C", "A" * (EasyHLA.EXON3_LENGTH - 1), "myseq-c03-good-exon3-short", 1),
        ],
        indirect=["easyhla"],
    )
    def test_check_length(
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

    @pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
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

    @pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
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

    @pytest.mark.parametrize("easyhla", ["A", "B", "C"], indirect=True)
    @pytest.mark.parametrize(
        "standard, sequence, exp_left_pad, exp_right_pad",
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
        standard: str,
        sequence: str,
        exp_left_pad: int,
        exp_right_pad: int,
    ):
        std = easyhla.nuc2bin(standard)
        seq = easyhla.nuc2bin(sequence)
        left_pad, right_pad = easyhla.calc_padding(std, seq)
        print(left_pad, right_pad)
        assert left_pad == exp_left_pad
        assert right_pad == exp_right_pad
