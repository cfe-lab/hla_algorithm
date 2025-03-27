import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import pytz

from easyhla.easyhla import DATE_FORMAT, EXON_NAME, EasyHLA
from easyhla.models import (
    HLACombinedStandard,
    HLAProteinPair,
    HLAStandard,
    HLAStandardMatch,
)

from .conftest import compare_ref_vs_test


@pytest.fixture(scope="module")
def easyhla(request: pytest.FixtureRequest):
    easyhla = EasyHLA(locus=request.param)
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
    p.write_text("2233,1423,5701,5703,4043,2529")

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


class TestCombineStandards:
    @pytest.mark.parametrize(
        "sequence, matching_standards, thresholds, exp_result",
        [
            # Simple case
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                ],
                [0, 1, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 4, 2, 8]),
                        mismatch=2,
                    ),
                ],
                [0, 1, 2, 3, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 4, 2, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 2,
                },
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_allmatch2",
                        sequence=np.array([1, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [0],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_allmatch2",
                        sequence=np.array([1, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [1, 2, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                    HLACombinedStandard(
                        standard_bin=(1, 6, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch2"),),
                    ): 1,
                    HLACombinedStandard(
                        standard_bin=(1, 4, 4, 8),
                        possible_allele_pairs=(("std_allmatch2", "std_allmatch2"),),
                    ): 1,
                },
            ),
            #
            (
                np.array([9, 6, 4, 6]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch2",
                        sequence=np.array([8, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [0, 1, 2],
                {
                    HLACombinedStandard(
                        standard_bin=(9, 6, 4, 12),
                        possible_allele_pairs=(("std_1mismatch2", "std_allmatch"),),
                    ): 1,
                },
            ),
            #
            (
                np.array([9, 6, 4, 6]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch2",
                        sequence=np.array([8, 4, 4, 8]),
                        mismatch=1,
                    ),
                ],
                [3, 4, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(9, 6, 4, 12),
                        possible_allele_pairs=(("std_1mismatch2", "std_allmatch"),),
                    ): 1,
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 3,
                    HLACombinedStandard(
                        standard_bin=(8, 4, 4, 8),
                        possible_allele_pairs=(("std_1mismatch2", "std_1mismatch2"),),
                    ): 3,
                },
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    )
                ],
                [0, 1, 2, 5],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                    ): 1
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    )
                ],
                [0, 1, 3, 4, 5, 10],
                {
                    HLACombinedStandard(
                        standard_bin=(8, 4, 2, 1),
                        possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                    ): 4,
                },
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    ),
                ],
                [0],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    ),
                ],
                [1, 2, 3],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 12),
                        possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                    ): 1,
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                    ): 1,
                },
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    ),
                ],
                [4, 5, 10],
                {
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 8),
                        possible_allele_pairs=(("std_allmatch", "std_allmatch"),),
                    ): 0,
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 12),
                        possible_allele_pairs=(("std_1mismatch", "std_allmatch"),),
                    ): 1,
                    HLACombinedStandard(
                        standard_bin=(1, 2, 4, 4),
                        possible_allele_pairs=(("std_1mismatch", "std_1mismatch"),),
                    ): 1,
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 9),
                        possible_allele_pairs=(("std_allmatch", "std_allmismatch"),),
                    ): 4,
                    HLACombinedStandard(
                        standard_bin=(9, 6, 6, 5),
                        possible_allele_pairs=(("std_1mismatch", "std_allmismatch"),),
                    ): 4,
                    HLACombinedStandard(
                        standard_bin=(8, 4, 2, 1),
                        possible_allele_pairs=(("std_allmismatch", "std_allmismatch"),),
                    ): 4,
                },
            ),
        ],
    )
    def test_combine_standards(
        self,
        sequence: list[int],
        matching_standards: list[HLAStandardMatch],
        thresholds: list[int],
        exp_result: dict[int, list[int]],
    ):
        for threshold in thresholds:
            result = EasyHLA.combine_standards(
                matching_stds=matching_standards,
                seq=sequence,
                mismatch_threshold=threshold,
            )
            assert result == exp_result


class TestEasyHLAMisc:
    def test_unknown_hla_gene(self):
        """
        Assert we raise a value error if we put in an unknown HLA gene.
        """
        with pytest.raises(ValueError):
            _ = EasyHLA("D")  # type: ignore[arg-type]

    def test_known_hla_gene_lowercase(self):
        """
        Assert we raise a value error if we put in an HLA gene with wrong case.
        """
        with pytest.raises(ValueError):
            _ = EasyHLA("a")  # type: ignore[arg-type]

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
class TestEasyHLADiscreteHLALocusA:
    """
    Testing EasyHLA where tests require HLA-A.
    """

    # Tests of check_length:
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
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="22",
                first_field_2="33",
                second_field_1="14",
                second_field_2="23",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
        assert result == exp_result


@pytest.mark.parametrize("easyhla", ["B"], indirect=True)
class TestEasyHLADiscreteHLALocusB:
    """
    Testing EasyHLA where tests require HLA-B.
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
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="57",
                first_field_2="01",
                second_field_1="57",
                second_field_2="03",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
        assert result == exp_result


@pytest.mark.parametrize("easyhla", ["C"], indirect=True)
class TestEasyHLADiscreteHLALocusC:
    """
    Testing EasyHLA where tests require HLA-C.
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
            easyhla.check_length(seq=sequence, name=name)
        else:
            with pytest.raises(ValueError):
                easyhla.check_length(seq=sequence, name=name)

    def test_load_hla_freqs(self, easyhla, hla_frequency_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_frequency_file)
        exp_result = {
            HLAProteinPair(
                first_field_1="40",
                first_field_2="43",
                second_field_1="25",
                second_field_2="29",
            ): 1,
        }
        result = easyhla.load_hla_frequencies()
        assert result == exp_result

    # @pytest.mark.integration
    # def test_run(self, easyhla: EasyHLA):
    #     """
    #     Integration test, assert that pyEasyHLA produces an identical output to
    #     the original Ruby output.
    #     """
    #     input_file = os.path.dirname(__file__) + "/input/test.fasta"
    #     ref_output_file = os.path.dirname(__file__) + "/output/hla-c-output.csv"
    #     output_file = os.path.dirname(__file__) + "/output/test.csv"

    #     easyhla.run(
    #         input_file,
    #         output_file,
    #         0,
    #     )

    #     compare_ref_vs_test(
    #         easyhla=easyhla,
    #         reference_output_file=ref_output_file,
    #         output_file=output_file,
    #     )


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
            easyhla.check_bases(seq=sequence)
        else:
            with pytest.raises(ValueError):
                easyhla.check_bases(seq=sequence)

    # FIXME: something with a V and something with a B
    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("ACGT", np.array([1, 2, 4, 8])),
            ("YANK", np.array([10, 1, 15, 12])),
            ("TARDY", np.array([8, 1, 5, 11, 10])),
            ("MRMAN", np.array([3, 5, 3, 1, 15])),
            ("MYSWARD", np.array([3, 10, 6, 9, 1, 5, 11])),
            # This is where I pull out scrabblewordfinder.org
            ("GANTRY", np.array([4, 1, 15, 8, 5, 10])),
            ("SKYWATCH", np.array([6, 12, 10, 9, 1, 8, 2, 13])),
            ("THWACK", np.array([8, 13, 9, 1, 2, 12])),
        ],
    )
    def test_nuc2bin_bin2nuc(
        self, easyhla: EasyHLA, sequence_str: str, sequence_list: np.ndarray
    ):
        """
        Test that we can convert back and forth between a list of binary values
        and strings
        """
        result_str = easyhla.bin2nuc(sequence_list)
        assert result_str == sequence_str
        result_list = easyhla.nuc2bin(sequence_str)
        assert np.array_equal(result_list, sequence_list)

    # FIXME: add some tests when there are ties
    @pytest.mark.parametrize(
        "hla_standard, sequence, exp_left_pad, exp_right_pad",
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
        hla_standard: str,
        sequence: str,
        exp_left_pad: int,
        exp_right_pad: int,
    ):
        std = easyhla.nuc2bin(hla_standard)
        seq = easyhla.nuc2bin(sequence)
        left_pad, right_pad = easyhla.calc_padding(std, seq)
        print(left_pad, right_pad)
        assert left_pad == exp_left_pad
        assert right_pad == exp_right_pad

    # FIXME: add some tests for mixtures
    @pytest.mark.parametrize(
        "sequence, standard, exp_result",
        [
            #
            ([1, 2, 4, 8], [1, 2, 4, 8], 0),
            ([1, 2, 4, 8], [1, 2, 4, 4], 1),
            ([1, 2, 4, 8], [8, 4, 2, 1], 4),
            ([1, 2, 4, 8], [5, 2, 6, 12], 0),
            ([5, 2, 6, 12], [1, 2, 4, 8], 0),
            ([1, 2, 7, 2], [2, 3, 6, 2], 1),
        ],
    )
    def test_std_match(
        self,
        easyhla: EasyHLA,
        sequence: list[int],
        standard: list[int],
        exp_result: int,
    ):
        result = easyhla.std_match(std=np.array(standard), seq=np.array(sequence))
        print(result)
        assert result == exp_result

    # FIXME: some mixtures here too? and maybe some ties?
    @pytest.mark.parametrize(
        "sequence, hla_stds, exp_result",
        [
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([1, 2, 4, 8])
                    )
                ],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    )
                ],
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([1, 2, 4, 4])
                    )
                ],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    )
                ],
            ),
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([8, 4, 2, 1])
                    )
                ],
                [
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
                    )
                ],
            ),
            #
            (
                np.array([1, 2, 4, 8]),
                [
                    HLAStandard(allele="std_allmatch", sequence=np.array([1, 2, 4, 8])),
                    HLAStandard(
                        allele="std_1mismatch", sequence=np.array([1, 2, 4, 4])
                    ),
                    HLAStandard(
                        allele="std_allmismatch", sequence=np.array([8, 4, 2, 1])
                    ),
                ],
                [
                    HLAStandardMatch(
                        allele="std_allmatch",
                        sequence=np.array([1, 2, 4, 8]),
                        mismatch=0,
                    ),
                    HLAStandardMatch(
                        allele="std_1mismatch",
                        sequence=np.array([1, 2, 4, 4]),
                        mismatch=1,
                    ),
                    HLAStandardMatch(
                        allele="std_allmismatch",
                        sequence=np.array([8, 4, 2, 1]),
                        mismatch=4,
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
        sequence: list[int],
        hla_stds: list[HLAStandard],
        exp_result: list[HLAStandardMatch],
    ):
        result = easyhla.get_matching_stds(seq=sequence, hla_stds=hla_stds)  # type: ignore
        print(result)
        assert result == exp_result

    def test_load_hla_stds(self, easyhla, hla_standard_file, mocker):
        mocker.patch.object(os.path, "join", return_value=hla_standard_file)
        exp_result = [
            HLAStandard(
                allele="HELLO-WORLD", sequence=np.array([1, 1, 1, 1, 2, 1, 5, 8, 10])
            )
        ]

        result = easyhla.load_hla_stds()
        assert result == exp_result

    @pytest.mark.parametrize(
        "sequence, exon, hla_std, exp_result",
        [
            #
            (
                "ACGT",
                None,
                HLAStandard(allele="std", sequence=np.array([1, 2, 4, 8])),
                np.array([1, 2, 4, 8]),
            ),
            (
                "ACGT",
                "exon2",
                HLAStandard(allele="std", sequence=np.array([1, 2, 4, 8])),
                np.array([1, 2, 4, 8]),
            ),
            (
                "ACGT",
                "exon3",
                HLAStandard(allele="std", sequence=np.array([1, 2, 4, 8])),
                np.array([1, 2, 4, 8]),
            ),
            # This is going to be an absolute nightmare to test
            # Full test with intron
            (
                "A" * EasyHLA.EXON2_LENGTH + "ACGT" + "A" * EasyHLA.EXON3_LENGTH,
                None,
                HLAStandard(
                    allele="std",
                    sequence=np.array([1, 2, 4, 8]),
                ),
                np.array(
                    [
                        *([1] * EasyHLA.EXON2_LENGTH),
                        1,
                        2,
                        4,
                        8,
                        *([1] * EasyHLA.EXON3_LENGTH),
                    ]
                ),
            ),
            # Full test with exon2
            (
                "A" * (EasyHLA.EXON2_LENGTH - 4) + "ACGT",
                "exon2",
                HLAStandard(
                    allele="std",
                    sequence=np.array(
                        [
                            *([1] * EasyHLA.EXON2_LENGTH),
                            *([1] * EasyHLA.EXON3_LENGTH),
                        ]
                    ),
                ),
                np.array(
                    [
                        *([1] * int(EasyHLA.EXON2_LENGTH - 4)),
                        1,
                        2,
                        4,
                        8,
                    ]
                ),
            ),
            # Full test with exon3
            (
                "ACGT" + "A" * (EasyHLA.EXON3_LENGTH - 4),
                "exon3",
                HLAStandard(
                    allele="std",
                    sequence=np.array(
                        [
                            *([1] * EasyHLA.EXON2_LENGTH),
                            *([1] * EasyHLA.EXON3_LENGTH),
                        ]
                    ),
                ),
                np.array(
                    [
                        1,
                        2,
                        4,
                        8,
                        *([1] * int(EasyHLA.EXON3_LENGTH - 4)),
                    ]
                ),
            ),
            # Full test two possible choices, should select the best match which
            # is the second string of 5s
            (
                "A" * (EasyHLA.EXON2_LENGTH)
                + "RRRRRR"
                + "A" * (EasyHLA.EXON3_LENGTH - 6),
                None,
                HLAStandard(
                    allele="std",
                    sequence=np.array(
                        [
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
                        ]
                    ),
                ),
                np.array(
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
                    ]
                ),
            ),
        ],
    )
    def test_pad_short(
        self,
        easyhla: EasyHLA,
        sequence: str,
        exon: EXON_NAME,
        hla_std: HLAStandard,
        exp_result: np.ndarray,
    ):
        bin_list = easyhla.nuc2bin(sequence)
        result = easyhla.pad_short(seq=bin_list, exon=exon, hla_std=hla_std)
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
        assert np.array_equal(result, exp_result)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self, easyhla: EasyHLA):
        """
        Integration test, assert that pyEasyHLA produces an identical output to
        the original Ruby output.
        """

        input_file = (
            os.path.dirname(__file__) + f"/input/hla-{easyhla.locus.lower()}-seqs.fasta"
        )
        ref_output_file = (
            os.path.dirname(__file__)
            + f"/output/hla-{easyhla.locus.lower()}-output-ref.csv"
        )
        output_file = (
            os.path.dirname(__file__) + f"/output/hla-{easyhla.locus.lower()}-test.csv"
        )

        if not os.path.exists(input_file):
            pytest.skip("Input sequence does not exist!")
        if not os.path.exists(ref_output_file):
            pytest.skip("Reference output does not exist!")

        start_time = datetime.now()
        print(f"Test started at {start_time.isoformat()}")

        easyhla.run(
            input_file,
            output_file,
            0,
        )

        end_time = datetime.now()

        print(f"Interpretation ended at {end_time.isoformat()}")

        compare_ref_vs_test(
            easyhla=easyhla,
            reference_output_file=ref_output_file,
            output_file=output_file,
        )

        end_compare_time = datetime.now()

        print(f"Test ended at {end_compare_time.isoformat()}")

        print(f"Time elapsed: {(end_compare_time - start_time).total_seconds()}")
        print(
            f"Time elapsed for interpretation: {(end_time - start_time).total_seconds()}"
        )
        print(
            f"Time elapsed for output comparison: {(end_compare_time - end_time).total_seconds()}"
        )
