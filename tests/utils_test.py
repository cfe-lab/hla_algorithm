import numpy as np
import pytest

from easyhla.utils import bin2nuc, nuc2bin


class TestBinaryNucleotideTranslation:
    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("ACGT", np.array([1, 2, 4, 8])),
            ("YANK", np.array([10, 1, 15, 12])),
            ("TARDY", np.array([8, 1, 5, 13, 10])),
            ("MRMAN", np.array([3, 5, 3, 1, 15])),
            ("MYSWARD", np.array([3, 10, 6, 9, 1, 5, 13])),
            # This is where I pull out scrabblewordfinder.org
            ("GANTRY", np.array([4, 1, 15, 8, 5, 10])),
            ("SKYWATCH", np.array([6, 12, 10, 9, 1, 8, 2, 11])),
            ("THWACK", np.array([8, 11, 9, 1, 2, 12])),
            ("VAN", np.array([7, 1, 15])),
            ("ABRA", np.array([1, 14, 5, 1])),
        ],
    )
    def test_nuc2bin_bin2nuc_good_cases(
        self, sequence_str: str, sequence_list: np.ndarray
    ):
        """
        Test that we can convert back and forth between a list of binary values
        and strings.
        """
        result_str = bin2nuc(sequence_list)
        assert result_str == sequence_str
        result_list = nuc2bin(sequence_str)
        assert np.array_equal(result_list, sequence_list)

    @pytest.mark.parametrize(
        "sequence_str, sequence_list",
        [
            ("E", np.array([0])),
            ("a", np.array([0])),
            ("123", np.array([0, 0, 0])),
            ("AC_TT_G", np.array([1, 2, 0, 8, 8, 0, 4])),
        ],
    )
    def test_nuc2bin_bad_characters(self, sequence_str: str, sequence_list: np.ndarray):
        """
        Translating characters that aren't in the mapping turns them into 0s.
        """
        result_list = nuc2bin(sequence_str)
        assert np.array_equal(result_list, sequence_list)

    @pytest.mark.parametrize(
        "sequence_list, sequence_str",
        [
            (np.array([0]), "_"),
            (np.array([16]), "_"),
            (np.array([17]), "_"),
            (np.array([250]), "_"),
            (np.array([1, 2, 0, 8, 8, 0, 4]), "AC_TT_G"),
            (np.array([1, 14, 2, 100, 8, 17, 4]), "ABC_T_G"),
        ],
    )
    def test_bin2nuc_bad_indices(self, sequence_list: np.ndarray, sequence_str: str):
        """
        Indices not in our mapping become underscores.
        """
        result_str = bin2nuc(sequence_list)
        assert np.array_equal(result_str, sequence_str)
