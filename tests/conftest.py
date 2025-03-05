from typing import Tuple

from easyhla import EasyHLA


def make_comparison(easyhla: EasyHLA, ref_seq: str, test_seq: str) -> str:
    """
    Compares two sequences for differences

    :param easyhla: EasyHLA object
    :type easyhla: EasyHLA
    :param ref_seq: ...
    :type ref_seq: str
    :param test_seq: ...
    :type test_seq: str
    :return: A sequence where mismatches are replaced with '_'
    :rtype: str
    """
    ref, test = easyhla.nuc2bin(ref_seq.strip()), easyhla.nuc2bin(test_seq.strip())
    masked_seq = []

    for i in range(max(len(ref), len(test))):
        _r = ref[i] if i < len(ref) else 0
        _t = test[i] if i < len(test) else 0

        masked_seq.append(_r & _t)

    if len(ref) != len(test):
        if len(ref) > len(test):
            side_is_short = "test"
        elif len(ref) < len(test):
            side_is_short = "ref"
        return easyhla.bin2nuc(masked_seq) + f" [{side_is_short} is short]"  # type: ignore
    return easyhla.bin2nuc(masked_seq)  # type: ignore


def compare_ref_vs_test(
    easyhla: EasyHLA,
    reference_output_file: str,
    output_file: str,
    skip_preamble: Tuple[bool, bool] = (True, True),
):
    """
    Compares a reference output file versus a newly generated output file.

    Checks for length, content, and order between each element in a row for all
    rows.

    :param easyhla: ...
    :type easyhla: EasyHLA
    :param reference_output_file: ...
    :type reference_output_file: str
    :param output_file: ...
    :type output_file: str
    """
    with open(reference_output_file, "r", encoding="utf-8") as f_reference:
        reference_file = f_reference.readlines()[int(skip_preamble[0]) :]
    with open(output_file, "r", encoding="utf-8") as f_test_output:
        test_output_file = f_test_output.readlines()[int(skip_preamble[1]) :]

    column_names = reference_file[0].strip().split(",")

    assert len(reference_file) == len(test_output_file), (
        "Size of test output does not match reference file!"
    )

    for row_num, (ref, test) in enumerate(zip(reference_file, test_output_file)):
        try:
            for col_num, (_ref, _test) in enumerate(
                zip(ref.strip().split(","), test.strip().split(","))
            ):
                if row_num > 0 and column_names[col_num] in [
                    "EXON2",
                    "INTRON",
                    "EXON3",
                ]:
                    comparison = make_comparison(easyhla, _ref, _test)
                    if "_" in comparison:
                        print(
                            ">>>",
                            column_names[col_num],
                            comparison,
                            len(_ref),
                            len(_test),
                        )

                _row_num = row_num + 1 + int(skip_preamble[1])
                # Check that there is no strippable whitespace when there shouldn't be.
                assert _test == _test.strip(), (
                    f"Whitespace detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )
                assert _ref == _ref.strip(), (
                    f"[REFERENCE FILE] Whitespace detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )

                assert len(_ref.strip()) == len(_test.strip()), (
                    f"Length mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )
                assert _ref.strip() == _test.strip(), (
                    f"Content mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )

                if row_num > 0 and column_names[col_num] in [
                    "ALLELES_CLEAN",
                    "ALLELES",
                ]:
                    assert (
                        set(_ref.strip().split(";")) ^ set(_test.strip().split(";"))
                        == set()
                    ), (
                        f"Order mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                    )
        except AssertionError as e:
            print("REF >>>", ref)
            print("OUT >>>", test)
            raise e
