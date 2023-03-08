"""
Perform a manual comparison between two output files.

For help use `python tests/manual_comparison.py --help`.
"""

import os
from src.easyhla.easyhla import EasyHLA
from conftest import compare_ref_vs_test, make_comparison
import typer


def main(
    reference_file: str = typer.Option(
        "/output/hla-a-output.csv",
        "--ref",
        "--reference",
        "-r",
        help="Reference file relative to this file",
    ),
    test_file: str = typer.Option(
        "/output/output.csv",
        "--file",
        "-f",
        help="Test output file relative to this file",
    ),
    skip_preamble: bool = typer.Option(
        False,
        "--no-skip-preamble",
        "-s",
        is_flag=True,
        flag_value=True,
        help="If both files begin with a timestamp, skip that line",
    ),
    skip_preamble_ref: bool = typer.Option(
        False,
        "--no-skip-preamble-ref",
        is_flag=True,
        flag_value=True,
        help="If reference file begins with a timestamp, skip that line",
    ),
    skip_preamble_out: bool = typer.Option(
        False,
        "--no-skip-preamble-output",
        is_flag=True,
        flag_value=True,
        help="If output file begins with a timestamp, skip that line",
    ),
) -> None:
    ref_output_file = os.path.dirname(__file__) + reference_file
    output_file = os.path.dirname(__file__) + test_file

    easyhla = EasyHLA("A")

    _skip_preamble = (
        skip_preamble or skip_preamble_ref,
        skip_preamble or skip_preamble_out,
    )

    with open(ref_output_file, "r", encoding="utf-8") as f_reference:
        reference_file = f_reference.readlines()[int(_skip_preamble[0]) :]
    with open(output_file, "r", encoding="utf-8") as f_test_output:
        test_output_file = f_test_output.readlines()[int(_skip_preamble[1]) :]

    column_names = reference_file[0].strip().split(",")

    assert len(reference_file) == len(
        test_output_file
    ), "Size of test output does not match reference file!"

    for row_num, (ref, test) in enumerate(zip(reference_file, test_output_file)):
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

            _row_num = row_num + 1 + int(_skip_preamble[1])
            # Check that there is no strippable whitespace when there shouldn't be.
            if _test != _test.strip():
                print(
                    f"Whitespace detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )
            if _ref != _ref.strip():
                print(
                    f"[REFERENCE FILE] Whitespace detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )

            if len(_ref.strip()) != len(_test.strip()):
                print(
                    f"Length mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )
            if _ref.strip() != _test.strip():
                print(
                    f"Content mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                )

            if row_num > 0 and column_names[col_num] in [
                "ALLELES_CLEAN",
                "ALLELES",
            ]:
                if (
                    _ref.strip() != _test.strip()
                    and len(_ref.strip()) == len(_test.strip())
                    and set(_ref.strip().split(";")) ^ set(_test.strip().split(";"))
                    == set()
                ):
                    print(
                        f"Order mismatch detected at row {_row_num}, column {col_num} ('{column_names[col_num]}')"
                    )


if __name__ == "__main__":
    typer.run(main)
