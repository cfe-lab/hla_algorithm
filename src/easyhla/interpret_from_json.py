#! /usr/bin/env python

import argparse
import json
import logging
from typing import Optional

from .easyhla import EasyHLA
from .interpret_from_json_lib import HLAInput, HLAResult
from .models import (
    HLAInterpretation,
    HLAProteinPair,
    HLAStandard,
)

logger: logging.Logger = logging.getLogger(__name__)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Produce an HLA interpretation from a JSON input"
    )
    parser.add_argument(
        "infile",
        type=argparse.FileType("r"),
        help='Input file containing the JSON input (use "-" to read from stdin)',
    )
    args: argparse.Namespace = parser.parse_args()

    hla_input_str: str = ""
    with args.infile:
        for line in args.infile:
            hla_input_str += f"{line}\n"

    hla_input: HLAInput = HLAInput(**json.loads(hla_input_str))

    errors: list[str] = hla_input.check_sequences()
    if len(errors) > 0:
        error_result: HLAResult = HLAResult(errors=errors)
        print(error_result.model_dump_json())
    else:
        curr_standards: Optional[dict[str, HLAStandard]] = None
        if hla_input.hla_std_path is not None:
            with open(hla_input.hla_std_path) as f:
                curr_standards = EasyHLA.read_hla_standards(f)

        curr_frequencies: Optional[dict[HLAProteinPair, int]] = None
        if hla_input.hla_freq_path is not None:
            with open(hla_input.hla_freq_path) as f:
                curr_frequencies = EasyHLA.read_hla_frequencies(hla_input.locus, f)

        easyhla: EasyHLA = EasyHLA(
            locus=hla_input.locus,
            hla_standards=curr_standards,
            hla_frequencies=curr_frequencies,
        )
        interp: HLAInterpretation = easyhla.interpret(hla_input.hla_sequence())
        print(HLAResult.build_from_interpretation(interp).model_dump_json())


if __name__ == "__main__":
    main()
