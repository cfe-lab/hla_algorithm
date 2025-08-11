#! /usr/bin/env python

import argparse
import json
import logging

from .hla_algorithm import HLAAlgorithm
from .interpret_from_json_lib import HLAInput, HLAResult
from .models import HLAInterpretation

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
        hla_alg: HLAAlgorithm = HLAAlgorithm.use_config(
            hla_input.hla_std_path,
            hla_input.hla_freq_path,
        )
        interp: HLAInterpretation = hla_alg.interpret(hla_input.hla_sequence())
        print(HLAResult.build_from_interpretation(interp).model_dump_json())


if __name__ == "__main__":
    main()
