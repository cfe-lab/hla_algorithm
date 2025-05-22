#! /usr/bin/env python

import fileinput
import json
import os
from typing import Final, Optional

from easyhla.easyhla import EasyHLA
from easyhla.models import (
    HLAInterpretation,
    HLAProteinPair,
    HLAStandard,
)
from easyhla.utils import HLA_LOCUS
from easyhla.ruby_adaptor_lib import HLAInput, HLAResult

# These are the "configuration files" that the algorithm uses; these are or may
# be updated, in which case you specify the path to the new version in the
# environment.
HLA_STANDARDS: Final[dict[HLA_LOCUS, Optional[str]]] = {
    "A": os.environ.get("HLA_STANDARDS_A"),
    "B": os.environ.get("HLA_STANDARDS_B"),
    "C": os.environ.get("HLA_STANDARDS_C"),
}
HLA_FREQUENCIES: Final[str] = os.environ.get("HLA_FREQUENCIES")


def main():
    hla_input_str: str = ""
    with fileinput.input() as f:
        for line in f:
            hla_input_str += f"{line}\n"

    hla_input: HLAInput = HLAInput(**json.loads(hla_input_str))

    errors: list[str] = hla_input.check_sequences()
    if len(errors) > 0:
        error_result: HLAResult = HLAResult(errors=errors)
        print(error_result.model_dump_json())
    else:
        curr_standards: Optional[dict[str, HLAStandard]] = None
        curr_frequencies: Optional[dict[HLAProteinPair, int]] = None
        if HLA_FREQUENCIES is not None:
            with open(HLA_FREQUENCIES) as f:
                curr_frequencies = EasyHLA.read_hla_frequencies(hla_input.locus, f)
        if HLA_STANDARDS[hla_input.locus] is not None:
            with open(HLA_STANDARDS[hla_input.locus]) as f:
                curr_standards = EasyHLA.read_hla_standards(f)
        easyhla: EasyHLA = EasyHLA(
            locus=hla_input.locus,
            hla_standards=curr_standards,
            hla_frequencies=curr_frequencies,
        )
        interp: HLAInterpretation = easyhla.interpret(hla_input.hla_sequence())
        print(HLAResult.build_from_interpretation(interp).model_dump_json())


if __name__ == "__main__":
    main()
