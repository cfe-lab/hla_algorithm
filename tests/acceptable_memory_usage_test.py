import numpy as np
import pytest

from hla_algorithm.hla_algorithm import HLAAlgorithm
from hla_algorithm.models import HLASequence, HLAStandard


@pytest.mark.slow
@pytest.mark.limit_memory("500 MB")
def test_acceptable_memory_usage():
    # We process a sequence produced by "mushing together" B*07:02:01G
    # and B*45:01:01G, which as of the v2.63.0-alpha HLA alleles produces
    # an expensive calculation.
    hla_alg = HLAAlgorithm()

    allele_1: HLAStandard = hla_alg.hla_standards["B"]["B*07:02:01G"]
    allele_2: HLAStandard = hla_alg.hla_standards["B"]["B*45:01:01G"]

    # "Mush" together the two sequences by doing a bitwise or of the binary
    # sequences.
    exon2_bin: np.ndarray = np.array(allele_1.two) | np.array(allele_2.two)
    exon3_bin: np.ndarray = np.array(allele_1.three) | np.array(allele_2.three)

    expensive_sequence = HLASequence(
        two=tuple(int(s) for s in exon2_bin),
        intron=(),
        three=tuple(int(s) for s in exon3_bin),
        name="expensive_sequence",
        locus="B",
    )

    hla_alg.interpret(expensive_sequence)
