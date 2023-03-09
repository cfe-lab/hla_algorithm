import os
from easyhla.easyhla import EasyHLA

letter = "C"

input_file = os.path.join(
    os.path.dirname(__file__), f"../tests/input/hla-{letter.lower()}-seqs.fasta"
)
output_file = os.path.join(
    os.path.dirname(__file__), f"../tests/output/hla-{letter.lower()}-output-test.csv"
)

easyhla = EasyHLA(letter.upper())

easyhla.run(
    easyhla.letter,
    input_file,
    output_file,
    0,
)
