from easyhla.easyhla import EasyHLA

input_file = "tests/input/hla-a-seqs.fasta"
output_file = "tests/output/output.csv"

easyhla = EasyHLA("A")

easyhla.run(
    easyhla.letter,
    input_file,
    output_file,
    0,
)
