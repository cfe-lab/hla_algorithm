"""
This converts data from the BC CfE COVID19 immunity study into an input/output file.
"""

import os
import numpy as np
import pandas as pd
from random import Random
from typing import Dict, List

###

# This can be A, B, or C
HLA_TYPE: str = "A"

###

assert HLA_TYPE.upper() in "ABC", "Incorrect HLA type specified!"

df = pd.read_csv(
    os.path.dirname(__file__) + f"/output/raw_hla_{HLA_TYPE.lower()}_data.csv"
)
df = df.replace(np.nan, "")

# 0-1, percentage of input samples that should be suffixed '_exon#'
# HLA Type A has to include the intron
if HLA_TYPE.upper() == "A":
    perc_pure_exon: float = 0.0
else:
    perc_pure_exon: float = 0.4

sample_input_df = df[["E-Number", "EXON2", "INTRON", "EXON3"]]
sample_input_seqs: Dict[str, str] = {}
pure_exon_samples: List[str] = []

for row in sample_input_df.iterrows():
    chance = Random().random()
    if chance > perc_pure_exon:
        sample_input_seqs[
            row[1]["E-Number"]
        ] = f"{row[1]['EXON2']}{row[1]['INTRON']}{row[1]['EXON3']}"
    else:
        pure_exon_samples.append(row[1]["E-Number"])
        sample_input_seqs[row[1]["E-Number"] + "_exon2"] = f"{row[1]['EXON2']}"
        sample_input_seqs[row[1]["E-Number"] + "_exon3"] = f"{row[1]['EXON3']}"

with open(
    os.path.dirname(__file__) + f"/input/hla-{HLA_TYPE.lower()}-seqs.fasta",
    "w",
    encoding="utf-8",
) as f:
    for name, seq in sample_input_seqs.items():
        f.write(f">{name}\n")
        f.write(seq + "\n")

output_columns = "E-Number,ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3".split(
    ","
)

sample_output_df = df[output_columns]

with open(
    os.path.dirname(__file__) + f"/output/hla-{HLA_TYPE.lower()}-output.csv",
    "w",
    encoding="utf-8",
) as f:
    f.write("# Preamble timestamp goes here")
    f.write(",".join(output_columns).replace("E-Number", "ENUM") + "\n")
    for row in sample_output_df.iterrows():
        for col in output_columns:
            if row[1]["E-Number"] in pure_exon_samples and col == "INTRON":
                f.write(f",")
            else:
                f.write(f"{row[1][col]},")
        f.write("\n")
