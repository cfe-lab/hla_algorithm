"""
This converts data from the BC CfE COVID19 immunity study into an input/output file.
"""

import os
from random import Random
from typing import Dict, List

import numpy as np
import pandas as pd

###

# This can be A, B, or C
HLA_TYPE: str = "A"

###

assert HLA_TYPE.upper() in "ABC", "Incorrect HLA type specified!"

ENUM_IDENTIFIER = {
    "A": "E-Number",
    "B": "ENUM",
    "C": "E-Number",
}

df = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        f"../tests/output/raw_hla_{HLA_TYPE.lower()}_data.csv",
    )
)
df = df.replace(np.nan, "")

# 0-1, percentage of input samples that should be suffixed '_exon#'
# HLA Type A has to include the intron
if HLA_TYPE.upper() == "A":
    perc_pure_exon: float = 0.0
else:
    perc_pure_exon: float = 0.0

sample_input_df = df[[ENUM_IDENTIFIER[HLA_TYPE], "EXON2", "INTRON", "EXON3"]]
sample_input_seqs: Dict[str, str] = {}
pure_exon_samples: List[str] = []

for row in sample_input_df.iterrows():
    chance = Random().random()
    if chance > perc_pure_exon and row[1]["INTRON"] != "":
        if HLA_TYPE == "A":
            sample_input_seqs[row[1][ENUM_IDENTIFIER[HLA_TYPE]]] = (
                f"{row[1]['EXON2']}{row[1]['INTRON']}{row[1]['EXON3']}"
            )
        else:
            sample_input_seqs[row[1][ENUM_IDENTIFIER[HLA_TYPE]]] = (
                f"{row[1]['EXON2']}{row[1]['EXON3']}"
            )
    else:
        pure_exon_samples.append(row[1][ENUM_IDENTIFIER[HLA_TYPE]])
        sample_input_seqs[row[1][ENUM_IDENTIFIER[HLA_TYPE]] + "_exon2"] = (
            f"{row[1]['EXON2']}"
        )
        sample_input_seqs[row[1][ENUM_IDENTIFIER[HLA_TYPE]] + "_exon3"] = (
            f"{row[1]['EXON3']}"
        )

with open(
    os.path.join(
        os.path.dirname(__file__), f"../tests/input/hla-{HLA_TYPE.lower()}-seqs.fasta"
    ),
    "w",
    encoding="utf-8",
) as f:
    for name, seq in sample_input_seqs.items():
        f.write(f">{name}\n")
        f.write(seq + "\n")

output_columns = f"{ENUM_IDENTIFIER[HLA_TYPE]},ALLELES_CLEAN,ALLELES,AMBIGUOUS,HOMOZYGOUS,MISMATCH_COUNT,MISMATCHES,EXON2,INTRON,EXON3".split(
    ","
)

sample_output_df = df[output_columns]

with open(
    os.path.join(
        os.path.dirname(__file__),
        f"../tests/output/hla-{HLA_TYPE.lower()}-output-ref.csv",
    ),
    "w",
    encoding="utf-8",
) as f:
    f.write("# Preamble timestamp goes here\n")
    f.write(",".join(output_columns).replace("E-Number", "ENUM") + "\n")
    for row in sample_output_df.iterrows():
        if row[1]["EXON2"] == "" or row[1]["EXON3"] == "":
            continue
        for col in output_columns:
            if (
                row[1][ENUM_IDENTIFIER[HLA_TYPE]] in pure_exon_samples
                and col == "INTRON"
            ):
                f.write(",")
            elif col == output_columns[-1]:
                f.write(f"{row[1][col]}")
            else:
                f.write(f"{row[1][col]},")
        f.write("\n")
