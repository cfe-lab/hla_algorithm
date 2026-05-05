#! /usr/bin/env python

import argparse
import csv
import glob
import json
import os.path
import re
import subprocess
from typing import TypedDict


TIME_REGEX = re.compile(
    r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.*)$",
    flags=re.MULTILINE,
)
MEMORY_REGEX = re.compile(
    r"^\s*Maximum resident set size \(kbytes\): (.*)$",
    flags=re.MULTILINE,
)

def get_wall_clock_time(time_output: str) -> str:
    return TIME_REGEX.search(time_output).group(1)

def get_max_memory_usage(time_output: str) -> str:
    return MEMORY_REGEX.search(time_output).group(1)


class ResourceSummary(TypedDict):
    sample_name: str
    wall_clock_time: str
    max_memory_usage_kb: str


def main():
    parser = argparse.ArgumentParser(
        "Process HLA sequences and report the resource usage."
    )
    parser.add_argument("input_dir", help="Directory to scan for HLA sequences")
    parser.add_argument("--output_csv", help="CSV file summary", default="out.csv")
    args = parser.parse_args()

    resource_summaries: list[ResourceSummary] = []
    sample_regex = re.compile(r"^.*/(.*)\.BA\.txt$")
    for exon1_filename in glob.glob(f"{args.input_dir}/*.BA.txt"):
        sample_name: str = sample_regex.match(exon1_filename).group(1)
        exon2_filename: str = os.path.join(args.input_dir, f"{sample_name}.BB.txt")
        with open(exon1_filename) as f:
            exon1: str = f.read().strip()
        with open(exon2_filename) as f:
            exon2: str = f.read().strip()

        json_input = {
            "seq1": exon1,
            "seq2": exon2,
            "locus": "B",
        }
        json_filename: str = os.path.join(args.input_dir, f"{sample_name}.json")
        with open(json_filename, "w") as f:
            json.dump(json_input, f)

        print(f"----\nSample {sample_name}:")
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-v",
                "interpret_from_json",
                json_filename,
            ],
            capture_output=True,
            text=True,
        )
        print("stdout:")
        print(result.stdout)
        print("stderr:")
        print(result.stderr)

        resource_summaries.append(
            {
                "sample_name": sample_name,
                "wall_clock_time": get_wall_clock_time(result.stderr),
                "max_memory_usage_kb": get_max_memory_usage(result.stderr),
            }
        )

    with open(args.output_csv, "w") as f:
        resource_summary_writer = csv.DictWriter(
            f,
            fieldnames=("sample_name", "wall_clock_time", "max_memory_usage_kb"),
        )
        resource_summary_writer.writeheader()
        resource_summary_writer.writerows(resource_summaries)


if __name__ == "__main__":
    main()
