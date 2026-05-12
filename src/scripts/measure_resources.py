#! /usr/bin/env python

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import TypedDict

TIME_REGEX = re.compile(
    r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.*)$",
    flags=re.MULTILINE,
)
MEMORY_REGEX = re.compile(
    r"^\s*Maximum resident set size \(kbytes\): (.*)$",
    flags=re.MULTILINE,
)
SAMPLE_REGEX = re.compile(r"^(.*)\.BA\.txt$")


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
    parser.add_argument(
        "input_dir",
        help="Directory to scan for HLA sequences",
        type=Path,
    )
    parser.add_argument(
        "--output_csv",
        help="CSV file summary",
        type=Path,
        default=Path("out.csv"),
    )
    args = parser.parse_args()

    resource_summaries: list[ResourceSummary] = []

    for exon1_filepath in args.input_dir.glob("*.BA.txt"):
        sample_name: str = SAMPLE_REGEX.match(exon1_filepath.name).group(1)
        exon2_filepath: Path = args.input_dir / f"{sample_name}.BB.txt"
        exon1: str = exon1_filepath.read_text().strip()
        exon2: str = exon2_filepath.read_text().strip()

        json_input = {
            "seq1": exon1,
            "seq2": exon2,
            "locus": "B",
        }
        json_filepath: Path = args.input_dir / f"{sample_name}.json"
        json_filepath.write_text(json.dumps(json_input))

        print(f"----\nSample {sample_name}:")
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-v",
                "interpret_from_json",
                json_filepath.as_posix(),
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

    with args.output_csv.open("w") as f:
        resource_summary_writer = csv.DictWriter(
            f,
            fieldnames=("sample_name", "wall_clock_time", "max_memory_usage_kb"),
        )
        resource_summary_writer.writeheader()
        resource_summary_writer.writerows(resource_summaries)


if __name__ == "__main__":
    main()
