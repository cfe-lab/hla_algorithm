import pytest
import json
import os
from easyhla import EasyHLA


def test_easyhla():
    ehla = EasyHLA("A")

    with open(
        os.path.join("output/hla_stds.json"),
        "w",
        encoding="utf-8",
    ) as f:
        dat = ehla.load_hla_stds("A")
        json.dump(dat, f, indent=2)

    with open(
        os.path.join("output/hla_freqs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        dat = ehla.load_hla_frequencies("A")
        json.dump(dat, f, indent=2)
