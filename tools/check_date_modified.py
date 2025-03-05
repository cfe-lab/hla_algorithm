"""
This script checks whether our HLA standards have been modified.

This is intended to be used as part of our CI pipeline.

For available commands use `python check_date_modified.py --help`

---

For CI purposes we just need `python check_date_modified.py check-dates`

---

If you need to update the HLA references, perform
`python check_date_modified.py update-last-recorded-mtime`, this will prompt you
to confirm whether you want to replace the contents of the `hla_nuc.fasta.mtime`
file.
"""

import os
from datetime import datetime
from typing import List, Tuple

import typer

from easyhla.easyhla import DATE_FORMAT

app = typer.Typer()


def _list_files() -> List[Tuple[str, int]]:
    """
    List all .csv files, returning a list of filenames and their last-modified
    date as an ISO timestamp.
    """
    file_mtime: List[Tuple[str, int]] = []
    dir = os.path.join(os.path.dirname(__file__), "../src/easyhla/")
    for file_name in os.listdir(dir):
        if file_name.endswith(".csv"):
            file_mtime.append(
                (file_name, int(os.path.getmtime(os.path.join(dir, file_name))))
            )

    return file_mtime


@app.command()
def list_files():
    return _list_files()


def _get_latest_mtime() -> datetime:
    """
    Get the date of the latest modified file.
    """
    file_mtimes = _list_files()
    return max([datetime.fromtimestamp(mtime) for file, mtime in file_mtimes])


@app.command()
def get_latest_mtime():
    return _get_latest_mtime()


def _get_mtime_file() -> str:
    return os.path.join(os.path.dirname(__file__), "../src/easyhla/hla_nuc.fasta.mtime")


@app.command()
def get_mtime_file():
    return _get_mtime_file()


def _get_last_recorded_mtime() -> datetime:
    """
    Return the date of the last recorded file modified date. This represents
    the time when our HLA standards were last updated.
    """
    last_modified_file_date: str = _get_mtime_file()
    if not os.path.exists(last_modified_file_date):
        raise FileNotFoundError("Last modified time does not exist!")

    with open(last_modified_file_date, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line]
        assert len(lines) == 1
        last_mod_time = datetime.strptime(lines[0], DATE_FORMAT)

    return last_mod_time


@app.command()
def get_last_recorded_mtime():
    return _get_last_recorded_mtime()


def _update_last_recorded_mtime(confirm: bool) -> None:
    """
    Updates the recorded last modified time file.
    """
    if not confirm:
        raise RuntimeError("You must confirm you want to update the time.")

    with open(_get_mtime_file(), "w", encoding="utf-8") as f:
        f.write(_get_latest_mtime().strftime(DATE_FORMAT))


@app.command()
def update_last_recorded_mtime():
    confirm = typer.confirm(
        "Are you sure you want to update the last recorded"
        "date modified? This will replace the contents of hla_nuc.fasta.mtime",
        abort=True,
    )
    return _update_last_recorded_mtime(confirm)


def _check_dates() -> bool:
    """
    Compare the date on record to the last modified date.
    """
    latest_file_date = int(_get_latest_mtime().strftime("%Y-%m-%d"))
    last_recorded_date = int(_get_last_recorded_mtime().strftime("%Y-%m-%d"))
    print("Last Modified Date:", latest_file_date)
    print("Last Recorded Date:", last_recorded_date)
    if latest_file_date == last_recorded_date:
        return True
    return False


@app.command()
def check_dates():
    if not _check_dates():
        warning_msg = "WARNING: The last modified file date has changed!"
        # preamble_height = 4
        # len_msg = ceil(len(warning_msg) / preamble_height)
        print("#" * len(warning_msg))
        print("#" * len(warning_msg))
        # if you want a silly message on an angle, uncomment this.
        # for i in range(preamble_height):
        #     n = i*len_msg
        #     print(' '*(n) + warning_msg[n:n+len_msg])
        print(f"\n{warning_msg}\n")
        print("#" * len(warning_msg))
        print("#" * len(warning_msg))
        exit(1)
    exit(0)


if __name__ == "__main__":
    app()
