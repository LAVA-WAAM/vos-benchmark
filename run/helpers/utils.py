import csv
import datetime
from pathlib import Path
from typing import Dict, List

from tabulate import tabulate


def prefix_subsets_paths(
    subsets: Dict[str, List[str]], prefix: Path
) -> Dict[str, List[Path]]:
    new_subsets = {}
    for key, dir_list in subsets.items():
        new_subsets[key] = [prefix / dir_name for dir_name in dir_list]
    return new_subsets


def timestamp_to_filename() -> str:
    now = datetime.datetime.now()

    # format it as YYYYMMDD_HHMMSS
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return timestamp


def parse_files(base_dir: Path) -> None:
    csv_paths = sorted(list(base_dir.rglob("*.csv")))
    for path in csv_paths:
        print_table(csv_path=path, fmt=".1f")


def print_table(csv_path: Path, fmt=".3f") -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    table_lines = tabulate(
        rows[1:], headers=rows[0], tablefmt="mixed_outline", floatfmt=fmt
    ).splitlines()

    sep = next(line for line in table_lines if line.startswith("┝"))

    # insert that divider right before the last data-row (the “average”)
    table_lines.insert(-2, sep)

    print()
    print(f"{csv_path.name}:")
    print("\n".join(table_lines))
    print()
