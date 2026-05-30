from pathlib import Path

from helpers.utils import parse_files

if __name__ == "__main__":
    parse_files(
        base_dir=Path(
            "dataset_representativeness/LAVA_WAAM__org"
        )
    )
