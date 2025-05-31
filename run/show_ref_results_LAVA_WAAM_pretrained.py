from pathlib import Path

from helpers.utils import parse_files

if __name__ == "__main__":
    parse_files(
        base_dir=Path(
            "reference_results/LAVA_WAAM__pretrained_models"
        )
    )
