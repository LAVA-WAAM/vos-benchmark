from argparse import ArgumentParser
from pathlib import Path

from helpers.utils import prefix_subsets_paths, timestamp_to_filename

from vos_benchmark.benchmark_helpers import (
    benchmark_multidir,
    benchmark_muliroot,
)

subsets = {
    "S-1": [
        "E2_S3_W1_L1",
        "E2_S3_W1_L2",
        "E2_S3_W1_L3",
        "E2_S3_W1_L4",
        "E2_S3_W1_L5",
        "E2_S3_W1_L9",
        "E2_S3_W1_L13",
        "E2_S3_W1_L15",
    ],
    "S-2": [
        "E3_S5_C2_P0",
        "E3_S5_C2_P1",
        "E3_S5_C2_P2",
        "E3_S5_C2_P3",
        "E3_S5_C2_P4",
        "E3_S5_C2_P5",
        "E3_S5_C2_P6",
        "E3_S5_C2_P7",
        "E3_S5_C2_P8",
    ],
    "S-3": [
        "E5_S8_C0_P1",
        "E5_S8_C0_P5",
        "E5_S8_C0_P6",
    ],
    "S-4": [
        "E5_S9_C1_P2",
        "E5_S9_C1_P3",
        "E5_S9_C1_P4",
        "E5_S9_C1_P7",
        "E5_S9_C1_P10",
        "E5_S9_C1_P13",
        "E5_S9_C1_P14",
        "E5_S9_C1_P16",
        "E5_S9_C1_P19",
        "E5_S9_C1_P25",
    ],
    "S-5": [
        "E6_S13_C1_V1_9",
        "E6_S13_C1_V1_10",
    ],
}


def eval_LAVA_WAAM(data_root: str, pred_root: str) -> None:
    gt_root = Path(data_root) / "test/Annotations/"
    pred_root = Path(pred_root)
    gt_dirs = prefix_subsets_paths(subsets=subsets, prefix=gt_root)
    pred_dirs = prefix_subsets_paths(subsets=subsets, prefix=pred_root)

    # benchmark specific dirs
    timestamp = timestamp_to_filename()
    for subset in gt_dirs.keys():
        benchmark_multidir(
            gt_dirs=gt_dirs[subset],
            pred_dirs=pred_dirs[subset],
            file_name=f"{timestamp}_detailed_{subset}",
            strict=args.strict,
            num_processes=args.num_processes,
            verbose=True,
            skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        )

    # benchmark complete dataset
    benchmark_muliroot(
        gt_roots=[gt_root],
        mask_roots=[pred_root],
        strict=args.strict,
        num_processes=args.num_processes,
        verbose=True,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        file_name=f"{timestamp}_summary",
    )


if __name__ == "__main__":
    # ADAPT PATHS

    # LAVA_WAAM - pretrained model:
    data_dir = "/home/docker/xmem_datasets/LAVA_WAAM/"
    pred_dir = "/home/docker/inference_results_xmem/LAVA_WAAM_xmem_pth/"

    # LAVA_WAAM - fine-tuned model:
    # data_dir = "/home/docker/xmem_datasets/LAVA_WAAM/"
    # pred_dir = "/home/docker/inference_results_xmem/LAVA_WAAM_fine-tuned_pth/"

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=data_dir,
        help="Path to a dataset folder",
    )
    parser.add_argument(
        "--pred_dir",
        default=pred_dir,
        help="Path to a folder containing folders of masks to be evaluated",
    )
    parser.add_argument(
        "-n",
        "--num_processes",
        default=16,
        type=int,
        help="Number of concurrent processes",
    )
    parser.add_argument(
        "-s",
        "--strict",
        help="Make sure every video in the ground-truth has a corresponding video in the prediction",
        action="store_true",
    )

    # https://github.com/davisvideochallenge/davis2017-evaluation/blob/d34fdef71ce3cb24c1a167d860b707e575b3034c/davis2017/evaluation.py#L85
    parser.add_argument(
        "-d",
        "--do_not_skip_first_and_last_frame",
        help="By default, we skip the first and the last frame in evaluation following DAVIS semi-supervised evaluation."
        "They should not be skipped in unsupervised evaluation.",
        action="store_true",
    )
    args = parser.parse_args()
    eval_LAVA_WAAM(data_root=args.data_dir, pred_root=args.pred_dir)
