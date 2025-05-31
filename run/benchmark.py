from argparse import ArgumentParser
from pathlib import Path

from vos_benchmark.benchmark_helpers import benchmark_muliroot

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

    benchmark_muliroot(
        [Path(args.data_dir) / "test/Annotations/"],
        [Path(args.pred_dir)],
        args.strict,
        args.num_processes,
        verbose=True,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
    )
