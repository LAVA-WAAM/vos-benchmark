import os
import time
from multiprocessing import Pool
from os import path
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tqdm
from helpers.utils import print_table
from natsort import natsorted
from PIL import Image

from .evaluator import Evaluator


class VideoEvaluator:
    """
    A processing function object.
    This returns metrics for a single video.
    """

    def __init__(self, gt_root, mask_root, skip_first_and_last=True):
        self.gt_root = gt_root
        self.mask_root = mask_root
        self.skip_first_and_last = skip_first_and_last

    def __call__(self, vid_name):
        vid_gt_path = path.join(self.gt_root, vid_name)
        vid_mask_path = path.join(self.mask_root, vid_name)

        frames = sorted(os.listdir(vid_gt_path))
        if self.skip_first_and_last:
            # the first and the last frames are skipped in DAVIS semi-supervised evaluation
            frames = frames[1:-1]
        evaluator = Evaluator(name=vid_name)
        for f in frames:
            try:
                gt_array = np.array(Image.open(path.join(vid_gt_path, f)))
                mask_array = np.array(Image.open(path.join(vid_mask_path, f)))
                assert gt_array.shape[-2:] == mask_array.shape[-2:], (
                    f"Dimensions mismatch: GT: {gt_array.shape}, predicted: {mask_array.shape}. "
                    f"GT path: {path.join(vid_gt_path, f)}; "
                    f"predicted path: {path.join(vid_mask_path, f)}"
                )
            except FileNotFoundError:
                print(f"{f} not found in {vid_mask_path}.")
                exit(1)

            evaluator.feed_frame(mask_array, gt_array)
        iou, boundary_f = evaluator.conclude()
        return vid_name, iou, boundary_f


# ──────────────────────────────────────────────────────────────────────────────
def validate_and_list_videos(
    gt_videos: List[Path],
    mask_videos: List[Path],
    strict: bool,
    verbose: bool,
) -> Tuple[Path, List[Path]]:
    """
    Given:
      - gt_root (possibly a top-level folder, maybe containing an 'Annotations' subfolder)
      - mask_root
      - gt_videos:  list of directory-names under gt_root (i.e. os.listdir(gt_root))
      - mask_videos: list of directory-names under mask_root (i.e. os.listdir(mask_root))
      - strict, verbose flags

    This function checks:
      1) If len(gt_videos) != len(mask_videos) and 'Annotations' is in gt_videos,
         then descend into gt_root/Annotations, re-compute gt_videos = os.listdir(gt_root/Annotations).
      2) Filter out any non-directories from both lists.
      3) If strict=True, enforce exact match (else print mismatches and exit).
         If strict=False, take the intersection of the two lists.
      4) Return (possibly updated) gt_root and the final sorted `videos` list.

    Note: You only call os.listdir(...) when “descending” into Annotations. Otherwise,
    you work with whatever lists were passed in.
    """

    # 1) Derive the roots from the .parent of the first Path in each list
    if not gt_videos or not mask_videos:
        raise ValueError(
            "gt_videos and mask_videos must each be a non-empty list of Path objects."
        )
    gt_root = gt_videos[0].parent
    mask_root = mask_videos[0].parent

    # 2) If the counts differ and an “Annotations” subfolder exists, descend into it
    if len(gt_videos) != len(mask_videos):
        ann_path = gt_root / "Annotations"
        if ann_path.exists() and ann_path.is_dir():
            # Gather everything under gt_root/Annotations
            ann_contents = sorted(ann_path.iterdir())
            # Original logic “only checked the first item for .png”
            if ann_contents and not ann_contents[0].name.endswith(".png"):
                # Now treat gt_root as the “Annotations” folder
                gt_root = ann_path
                gt_videos = sorted(gt_root.iterdir())

    # 3) Filter out anything that is not a directory
    gt_videos = [p for p in gt_videos if p.is_dir()]
    mask_videos = [p for p in mask_videos if p.is_dir()]

    # 4) Build sets of “names” to do strict / non-strict matching
    gt_names = {p.name for p in gt_videos}
    mask_names = {p.name for p in mask_videos}

    if not strict:
        common = natsorted(gt_names.intersection(mask_names))
        # Build Path objects for each common subfolder name under gt_root
        videos = [gt_root / name for name in common]
    else:
        gt_extras = gt_names - mask_names
        mask_extras = mask_names - gt_names

        if gt_extras:
            print(f"Videos in {gt_root} but not in {mask_root}: {gt_extras}")
        if mask_extras:
            print(f"Videos in {mask_root} but not in {gt_root}: {mask_extras}")
        if gt_extras or mask_extras:
            print("Validation failed. Exiting.")
            exit(1)

        # If strict AND no mismatches, we simply sort gt_names
        videos = [gt_root / name for name in natsorted(gt_names)]

    # 5) Verbose print
    if verbose:
        print(
            f"In dataset {gt_root}, evaluating on {len(videos)} videos: {[p.name for p in videos]}"
        )

    return gt_root, videos


# ──────────────────────────────────────────────────────────────────────────────
def quantify(
    results,  # List[ (name, iou_dict, boundary_f_dict) ]
    save_path: Path,
    start_time: float,
    verbose: bool,
    all_global_jf: list[float],
    all_global_j: list[float],
    all_global_f: list[float],
    all_object_metrics: list[dict],
):
    """
    Given a completed `results` (a list of (video_name, iou_dict, boundary_f_dict)),
    compute global metrics, print to console (if verbose), write `results.csv` under save_dir,
    and append metrics into the `all_global_*` and `all_object_metrics` lists.
    """
    all_iou = []
    all_boundary_f = []
    object_metrics = {}

    for name, iou_dict, boundary_f_dict in results:
        all_iou.extend(iou_dict.values())
        all_boundary_f.extend(boundary_f_dict.values())
        object_metrics[name] = (iou_dict, boundary_f_dict)

    # Avoid division‐by‐zero if lists empty
    global_j = float(np.mean(all_iou)) if all_iou else 0.0
    global_f = float(np.mean(all_boundary_f)) if all_boundary_f else 0.0
    global_jf = (global_j + global_f) / 2

    time_taken = time.time() - start_time

    # Build the CSV text
    ml = max(*(len(n) for n in object_metrics.keys()), len("Global score"))
    out_lines = []
    out_lines.append(
        f'{"sequence":<{ml}},{"obj":>3}, {"J&F":>4}, {"J":>4}, {"F":>4}'
    )
    for name, (iou_dict, boundary_f_dict) in object_metrics.items():
        for obj_idx, j in iou_dict.items():
            f = boundary_f_dict[obj_idx]
            jf = (j + f) / 2
            out_lines.append(
                f"{name:<{ml}},{obj_idx:03}, {jf:>4.1f}, {j:>4.1f}, {f:>4.1f}"
            )
    out_lines.append(
        f'{"Global score":<{ml}},{"":>3}, {global_jf:.1f}, {global_j:.1f}, {global_f:.1f}'
    )
    out_string = "\n".join(out_lines) + "\n"

    # Write CSV
    out_string.replace(",", " ")
    csv_path = f"{save_path}.csv"
    with open(csv_path, "w") as fh:
        fh.write(out_string)

    if verbose:
        print_table(csv_path=Path(csv_path), fmt=".1f")
        print(
            f"Global score: J&F: {global_jf:.1f} J: {global_j:.1f} F: {global_f:.1f}"
        )
        print(f"Time taken: {time_taken:.2f}s \n")


    # Append to accumulators
    all_global_jf.append(global_jf)
    all_global_j.append(global_j)
    all_global_f.append(global_f)
    all_object_metrics.append(object_metrics)


# ──────────────────────────────────────────────────────────────────────────────


def benchmark_single_root(
    gt_videos: Iterable[Path],
    mask_videos: Iterable[Path],
    strict: bool,
    verbose: bool,
    skip_first_and_last: bool,
    pool: Pool,
    single_dataset: bool,
):
    """
    1) Sort GT and mask video lists
    2) Validate and list videos
    3) Build VideoEvaluator
    4) Dispatch (sync if single_dataset else async)

    Returns either a tqdm iterator (if single_dataset and verbose),
    a list of results (if single_dataset and not verbose), or an AsyncResult.
    """
    # 1) Sort
    gt_videos = natsorted(gt_videos)
    mask_videos = natsorted(mask_videos)

    # 2) Validate & list
    gt_root_used, videos_to_eval = validate_and_list_videos(
        gt_videos=gt_videos,
        mask_videos=mask_videos,
        strict=strict,
        verbose=verbose,
    )

    # 3) Build evaluator
    evaluator = VideoEvaluator(
        gt_root=str(gt_root_used),  # cast Path to str if needed
        mask_root=str(mask_videos[0].parent),
        skip_first_and_last=skip_first_and_last,
    )

    names_to_eval = [p.name for p in videos_to_eval]

    # 4) Dispatch
    if single_dataset:
        if verbose:
            return tqdm.tqdm(
                pool.imap(evaluator, names_to_eval), total=len(videos_to_eval)
            )
        else:
            return pool.map(evaluator, names_to_eval)
    else:
        return pool.map_async(evaluator, names_to_eval)


# ──────────────────────────────────────────────────────────────────────────────


def quantify_single_root(
    save_path: Path,
    handle,
    single_dataset: bool,
    start_time: float,
    verbose: bool,
    all_global_jf: list,
    all_global_j: list,
    all_global_f: list,
    all_object_metrics: list,
):
    """
    For one dataset (mask_root + its corresponding handle), do:
      - if multi-dataset, wait for the async job to finish (handle.get())
      - if single_dataset, ensure handle is a list or convert it to a list
      - call quantify(...) with exactly the same arguments as before,
        letting quantify() append into the global lists.
    """
    # 1) “unwrap” the results_i from the handle
    if not single_dataset:
        # this was the async path in your original loop
        results_i = handle.get()
    else:
        # this was your “single‐dataset” path
        if isinstance(handle, list):
            results_i = handle
        else:
            # tqdm‐wrapped generator → convert to list
            results_i = list(handle)

    # 2) Now call quantify(...) exactly as before
    quantify(
        results=results_i,
        save_path=save_path,
        start_time=start_time,
        verbose=verbose,
        all_global_jf=all_global_jf,
        all_global_j=all_global_j,
        all_global_f=all_global_f,
        all_object_metrics=all_object_metrics,
    )


# ──────────────────────────────────────────────────────────────────────────────


def verbose_msg(skip_first_and_last: bool):
    if skip_first_and_last:
        print(
            "We are *SKIPPING* the evaluation of the first and the last frame (standard for semi-supervised video object segmentation)."
        )
    else:
        print(
            "We are *NOT SKIPPING* the evaluation of the first and the last frame (*NOT STANDARD* for semi-supervised video object segmentation)."
        )

