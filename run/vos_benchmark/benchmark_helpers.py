import time
from multiprocessing import Pool
from pathlib import Path

from .benchmark_common import (
    quantify_single_root,
    benchmark_single_root,
    verbose_msg,
)


def benchmark_muliroot(
    gt_roots: list[Path],
    mask_roots: list[Path],
    strict: bool = True,
    num_processes: int = None,
    *,
    verbose: bool = True,
    skip_first_and_last: bool = True,
    file_name=None,
):
    assert len(gt_roots) == len(mask_roots)
    single_dataset = len(gt_roots) == 1

    if verbose:
        verbose_msg(skip_first_and_last)

    pool = Pool(num_processes)
    start_time = time.time()
    results_handles = []

    for gt_root, mask_root in zip(gt_roots, mask_roots):
        # 1) Build lists of Path objects under each root
        gt_videos = gt_root.iterdir()
        mask_videos = mask_root.iterdir()

        handle = benchmark_single_root(
            gt_videos,
            mask_videos,
            strict,
            verbose,
            skip_first_and_last,
            pool,
            single_dataset,
        )
        results_handles.append(handle)

    pool.close()

    # Now collect & quantify each dataset’s results
    all_global_jf = []
    all_global_j = []
    all_global_f = []
    all_object_metrics = []

    for mask_root, handle in zip(mask_roots, results_handles):
        if file_name is None:
            save_path = Path(mask_root) / "results"
        else:
            save_path = Path("results") / file_name

        quantify_single_root(
            save_path=save_path,
            handle=handle,
            single_dataset=single_dataset,
            start_time=start_time,
            verbose=verbose,
            all_global_jf=all_global_jf,
            all_global_j=all_global_j,
            all_global_f=all_global_f,
            all_object_metrics=all_object_metrics,
        )

    return all_global_jf, all_global_j, all_global_f, all_object_metrics


# ──────────────────────────────────────────────────────────────────────────────


def benchmark_multidir(
    gt_dirs: list[Path],
    pred_dirs: list[Path],
    file_name: str,
    strict: bool = True,
    num_processes: int = None,
    *,
    verbose: bool = True,
    skip_first_and_last: bool = True,
):
    assert len(gt_dirs) == len(pred_dirs)
    single_dataset = True

    if verbose:
        verbose_msg(skip_first_and_last)

    pool = Pool(num_processes)
    start_time = time.time()
    results_handles = []

    handle = benchmark_single_root(
        gt_dirs,
        pred_dirs,
        strict,
        verbose,
        skip_first_and_last,
        pool,
        single_dataset,
    )
    results_handles.append(handle)

    pool.close()

    # Now collect & quantify each dataset’s results
    all_global_jf = []
    all_global_j = []
    all_global_f = []
    all_object_metrics = []

    quantify_single_root(
        save_path=Path(f"results/{file_name}"),
        handle=handle,
        single_dataset=single_dataset,
        start_time=start_time,
        verbose=verbose,
        all_global_jf=all_global_jf,
        all_global_j=all_global_j,
        all_global_f=all_global_f,
        all_object_metrics=all_object_metrics,
    )
