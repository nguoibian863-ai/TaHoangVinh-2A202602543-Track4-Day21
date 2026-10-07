"""Comparative Benchmark Across Real Datasets: KITTI vs nuScenes (Bonus B5).

This script executes identical extrinsic perturbation sweeps across both
KITTI (64-beam, 1242x375, daytime) and nuScenes (32-beam, 1600x900, day & night),
and analyzes the effect of ego-motion compensation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import list_frames, load_frame
from starter.projection import perturb_extrinsic
from src.projection_qa import evaluate_frame_projection, save_results_csv


def evaluate_dataset_sweep(
    data_root: str | Path,
    frame_ids: List[str],
    yaw_angles: List[float],
    use_ego_motion: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ds_name = Path(data_root).name

    for yaw in yaw_angles:
        metrics: List[Dict[str, float]] = []
        for fid in frame_ids:
            kwargs = {"use_ego_motion": use_ego_motion} if "nuscenes" in ds_name else {}
            fr = load_frame(data_root, fid, **kwargs)
            calib_pert = perturb_extrinsic(fr["calib"], yaw_deg=yaw)
            m = evaluate_frame_projection(fr, calib_pert, min_pts_per_obj=3)
            metrics.append(m)

        avg_pbr = float(np.mean([m["point_in_box_retention"] for m in metrics]))
        avg_iou = float(np.mean([m["projected_box_iou"] for m in metrics]))
        avg_shift = float(np.mean([m["mean_shift_overall_px"] for m in metrics]))
        avg_fov = float(np.mean([m["fov_ratio"] for m in metrics]))
        total_objs = sum(int(m["evaluated_objects"]) for m in metrics)

        results.append({
            "dataset": ds_name,
            "ego_motion": use_ego_motion,
            "yaw_deg": yaw,
            "point_in_box_retention": avg_pbr,
            "projected_box_iou": avg_iou,
            "mean_shift_px": avg_shift,
            "fov_ratio": avg_fov,
            "evaluated_objects": total_objs,
        })
    return results


def plot_comparison(
    records_kitti: List[Dict[str, Any]],
    records_nusc: List[Dict[str, Any]],
    records_nusc_noego: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r["yaw_deg"] for r in records_kitti]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Point-in-Box Retention Comparison
    ax1 = axes[0]
    ax1.plot(x, [r["point_in_box_retention"] * 100 for r in records_kitti], "b-o", linewidth=2.2, label="KITTI (64-beam)")
    ax1.plot(x, [r["point_in_box_retention"] * 100 for r in records_nusc], "g-s", linewidth=2.2, label="nuScenes (32-beam, Ego-Comp)")
    ax1.plot(x, [r["point_in_box_retention"] * 100 for r in records_nusc_noego], "r--^", linewidth=2.0, label="nuScenes (No Ego-Comp)")
    ax1.set_xlabel("Extrinsic Yaw Drift (degrees)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Point-in-Box Retention (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Retention Sensitivity: KITTI vs nuScenes", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower center", framealpha=0.9)

    # Plot 2: Mean Pixel Displacement
    ax2 = axes[1]
    ax2.plot(x, [r["mean_shift_px"] for r in records_kitti], "b-o", linewidth=2.2, label="KITTI (f~721px)")
    ax2.plot(x, [r["mean_shift_px"] for r in records_nusc], "g-s", linewidth=2.2, label="nuScenes (f~1260px)")
    ax2.set_xlabel("Extrinsic Yaw Drift (degrees)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Pixel Shift (px)", fontsize=11, fontweight="bold")
    ax2.set_title("Pixel Displacement (Focal Length Effect)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper center", framealpha=0.9)

    plt.suptitle("Cross-Dataset Calibration Robustness & Temporal Desync (B5)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Comparison Plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Dataset Benchmark: KITTI vs nuScenes")
    parser.add_argument("--kitti-root", default="data/kitti_mini")
    parser.add_argument("--nusc-root", default="data/nuscenes_mini_subset")
    parser.add_argument("--out-csv", default="results/dataset_comparison.csv")
    parser.add_argument("--out-figure", default="results/figures/kitti_vs_nuscenes.png")
    args = parser.parse_args()

    kitti_frames = ["000001", "000011", "000021", "000032", "000049"]
    nusc_frames = ["scene-0103_000", "scene-0103_010", "scene-0103_020", "scene-1094_000", "scene-1094_010"]
    yaw_angles = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    print("Evaluating KITTI...")
    rec_kitti = evaluate_dataset_sweep(args.kitti_root, kitti_frames, yaw_angles)

    print("Evaluating nuScenes (with Ego Motion)...")
    rec_nusc = evaluate_dataset_sweep(args.nusc_root, nusc_frames, yaw_angles, use_ego_motion=True)

    print("Evaluating nuScenes (WITHOUT Ego Motion)...")
    rec_nusc_noego = evaluate_dataset_sweep(args.nusc_root, nusc_frames, yaw_angles, use_ego_motion=False)

    all_records = rec_kitti + rec_nusc + rec_nusc_noego
    save_results_csv(all_records, Path(args.out_csv))
    plot_comparison(rec_kitti, rec_nusc, rec_nusc_noego, Path(args.out_figure))


if __name__ == "__main__":
    main()
