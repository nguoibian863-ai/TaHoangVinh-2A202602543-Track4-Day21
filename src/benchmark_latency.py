"""Latency Benchmark Tool for LiDAR-Camera Projection Pipeline (Bonus B3).

Strictly follows B3 requirements:
- Discards first run (warmup).
- Executes at least 20 runs (50 iterations evaluated).
- Computes p50 (median) and p95 latency.
- Records and exports hardware configuration (CPU, OS, Python).
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import load_frame
from starter.projection import cam_to_image, overlay_points, velo_to_cam


def get_hardware_info() -> Dict[str, str]:
    """Retrieve detailed system and hardware metadata."""
    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def benchmark_pipeline(
    data_root: str = "data/kitti_mini",
    frame_id: str = "000011",
    num_iterations: int = 50,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    fr = load_frame(data_root, frame_id)
    pts = fr["points"][:, :3]
    calib = fr["calib"]
    img = fr["image"]
    img_shape = img.shape
    n_points = len(pts)

    print(f"Benchmarking on {data_root} frame {frame_id} ({n_points:,} points)...")

    # Warmup run (discarded)
    p_cam = velo_to_cam(pts, calib)
    uv, depth, mask = cam_to_image(p_cam, calib.P2, img_shape)
    _ = overlay_points(img, uv, depth)

    timings_velo_cam: List[float] = []
    timings_cam_img: List[float] = []
    timings_e2e_proj: List[float] = []
    timings_overlay: List[float] = []
    timings_total: List[float] = []

    for _ in range(num_iterations):
        # 1. velo_to_cam
        t0 = time.perf_counter()
        p_cam = velo_to_cam(pts, calib)
        t1 = time.perf_counter()

        # 2. cam_to_image
        uv, depth, mask = cam_to_image(p_cam, calib.P2, img_shape)
        t2 = time.perf_counter()

        # 3. overlay_points
        _ = overlay_points(img, uv, depth)
        t3 = time.perf_counter()

        timings_velo_cam.append((t1 - t0) * 1000.0)
        timings_cam_img.append((t2 - t1) * 1000.0)
        timings_e2e_proj.append((t2 - t0) * 1000.0)
        timings_overlay.append((t3 - t2) * 1000.0)
        timings_total.append((t3 - t0) * 1000.0)

    stages = {
        "velo_to_cam": timings_velo_cam,
        "cam_to_image": timings_cam_img,
        "projection_total": timings_e2e_proj,
        "overlay_rendering": timings_overlay,
        "end_to_end_total": timings_total,
    }

    hw = get_hardware_info()
    hw["n_points"] = str(n_points)
    hw["num_iterations"] = str(num_iterations)
    return stages, hw


def compute_stage_stats(timings: List[float]) -> Dict[str, float]:
    arr = np.array(timings)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def save_latency_csv(
    stages: Dict[str, List[float]], hw: Dict[str, str], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("# Hardware Metadata\n")
        for k, v in hw.items():
            fp.write(f"# {k}: {v}\n")
        fp.write("stage,p50_ms,p95_ms,mean_ms,std_ms,min_ms,max_ms,iterations\n")
        for stage_name, times in stages.items():
            st = compute_stage_stats(times)
            fp.write(
                f"{stage_name},{st['p50_ms']:.3f},{st['p95_ms']:.3f},{st['mean_ms']:.3f},"
                f"{st['std_ms']:.3f},{st['min_ms']:.3f},{st['max_ms']:.3f},{len(times)}\n"
            )
    print(f"[OK] Saved Latency CSV: {out_path}")


def plot_latency(stages: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    # Boxplot of stages
    ax1 = axes[0]
    stage_names = ["velo_to_cam", "cam_to_image", "projection_total", "overlay_rendering"]
    data = [stages[k] for k in stage_names]
    stage_labels = ["velo->cam", "cam->img", "Total Proj", "Overlay"]
    try:
        bplot = ax1.boxplot(data, tick_labels=stage_labels, patch_artist=True)
    except TypeError:
        bplot = ax1.boxplot(data, labels=stage_labels, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Stage-wise Latency Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Histogram of end-to-end projection
    ax2 = axes[1]
    proj_times = stages["projection_total"]
    st = compute_stage_stats(proj_times)
    ax2.hist(proj_times, bins=15, color="#C44E52", alpha=0.75, edgecolor="black")
    ax2.axvline(st["p50_ms"], color="blue", linestyle="--", linewidth=2.0, label=f"p50: {st['p50_ms']:.2f} ms")
    ax2.axvline(st["p95_ms"], color="red", linestyle="-.", linewidth=2.0, label=f"p95: {st['p95_ms']:.2f} ms")
    ax2.set_xlabel("Projection Execution Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title(f"Projection Latency (108k points, n={len(proj_times)})", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Projection Performance & Real-Time Feasibility (Bonus B3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Latency Figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR Projection Latency Benchmark (p50/p95)")
    parser.add_argument("--data-root", default="data/kitti_mini")
    parser.add_argument("--frame", default="000011")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--out-csv", default="results/latency_benchmark.csv")
    parser.add_argument("--out-figure", default="results/figures/latency_distribution.png")
    args = parser.parse_args()

    stages, hw = benchmark_pipeline(args.data_root, args.frame, args.iterations)
    save_latency_csv(stages, hw, Path(args.out_csv))
    plot_latency(stages, Path(args.out_figure))


if __name__ == "__main__":
    main()
