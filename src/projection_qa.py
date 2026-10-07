"""LiDAR-Camera Projection QA and Calibration Sensitivity Analysis Tool.

This module performs quantitative benchmark of projection consistency and
sensitivity to extrinsic calibration drift (yaw, pitch, roll, translation)
across camera and LiDAR sensor frames.

Supports KITTI, nuScenes, and synthetic datasets.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add repository root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import dataset_type, list_frames, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    cam_to_image,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    velo_to_cam,
)


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute 2D Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def get_points_in_object_3d(pts_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Determine boolean mask of points strictly inside an object's 3D oriented bounding box.

    In KITTI camera frame:
      - obj.location: bottom center [cx, cy_bottom, cz]
      - dimensions: [h, w, l]
      - rotation_y: yaw around camera y-axis
    """
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    p_rel = pts_cam - obj.location
    p_local = (R_y.T @ p_rel.T).T

    in_x = np.abs(p_local[:, 0]) <= (l / 2.0)
    in_y = (p_local[:, 1] >= -h) & (p_local[:, 1] <= 0.0)
    in_z = np.abs(p_local[:, 2]) <= (w / 2.0)
    return in_x & in_y & in_z


def evaluate_frame_projection(
    frame: Dict[str, Any],
    calib_perturbed: KittiCalib,
    min_pts_per_obj: int = 5,
) -> Dict[str, float]:
    """Evaluate projection metrics for a single frame under given calibration."""
    pts_raw = frame["points"][:, :3]
    calib_orig = frame["calib"]
    img_shape = frame["image"].shape

    # Baseline projection
    pts_cam_orig = velo_to_cam(pts_raw, calib_orig)
    uv_orig, depth_orig, mask_orig = cam_to_image(pts_cam_orig, calib_orig.P2, img_shape)

    # Perturbed projection
    pts_cam_pert = velo_to_cam(pts_raw, calib_perturbed)
    uv_pert, depth_pert, mask_pert = cam_to_image(pts_cam_pert, calib_perturbed.P2, img_shape)

    # 1. FOV retention
    fov_ratio = float(mask_pert.sum() / len(pts_raw)) if len(pts_raw) > 0 else 0.0

    # 2. Distance-stratified pixel shift on common visible points
    common_mask = mask_orig & mask_pert
    if np.any(common_mask):
        uv_o_full = np.zeros((len(pts_raw), 2), dtype=np.float32)
        uv_p_full = np.zeros((len(pts_raw), 2), dtype=np.float32)
        uv_o_full[mask_orig] = uv_orig
        uv_p_full[mask_pert] = uv_pert

        shifts = np.linalg.norm(uv_o_full[common_mask] - uv_p_full[common_mask], axis=1)
        depths = pts_cam_orig[common_mask, 2]

        near_pts = shifts[depths < 15.0]
        mid_pts = shifts[(depths >= 15.0) & (depths < 30.0)]
        far_pts = shifts[depths >= 30.0]

        mean_shift_near = float(near_pts.mean()) if len(near_pts) > 0 else 0.0
        mean_shift_mid = float(mid_pts.mean()) if len(mid_pts) > 0 else 0.0
        mean_shift_far = float(far_pts.mean()) if len(far_pts) > 0 else 0.0
        mean_shift_overall = float(shifts.mean())
    else:
        mean_shift_near = mean_shift_mid = mean_shift_far = mean_shift_overall = 0.0

    # 3. Object-level metrics: Point-in-Box Retention (PBR) & Box IoU
    obj_pbr_list: List[float] = []
    obj_iou_list: List[float] = []

    for obj in frame["labels"]:
        in_3d = get_points_in_object_3d(pts_cam_orig, obj)
        if in_3d.sum() < min_pts_per_obj:
            continue

        pts_obj_cam_pert = pts_cam_pert[in_3d]
        uv_obj, _, _ = cam_to_image(pts_obj_cam_pert, calib_perturbed.P2, img_shape)

        if len(uv_obj) == 0:
            obj_pbr_list.append(0.0)
            obj_iou_list.append(0.0)
            continue

        x1, y1, x2, y2 = obj.bbox
        in_2d = (
            (uv_obj[:, 0] >= x1)
            & (uv_obj[:, 0] <= x2)
            & (uv_obj[:, 1] >= y1)
            & (uv_obj[:, 1] <= y2)
        )
        obj_pbr_list.append(float(in_2d.mean()))

        proj_bbox = np.array([
            uv_obj[:, 0].min(),
            uv_obj[:, 1].min(),
            uv_obj[:, 0].max(),
            uv_obj[:, 1].max(),
        ])
        obj_iou_list.append(compute_box_iou(proj_bbox, obj.bbox))

    mean_pbr = float(np.mean(obj_pbr_list)) if obj_pbr_list else 1.0
    mean_iou = float(np.mean(obj_iou_list)) if obj_iou_list else 1.0

    return {
        "fov_ratio": fov_ratio,
        "mean_shift_near_px": mean_shift_near,
        "mean_shift_mid_px": mean_shift_mid,
        "mean_shift_far_px": mean_shift_far,
        "mean_shift_overall_px": mean_shift_overall,
        "point_in_box_retention": mean_pbr,
        "projected_box_iou": mean_iou,
        "evaluated_objects": len(obj_pbr_list),
    }


def run_perturbation_sweep(
    data_root: str | Path,
    frame_ids: List[str],
    param_name: str = "yaw_deg",
    values: List[float] | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run controlled perturbation sweep over specified frames."""
    np.random.seed(seed)
    if values is None:
        values = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    ds_type = dataset_type(data_root)
    kwargs = {"use_ego_motion": True} if ds_type == "nuscenes" else {}

    results: List[Dict[str, Any]] = []

    for val in values:
        p_kwargs = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0.0, 0.0, 0.0)}
        if param_name in p_kwargs:
            p_kwargs[param_name] = val
        elif param_name == "tx":
            p_kwargs["t_xyz_m"] = (val, 0.0, 0.0)
        elif param_name == "ty":
            p_kwargs["t_xyz_m"] = (0.0, val, 0.0)
        elif param_name == "tz":
            p_kwargs["t_xyz_m"] = (0.0, 0.0, val)

        frame_metrics: List[Dict[str, float]] = []
        for fid in frame_ids:
            fr = load_frame(data_root, fid, **kwargs)
            calib_pert = perturb_extrinsic(fr["calib"], **p_kwargs)
            m = evaluate_frame_projection(fr, calib_pert)
            frame_metrics.append(m)

        # Aggregate across frames
        avg_fov = np.mean([m["fov_ratio"] for m in frame_metrics])
        avg_shift_near = np.mean([m["mean_shift_near_px"] for m in frame_metrics])
        avg_shift_mid = np.mean([m["mean_shift_mid_px"] for m in frame_metrics])
        avg_shift_far = np.mean([m["mean_shift_far_px"] for m in frame_metrics])
        avg_shift_overall = np.mean([m["mean_shift_overall_px"] for m in frame_metrics])
        avg_pbr = np.mean([m["point_in_box_retention"] for m in frame_metrics])
        avg_iou = np.mean([m["projected_box_iou"] for m in frame_metrics])
        total_eval_objs = sum(int(m["evaluated_objects"]) for m in frame_metrics)

        results.append({
            "param": param_name,
            "perturb_value": val,
            "fov_ratio": avg_fov,
            "mean_shift_near_px": avg_shift_near,
            "mean_shift_mid_px": avg_shift_mid,
            "mean_shift_far_px": avg_shift_far,
            "mean_shift_overall_px": avg_shift_overall,
            "point_in_box_retention": avg_pbr,
            "projected_box_iou": avg_iou,
            "evaluated_objects": total_eval_objs,
        })

    return results


def save_results_csv(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Save records to CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    headers = list(records[0].keys())
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(",".join(headers) + "\n")
        for r in records:
            fp.write(",".join(f"{r[h]:.6f}" if isinstance(r[h], float) else str(r[h]) for h in headers) + "\n")
    print(f"[OK] Saved CSV: {out_path} ({len(records)} rows)")


def plot_drift_benchmark(
    records: List[Dict[str, Any]],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Generate professional 3-panel figure showing calibration drift sensitivity."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r["perturb_value"] for r in records]
    param_name = records[0]["param"]
    unit = "deg" if "deg" in param_name else "m"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # Subplot 1: Point-in-Box Retention & Box IoU
    pbr = [r["point_in_box_retention"] * 100 for r in records]
    iou = [r["projected_box_iou"] for r in records]

    ax1 = axes[0]
    l1 = ax1.plot(x, pbr, "b-o", linewidth=2.2, label="Point-in-Box Retention (%)")
    ax1.set_xlabel(f"Perturbation {param_name} ({unit})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Point Retention (%)", color="b", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x, iou, "r--s", linewidth=2.0, label="Projected Box IoU")
    ax1_twin.set_ylabel("Bounding Box IoU", color="r", fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="r")

    # Joint legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", framealpha=0.9)
    ax1.set_title("Projection Alignment Sensitivity", fontsize=12, fontweight="bold")

    # Subplot 2: Distance-stratified Pixel Displacement
    ax2 = axes[1]
    s_near = [r["mean_shift_near_px"] for r in records]
    s_mid = [r["mean_shift_mid_px"] for r in records]
    s_far = [r["mean_shift_far_px"] for r in records]

    ax2.plot(x, s_near, "g-^", linewidth=2.0, label="Near (<15 m)")
    ax2.plot(x, s_mid, "y-d", linewidth=2.0, label="Mid (15-30 m)")
    ax2.plot(x, s_far, "m-x", linewidth=2.0, label="Far (>30 m)")
    ax2.set_xlabel(f"Perturbation {param_name} ({unit})", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Pixel Shift (px)", fontsize=11, fontweight="bold")
    ax2.set_title("Pixel Displacement by Distance Bucket", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper center", framealpha=0.9)

    # Subplot 3: FOV Retention & Drift Threshold
    ax3 = axes[2]
    fov = [r["fov_ratio"] * 100 for r in records]
    ax3.plot(x, fov, "k-o", linewidth=2.0, label="% LiDAR Points in Image")
    ax3.axhline(y=fov[x.index(0.0)], color="gray", linestyle=":", label="Nominal FOV")
    ax3.set_xlabel(f"Perturbation {param_name} ({unit})", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Points inside Image FOV (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Overall FOV Point Retention", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower center", framealpha=0.9)

    plt.suptitle(
        f"Extrinsic Calibration Sensitivity QA {title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure: {out_path}")


def save_visual_demo(
    data_root: str | Path,
    frame_id: str,
    out_dir: Path,
    yaw_deg: float = 0.0,
    prefix: str = "overlay",
) -> Path:
    """Save overlay visualization for report evidence."""
    ds_type = dataset_type(data_root)
    kwargs = {"use_ego_motion": True} if ds_type == "nuscenes" else {}
    fr = load_frame(data_root, frame_id, **kwargs)

    calib_pert = perturb_extrinsic(fr["calib"], yaw_deg=yaw_deg)
    uv, depth, mask = cam_to_image(
        velo_to_cam(fr["points"][:, :3], calib_pert), calib_pert.P2, fr["image"].shape
    )
    vis = overlay_points(fr["image"], uv, depth, radius=2)

    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, color=(0, 255, 0), label=obj.type)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{prefix}_{frame_id}_yaw_{yaw_deg:+.1f}deg.png"
    cv2.imwrite(str(out_file), vis)
    print(f"[OK] Saved Visual Demo: {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiDAR-Camera Projection QA Benchmark and Extrinsic Drift Analysis"
    )
    parser.add_argument(
        "--data-root",
        default="data/kitti_mini",
        help="Path to dataset root (default: data/kitti_mini)",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="List of frame IDs to evaluate (default: all available frames or representative sample)",
    )
    parser.add_argument(
        "--param",
        default="yaw_deg",
        choices=["yaw_deg", "pitch_deg", "roll_deg", "tx", "ty", "tz"],
        help="Calibration parameter to perturb (default: yaw_deg)",
    )
    parser.add_argument(
        "--values",
        nargs="*",
        type=float,
        default=[-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        help="Perturbation values to sweep",
    )
    parser.add_argument(
        "--out-csv",
        default="results/projection_benchmark.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--out-figure",
        default="results/figures/projection_drift_analysis.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save visual overlay comparison images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    available_frames = list_frames(args.data_root)
    if args.frames:
        target_frames = [f for f in args.frames if f in available_frames]
    else:
        # Pick 5 diverse frames if KITTI
        if len(available_frames) > 5:
            target_frames = ["000001", "000011", "000021", "000032", "000049"]
            target_frames = [f for f in target_frames if f in available_frames] or available_frames[:5]
        else:
            target_frames = available_frames

    print(f"=== Running Projection QA on {args.data_root} ===")
    print(f"Target Frames ({len(target_frames)}): {target_frames}")
    print(f"Perturbing {args.param} with values: {args.values}")

    records = run_perturbation_sweep(
        args.data_root, target_frames, param_name=args.param, values=args.values, seed=args.seed
    )

    save_results_csv(records, Path(args.out_csv))
    plot_drift_benchmark(records, Path(args.out_figure), title_suffix=f"({Path(args.data_root).name})")

    if args.save_overlays and target_frames:
        rep_frame = target_frames[0]
        fig_dir = Path("results/figures")
        save_visual_demo(args.data_root, rep_frame, fig_dir, yaw_deg=0.0, prefix="demo_nominal")
        save_visual_demo(args.data_root, rep_frame, fig_dir, yaw_deg=1.0, prefix="demo_yaw_1deg")
        save_visual_demo(args.data_root, rep_frame, fig_dir, yaw_deg=2.0, prefix="fail_yaw_2deg")


if __name__ == "__main__":
    main()
