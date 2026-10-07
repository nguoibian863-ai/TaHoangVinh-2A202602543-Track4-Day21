"""Generate high-contrast, annotated failure case figures for REPORT.md.

Creates:
1. fail_01_yaw_drift_mismatch.png (Geometry layer: +2.0 deg extrinsic drift)
2. fail_02_ego_motion_time_desync.png (Time layer: nuScenes time desync without ego compensation)
3. fail_03_metric_failure_textureless.png (Metric layer: edge-metric breakdown in textureless region)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import load_frame
from starter.projection import (
    cam_to_image,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    velo_to_cam,
)


def create_fail_01_yaw_drift(out_path: Path) -> None:
    """Compare nominal projection vs 2.0 deg yaw drift on KITTI frame 000011."""
    fr = load_frame("data/kitti_mini", "000011")
    img = fr["image"]
    pts = fr["points"][:, :3]

    # Nominal
    uv_nom, d_nom, _ = cam_to_image(velo_to_cam(pts, fr["calib"]), fr["calib"].P2, img.shape)
    vis_nom = overlay_points(img, uv_nom, d_nom, radius=2)
    for obj in fr["labels"]:
        vis_nom = draw_box2d(vis_nom, obj.bbox, color=(0, 255, 0), label=obj.type)

    # 2.0 deg drift
    calib_pert = perturb_extrinsic(fr["calib"], yaw_deg=2.0)
    uv_pert, d_pert, _ = cam_to_image(velo_to_cam(pts, calib_pert), calib_pert.P2, img.shape)
    vis_pert = overlay_points(img, uv_pert, d_pert, radius=2)
    for obj in fr["labels"]:
        vis_pert = draw_box2d(vis_pert, obj.bbox, color=(0, 0, 255), label=f"{obj.type} [DRIFT]")

    # Crop interesting region around pedestrians and car (e.g. y: 150..370, x: 200..900)
    crop_nom = vis_nom[140:370, 200:850]
    crop_pert = vis_pert[140:370, 200:850]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300)
    axes[0].imshow(cv2.cvtColor(crop_nom, cv2.COLOR_BGR2RGB))
    axes[0].set_title("[NOMINAL] Calibration Chuẩn (Yaw = 0.0°) — Điểm LiDAR bám khít người và xe", fontsize=11, fontweight="bold", color="darkgreen")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(crop_pert, cv2.COLOR_BGR2RGB))
    axes[1].set_title("[FAILURE: GEOMETRY] Lệch Yaw +2.0° — Điểm LiDAR lệch ngang ~31px, rơi ra khỏi 2D Box của người và xe", fontsize=11, fontweight="bold", color="darkred")
    axes[1].axis("off")

    plt.suptitle("Failure Case 1 (Lớp Geometry): Sai lệch Extrinsic Calibration Drift trên KITTI 000011", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def create_fail_02_time_desync(out_path: Path) -> None:
    """Compare nuScenes projection with vs without ego-motion compensation."""
    fr_comp = load_frame("data/nuscenes_mini_subset", "scene-0103_010", use_ego_motion=True)
    fr_nocomp = load_frame("data/nuscenes_mini_subset", "scene-0103_010", use_ego_motion=False)

    dt_ms = (fr_comp["timestamp_camera_us"] - fr_comp["timestamp_lidar_us"]) / 1000.0

    uv_c, d_c, _ = cam_to_image(velo_to_cam(fr_comp["points"][:, :3], fr_comp["calib"]), fr_comp["calib"].P2, fr_comp["image"].shape)
    vis_c = overlay_points(fr_comp["image"], uv_c, d_c, radius=3)

    uv_nc, d_nc, _ = cam_to_image(velo_to_cam(fr_nocomp["points"][:, :3], fr_nocomp["calib"]), fr_nocomp["calib"].P2, fr_nocomp["image"].shape)
    vis_nc = overlay_points(fr_nocomp["image"], uv_nc, d_nc, radius=3)

    # Focus crop on cyclist / dynamic street section
    h, w = fr_comp["image"].shape[:2]
    crop_c = vis_c[350:800, 450:1250]
    crop_nc = vis_nc[350:800, 450:1250]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), dpi=300)
    axes[0].imshow(cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"[NOMINAL] Đã bù chuyển động Ego-motion (LiDAR-Camera Δt = {dt_ms:.1f} ms) — Khớp vật thể", fontsize=11, fontweight="bold", color="darkgreen")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(crop_nc, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"[FAILURE: TIME] Bỏ bù Ego-motion (Δt = {dt_ms:.1f} ms, xe di chuyển ~0.5m) — Điểm LiDAR trượt tụt hậu phía sau", fontsize=11, fontweight="bold", color="darkred")
    axes[1].axis("off")

    plt.suptitle("Failure Case 2 (Lớp Time): Lỗi đồng bộ thời gian LiDAR-Camera trên nuScenes scene-0103_010", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def create_fail_03_metric_breakdown(out_path: Path) -> None:
    """Demonstrate failure mode of edge alignment score in textureless/smooth region."""
    fr = load_frame("data/kitti_mini", "000021")
    img = fr["image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    dist_map = cv2.distanceTransform(255 - canny, cv2.DIST_L2, 3)

    # Pick a textureless asphalt ground patch (y: 280..360, x: 500..800)
    crop_img = img[260:360, 500:850]
    crop_canny = canny[260:360, 500:850]
    crop_dist = dist_map[260:360, 500:850]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    axes[0].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Vùng mặt đường nhựa không texture", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(crop_canny, cmap="gray")
    axes[1].set_title("2. Canny Edge rỗng (0 edge visual)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(crop_dist, cmap="viridis")
    axes[2].set_title("3. Distance Transform suy biến (Blind Spot)", fontsize=10, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Failure Case 3 (Lớp Metric/Preprocess): Thuật toán Edge-Alignment bất lực trước bề mặt đồng nhất",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    create_fail_01_yaw_drift(fig_dir / "fail_01_yaw_drift_mismatch.png")
    create_fail_02_time_desync(fig_dir / "fail_02_ego_motion_time_desync.png")
    create_fail_03_metric_breakdown(fig_dir / "fail_03_metric_failure_textureless.png")


if __name__ == "__main__":
    main()
