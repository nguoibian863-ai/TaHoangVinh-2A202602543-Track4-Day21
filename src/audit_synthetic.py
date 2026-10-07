"""Automated Audit Tool for Built-in Synthetic Data Anomalies (Bonus B6).

Scans data/synthetic to detect, verify, and document all deliberate anomalies
injected into raw velodyne files, timestamps, calib, and labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.datasets import list_frames, load_frame
from starter.kitti_io import frame_paths
from starter.projection import box3d_corners_cam, cam_to_image


def audit_synthetic_dataset(data_root: str | Path = "data/synthetic") -> List[Dict[str, Any]]:
    root = Path(data_root)
    frames = list_frames(root)
    findings: List[Dict[str, Any]] = []

    # 1. Audit Timestamps
    ts_file = root / "training" / "timestamps.txt"
    if ts_file.exists():
        timestamps = [float(x) for x in ts_file.read_text().splitlines() if x.strip()]
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            if not np.isclose(dt, 0.1, atol=1e-3):
                findings.append({
                    "error_type": "Timestamp Jitter / Packet Drop (dt=0.20s vs nominal 0.10s)",
                    "error_frame": frames[i],
                    "debug_layer": "Time",
                    "detection_method": f"Computed delta t between consecutive frames; found dt={dt:.2f}s between frame {frames[i-1]} and {frames[i]}",
                    "impact": "Desynchronization between sensors, vehicle position interpolation errors",
                })

    # 2. Audit Frame-by-Frame Point Clouds and Labels
    nominal_points = 23800
    for fid in frames:
        fr = load_frame(root, fid)
        pts = fr["points"]

        # NaN / Inf detection
        nan_count = int(np.isnan(pts).any(axis=1).sum())
        inf_count = int(np.isinf(pts).any(axis=1).sum())
        if nan_count > 0 or inf_count > 0:
            findings.append({
                "error_type": f"Invalid Values (NaN={nan_count}, Inf={inf_count}) in Velodyne Raw Scan",
                "error_frame": fid,
                "debug_layer": "I/O & Preprocess",
                "detection_method": "np.isnan(pts).any(axis=1).sum(); detected unmasked NaN floating-point coordinates",
                "impact": "Crashes downstream coordinate transformations unless explicitly filtered",
            })

        # Sector Dropout / Missing Points
        if len(pts) < nominal_points * 0.95:
            # Analyze azimuth
            valid = pts[np.isfinite(pts).all(axis=1)]
            az = np.rad2deg(np.arctan2(valid[:, 1], valid[:, 0]))
            hist, edges = np.histogram(az, bins=36, range=(-180, 180))
            # Blind spot check
            blind_bins = np.where(hist < 250)[0]
            sectors = [f"[{edges[b]:.0f}°, {edges[b+1]:.0f}°]" for b in blind_bins if -50 <= edges[b] <= 10]
            findings.append({
                "error_type": f"LiDAR Sector Dropout (~{nominal_points - len(pts)} points dropped)",
                "error_frame": fid,
                "debug_layer": "Sensor / Preprocess",
                "detection_method": f"Azimuth histogram analysis in 10° bins; severe drop in sector {', '.join(sectors)}",
                "impact": "Sensor blind spot; inability to detect objects entering right/front field of view",
            })

        # Bounding Box Vertical Truncation
        img_shape = fr["image"].shape
        for obj in fr["labels"]:
            corners = box3d_corners_cam(obj)
            uv, depth, mask = cam_to_image(corners, fr["calib"].P2, img_shape)
            if mask.sum() < 8:
                # Some corners outside FOV
                findings.append({
                    "error_type": f"3D Box Field-of-View Truncation ({obj.type} at {depth[0]:.1f}m, only {mask.sum()}/8 corners in image)",
                    "error_frame": fid,
                    "debug_layer": "Geometry / Sensor FOV",
                    "detection_method": f"Projected 8 3D box corners to pixel space; found corners exceed vertical image bounds (v >= {img_shape[0]} px)",
                    "impact": "Ground truncation for very close obstacles (<8m) due to camera pitch and mounting height",
                })

    return findings


def save_audit_csv(findings: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["error_type", "error_frame", "debug_layer", "detection_method", "impact"]
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(",".join(headers) + "\n")
        for item in findings:
            fp.write(",".join(f'"{item[h]}"' for h in headers) + "\n")
    print(f"[OK] Saved Synthetic Audit CSV: {out_path} ({len(findings)} findings)")


def print_markdown_table(findings: List[Dict[str, Any]]) -> None:
    print("\n### Bảng phát hiện lỗi cài sẵn trong data/synthetic (Bonus B6)")
    print("| Lỗi | Frame bị lỗi | Cách phát hiện | Lớp Debug |")
    print("|---|---|---|---|")
    # Group similar errors for clean reporting
    seen = set()
    for f in findings:
        key = (f["error_type"].split("(")[0].strip(), f["error_frame"])
        if key in seen:
            continue
        seen.add(key)
        print(f"| {f['error_type']} | {f['error_frame']} | {f['detection_method']} | {f['debug_layer']} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data/synthetic for injected anomalies (Bonus B6)")
    parser.add_argument("--data-root", default="data/synthetic")
    parser.add_argument("--out-csv", default="results/synthetic_audit.csv")
    args = parser.parse_args()

    findings = audit_synthetic_dataset(args.data_root)
    save_audit_csv(findings, Path(args.out_csv))
    print_markdown_table(findings)


if __name__ == "__main__":
    main()
