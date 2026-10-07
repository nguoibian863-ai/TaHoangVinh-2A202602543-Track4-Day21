"""Unit and Edge-Case Test Suite for LiDAR Projection Functions."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.kitti_io import KittiCalib, load_calib
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam


class TestProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calib_path = ROOT / "data" / "synthetic" / "training" / "calib" / "000000.txt"
        cls.calib = load_calib(calib_path)
        cls.img_shape = (375, 1242, 3)

    def test_cp2_manual_point(self):
        """Test checkpoint CP2 hand-check: (10, 0, 0) -> z_cam ~ 9.73, (u, v) ~ (614, 175)."""
        pt = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        cam = velo_to_cam(pt, self.calib)
        self.assertAlmostEqual(float(cam[0, 2]), 9.727, places=2)

        uv, depth, mask = cam_to_image(cam, self.calib.P2, self.img_shape)
        self.assertTrue(mask[0])
        self.assertEqual(len(uv), 1)
        self.assertAlmostEqual(float(uv[0, 0]), 613.96, delta=1.0)
        self.assertAlmostEqual(float(uv[0, 1]), 175.01, delta=1.0)

    def test_empty_points(self):
        """Verify behavior on empty inputs."""
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        cam = velo_to_cam(empty_pts, self.calib)
        self.assertEqual(cam.shape, (0, 3))

        uv, depth, mask = cam_to_image(cam, self.calib.P2, self.img_shape)
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(mask.shape, (0,))

    def test_points_behind_camera(self):
        """Verify points behind camera (z_cam <= 0) are strictly filtered out."""
        # A point with x = -10 (behind vehicle)
        pts_behind = np.array([[-10.0, 0.0, 0.0]], dtype=np.float32)
        cam = velo_to_cam(pts_behind, self.calib)
        self.assertLess(cam[0, 2], 0.0)

        uv, depth, mask = cam_to_image(cam, self.calib.P2, self.img_shape)
        self.assertFalse(mask[0])
        self.assertEqual(len(uv), 0)

    def test_nan_and_inf_handling(self):
        """Verify points with NaN or Inf do not crash or produce invalid projections."""
        pts_corrupt = np.array([
            [10.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [10.0, np.inf, 0.0],
            [0.0, 0.0, np.nan],
        ], dtype=np.float32)
        cam = velo_to_cam(pts_corrupt, self.calib)
        self.assertEqual(len(cam), 4)

        uv, depth, mask = cam_to_image(cam, self.calib.P2, self.img_shape)
        # Only the first point is valid
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertEqual(len(uv), 1)

    def test_outside_fov_bounds(self):
        """Verify points projecting outside image dimensions are filtered out."""
        # Point far to the right in velodyne frame
        pts_out = np.array([[5.0, 50.0, 0.0]], dtype=np.float32)
        cam = velo_to_cam(pts_out, self.calib)
        uv, depth, mask = cam_to_image(cam, self.calib.P2, self.img_shape)
        self.assertFalse(mask[0])
        self.assertEqual(len(uv), 0)

    def test_perturb_extrinsic_consistency(self):
        """Verify perturb_extrinsic modifies calibration without mutating original."""
        calib_copy = perturb_extrinsic(self.calib, yaw_deg=1.5, t_xyz_m=(0.05, 0.0, 0.0))
        # Original should remain unchanged
        self.assertTrue(np.allclose(self.calib.Tr_velo_to_cam, self.calib.Tr_velo_to_cam))
        # Perturbed should differ
        self.assertFalse(np.allclose(self.calib.Tr_velo_to_cam, calib_copy.Tr_velo_to_cam))


if __name__ == "__main__":
    unittest.main()
