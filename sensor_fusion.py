"""
sensor_fusion.py — Camera-LiDAR Association Module

Why this module exists:
  YOLO gives bounding boxes in 2D pixel space (WHERE in the image),
  but has no knowledge of physical distance.
  LiDAR gives 3D point clouds (HOW FAR), but doesn't know what object it hit.
  By projecting LiDAR points into the camera image, we can ask:
  "Which LiDAR points land inside this YOLO bounding box?"
  The depth of those points = the physical distance to that object.

Coordinate Systems (CARLA / Unreal Engine 4):
  - CARLA uses a LEFT-HANDED coordinate system (same as UE4):
      X = forward,  Y = right,  Z = up
  - LiDAR raw data is in SENSOR-LOCAL frame (same convention as world, but relative to sensor origin)
  - Camera raw data same.
  - OpenCV camera model uses RIGHT-HANDED:
      X = right,  Y = down,  Z = forward (depth)
  - We apply R_carla2cv to convert the camera-frame points before projection.
"""

import numpy as np
import config

# Rotation: CARLA cam frame (X=fwd, Y=right, Z=up) → OpenCV frame (X=right, Y=down, Z=fwd)
R_CARLA2CV = np.array([
    [0,  1,  0],   # x_cv  = y_carla
    [0,  0, -1],   # y_cv  = -z_carla
    [1,  0,  0],   # z_cv  = x_carla (depth/forward)
], dtype=np.float32)


def build_intrinsics(width, height, fov_deg):
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width/2],
                     [0, f, height/2],
                     [0, 0, 1]], dtype=np.float32)


def transform_to_matrix(t):
    """carla.Transform → 4x4 float32 numpy (local→world)."""
    return np.array(t.get_matrix(), dtype=np.float32)


class SensorFusion:
    def __init__(self):
        self.K = build_intrinsics(config.CAMERA_RES_X, config.CAMERA_RES_Y, config.CAMERA_FOV)

    def fuse(self, raw_lidar_data, detections, lidar_transform, camera_transform):
        """
        For each YOLO detection, find LiDAR points inside its bounding box
        and return a physical distance estimate.

        Pipeline:
          LiDAR pts (sensor local)
            → World frame  (via lidar_transform matrix)
            → Camera local (via inverse camera_transform)
            → OpenCV frame (via R_CARLA2CV rotation)
            → 2D pixels    (via camera intrinsics K)
            → Distance     (depth of points inside bbox)
        """
        if raw_lidar_data is None or len(detections) == 0:
            return [dict(d, distance_m=None) for d in detections]

        # --- Parse LiDAR buffer ---
        pts = np.frombuffer(raw_lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3].copy()                        # (N, 3)
        if xyz.shape[0] < 10:
            return [dict(d, distance_m=None) for d in detections]

        # Ground filter: discard points very low relative to sensor
        xyz = xyz[xyz[:, 2] > -2.0]                    # sensor Z > -2m

        N = xyz.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        xyz_h = np.hstack([xyz, ones]).T               # (4, N)

        # --- Step 1: LiDAR sensor local → World ---
        T_L2W = transform_to_matrix(lidar_transform)   # (4,4)
        pts_world = T_L2W @ xyz_h                      # (4, N)

        # --- Step 2: World → Camera sensor local (CARLA convention) ---
        T_C2W = transform_to_matrix(camera_transform)
        T_W2C = np.linalg.inv(T_C2W)
        pts_cam_carla = (T_W2C @ pts_world)[:3, :].T  # (N, 3)

        # --- Step 3: CARLA cam frame → OpenCV cam frame ---
        pts_cv = (R_CARLA2CV @ pts_cam_carla.T).T      # (N, 3)

        # Keep only points in front of camera
        in_front = pts_cv[:, 2] > 0.1
        if in_front.sum() == 0:
            return [dict(d, distance_m=None) for d in detections]

        depths = pts_cv[:, 2]                          # (N,) — OpenCV depth

        # --- Step 4: Project to 2D ---
        pixels = np.zeros((N, 2), dtype=np.int32)
        fwd_pts = pts_cv[in_front]                     # (M, 3)
        uvw = self.K @ fwd_pts.T                       # (3, M)
        uv = (uvw[:2] / uvw[2]).T.astype(np.int32)     # (M, 2)
        pixels[in_front] = uv

        W, H = config.CAMERA_RES_X, config.CAMERA_RES_Y

        # --- Step 5: For each detection, find matching LiDAR points ---
        fused = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            inside = []
            for i in range(N):
                if not in_front[i]:
                    continue
                px, py = pixels[i]
                # Check pixel is inside image AND inside bounding box
                if 0 <= px < W and 0 <= py < H:
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        inside.append(depths[i])

            dist = None
            if len(inside) >= 3:
                arr = np.array(inside)
                # Median of the closest 20th percentile = robust object surface estimate
                p20 = np.percentile(arr, 20)
                close = arr[arr <= p20 * 1.5]
                dist = float(np.median(close)) if len(close) > 0 else float(np.median(arr))

            fused.append({**det, 'distance_m': dist})

        return fused
