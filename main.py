"""
main.py — Orchestrator for the CARLA Adaptive Perception & Closed-Loop Control System

Data flow every simulation tick:
  CARLA Tick
    → RGB Camera + LiDAR
    → YOLO Object Detection          (perception.py)
    → Camera-LiDAR Fusion            (sensor_fusion.py)
    → Safety Manager / TTC           (safety.py)
    → Longitudinal PID (speed)       (controller.py)
    → Lateral PID (steering)         (controller.py)
    → Apply VehicleControl to CARLA
    → Render Dashboard               (visualization.py)
    → Log Telemetry to CSV
"""

import csv
import os
import time
import numpy as np
import carla
import config
from carla_env import CarlaEnvironment
from visualization import Dashboard


def _open_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    fname = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(fname, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'vehicle_speed', 'target_speed',
        'throttle', 'brake', 'steering',
        'nearest_obstacle_m', 'ttc', 'safety_state',
        'detections', 'fps',
    ])
    print(f"Logging telemetry to: {fname}")
    return f, writer


def main():
    env       = None
    dashboard = None
    log_file  = None

    latest_image = None
    latest_lidar = None

    def camera_callback(data):
        nonlocal latest_image
        latest_image = data

    def lidar_callback(data):
        nonlocal latest_lidar
        latest_lidar = data

    try:
        # ── Init ──────────────────────────────────────────────────────────────
        env       = CarlaEnvironment()
        dashboard = Dashboard()
        log_file, log_writer = _open_logger()

        from perception import PerceptionSystem
        from sensor_fusion import SensorFusion
        from safety import SafetyManager
        from controller import (LateralPIDController, LongitudinalPIDController,
                                get_speed_kmh, get_target_waypoint)

        perception   = PerceptionSystem(config.YOLO_MODEL)
        fusion       = SensorFusion()
        safety_mgr   = SafetyManager()
        lat_pid      = LateralPIDController()
        lon_pid      = LongitudinalPIDController()

        # ── Spawn actors ──────────────────────────────────────────────────────
        env.spawn_ego_vehicle()
        env.spawn_camera(camera_callback)
        env.spawn_lidar(lidar_callback)
        env.spawn_npc_vehicles()
        env.spawn_lead_vehicle(18.0)   # stationary obstacle for TTC/safety testing

        print("\n" + "="*60)
        print("  CARLA Adaptive Control System — Running")
        print(f"  Target Speed : {config.TARGET_SPEED_KMH} km/h")
        print(f"  Town         : {config.TOWN}")
        print(f"  YOLO Model   : {config.YOLO_MODEL}")
        print("  Press Ctrl+C to stop.")
        print("="*60 + "\n")

        tick_count   = 0
        fused_objects = []
        safety_result = {}
        dt = config.FIXED_DELTA_SECONDS
        prev_tick_time = time.time()

        # ── MAIN SYNCHRONOUS LOOP ─────────────────────────────────────────────
        while True:
            # 1. Advance CARLA by one timestep
            env.tick()
            tick_count += 1

            # Measure actual wall-clock dt for TTC calculation
            now = time.time()
            dt  = max(now - prev_tick_time, 1e-4)
            prev_tick_time = now

            if latest_image is None:
                continue

            # 2. Convert CARLA image → OpenCV BGR numpy
            arr = np.frombuffer(latest_image.raw_data, dtype=np.uint8)
            arr = arr.reshape((latest_image.height, latest_image.width, 4))
            bgr = np.ascontiguousarray(arr[:, :, :3])

            # 3. YOLO Perception
            detections = perception.detect(bgr)

            # 4. Camera-LiDAR Sensor Fusion
            if latest_lidar is not None:
                fused_objects = fusion.fuse(
                    latest_lidar,
                    detections,
                    env.lidar.get_transform(),
                    env.camera.get_transform(),
                )
            else:
                fused_objects = [dict(d, distance_m=None) for d in detections]

            # 5. Safety Manager → adaptive target speed & TTC
            current_speed = get_speed_kmh(env.ego_vehicle)
            safety_result = safety_mgr.update(
                fused_objects, current_speed, dt,
                img_width=config.CAMERA_RES_X,
            )
            target_speed   = safety_result['target_speed']
            emergency_brake = safety_result['emergency_brake']

            # 6. Lateral PID → steering
            target_wp = get_target_waypoint(env.world, env.ego_vehicle)
            steering  = 0.0
            if target_wp is not None:
                steering = lat_pid.run(env.ego_vehicle.get_transform(), target_wp)

            # 7. Longitudinal PID → throttle / brake
            if emergency_brake:
                throttle, brake = 0.0, 1.0
            else:
                throttle, brake = lon_pid.run(target_speed, current_speed)

            # 8. Apply control
            env.ego_vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steering,
                brake=brake,
                hand_brake=False,
                manual_gear_shift=False,
            ))

            # 9. Render full dashboard
            telemetry = {
                'speed':        current_speed,
                'target_speed': target_speed,
                'throttle':     throttle,
                'brake':        brake,
                'steering':     steering,
                'safety_state': safety_result.get('safety_state', 'CLEAR'),
                'ttc':          safety_result.get('ttc'),
                'nearest_dist': safety_result.get('nearest_dist'),
            }
            dashboard.render(bgr, fused_objects, telemetry)

            # 10. Log every 5 ticks (~4 Hz) to avoid huge CSV files
            if tick_count % 5 == 0:
                log_writer.writerow([
                    time.strftime('%H:%M:%S'),
                    f"{current_speed:.2f}",
                    f"{target_speed:.2f}",
                    f"{throttle:.3f}",
                    f"{brake:.3f}",
                    f"{steering:.3f}",
                    f"{safety_result.get('nearest_dist') or ''}",
                    f"{safety_result.get('ttc') or ''}",
                    safety_result.get('safety_state', ''),
                    len(detections),
                    f"{dashboard._fps:.1f}",
                ])

            # 11. Console telemetry every second (20 ticks)
            if tick_count % 20 == 0:
                st   = safety_result.get('safety_state', 'CLEAR')
                nd   = safety_result.get('nearest_dist')
                ttc  = safety_result.get('ttc')
                print(
                    f"[{st:<9}] "
                    f"Speed={current_speed:5.1f}→{target_speed:5.1f} km/h | "
                    f"Steer={steering:+.2f} | T={throttle:.2f} B={brake:.2f} | "
                    f"Obs={f'{nd:.1f}m' if nd else 'none':>7} | "
                    f"TTC={f'{ttc:.1f}s' if ttc else 'inf':>6} | "
                    f"Det={len(detections)} | FPS={dashboard._fps:.1f}"
                )

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        if log_file:
            log_file.close()
        if dashboard:
            dashboard.destroy()
        if env:
            env.cleanup()


if __name__ == '__main__':
    main()
