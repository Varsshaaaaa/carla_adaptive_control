# CARLA Adaptive Perception & Closed-Loop Control System

A complete, modular **Real-Time Adaptive Perception and Closed-Loop Control System** built on **CARLA 0.9.15**, demonstrated on an NVIDIA RTX 4050 Laptop GPU (6GB VRAM).

---

## Project Architecture

```
carla_adaptive_control/
├── config.py           # All parameters centralized
├── carla_env.py        # CARLA connection, world, vehicle & sensor spawning
├── perception.py       # YOLOv8n GPU inference (object detection)
├── sensor_fusion.py    # LiDAR-Camera projection & distance estimation
├── safety.py           # TTC calculation & adaptive speed safety manager
├── controller.py       # Lateral PID (steering) + Longitudinal PID (throttle/brake)
├── visualization.py    # Pygame real-time dashboard
├── main.py             # Orchestrator (main loop)
├── validate_environment.py
├── requirements.txt
├── install.bat
├── run_server.bat
├── run_client.bat
├── logs/               # Auto-generated CSV telemetry files
└── README.md
```

---

## Data Flow

```
CARLA Tick
  ↓
RGB Camera + LiDAR
  ↓
YOLO Object Detection       →  class, bbox, confidence
  ↓
Camera-LiDAR Fusion         →  distance_m per detection
  ↓
Safety Manager (TTC)        →  target_speed, emergency_brake, safety_state
  ↓
Longitudinal PID            →  throttle, brake
CARLA Waypoints
  ↓
Lateral PID                 →  steering
  ↓
Ego Vehicle Control
  ↓
Dashboard + CSV Logger
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
install.bat
```

### Step 2 — Validate environment
```bash
python validate_environment.py
```

### Step 3 — Start the CARLA Server
```bash
run_server.bat
```
Wait for the 3D window to appear (1-2 minutes first time).

### Step 4 — Run the client
```bash
run_client.bat
# or
python main.py
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `TARGET_SPEED_KMH` | 30.0 | Normal cruise speed |
| `NPC_VEHICLES_COUNT` | 20 | Number of NPC vehicles |
| `YOLO_MODEL` | yolov8n.pt | YOLO model file |
| `YOLO_CONF_THRESHOLD` | 0.50 | Detection confidence cutoff |
| `TTC_CAUTION` | 8.0s | Start reducing speed |
| `TTC_BRAKE` | 4.0s | Hard speed reduction |
| `TTC_EMERGENCY` | 2.0s | Full emergency stop |
| `DIST_CAUTION` | 20m | Distance-based caution |
| `DIST_BRAKE` | 10m | Distance-based brake |
| `DIST_EMERGENCY` | 5m | Distance-based emergency |
| `LAT_KP/KI/KD` | 0.8/0.05/0.2 | Lateral PID gains |
| `LON_KP/KI/KD` | 0.15/0.02/0.05 | Longitudinal PID gains |

---

## How Each Component Works 

### YOLO (perception.py)
YOLOv8n runs on GPU (CUDA). It takes a 640x360 RGB image and outputs bounding boxes, class labels, and confidence scores for vehicles and pedestrians. It does **not** produce distance.

### LiDAR Sensor Fusion (sensor_fusion.py)
The 16-channel LiDAR generates a 3D point cloud in sensor-local coordinates. We project every point through the camera intrinsic matrix (after correcting for CARLA's left-handed → OpenCV right-handed coordinate system). Points that land inside a YOLO bounding box belong to that object. We take the median of the closest 20th percentile of their depths as the distance estimate.

### Safety Manager (safety.py)
Implements TTC = distance / closing_speed, where closing_speed is estimated from two consecutive distance measurements. Based on TTC and raw distance, the manager transitions through 5 safety states (CLEAR → SAFE → CAUTION → BRAKE → EMERGENCY) and overrides the cruise speed command accordingly.

### PID Controllers (controller.py)
- **Lateral PID**: Computes heading error between the vehicle's current yaw and the angle toward a look-ahead waypoint. Feeds through P+I+D and applies steering smoothing.
- **Longitudinal PID**: Computes speed error. Positive error → throttle. Negative error → proportional braking.

### Telemetry (logs/)
Every run generates a CSV with timestamp, speed, target_speed, throttle, brake, steering, obstacle distance, TTC, safety_state, detections, and FPS. Use this to evaluate control stability and braking response.

---

## Hardware Requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA GTX 1060 6GB |
| RAM | 16 GB |
| Disk | 25 GB free |
| OS | Windows 10/11 or Ubuntu 22.04 |

Tested on: AMD Ryzen 7, NVIDIA RTX 4050 6GB, Windows 11.
