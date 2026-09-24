# config.py
# Centralized parameters for the CARLA Adaptive Control project

# Environment
HOST = 'localhost'
PORT = 2000
TIMEOUT = 60.0
TOWN = 'Town01'
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05  # 20 Hz

# Vehicle
VEHICLE_MODEL = 'vehicle.tesla.model3'

# Camera Sensor
CAMERA_RES_X = 640
CAMERA_RES_Y = 360
CAMERA_FOV = 90

# LiDAR Sensor
LIDAR_RANGE = 50.0
LIDAR_ROTATION_FREQ = 20.0
LIDAR_CHANNELS = 16
LIDAR_POINTS_PER_SEC = 50000

# Perception
YOLO_CONF_THRESHOLD = 0.50
YOLO_MODEL = 'yolov8n.pt'

# Traffic
NPC_VEHICLES_COUNT = 20   # Number of NPC vehicles to spawn around the map

# --- Controller ---
# Lateral PID (steering)
LAT_KP = 0.8
LAT_KI = 0.05
LAT_KD = 0.2
STEERING_SMOOTHING = 0.4   # 0 = no smoothing, 1 = never changes

# Longitudinal PID (throttle/brake)
LON_KP = 0.15
LON_KI = 0.02
LON_KD = 0.05

# Waypoint navigation
MIN_LOOKAHEAD_M = 5.0          # Minimum lookahead distance
LOOKAHEAD_SPEED_FACTOR = 0.3   # lookahead_m += speed_kmh * factor

# Target cruise speed (km/h) — safety manager can reduce this
TARGET_SPEED_KMH = 30.0

# --- Safety Manager Thresholds ---
# TTC thresholds (seconds)
TTC_CAUTION   = 8.0    # Reduce speed
TTC_BRAKE     = 4.0    # Strong reduction
TTC_EMERGENCY = 2.0    # Full emergency stop

# Distance thresholds (metres)
DIST_CAUTION   = 20.0  # Start reducing speed
DIST_BRAKE     = 10.0  # Hard brake
DIST_EMERGENCY =  5.0  # Emergency stop
