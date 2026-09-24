"""
controller.py — Lateral and Longitudinal PID Controllers

Why two separate PIDs?
  Lateral  PID → steers the car to follow road waypoints (heading error → steering angle)
  Longitudinal PID → controls speed via throttle/brake (speed error → throttle or brake)

This separation mirrors real AV architectures where steering and speed are
independent control loops, allowing the safety manager to override speed
without affecting steering.
"""

import math
import collections
import carla
import config


class LateralPIDController:
    """
    Steers the ego vehicle toward a target waypoint by computing
    the cross-track / heading error and running a PID on it.

    Output: steering in [-1, 1] where -1 = full left, +1 = full right.
    """

    def __init__(self):
        self.Kp = config.LAT_KP
        self.Ki = config.LAT_KI
        self.Kd = config.LAT_KD
        self._error_history = collections.deque(maxlen=10)
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_steering = 0.0

    def run(self, vehicle_transform, target_waypoint):
        """
        Args:
            vehicle_transform : carla.Transform of the ego vehicle
            target_waypoint   : carla.Waypoint to steer toward

        Returns:
            steering : float in [-1, 1]
        """
        v_loc = vehicle_transform.location
        v_yaw = math.radians(vehicle_transform.rotation.yaw)  # current heading

        wp_loc = target_waypoint.transform.location

        # Vector from vehicle to waypoint in world frame
        dx = wp_loc.x - v_loc.x
        dy = wp_loc.y - v_loc.y

        # Angle from vehicle to waypoint
        target_angle = math.atan2(dy, dx)

        # Heading error (how much we need to turn)
        error = target_angle - v_yaw

        # Normalize to [-pi, pi] to avoid wrap-around jumps
        error = math.atan2(math.sin(error), math.cos(error))

        # PID terms
        self._integral += error * config.FIXED_DELTA_SECONDS
        # Integral windup clamp: prevents accumulation during long straights
        self._integral = max(-1.0, min(1.0, self._integral))

        derivative = (error - self._prev_error) / max(config.FIXED_DELTA_SECONDS, 1e-6)
        self._prev_error = error

        raw = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # Saturate to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        # Smooth steering to avoid sharp jerks (exponential smoothing)
        alpha = config.STEERING_SMOOTHING
        steering = alpha * self._prev_steering + (1.0 - alpha) * raw
        self._prev_steering = steering

        return float(steering)


class LongitudinalPIDController:
    """
    Controls vehicle speed by computing the error between target and current
    speed, then outputting throttle (positive error) or brake (negative error).

    Output: (throttle, brake) each in [0, 1].
    """

    def __init__(self):
        self.Kp = config.LON_KP
        self.Ki = config.LON_KI
        self.Kd = config.LON_KD
        self._integral = 0.0
        self._prev_error = 0.0

    def run(self, target_speed_kmh, current_speed_kmh):
        """
        Args:
            target_speed_kmh  : desired speed in km/h (from safety manager)
            current_speed_kmh : actual speed in km/h (from vehicle velocity)

        Returns:
            (throttle, brake) : both floats in [0, 1]
        """
        error = target_speed_kmh - current_speed_kmh

        self._integral += error * config.FIXED_DELTA_SECONDS
        # Anti-windup: clamp integral
        self._integral = max(-10.0, min(10.0, self._integral))

        derivative = (error - self._prev_error) / max(config.FIXED_DELTA_SECONDS, 1e-6)
        self._prev_error = error

        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        if output >= 0:
            throttle = min(output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            # Apply proportional braking — stronger when further above target speed
            brake = min(abs(output), 1.0)

        return float(throttle), float(brake)


def get_speed_kmh(vehicle):
    """Returns the current speed of a CARLA vehicle in km/h."""
    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    return speed_ms * 3.6


def get_target_waypoint(world, vehicle, lookahead_m=None):
    """
    Finds the waypoint `lookahead_m` ahead of the vehicle on the current road.
    Scales look-ahead distance with current speed for dynamic behavior.

    Returns a carla.Waypoint or None if no valid waypoint found.
    """
    if lookahead_m is None:
        speed_kmh = get_speed_kmh(vehicle)
        # Dynamic lookahead: minimum 5m, scales up with speed
        lookahead_m = max(config.MIN_LOOKAHEAD_M,
                          speed_kmh * config.LOOKAHEAD_SPEED_FACTOR)

    carla_map = world.get_map()
    vehicle_transform = vehicle.get_transform()

    # Find the nearest waypoint on the road
    current_wp = carla_map.get_waypoint(
        vehicle_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if current_wp is None:
        return None

    # Walk forward along the waypoint graph
    # next() returns a list of possible next waypoints (branching at intersections)
    next_wps = current_wp.next(lookahead_m)
    if not next_wps:
        return None

    # At intersections, prefer the waypoint that is most aligned with current heading
    return next_wps[0]
