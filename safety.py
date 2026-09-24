"""
safety.py — Safety Manager (TTC + Adaptive Speed Control)

Why this module exists:
  The controller alone just targets a fixed cruise speed.
  The Safety Manager sits between the perception pipeline and the controller,
  overriding the target speed based on how dangerous the current situation is.

TTC (Time-To-Collision) = distance / closing_speed
  If TTC is large → safe to cruise.
  If TTC is small → reduce speed or emergency brake.

Closing speed is estimated from consecutive distance measurements:
  closing_speed = (prev_distance - curr_distance) / dt
  Positive closing speed = we are approaching the obstacle.

Safety States:
  CLEAR      → no frontal obstacle detected
  SAFE       → obstacle present, comfortable distance
  CAUTION    → obstacle approaching, reduce speed
  BRAKE      → low TTC, significant speed reduction
  EMERGENCY  → critical TTC, full emergency stop
"""

import math
import config


# ── Safety state labels ────────────────────────────────────────────────────────
CLEAR     = "CLEAR"
SAFE      = "SAFE"
CAUTION   = "CAUTION"
BRAKE     = "BRAKE"
EMERGENCY = "EMERGENCY"

# State colors for visualization (RGB)
STATE_COLORS = {
    CLEAR:     (0,   200, 0),
    SAFE:      (100, 220, 100),
    CAUTION:   (255, 200, 0),
    BRAKE:     (255, 100, 0),
    EMERGENCY: (220, 0,   0),
}


class SafetyManager:
    """
    Inputs every frame:
      - fused_objects   : list of dicts with 'distance_m', 'class_name', 'bbox'
      - ego_speed_kmh   : current ego vehicle speed
      - dt              : time delta (seconds) since last call

    Outputs every frame:
      - target_speed    : float km/h (overrides cruise speed)
      - emergency_brake : bool
      - safety_state    : str (one of CLEAR/SAFE/CAUTION/BRAKE/EMERGENCY)
      - nearest_dist    : float or None
      - ttc             : float or None (seconds)
    """

    def __init__(self):
        self._prev_distance = None   # for closing speed estimation

    def _find_frontal_obstacle(self, fused_objects, img_width):
        """
        From all detected+fused objects, selects the most dangerous one
        in the frontal lane (center 50% of image width).

        Only objects with a known distance are considered.
        Returns the closest frontal obstacle dict or None.
        """
        if not fused_objects:
            return None

        center_x = img_width / 2.0
        # Frontal zone: center ± 25% of image width
        left_bound  = center_x - img_width * 0.25
        right_bound = center_x + img_width * 0.25

        frontal = []
        for obj in fused_objects:
            if obj.get('distance_m') is None:
                continue
            x1, y1, x2, y2 = obj['bbox']
            obj_cx = (x1 + x2) / 2.0
            # Check if object center is in the frontal lane
            if left_bound <= obj_cx <= right_bound:
                frontal.append(obj)

        if not frontal:
            return None

        # Return the closest one
        return min(frontal, key=lambda o: o['distance_m'])

    def _compute_ttc(self, distance, prev_distance, dt):
        """
        TTC = distance / closing_speed
        closing_speed = (prev_distance - distance) / dt
        If object is moving away (closing_speed <= 0), TTC = infinity.
        """
        if prev_distance is None or dt <= 0:
            return None

        closing_speed = (prev_distance - distance) / dt   # m/s

        if closing_speed <= 0:
            return None   # object moving away or stationary — no collision risk

        ttc = distance / closing_speed
        return ttc

    def update(self, fused_objects, ego_speed_kmh, dt, img_width=None):
        """
        Main safety update. Call once per simulation tick.

        Returns dict with:
            target_speed    : float km/h
            emergency_brake : bool
            safety_state    : str
            nearest_dist    : float or None
            ttc             : float or None
        """
        if img_width is None:
            img_width = config.CAMERA_RES_X

        obstacle = self._find_frontal_obstacle(fused_objects, img_width)

        if obstacle is None:
            # No frontal obstacle → cruise at normal speed
            self._prev_distance = None
            return {
                'target_speed':    config.TARGET_SPEED_KMH,
                'emergency_brake': False,
                'safety_state':    CLEAR,
                'nearest_dist':    None,
                'ttc':             None,
            }

        distance = obstacle['distance_m']
        ttc = self._compute_ttc(distance, self._prev_distance, dt)
        self._prev_distance = distance

        # ── Decision logic ─────────────────────────────────────────────────────
        emergency_brake = False
        state = SAFE
        target_speed = config.TARGET_SPEED_KMH

        if ttc is not None:
            if ttc < config.TTC_EMERGENCY:
                state = EMERGENCY
                target_speed = 0.0
                emergency_brake = True
            elif ttc < config.TTC_BRAKE:
                state = BRAKE
                # Aggressive speed reduction proportional to TTC
                ratio = ttc / config.TTC_BRAKE
                target_speed = config.TARGET_SPEED_KMH * ratio * 0.4
            elif ttc < config.TTC_CAUTION:
                state = CAUTION
                ratio = ttc / config.TTC_CAUTION
                target_speed = config.TARGET_SPEED_KMH * (0.4 + 0.4 * ratio)
            else:
                state = SAFE

        # Also apply distance-based speed reduction (regardless of TTC)
        if distance < config.DIST_EMERGENCY:
            state = EMERGENCY
            target_speed = 0.0
            emergency_brake = True
        elif distance < config.DIST_BRAKE and state not in (EMERGENCY,):
            state = BRAKE if state != EMERGENCY else state
            target_speed = min(target_speed, 5.0)
        elif distance < config.DIST_CAUTION and state not in (EMERGENCY, BRAKE):
            state = CAUTION
            # Scale speed with distance
            ratio = (distance - config.DIST_BRAKE) / (config.DIST_CAUTION - config.DIST_BRAKE)
            target_speed = min(target_speed, config.TARGET_SPEED_KMH * ratio)

        return {
            'target_speed':    max(0.0, target_speed),
            'emergency_brake': emergency_brake,
            'safety_state':    state,
            'nearest_dist':    distance,
            'ttc':             ttc,
        }
