"""
visualization.py — Real-Time Telemetry Dashboard using Pygame

Displays:
  - Live camera feed (640x360)
  - YOLO bounding boxes with class, confidence, distance (color-coded by safety state)
  - Telemetry overlay panel: speed, target, throttle, brake, steering, TTC, FPS, safety state
  - Frontal zone indicator (center lane boundary lines)
"""

import time
import pygame
import numpy as np
import config
from safety import STATE_COLORS, CLEAR


# Telemetry panel height below the camera feed
PANEL_HEIGHT = 130
FONT_SIZE_LARGE = 18
FONT_SIZE_SMALL = 14

# Colors
WHITE  = (255, 255, 255)
GRAY   = (180, 180, 180)
DARK   = (30,  30,  30)
BLACK  = (0,   0,   0)
CYAN   = (0,   220, 220)


class Dashboard:
    def __init__(self):
        pygame.init()
        self.w = config.CAMERA_RES_X
        self.h = config.CAMERA_RES_Y
        self.total_h = self.h + PANEL_HEIGHT

        self.display = pygame.display.set_mode(
            (self.w, self.total_h),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Adaptive Control — Live Dashboard")

        self.font_lg = pygame.font.SysFont("Consolas", FONT_SIZE_LARGE, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", FONT_SIZE_SMALL)

        self._fps_timer = time.time()
        self._frame_count = 0
        self._fps = 0.0

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

    def render(self, bgr_image, fused_objects=None, telemetry=None):
        """
        Main render method.

        Args:
            bgr_image     : OpenCV BGR numpy array (H, W, 3)
            fused_objects : list of dicts with bbox, class_name, confidence, distance_m
            telemetry     : dict with speed, target_speed, throttle, brake,
                            steering, safety_state, ttc, nearest_dist
        """
        pygame.event.pump()
        self._update_fps()

        if fused_objects is None:
            fused_objects = []
        if telemetry is None:
            telemetry = {}

        safety_state = telemetry.get('safety_state', CLEAR)
        state_color  = STATE_COLORS.get(safety_state, (200, 200, 200))

        # ── Camera surface ─────────────────────────────────────────────────────
        rgb_array = bgr_image[:, :, ::-1]                            # BGR → RGB
        cam_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))

        # ── Frontal zone indicator lines ───────────────────────────────────────
        cx = self.w // 2
        zone_half = int(self.w * 0.25)
        pygame.draw.line(cam_surface, (80, 80, 255), (cx - zone_half, 0), (cx - zone_half, self.h), 1)
        pygame.draw.line(cam_surface, (80, 80, 255), (cx + zone_half, 0), (cx + zone_half, self.h), 1)

        # ── YOLO bounding boxes ────────────────────────────────────────────────
        for det in fused_objects:
            x1, y1, x2, y2 = det['bbox']
            dist = det.get('distance_m')
            conf = det.get('confidence', 0)
            cls  = det.get('class_name', '?')

            # Color depends on whether distance is known and safety state
            color = (0, 220, 0) if dist is not None else (255, 140, 0)

            # Draw box
            rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(cam_surface, color, rect, 2)

            # Label: class + conf + distance
            dist_str = f"{dist:.1f}m" if dist is not None else "?"
            label = f"{cls} {conf:.2f} | {dist_str}"
            txt = self.font_sm.render(label, True, color)
            cam_surface.blit(txt, (x1, max(0, y1 - 16)))

        self.display.blit(cam_surface, (0, 0))

        # ── Telemetry panel ────────────────────────────────────────────────────
        panel_rect = pygame.Rect(0, self.h, self.w, PANEL_HEIGHT)
        pygame.draw.rect(self.display, DARK, panel_rect)

        # Safety state bar (colored strip at top of panel)
        bar_rect = pygame.Rect(0, self.h, self.w, 6)
        pygame.draw.rect(self.display, state_color, bar_rect)

        # Build telemetry text columns
        speed     = telemetry.get('speed', 0.0)
        target    = telemetry.get('target_speed', 0.0)
        throttle  = telemetry.get('throttle', 0.0)
        brake     = telemetry.get('brake', 0.0)
        steering  = telemetry.get('steering', 0.0)
        ttc       = telemetry.get('ttc', None)
        nearest   = telemetry.get('nearest_dist', None)

        ttc_str     = f"{ttc:.1f}s" if ttc is not None else "  inf"
        nearest_str = f"{nearest:.1f}m" if nearest is not None else "  N/A"

        col1 = [
            f"Speed   : {speed:5.1f} km/h",
            f"Target  : {target:5.1f} km/h",
            f"Throttle: {throttle:.3f}",
        ]
        col2 = [
            f"Brake   : {brake:.3f}",
            f"Steering: {steering:+.3f}",
            f"FPS     : {self._fps:4.1f}",
        ]
        col3 = [
            f"Obstacle: {nearest_str}",
            f"TTC     : {ttc_str}",
            f"Safety  : {safety_state}",
        ]

        y0   = self.h + 12
        line_h = FONT_SIZE_LARGE + 4

        for i, (t1, t2, t3) in enumerate(zip(col1, col2, col3)):
            y = y0 + i * line_h
            self.display.blit(self.font_lg.render(t1, True, WHITE), (12, y))
            self.display.blit(self.font_lg.render(t2, True, WHITE), (self.w // 3 + 12, y))
            # Safety state column colored
            col3_color = state_color if i == 2 else WHITE
            self.display.blit(self.font_lg.render(t3, True, col3_color), (2 * self.w // 3 + 12, y))

        # FPS bar (small visual indicator)
        fps_frac = min(self._fps / 30.0, 1.0)
        pygame.draw.rect(self.display, (60, 60, 60), pygame.Rect(12, self.h + PANEL_HEIGHT - 12, 120, 6))
        pygame.draw.rect(self.display, CYAN, pygame.Rect(12, self.h + PANEL_HEIGHT - 12, int(120 * fps_frac), 6))

        pygame.display.flip()

    # Keep backward compat with old calls
    def render_camera(self, bgr_image, detections=None):
        self.render(bgr_image, fused_objects=detections)

    def destroy(self):
        pygame.quit()
