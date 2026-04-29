"""
AATIS - Predictive Interception System (Simulation)
====================================================
Instead of chasing a fast target, the system:
  1. Detects & tracks the moving target over time.
  2. Calculates velocity/direction, predicts future (x,y).
  3. Generates a Kill Zone corridor ahead of the target's path.
  4. When the target enters the corridor, it gets TAGGED.
  5. An interceptor drone moves to the predicted position, not the current one.

Controls: q=quit  r=reset  +/-=prediction horizon  f=smoothing  space=pause
"""

import cv2
import numpy as np
import math
import time
from collections import deque

# ── Dimensions ────────────────────────────────────────────────────────────
W, H = 900, 650
PRED_DT = 18
SMOOTH_ALPHA = 0.35
HISTORY = 80
CORRIDOR_NARROW = 16           # Funnel narrow end (near target)
CORRIDOR_WIDE = 80             # Funnel wide end (at predicted pos)
CORRIDOR_LENGTH_FACTOR = 1.3
CORRIDOR_CURVE_PTS = 30        # Smoothness of curved edges
INTERCEPT_SPEED = 3.5          # interceptor drone max speed (px/frame)
ARC_OFFSET = 60                # how much the drone arcs sideways

# ── Colors ────────────────────────────────────────────────────────────────
C_BG1 = (25, 28, 30); C_BG2 = (35, 38, 42); C_GRID = (42, 45, 48)
C_TARGET = (0, 0, 230); C_TARGET_TAG = (0, 255, 255)
C_DETECT = (0, 255, 100); C_PRED = (0, 160, 255)
C_TRAIL = (255, 200, 50); C_PATH = (80, 80, 255)
C_CORRIDOR = (0, 80, 180); C_CORRIDOR_ACTIVE = (0, 180, 255)
C_DRONE = (255, 180, 0); C_DRONE_TRAIL = (200, 140, 0)
C_BBOX = (255, 255, 0); C_VEL = (150, 255, 0)
C_PNL = (20, 22, 25); C_TXT = (200, 200, 200); C_HDR = (0, 240, 255)
C_ACC = (0, 200, 255); C_TAG_TEXT = (0, 255, 255); C_RING = (0, 0, 255)
C_ZONE_FILL = (0, 50, 120)

# HSV range for red ball detection
HSV_LO1 = (0, 100, 100);  HSV_HI1 = (10, 255, 255)
HSV_LO2 = (170, 100, 100); HSV_HI2 = (180, 255, 255)


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATED TARGET (fast-moving ball)
# ═══════════════════════════════════════════════════════════════════════════
class Target:
    def __init__(self):
        self.r = 18
        self.x = 30.0                          # start from left side
        self.y = float(H // 2)                  # fixed Y — middle of screen
        self.vx = np.random.uniform(4, 6)       # always moving right (+X)
        self.vy = 0.0                            # no vertical movement
        self.tagged = False
        self.tag_timer = 0

    def update(self):
        # Small horizontal speed variation for realism
        self.vx += np.random.uniform(-0.1, 0.1)
        self.vx = np.clip(self.vx, 3, 8)        # keep moving right
        self.x += self.vx
        # When target goes off the right edge, reset to left side
        if self.x - self.r > W:
            self.x = float(-self.r)
        if self.tag_timer > 0:
            self.tag_timer -= 1
        else:
            self.tagged = False

    def draw(self, frame):
        color = C_TARGET_TAG if self.tagged else C_TARGET
        p = self.pos
        # Drone body
        cv2.circle(frame, p, int(self.r * 0.5), color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, int(self.r * 0.75), color, 2, cv2.LINE_AA)
        # Drone arms and rotors
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            ex = int(self.x + self.r * 1.2 * math.cos(rad))
            ey = int(self.y + self.r * 1.2 * math.sin(rad))
            cv2.line(frame, p, (ex, ey), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 7, color, 1, cv2.LINE_AA)
            
        if self.tagged:
            cv2.circle(frame, self.pos, int(self.r * 1.5) + 6, C_TAG_TEXT, 2, cv2.LINE_AA)
            cv2.putText(frame, "TAGGED", (self.pos[0] - 25, self.pos[1] - int(self.r * 1.5) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TAG_TEXT, 2, cv2.LINE_AA)

    @property
    def pos(self):
        return (int(self.x), int(self.y))

    def tag(self):
        self.tagged = True
        self.tag_timer = 30  # stay tagged for 30 frames


# ═══════════════════════════════════════════════════════════════════════════
#  INTERCEPTOR DRONE — moves toward predicted position
# ═══════════════════════════════════════════════════════════════════════════
class Interceptor:
    """Drone that stays still until engaged, then follows a curved arc to predicted pos."""
    def __init__(self):
        self.x = float(W // 2); self.y = float(50)
        self.home_x = self.x; self.home_y = self.y
        self.target_x = self.x; self.target_y = self.y
        self.speed = INTERCEPT_SPEED
        self.trail = deque(maxlen=60)
        self.active = False
        self.engaged = False       # stays still until E is pressed
        self.arc_pts = []
        self.arc_idx = 0
        self._last_tgt = (0, 0)

    def reset(self):
        self.x = self.home_x; self.y = self.home_y
        self.trail.clear(); self.active = False; self.engaged = False
        self.arc_pts = []; self.arc_idx = 0

    def _build_arc(self):
        """Build a quadratic Bezier curve from drone to predicted pos."""
        p0 = np.array([self.x, self.y])
        p2 = np.array([self.target_x, self.target_y])
        mid = (p0 + p2) / 2
        # Perpendicular offset for the arc bend
        dx = p2[0] - p0[0]; dy = p2[1] - p0[1]
        perp = np.array([-dy, dx])
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm * ARC_OFFSET
        ctrl = mid + perp  # control point offset to one side
        # Sample Bezier
        pts = []
        steps = 40
        for i in range(steps + 1):
            t = i / steps
            b = (1-t)**2 * p0 + 2*(1-t)*t * ctrl + t**2 * p2
            pts.append(b)
        self.arc_pts = pts
        self.arc_idx = 0

    def engage(self, px, py):
        """Engage! Move toward the PREDICTED position via a curved arc."""
        self.engaged = True
        self.active = True
        new_tgt = (float(px), float(py))
        self.target_x = new_tgt[0]; self.target_y = new_tgt[1]
        self._last_tgt = new_tgt
        self._build_arc()

    def update(self):
        if not self.engaged or not self.arc_pts:
            return
        # Follow the arc path point by point
        if self.arc_idx < len(self.arc_pts):
            goal = self.arc_pts[self.arc_idx]
            dx = goal[0] - self.x; dy = goal[1] - self.y
            dist = math.hypot(dx, dy)
            if dist < self.speed * 1.5:
                self.arc_idx += 1
            if dist > 1:
                self.x += (dx / dist) * self.speed
                self.y += (dy / dist) * self.speed
        else:
            # Reached end — track directly
            dx = self.target_x - self.x; dy = self.target_y - self.y
            dist = math.hypot(dx, dy)
            if dist > 2:
                self.x += (dx / dist) * self.speed
                self.y += (dy / dist) * self.speed
        self.trail.append((int(self.x), int(self.y)))

    def draw(self, frame):
        if not self.active:
            return
        # Draw planned arc path (faint)
        if self.arc_pts and self.arc_idx < len(self.arc_pts):
            for i in range(max(0, self.arc_idx), len(self.arc_pts) - 1):
                p1 = (int(self.arc_pts[i][0]), int(self.arc_pts[i][1]))
                p2 = (int(self.arc_pts[i+1][0]), int(self.arc_pts[i+1][1]))
                cv2.line(frame, p1, p2, (100, 80, 0), 1, cv2.LINE_AA)
        # Drone trail (actual path taken)
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = tuple(int(v * a) for v in C_DRONE_TRAIL)
            cv2.line(frame, pts[i-1], pts[i], c, 2, cv2.LINE_AA)
        # Drone body — X shape like a quadcopter
        p = (int(self.x), int(self.y))
        cv2.circle(frame, p, 10, C_DRONE, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 14, C_DRONE, 2, cv2.LINE_AA)
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            ex = int(self.x + 18 * math.cos(rad))
            ey = int(self.y + 18 * math.sin(rad))
            cv2.line(frame, p, (ex, ey), C_DRONE, 2, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 4, C_DRONE, -1, cv2.LINE_AA)
        cv2.putText(frame, "INTERCEPTOR", (p[0] + 22, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DRONE, 1, cv2.LINE_AA)

    @property
    def pos(self):
        return (int(self.x), int(self.y))

    def distance_to(self, px, py):
        return math.hypot(self.x - px, self.y - py)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1: DETECTION (HSV color)
# ═══════════════════════════════════════════════════════════════════════════
class Detector:
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, HSV_LO1, HSV_HI1)
        m2 = cv2.inRange(hsv, HSV_LO2, HSV_HI2)
        mask = cv2.bitwise_or(m1, m2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None, mask
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 200: return None, None, mask
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2), (x, y, w, h), mask


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2: TRACKING
# ═══════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.history = deque(maxlen=HISTORY)

    def reset(self): self.history.clear()

    def update(self, p): self.history.append(p)

    @property
    def cur(self): return self.history[-1] if self.history else None

    @property
    def prev(self): return self.history[-2] if len(self.history) >= 2 else None

    @property
    def ready(self): return len(self.history) >= 2

    def trail(self): return list(self.history)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3: PREDICTION + KILL ZONE CORRIDOR
# ═══════════════════════════════════════════════════════════════════════════
class Predictor:
    def __init__(self):
        self.vx_s = 0.0; self.vy_s = 0.0; self.smoothing = True

    def reset(self): self.vx_s = 0.0; self.vy_s = 0.0

    def velocity(self, cur, prev):
        vxr = cur[0] - prev[0]; vyr = cur[1] - prev[1]
        if self.smoothing:
            self.vx_s = SMOOTH_ALPHA * vxr + (1 - SMOOTH_ALPHA) * self.vx_s
            self.vy_s = SMOOTH_ALPHA * vyr + (1 - SMOOTH_ALPHA) * self.vy_s
            return self.vx_s, self.vy_s
        return float(vxr), float(vyr)

    def predict(self, cur, vx, vy, dt):
        return (int(cur[0] + vx * dt), int(cur[1] + vy * dt))

    def path(self, cur, vx, vy, dt, n=12):
        return [(int(cur[0] + vx * dt * i / n), int(cur[1] + vy * dt * i / n))
                for i in range(1, n + 1)]

    def corridor(self, cur, vx, vy, dt):
        """
        Build a curved funnel-shaped Kill Zone:
          - Narrow end near the target's current position
          - Wide end at the predicted future position
          - Curved edges (Bezier) connecting them
        Returns a numpy polygon (many points) for the curved funnel.
        """
        speed = math.hypot(vx, vy)
        if speed < 0.5:
            return None
        ux, uy = vx / speed, vy / speed  # unit direction
        px, py = -uy, ux                  # perpendicular

        start_dist = speed * 2
        end_dist = speed * dt * CORRIDOR_LENGTH_FACTOR
        narrow_hw = CORRIDOR_NARROW / 2
        wide_hw = CORRIDOR_WIDE / 2

        # Start (narrow) and end (wide) center points
        sx = cur[0] + ux * start_dist; sy = cur[1] + uy * start_dist
        ex = cur[0] + ux * end_dist;   ey = cur[1] + uy * end_dist

        # Build left edge: Bezier from narrow-left to wide-left
        # with a control point that curves outward
        n = CORRIDOR_CURVE_PTS
        left_edge = []
        right_edge = []
        for i in range(n + 1):
            t = i / n
            # Smoothly interpolate width from narrow to wide using ease-in
            w = narrow_hw + (wide_hw - narrow_hw) * (t ** 0.6)
            # Position along the corridor center line
            cx = sx + (ex - sx) * t
            cy = sy + (ey - sy) * t
            # Offset perpendicular
            left_edge.append([int(cx + px * w), int(cy + py * w)])
            right_edge.append([int(cx - px * w), int(cy - py * w)])

        # Combine: left edge forward, then right edge backward = closed polygon
        right_edge.reverse()
        all_pts = left_edge + right_edge
        return np.array(all_pts, dtype=np.int32)

    @staticmethod
    def point_in_corridor(point, corridor_pts):
        """Check if a point is inside the corridor polygon."""
        if corridor_pts is None: return False
        result = cv2.pointPolygonTest(corridor_pts.reshape(-1, 1, 2),
                                       (float(point[0]), float(point[1])), False)
        return result >= 0


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def draw_trail(f, trail, color=C_TRAIL):
    n = len(trail)
    for i in range(1, n):
        a = i / n
        cv2.line(f, trail[i-1], trail[i],
                 tuple(int(v * a) for v in color), max(1, int(a * 3)), cv2.LINE_AA)

def draw_pred_path(f, path):
    for i, pt in enumerate(path):
        fade = 1.0 - i / len(path) * 0.5
        c = tuple(int(v * fade) for v in C_PATH)
        cv2.circle(f, pt, max(2, 5 - i // 3), c, -1, cv2.LINE_AA)
        if i > 0 and i % 2 == 0:
            cv2.line(f, path[i-1], pt, c, 2, cv2.LINE_AA)

def draw_corridor(f, pts, active=False):
    """Draw the curved funnel Kill Zone with fill and glowing edges."""
    if pts is None: return
    overlay = f.copy()
    fill = C_CORRIDOR_ACTIVE if active else C_ZONE_FILL
    cv2.fillPoly(overlay, [pts], fill)
    alpha = 0.4 if active else 0.25
    cv2.addWeighted(overlay, alpha, f, 1 - alpha, 0, f)
    # Glowing border — draw twice for glow effect
    border = C_CORRIDOR_ACTIVE if active else C_CORRIDOR
    cv2.polylines(f, [pts], True, border, 2, cv2.LINE_AA)
    if active:
        glow = tuple(min(255, v + 60) for v in border)
        cv2.polylines(f, [pts], True, glow, 1, cv2.LINE_AA)
    # Label at center of funnel
    cx = int(np.mean(pts[:, 0])); cy = int(np.mean(pts[:, 1]))
    label = "!! KILL ZONE ACTIVE !!" if active else "KILL ZONE"
    color = C_TAG_TEXT if active else C_CORRIDOR
    cv2.putText(f, label, (cx - 70, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_crosshair(f, c, sz=20, clr=C_PRED):
    cv2.line(f, (c[0]-sz, c[1]), (c[0]+sz, c[1]), clr, 2, cv2.LINE_AA)
    cv2.line(f, (c[0], c[1]-sz), (c[0], c[1]+sz), clr, 2, cv2.LINE_AA)
    cv2.circle(f, c, sz // 2, clr, 1, cv2.LINE_AA)

def draw_panel(f, lines, org=(10, 10), lh=24, fs=0.44, pad=8):
    h = len(lines) * lh + pad * 2
    w = max(len(l) for l in lines) * 10 + pad * 2
    x0, y0 = org
    ov = f.copy()
    cv2.rectangle(ov, (x0, y0), (x0+w, y0+h), C_PNL, -1)
    cv2.addWeighted(ov, 0.8, f, 0.2, 0, ov)
    cv2.rectangle(ov, (x0, y0), (x0+w, y0+h), C_ACC, 1, cv2.LINE_AA)
    for i, l in enumerate(lines):
        cv2.putText(ov, l, (x0+pad, y0+pad+(i+1)*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    C_HDR if i == 0 else C_TXT, 1, cv2.LINE_AA)
    np.copyto(f, ov)

def make_bg():
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        v = int(25 + 18 * (y / H))
        bg[y, :] = (v, v + 4, v + 2)
    for gx in range(0, W, 50):
        cv2.line(bg, (gx, 0), (gx, H), C_GRID, 1)
    for gy in range(0, H, 50):
        cv2.line(bg, (0, gy), (W, gy), C_GRID, 1)
    return bg


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    target = Target()
    interceptor = Interceptor()
    detector = Detector()
    tracker = Tracker()
    predictor = Predictor()
    pred_dt = PRED_DT
    bg = make_bg()
    frame_n = 0; fps_avg = 0.0; paused = False
    corridor_active = False

    print("[AATIS] Predictive Interception System — Simulation")
    print("[INFO]  q=quit  r=reset  +/-=horizon  f=smoothing  space=pause")

    while True:
        t0 = time.perf_counter()
        frame_n += 1

        if not paused:
            target.update()
            interceptor.update()

        # Build scene (target drawn on clean background)
        scene = bg.copy()
        target.draw(scene)
        display = scene.copy()

        # ── Stage 1: Detect ──────────────────────────────────────────
        center, bbox, mask = detector.detect(scene)

        # ── Stage 2: Track ───────────────────────────────────────────
        if center and not paused:
            tracker.update(center)

        # ── Stage 3: Predict + Corridor + Intercept ──────────────────
        predicted = None; pred_path = []; corridor_pts = None
        vx = vy = speed = 0.0
        corridor_active = False

        if tracker.ready:
            cur, prev = tracker.cur, tracker.prev
            vx, vy = predictor.velocity(cur, prev)
            speed = math.hypot(vx, vy)
            predicted = predictor.predict(cur, vx, vy, pred_dt)
            pred_path = predictor.path(cur, vx, vy, pred_dt)

            # Build Kill Zone corridor ahead of target
            corridor_pts = predictor.corridor(cur, vx, vy, pred_dt)

            # Check if target has entered the corridor → TAG it
            if predictor.point_in_corridor(target.pos, corridor_pts):
                corridor_active = True

            # Auto-tag when interceptor reaches the target
            if interceptor.engaged and interceptor.distance_to(target.x, target.y) < 40:
                target.tag()
                target.tag_timer = 90
                interceptor.engaged = False
                interceptor.x = interceptor.home_x
                interceptor.y = interceptor.home_y
                interceptor.trail.clear()
                interceptor.arc_pts = []
                print("[INFO] Interceptor reached target — TAGGED & returned!")

        # ── Drawing ──────────────────────────────────────────────────
        # Trajectory trail
        trail = tracker.trail()
        if len(trail) > 1:
            draw_trail(display, trail)

        # Kill Zone corridor
        draw_corridor(display, corridor_pts, corridor_active)

        # Predicted path
        if pred_path:
            draw_pred_path(display, pred_path)

        # Bounding box
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x+w, y+h), C_BBOX, 2, cv2.LINE_AA)

        # Current detection dot
        if center:
            cv2.circle(display, center, 6, C_DETECT, -1, cv2.LINE_AA)
            cv2.circle(display, center, 10, C_DETECT, 2, cv2.LINE_AA)

        # Velocity arrow
        if center and speed > 0.5:
            ev = (int(center[0] + vx * 5), int(center[1] + vy * 5))
            cv2.arrowedLine(display, center, ev, C_VEL, 2, cv2.LINE_AA, tipLength=0.3)

        # Predicted position crosshair
        if predicted:
            draw_crosshair(display, predicted)
            cv2.circle(display, predicted, 5, C_PRED, -1, cv2.LINE_AA)
            cv2.circle(display, predicted, 16, C_RING, 2, cv2.LINE_AA)
            cv2.putText(display, f"PRED ({predicted[0]},{predicted[1]})",
                        (predicted[0] + 18, predicted[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_RING, 1, cv2.LINE_AA)

        # Connection line current → predicted
        if center and predicted:
            cv2.line(display, center, predicted, C_ACC, 1, cv2.LINE_AA)

        # Interceptor drone
        interceptor.draw(display)

        # Re-draw target on top so it's always visible
        target.draw(display)

        # FPS
        dt_f = time.perf_counter() - t0
        fps_avg = 0.9 * fps_avg + 0.1 / max(dt_f, 1e-9)
        direction = math.degrees(math.atan2(-vy, vx)) % 360 if speed > 0.3 else 0

        # ── Info Panel (top-left) ────────────────────────────────────
        info = [
            "AATIS Predictive Interception",
            f"FPS: {fps_avg:.0f}  Frame: {frame_n}",
            f"Smooth: {'ON' if predictor.smoothing else 'OFF'}  dt: {pred_dt}f",
            f"Target: {'TAGGED' if target.tagged else 'untagged'}",
            f"Corridor: {'ACTIVE' if corridor_active else 'waiting'}",
        ]
        if center:
            info.append(f"Detected : ({center[0]}, {center[1]})")
        if predicted:
            info.append(f"Predicted: ({predicted[0]}, {predicted[1]})")
        if speed > 0.3:
            info.append(f"v=({vx:.1f},{vy:.1f}) spd={speed:.1f} dir={direction:.0f}")
        d2t = interceptor.distance_to(target.x, target.y)
        info.append(f"Drone dist to target: {d2t:.0f}px")
        draw_panel(display, info)

        # ── Math Panel (bottom-right) ────────────────────────────────
        if center and predicted and speed > 0.3 and tracker.prev:
            ml = [
                "Prediction Math:",
                f"vx={tracker.cur[0]}-{tracker.prev[0]}={vx:.1f}",
                f"vy={tracker.cur[1]}-{tracker.prev[1]}={vy:.1f}",
                f"x_pred={center[0]}+{vx:.1f}*{pred_dt}={predicted[0]}",
                f"y_pred={center[1]}+{vy:.1f}*{pred_dt}={predicted[1]}",
            ]
            draw_panel(display, ml, org=(W - 340, H - 160))

        # ── Strategy Panel (top-right) ───────────────────────────────
        strat = [
            "Strategy:",
            "1. Predict target trajectory",
            "2. Place Kill Zone corridor ahead",
            "3. Target enters corridor -> TAG",
            "4. Drone -> predicted (x,y)",
            "Press [E] to manually TAG target",
        ]
        draw_panel(display, strat, org=(W - 290, 10))

        # Show
        cv2.imshow("AATIS - Predictive Interception", display)
        mc = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Detection Mask", cv2.resize(mc, (W // 3, H // 3)))

        # Keys
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            target = Target(); interceptor.reset()
            tracker.reset(); predictor.reset(); corridor_active = False
        elif key in (ord('+'), ord('=')): pred_dt = min(pred_dt + 2, 60)
        elif key in (ord('-'), ord('_')): pred_dt = max(pred_dt - 2, 1)
        elif key == ord('f'):
            predictor.smoothing = not predictor.smoothing
        elif key == ord(' '): paused = not paused
        elif key == ord('e') or key == ord('E'):
            if predicted and not interceptor.engaged:
                interceptor.engage(predicted[0], predicted[1])
                print(f"[INFO] ENGAGED! Interceptor -> predicted ({predicted[0]},{predicted[1]})")
            else:
                print("[INFO] No prediction available or already engaged.")

    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
