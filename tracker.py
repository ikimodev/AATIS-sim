"""
AATIS - 2D Moving Object Tracking & Prediction System
======================================================

A drone-style tracking system that detects a moving object, tracks its
position over time, and predicts its future trajectory using velocity
calculations — simulating how a drone would anticipate a target's
movement rather than merely reacting to its current position.

Three Stages:
  1. DETECTION  — Detect moving objects via background subtraction + contour
                   detection. Extract center position (x, y).
  2. TRACKING   — Store the object's position history over multiple frames.
                   Continuously update current position.
  3. PREDICTION — Estimate velocity from consecutive frames:
                     vx = x(t) - x(t-1)
                     vy = y(t) - y(t-1)
                   Predict future position:
                     x_pred = x + vx * dt
                     y_pred = y + vy * dt
                   Optional exponential smoothing for stability.

Controls:
  q       — Quit
  r       — Reset tracker
  s       — Save screenshot
  +/-     — Increase / decrease prediction time (dt)
  f       — Toggle smoothing filter on/off
"""

import cv2
import numpy as np
import math
import time
import os
from collections import deque
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SIMULATION_MODE = True
FRAME_W = 800
FRAME_H = 600

MIN_CONTOUR_AREA = 900          # Ignore contours smaller than this (px²)
HISTORY_LENGTH   = 80           # Number of past positions to remember
PREDICTION_DT    = 12           # How many frames ahead to predict
SMOOTHING_ALPHA  = 0.4          # Exponential smoothing factor (0 = full smooth, 1 = raw)
ENABLE_SMOOTHING = True         # Start with smoothing enabled

FRAMES_TO_TRACK  = 5            # Number of frames before considered TRACKED
FRAMES_TO_LOSE   = 15           # Number of frames without detection before considered LOST
YOLO_MODEL_PATH  = "yolov8n.pt" # Path to YOLO model

# Colors (BGR)
CLR_DETECTED     = (0, 255, 100)    # Green  — detected position
CLR_PREDICTED    = (0, 140, 255)    # Orange — predicted position
CLR_TRAIL        = (255, 220, 60)   # Cyan   — position history trail
CLR_PRED_LINE    = (80, 80, 255)    # Red    — prediction line / path
CLR_BBOX         = (255, 255, 0)    # Yellow — bounding box
CLR_VELOCITY     = (180, 255, 0)    # Lime   — velocity vector arrow
CLR_PANEL_BG     = (25, 25, 30)     # Dark   — info panel background
CLR_TEXT          = (210, 210, 210)  # Light  — info text
CLR_HEADER        = (0, 245, 255)   # Gold   — header text
CLR_ACCENT       = (0, 200, 255)    # Amber  — accent / highlights
CLR_FUTURE_RING  = (0, 0, 255)     # Red    — final predicted position ring

CLR_STATE_DET    = (0, 200, 255)   # Yellowish
CLR_STATE_TRK    = (0, 255, 100)   # Green
CLR_STATE_LST    = (100, 100, 100) # Gray


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATED TARGET
# ═══════════════════════════════════════════════════════════════════════════════
class SimTarget:
    def __init__(self):
        self.r = 15
        self.x = 50.0
        self.y = FRAME_H / 2.0
        self.vx = 8.0
        self.vy = 5.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Add random variations
        if np.random.random() < 0.05:
            self.vx += np.random.uniform(-1, 1)
            self.vy += np.random.uniform(-1, 1)
            
        speed = math.hypot(self.vx, self.vy)
        if speed > 12:
            self.vx = (self.vx / speed) * 12
            self.vy = (self.vy / speed) * 12
            
        # Bounce off walls
        if self.x - self.r < 0 or self.x + self.r > FRAME_W:
            self.vx *= -1
            self.x = np.clip(self.x, self.r, FRAME_W - self.r)
        if self.y - self.r < 0 or self.y + self.r > FRAME_H:
            self.vy *= -1
            self.y = np.clip(self.y, self.r, FRAME_H - self.r)

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, (255, 255, 255), -1)

def create_bg():
    bg = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for y in range(FRAME_H):
        v = int(25 + 15 * (y / FRAME_H))
        bg[y, :] = (v, v+2, v+4)
    return bg


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
class Detector:
    """
    Detects moving objects using MOG2 background subtraction + contour analysis.
    Returns the center (x, y) and bounding box of the largest moving region.
    """

    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def reset(self):
        """Reinitialize the background model."""
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def detect(self, frame):
        """
        Process a frame and return (center, bbox, mask).

        center : (x, y) tuple or None
        bbox   : (x, y, w, h) tuple or None
        mask   : binary foreground mask (for display)
        """
        # Apply background subtraction
        fg_mask = self.bg_sub.apply(frame)

        # Remove shadow pixels (MOG2 marks shadows as 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup to remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None, fg_mask

        # Select the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_CONTOUR_AREA:
            return None, None, fg_mask

        # Bounding box and center
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2

        return (cx, cy), (x, y, w, h), fg_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: TRACKING & IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
class Tracker:
    """
    Tracking & Identification Layer
    Stores the object's position over time, maintains identity, state, and classification.
    """

    def __init__(self, max_history=HISTORY_LENGTH):
        self.max_history = max_history
        self.object_id = 1
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.max_history)
        self.timestamps = deque(maxlen=self.max_history)
        self.bbox_history = deque(maxlen=self.max_history)
        self.velocity_history = deque(maxlen=self.max_history)
        self.state = "DETECTED"
        self.frames_since_seen = 0
        self.frames_seen = 0
        self.classification = "Unknown"
        self.yolo_class = None

    def update(self, position, bbox, frame_num, velocity=None, yolo_class=None):
        if position is not None:
            if self.state == "LOST" and self.frames_since_seen > FRAMES_TO_LOSE:
                old_id = self.object_id
                self.reset()
                self.object_id = old_id + 1
                
            self.history.append(position)
            self.timestamps.append(frame_num)
            if bbox is not None:
                self.bbox_history.append(bbox)
            if velocity is not None:
                self.velocity_history.append(velocity)
            if yolo_class is not None:
                self.yolo_class = yolo_class

            self.frames_since_seen = 0
            self.frames_seen += 1

            if self.frames_seen < FRAMES_TO_TRACK:
                self.state = "DETECTED"
            else:
                self.state = "TRACKED"
                
            self._update_classification()
        else:
            if self.frames_seen > 0:
                self.frames_since_seen += 1
                self.state = "LOST"

    def _update_classification(self):
        if self.state == "DETECTED" or len(self.velocity_history) < 5:
            self.classification = "Unknown"
            if self.yolo_class == "bird":
                self.classification = "Bird"
            return
            
        speeds = [math.hypot(vx, vy) for vx, vy in self.velocity_history]
        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds))
        
        areas = [w*h for _, _, w, h in self.bbox_history]
        mean_area = float(np.mean(areas))
        std_area = float(np.std(areas))
        
        cv_speed = std_speed / (mean_speed + 1e-5)
        cv_area = std_area / (mean_area + 1e-5)
        
        if self.yolo_class == "bird":
            self.classification = "Bird"
        elif self.yolo_class == "airplane" or self.yolo_class == "drone":
            self.classification = "Drone"
        else:
            if cv_area < 0.15 and cv_speed < 0.3 and mean_speed > 2.0:
                self.classification = "Drone"
            elif cv_area > 0.2 or cv_speed > 0.4:
                self.classification = "Bird"
            else:
                self.classification = "Unknown"

    @property
    def current(self):
        return self.history[-1] if self.history else None

    @property
    def previous(self):
        return self.history[-2] if len(self.history) >= 2 else None

    @property
    def has_enough_data(self):
        return len(self.history) >= 2

    def get_trail(self):
        return list(self.history)


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
class Predictor:
    """
    Estimates velocity from consecutive positions and predicts the future
    position of the moving object.
    """

    def __init__(self, smoothing_alpha=SMOOTHING_ALPHA):
        self.alpha = smoothing_alpha        # Smoothing factor
        self.vx_smooth = 0.0                # Smoothed velocity x
        self.vy_smooth = 0.0                # Smoothed velocity y
        self.smoothing_enabled = ENABLE_SMOOTHING

    def reset(self):
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0

    def compute_velocity(self, current, previous):
        """
        Compute raw velocity from two consecutive positions.
        """
        vx_raw = current[0] - previous[0]
        vy_raw = current[1] - previous[1]

        if self.smoothing_enabled:
            self.vx_smooth = self.alpha * vx_raw + (1 - self.alpha) * self.vx_smooth
            self.vy_smooth = self.alpha * vy_raw + (1 - self.alpha) * self.vy_smooth
            return self.vx_smooth, self.vy_smooth
        else:
            return float(vx_raw), float(vy_raw)

    def predict(self, current, vx, vy, dt):
        x_pred = current[0] + vx * dt
        y_pred = current[1] + vy * dt
        return int(round(x_pred)), int(round(y_pred))

    def predict_path(self, current, vx, vy, dt, steps=10):
        path = []
        for i in range(1, steps + 1):
            t = dt * (i / steps)
            px = int(round(current[0] + vx * t))
            py = int(round(current[1] + vy * t))
            path.append((px, py))
        return path


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def draw_trail(frame, trail, color=CLR_TRAIL, max_thickness=3):
    n = len(trail)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(alpha * max_thickness))
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)

def draw_prediction_line(frame, path, color=CLR_PRED_LINE):
    for i, pt in enumerate(path):
        fade = 1.0 - (i / len(path)) * 0.5
        c = tuple(int(v * fade) for v in color)
        radius = max(2, 5 - i // 3)
        cv2.circle(frame, pt, radius, c, -1, cv2.LINE_AA)
        if i > 0 and i % 2 == 0:
            cv2.line(frame, path[i - 1], pt, c, 2, cv2.LINE_AA)

def draw_crosshair(frame, center, size=20, color=CLR_PREDICTED, thickness=2):
    cx, cy = center
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, center, size // 2, color, 1, cv2.LINE_AA)

def draw_velocity_arrow(frame, origin, vx, vy, scale=5.0, color=CLR_VELOCITY):
    end = (int(origin[0] + vx * scale), int(origin[1] + vy * scale))
    cv2.arrowedLine(frame, origin, end, color, 2, cv2.LINE_AA, tipLength=0.3)

def draw_info_panel(frame, lines, origin=(10, 10), line_h=26, font_scale=0.50, pad=10):
    h = len(lines) * line_h + pad * 2
    w = max(len(l) for l in lines) * 11 + pad * 2
    x0, y0 = origin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), CLR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), CLR_ACCENT, 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        color = CLR_HEADER if i == 0 else CLR_TEXT
        cv2.putText(frame, line,
                    (x0 + pad, y0 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    frame_w = FRAME_W
    frame_h = FRAME_H
    print(f"[INFO] Running in SIMULATION MODE: {frame_w}x{frame_h}")
    print("[INFO] Controls:")
    print("         q = quit | r = reset | s = screenshot")
    
    target = SimTarget()
    bg = create_bg()
    print("         +/- = prediction horizon | f = toggle smoothing")

    detector  = Detector()       
    tracker   = Tracker()        
    predictor = Predictor()      
    
    print("[INFO] Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("[INFO] YOLO loaded.")

    prediction_dt = PREDICTION_DT
    frame_count = 0
    fps_avg = 0.0
    screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

    while True:
        t0 = time.perf_counter()

        # Simulate Frame
        target.update()
        frame = bg.copy()
        target.draw(frame)

        frame_count += 1
        display = frame.copy()

        # ══════════════════════════════════════════════════════════════════
        #  STAGE 1 — DETECTION & YOLO
        # ══════════════════════════════════════════════════════════════════
        center, bbox, fg_mask = detector.detect(frame)
        
        yolo_class = None
        if center is not None and bbox is not None:
            # Run YOLO on the bounding box to classify
            x, y, w, h = bbox
            pad = 20
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame_w, x+w+pad), min(frame_h, y+h+pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                results = yolo_model(crop, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    cls_id = int(results[0].boxes[0].cls[0])
                    yolo_class = yolo_model.names[cls_id]

        # ══════════════════════════════════════════════════════════════════
        #  STAGE 2 & 3 — PREDICTION & TRACKING
        # ══════════════════════════════════════════════════════════════════
        predicted = None
        pred_path = []
        vx, vy = 0.0, 0.0
        speed = 0.0

        if tracker.has_enough_data and center is not None:
            prev = tracker.current
            vx, vy = predictor.compute_velocity(center, prev)
            speed = math.hypot(vx, vy)
            predicted = predictor.predict(center, vx, vy, prediction_dt)
            pred_path = predictor.predict_path(center, vx, vy, prediction_dt, steps=prediction_dt)

        tracker.update(center, bbox, frame_count, velocity=(vx, vy) if center is not None else None, yolo_class=yolo_class)
        
        # Extrapolate if LOST
        if center is None and tracker.state == "LOST" and tracker.has_enough_data:
            last_pos = tracker.current
            if len(tracker.velocity_history) > 0:
                vx, vy = tracker.velocity_history[-1]
                speed = math.hypot(vx, vy)
                dt_lost = tracker.frames_since_seen
                extrapolated_pos = (int(last_pos[0] + vx * dt_lost), int(last_pos[1] + vy * dt_lost))
                predicted = predictor.predict(extrapolated_pos, vx, vy, prediction_dt)
                pred_path = predictor.predict_path(extrapolated_pos, vx, vy, prediction_dt, steps=prediction_dt)

        # ══════════════════════════════════════════════════════════════════
        #  DRAWING
        # ══════════════════════════════════════════════════════════════════

        trail = tracker.get_trail()
        if len(trail) > 1:
            draw_trail(display, trail)

        if pred_path:
            draw_prediction_line(display, pred_path)

        state_color = CLR_STATE_DET
        if tracker.state == "TRACKED": state_color = CLR_STATE_TRK
        elif tracker.state == "LOST": state_color = CLR_STATE_LST
        
        if bbox is not None and center is not None:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), state_color, 2, cv2.LINE_AA)
            label = f"ID:{tracker.object_id} | {tracker.classification} | {tracker.state}"
            cv2.putText(display, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)

        if center is not None:
            cv2.circle(display, center, 7, state_color, -1, cv2.LINE_AA)
            cv2.circle(display, center, 11, state_color, 2, cv2.LINE_AA)

        if center is None and tracker.state == "LOST" and tracker.has_enough_data:
            last_pos = tracker.current
            vx, vy = tracker.velocity_history[-1] if len(tracker.velocity_history) > 0 else (0, 0)
            dt_lost = tracker.frames_since_seen
            extrapolated_pos = (int(last_pos[0] + vx * dt_lost), int(last_pos[1] + vy * dt_lost))
            cv2.circle(display, extrapolated_pos, 7, state_color, 1, cv2.LINE_AA)
            cv2.putText(display, f"ID:{tracker.object_id} LOST", (extrapolated_pos[0]-20, extrapolated_pos[1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)

        if center is not None and speed > 0.5:
            draw_velocity_arrow(display, center, vx, vy, scale=6.0)

        if predicted is not None:
            draw_crosshair(display, predicted, size=22)
            cv2.circle(display, predicted, 5, CLR_PREDICTED, -1, cv2.LINE_AA)
            cv2.circle(display, predicted, 16, CLR_FUTURE_RING, 2, cv2.LINE_AA)
            cv2.putText(display, f"PREDICTED ({predicted[0]}, {predicted[1]})",
                        (predicted[0] + 20, predicted[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_FUTURE_RING, 2, cv2.LINE_AA)

        if center is not None and predicted is not None:
            cv2.line(display, center, predicted, CLR_ACCENT, 1, cv2.LINE_AA)
            dist = math.hypot(predicted[0] - center[0], predicted[1] - center[1])
            mid = ((center[0] + predicted[0]) // 2, (center[1] + predicted[1]) // 2)
            cv2.putText(display, f"{dist:.0f}px", (mid[0] + 5, mid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_ACCENT, 1, cv2.LINE_AA)

        dt_frame = time.perf_counter() - t0
        fps_now = 1.0 / max(dt_frame, 1e-9)
        fps_avg = 0.9 * fps_avg + 0.1 * fps_now

        direction = math.degrees(math.atan2(-vy, vx)) % 360 if speed > 0.3 else 0
        info = [
            "AATIS Tracking & Identification",
            f"FPS: {fps_avg:.1f}  |  Frame: {frame_count}",
            f"State: {tracker.state}",
            f"Class: {tracker.classification} " + (f"(YOLO: {tracker.yolo_class})" if tracker.yolo_class else ""),
            f"Object ID: {tracker.object_id}",
            f"Smoothing: {'ON' if predictor.smoothing_enabled else 'OFF'}",
            f"Prediction dt: {prediction_dt} frames",
        ]
        if center:
            info.append(f"Current  (x, y): ({center[0]}, {center[1]})")
        if predicted:
            info.append(f"Predicted(x, y): ({predicted[0]}, {predicted[1]})")
        if speed > 0.3:
            info.append(f"Velocity: vx={vx:.1f}  vy={vy:.1f}")
            info.append(f"Speed: {speed:.1f} px/f  Dir: {direction:.0f} deg")

        draw_info_panel(display, info)

        if center and predicted and speed > 0.3:
            formula_lines = [
                f"vx = {tracker.current[0]} - {tracker.previous[0]} => {vx:.1f}",
                f"vy = {tracker.current[1]} - {tracker.previous[1]} => {vy:.1f}",
                f"x_pred = {center[0]} + {vx:.1f} * {prediction_dt} = {predicted[0]}",
                f"y_pred = {center[1]} + {vy:.1f} * {prediction_dt} = {predicted[1]}",
            ]
            draw_info_panel(display, ["Math:"] + formula_lines,
                            origin=(frame_w - 380, frame_h - 170))

        cv2.imshow("AATIS - Drone Target Prediction", display)

        mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_color, (frame_w // 3, frame_h // 3))
        cv2.imshow("Foreground Mask", mask_small)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            tracker.reset()
            predictor.reset()
            print("[INFO] Tracker reset.")
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            fname = os.path.join(screenshot_dir, f"frame_{frame_count}.png")
            cv2.imwrite(fname, display)
            print(f"[INFO] Screenshot saved: {fname}")
        elif key in (ord('+'), ord('=')):
            prediction_dt = min(prediction_dt + 2, 60)
            print(f"[INFO] Prediction dt = {prediction_dt} frames")
        elif key in (ord('-'), ord('_')):
            prediction_dt = max(prediction_dt - 2, 1)
            print(f"[INFO] Prediction dt = {prediction_dt} frames")
        elif key == ord('f'):
            predictor.smoothing_enabled = not predictor.smoothing_enabled
            status = "ON" if predictor.smoothing_enabled else "OFF"
            print(f"[INFO] Smoothing: {status}")

    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
