"""
AATIS - Intelligent Moving Object Prediction System
====================================================

Uses OpenCV to detect, track, and predict the trajectory of moving objects
in real-time video. The system combines:

  1. Background Subtraction (MOG2) — isolates moving objects from the scene.
  2. Kalman Filter             — estimates the object's state (position + velocity)
                                  and produces a smooth, mathematically grounded
                                  prediction of where the object will be next.
  3. Trajectory Extrapolation  — extends the Kalman prediction further into the
                                  future to draw a projected path on screen.

Controls:
  - Press 'q' to quit.
  - Press 'r' to reset the tracker.
  - Press 's' to save a screenshot.
  - Press '+' / '-' to increase / decrease prediction horizon.
"""

import cv2
import numpy as np
import time
import math
import os
from collections import deque


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PREDICTION_STEPS = 15          # How many frames into the future to predict
MIN_CONTOUR_AREA = 800         # Ignore contours smaller than this (px²)
TRAIL_LENGTH = 60              # Number of past positions to keep for the trail
FPS_SMOOTHING = 0.9            # Exponential moving average factor for FPS display
KALMAN_PROCESS_NOISE = 1e-2    # Kalman process noise covariance scalar
KALMAN_MEASUREMENT_NOISE = 10  # Kalman measurement noise covariance scalar


# ──────────────────────────────────────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────────────────────────────────────
COLOR_CURRENT    = (0, 255, 120)   # Green — current detected position
COLOR_PREDICTED  = (0, 180, 255)   # Orange — predicted position
COLOR_TRAIL      = (255, 200, 50)  # Cyan-ish — past trajectory trail
COLOR_PRED_PATH  = (80, 80, 255)   # Red — predicted future path
COLOR_BBOX       = (255, 255, 0)   # Cyan — bounding box
COLOR_TEXT_BG    = (30, 30, 30)    # Dark background for text panels
COLOR_TEXT       = (220, 220, 220) # Light text
COLOR_ACCENT     = (0, 200, 255)   # Accent yellow-orange
COLOR_HEADER     = (0, 255, 255)   # Yellow header text


# ──────────────────────────────────────────────────────────────────────────────
# Kalman Filter Setup
# ──────────────────────────────────────────────────────────────────────────────
def create_kalman_filter():
    """
    Create a 4-state (x, y, vx, vy), 2-measurement (x, y) Kalman filter.

    State vector:       [x, y, vx, vy]^T
    Measurement vector: [x, y]^T

    Transition model assumes constant velocity:
        x(k+1) = x(k) + vx(k)*dt
        y(k+1) = y(k) + vy(k)*dt
        vx(k+1) = vx(k)
        vy(k+1) = vy(k)
    """
    kf = cv2.KalmanFilter(4, 2)

    # Transition matrix (constant-velocity model, dt = 1 frame)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement matrix — we observe x and y directly
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Process noise covariance
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE

    # Measurement noise covariance
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE

    # Error covariance post (initial uncertainty)
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


# ──────────────────────────────────────────────────────────────────────────────
# Multi-step Prediction
# ──────────────────────────────────────────────────────────────────────────────
def predict_future_path(kalman_filter, steps):
    """
    Given the current Kalman state, extrapolate *steps* frames into the future
    using the constant-velocity model.

    Returns a list of (x, y) integer tuples representing the predicted path.
    """
    state = kalman_filter.statePost.copy()          # [x, y, vx, vy]
    transition = kalman_filter.transitionMatrix      # F
    path = []
    for _ in range(steps):
        state = transition @ state                   # x(k+1) = F·x(k)
        px, py = int(state[0, 0]), int(state[1, 0])
        path.append((px, py))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Drawing Utilities
# ──────────────────────────────────────────────────────────────────────────────
def draw_trail(frame, trail, color, max_thickness=3):
    """Draw the historical trajectory as a fading polyline."""
    n = len(trail)
    for i in range(1, n):
        alpha = i / n  # 0 → oldest, 1 → newest
        thickness = max(1, int(alpha * max_thickness))
        # Fade colour toward dark
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)


def draw_predicted_path(frame, path, color):
    """Draw the predicted future trajectory as a dashed line with circles."""
    for i, pt in enumerate(path):
        alpha = 1.0 - (i / len(path)) * 0.6
        c = tuple(int(v * alpha) for v in color)
        radius = max(2, 6 - i // 3)
        cv2.circle(frame, pt, radius, c, -1, cv2.LINE_AA)
        if i > 0:
            # Dashed effect: draw every other segment
            if i % 2 == 0:
                cv2.line(frame, path[i - 1], pt, c, 2, cv2.LINE_AA)


def draw_crosshair(frame, center, size=20, color=(0, 255, 255), thickness=2):
    """Draw a targeting crosshair at the given center."""
    cx, cy = center
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, center, size // 2, color, 1, cv2.LINE_AA)


def draw_info_panel(frame, info_lines, origin=(10, 10), line_height=28,
                    font_scale=0.55, padding=10):
    """Draw a semi-transparent panel with text information."""
    h = len(info_lines) * line_height + padding * 2
    w = max(len(line) for line in info_lines) * 12 + padding * 2
    x0, y0 = origin

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), COLOR_ACCENT, 1, cv2.LINE_AA)

    # Text
    for i, line in enumerate(info_lines):
        color = COLOR_HEADER if i == 0 else COLOR_TEXT
        cv2.putText(frame, line,
                    (x0 + padding, y0 + padding + (i + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def draw_distance_line(frame, pt1, pt2, color):
    """Draw a line between two points and annotate the pixel distance."""
    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.putText(frame, f"{dist:.0f}px", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Video Source ──────────────────────────────────────────────────────
    # Use 0 for default webcam. Replace with a file path for video files.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video source opened: {frame_w}x{frame_h}")

    # ── Background Subtractor ─────────────────────────────────────────────
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    # ── Kalman Filter ─────────────────────────────────────────────────────
    kalman = create_kalman_filter()
    kalman_initialized = False

    # ── State ─────────────────────────────────────────────────────────────
    trail = deque(maxlen=TRAIL_LENGTH)
    prediction_steps = PREDICTION_STEPS
    fps_avg = 0.0
    frame_count = 0
    tracking_active = False
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")

    print("[INFO] Press 'q' to quit | 'r' to reset | 's' to screenshot")
    print("[INFO]       '+'/'-' to change prediction horizon")

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            # Loop video file; for webcam this means the device disconnected
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = frame.copy()
        frame_count += 1

        # ── 1. Background Subtraction ────────────────────────────────────
        fg_mask = bg_subtractor.apply(frame)

        # Remove shadows (shadow pixels have value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # ── 2. Find Contours ─────────────────────────────────────────────
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_center = None
        detected_bbox = None

        if contours:
            # Pick the largest moving region
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area >= MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2
                detected_center = (cx, cy)
                detected_bbox = (x, y, w, h)

        # ── 3. Kalman Update / Predict ───────────────────────────────────
        predicted_center = None
        future_path = []
        velocity = (0.0, 0.0)

        if detected_center is not None:
            measurement = np.array(
                [[np.float32(detected_center[0])],
                 [np.float32(detected_center[1])]]
            )

            if not kalman_initialized:
                # First detection — seed the Kalman state
                kalman.statePost = np.array(
                    [[np.float32(detected_center[0])],
                     [np.float32(detected_center[1])],
                     [0.0],
                     [0.0]], dtype=np.float32
                )
                kalman_initialized = True

            kalman.correct(measurement)
            prediction = kalman.predict()
            px, py = int(prediction[0, 0]), int(prediction[1, 0])
            predicted_center = (px, py)

            velocity = (float(kalman.statePost[2, 0]),
                        float(kalman.statePost[3, 0]))

            # Multi-step future path
            future_path = predict_future_path(kalman, prediction_steps)

            trail.append(detected_center)
            tracking_active = True

        elif kalman_initialized:
            # No detection — still predict based on last state
            prediction = kalman.predict()
            px, py = int(prediction[0, 0]), int(prediction[1, 0])
            predicted_center = (px, py)
            velocity = (float(kalman.statePost[2, 0]),
                        float(kalman.statePost[3, 0]))
            future_path = predict_future_path(kalman, prediction_steps)

        # ── 4. Drawing ───────────────────────────────────────────────────
        # Trail
        if len(trail) > 1:
            draw_trail(display, list(trail), COLOR_TRAIL)

        # Predicted future path
        if future_path:
            draw_predicted_path(display, future_path, COLOR_PRED_PATH)

        # Bounding box
        if detected_bbox is not None:
            x, y, w, h = detected_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          COLOR_BBOX, 2, cv2.LINE_AA)

        # Current position
        if detected_center is not None:
            cv2.circle(display, detected_center, 8, COLOR_CURRENT, -1, cv2.LINE_AA)
            cv2.circle(display, detected_center, 12, COLOR_CURRENT, 2, cv2.LINE_AA)

        # Predicted position
        if predicted_center is not None:
            draw_crosshair(display, predicted_center, size=22, color=COLOR_PREDICTED)
            cv2.circle(display, predicted_center, 6, COLOR_PREDICTED, -1, cv2.LINE_AA)

        # Distance line between current and predicted
        if detected_center is not None and predicted_center is not None:
            draw_distance_line(display, detected_center, predicted_center, COLOR_ACCENT)

        # Final predicted position marker
        if future_path:
            final = future_path[-1]
            cv2.circle(display, final, 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "PREDICTED", (final[0] + 18, final[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # ── 5. FPS Calculation ───────────────────────────────────────────
        dt = time.perf_counter() - t_start
        fps_now = 1.0 / max(dt, 1e-9)
        fps_avg = FPS_SMOOTHING * fps_avg + (1 - FPS_SMOOTHING) * fps_now

        # ── 6. Info Panel ────────────────────────────────────────────────
        speed = math.hypot(*velocity)
        angle_deg = math.degrees(math.atan2(-velocity[1], velocity[0])) % 360

        info = [
            "AATIS Object Predictor",
            f"FPS: {fps_avg:.1f}",
            f"Frame: {frame_count}",
            f"Status: {'TRACKING' if tracking_active and detected_center else 'SEARCHING'}",
            f"Prediction Horizon: {prediction_steps} frames",
        ]
        if detected_center:
            info.append(f"Detected (x,y): ({detected_center[0]}, {detected_center[1]})")
        if predicted_center:
            info.append(f"Predicted (x,y): ({predicted_center[0]}, {predicted_center[1]})")
        if speed > 0.1:
            info.append(f"Speed: {speed:.1f} px/frame")
            info.append(f"Direction: {angle_deg:.0f} deg")
        if future_path:
            fp = future_path[-1]
            info.append(f"Future ({prediction_steps}f): ({fp[0]}, {fp[1]})")

        draw_info_panel(display, info)

        # ── 7. Show Windows ──────────────────────────────────────────────
        cv2.imshow("AATIS - Object Prediction", display)

        # Show the foreground mask in a smaller window
        mask_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mask_display = cv2.resize(mask_display, (frame_w // 3, frame_h // 3))
        cv2.imshow("Foreground Mask", mask_display)

        # ── 8. Key Handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracker
            kalman = create_kalman_filter()
            kalman_initialized = False
            trail.clear()
            tracking_active = False
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
            print("[INFO] Tracker reset.")
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            fname = os.path.join(screenshot_dir,
                                 f"prediction_{frame_count}.png")
            cv2.imwrite(fname, display)
            print(f"[INFO] Screenshot saved: {fname}")
        elif key == ord('+') or key == ord('='):
            prediction_steps = min(prediction_steps + 5, 100)
            print(f"[INFO] Prediction horizon: {prediction_steps} frames")
        elif key == ord('-') or key == ord('_'):
            prediction_steps = max(prediction_steps - 5, 1)
            print(f"[INFO] Prediction horizon: {prediction_steps} frames")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
