import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

NOSE_PINCH_RATIO = 0.22

WAVE_WINDOW = 15
KNEE_WINDOW = 20

# Minimum number of direction reversals within the window to count as
# "waving" / "bobbing". Lower = easier to trigger, but more false positives.
WAVE_MIN_REVERSALS = 3
KNEE_MIN_REVERSALS = 2

# How many consecutive frames the full gesture must be held before
# the kitten clip plays.
HOLD_FRAMES = 6

# Minimum seconds between triggers, so it doesn't fire constantly.
COOLDOWN_SECONDS = 3.0

# Green screen color range (HSV) for chroma keying the kitten clip.
GREEN_LOWER = np.array([35, 60, 60])
GREEN_UPPER = np.array([85, 255, 255])

# Size of the kitten overlay, as a fraction of the webcam frame width.
OVERLAY_SCALE = 0.35

# Pose landmark indices (MediaPipe Pose)
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_WRIST, R_WRIST = 15, 16
L_KNEE, R_KNEE = 25, 26


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_reversals(values):
    """Count direction reversals in a sequence of numbers.

    Used to tell "swaying back and forth" apart from "moving steadily
    in one direction" or "not moving at all".
    """
    if len(values) < 3:
        return 0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h]), lm.visibility


def chroma_key_overlay(frame, overlay_bgr):
    """Remove green background from overlay_bgr and composite it onto the
    top-right corner of frame."""
    h, w = frame.shape[:2]
    ov_h, ov_w = overlay_bgr.shape[:2]

    target_w = int(w * OVERLAY_SCALE)
    target_h = max(1, int(ov_h * (target_w / ov_w)))
    overlay_resized = cv2.resize(overlay_bgr, (target_w, target_h))

    hsv = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask_inv = cv2.bitwise_not(mask)

    x0 = w - target_w
    roi = frame[0:target_h, x0:w]

    fg = cv2.bitwise_and(overlay_resized, overlay_resized, mask=mask_inv)
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    frame[0:target_h, x0:w] = cv2.add(fg, bg)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(kitten_path, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Could not open webcam at index {cam_index}")
        return

    kitten_cap = cv2.VideoCapture(kitten_path)
    if not kitten_cap.isOpened():
        print(f"Could not open kitten video: {kitten_path}")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    wave_hist = deque(maxlen=WAVE_WINDOW)
    knee_hist = deque(maxlen=KNEE_WINDOW)
    hold_counter = 0
    kitten_playing = False
    last_trigger_time = 0.0

    print("Scuba Dance Detector running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels more natural
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        gesture_detected = False

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            nose, _ = get_point(lms, NOSE, w, h)
            l_shoulder, _ = get_point(lms, L_SHOULDER, w, h)
            r_shoulder, _ = get_point(lms, R_SHOULDER, w, h)
            l_wrist, l_vis = get_point(lms, L_WRIST, w, h)
            r_wrist, r_vis = get_point(lms, R_WRIST, w, h)
            l_knee, lk_vis = get_point(lms, L_KNEE, w, h)
            r_knee, rk_vis = get_point(lms, R_KNEE, w, h)

            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            pinch_threshold = shoulder_width * NOSE_PINCH_RATIO if shoulder_width > 0 else 0

            l_dist_to_nose = np.linalg.norm(l_wrist - nose)
            r_dist_to_nose = np.linalg.norm(r_wrist - nose)

            nose_pinched = False
            waving_wrist = None

            if l_vis > 0.5 and pinch_threshold and l_dist_to_nose < pinch_threshold:
                nose_pinched = True
                if r_vis > 0.5:
                    waving_wrist = r_wrist
            elif r_vis > 0.5 and pinch_threshold and r_dist_to_nose < pinch_threshold:
                nose_pinched = True
                if l_vis > 0.5:
                    waving_wrist = l_wrist

            if waving_wrist is not None:
                wave_hist.append(waving_wrist[0])
            else:
                wave_hist.clear()

            if lk_vis > 0.5 and rk_vis > 0.5:
                knee_hist.append((l_knee[1] + r_knee[1]) / 2)
            else:
                knee_hist.clear()

            waving = count_reversals(list(wave_hist)) >= WAVE_MIN_REVERSALS
            bobbing = count_reversals(list(knee_hist)) >= KNEE_MIN_REVERSALS

            gesture_detected = nose_pinched and waving and bobbing

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Hold counter rises while gesture is active, decays otherwise
        hold_counter = hold_counter + 1 if gesture_detected else max(0, hold_counter - 1)

        now = time.time()
        if (
            hold_counter >= HOLD_FRAMES
            and not kitten_playing
            and (now - last_trigger_time) > COOLDOWN_SECONDS
        ):
            kitten_playing = True
            kitten_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_trigger_time = now

        if kitten_playing:
            ret_k, kitten_frame = kitten_cap.read()
            if not ret_k:
                kitten_playing = False
                hold_counter = 0
            else:
                frame = chroma_key_overlay(frame, kitten_frame)

        if gesture_detected:
            cv2.putText(frame, "SCUBA DETECTED!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Scuba Dance Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    kitten_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scuba dance gesture detector with kitten overlay")
    parser.add_argument("--kitten", required=True, help="Path to green-screen kitten scuba video")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args()
    main(args.kitten, args.camera)