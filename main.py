
import os
import csv
import time
import datetime
import cv2
import numpy as np
import mediapipe as mp
import pygame


# =========================================================
# CONFIGURATION
# =========================================================
CAMERA_INDEX = 0
ALARM_FILE = "beep.mp3"

EAR_THRESHOLD = 0.23
EYE_CLOSED_SECONDS = 5.0

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 1.5

GESTURE_CONFIRM_FRAMES = 5

LOG_DIR = "logs"
CSV_FILE = os.path.join(LOG_DIR, "driver_events.csv")

os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================================================
# CSV EVENT LOGGING
# =========================================================
CSV_FIELDS = [
    "timestamp",
    "session_elapsed_seconds",
    "event",
    "details",
    "ear",
    "mar",
    "head_yaw",
    "head_pitch"
]

if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()


session_start = time.time()


def log_event(event, details="", ear=None, mar=None,
              yaw=None, pitch=None):
    """Append one event to the session CSV."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        "session_elapsed_seconds": round(
            time.time() - session_start, 2
        ),
        "event": event,
        "details": details,
        "ear": round(ear, 3) if ear is not None else "",
        "mar": round(mar, 3) if mar is not None else "",
        "head_yaw": round(yaw, 2) if yaw is not None else "",
        "head_pitch": round(pitch, 2) if pitch is not None else ""
    }

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# =========================================================
# AUDIO
# =========================================================
pygame.mixer.init()

try:
    pygame.mixer.music.load(ALARM_FILE)
    pygame.mixer.music.set_volume(0.7)
    alarm_available = True
except pygame.error as error:
    print("Warning: Alarm audio unavailable:", error)
    alarm_available = False


def start_alarm():
    if alarm_available and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)


def stop_alarm():
    if alarm_available:
        pygame.mixer.music.stop()


# =========================================================
# FACE LANDMARKS
# =========================================================
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth corners and upper/lower inner lips.
MOUTH_VERTICAL = (13, 14)
MOUTH_HORIZONTAL = (78, 308)

# 2D face points for head-pose estimation.
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Approximate generic 3D face model points.
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)


def landmark_xy(face, index, width, height):
    point = face.landmark[index]
    return np.array(
        [point.x * width, point.y * height],
        dtype=np.float64
    )


def calculate_ear(points):
    p = np.asarray(points, dtype=np.float64)

    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(face, width, height):
    upper = landmark_xy(face, MOUTH_VERTICAL[0], width, height)
    lower = landmark_xy(face, MOUTH_VERTICAL[1], width, height)

    left = landmark_xy(face, MOUTH_HORIZONTAL[0], width, height)
    right = landmark_xy(face, MOUTH_HORIZONTAL[1], width, height)

    mouth_width = np.linalg.norm(left - right)

    if mouth_width < 1e-6:
        return 0.0

    return np.linalg.norm(upper - lower) / mouth_width


def estimate_head_pose(face, width, height):
    image_points = np.array([
        landmark_xy(face, index, width, height)
        for index in POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # OpenCV returns angles in degrees. Pitch and yaw are
    # approximate because the 3D face model is generic.
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


# =========================================================
# HAND GESTURE
# =========================================================
def is_thumbs_up(hand):
    lm = hand.landmark

    # Thumb tip above the thumb joints.
    thumb_up = lm[4].y < lm[3].y < lm[2].y

    # Four fingers folded.
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y

    # Require the thumb to be separated from the index finger.
    thumb_separated = abs(lm[4].x - lm[8].x) > 0.06

    return (
        thumb_up and index_folded and middle_folded
        and ring_folded and pinky_folded and thumb_separated
    )


# =========================================================
# SESSION STATISTICS
# =========================================================
stats = {
    "drowsiness_alerts": 0,
    "yawns": 0,
    "dismissals": 0
}

def record_event(event, details="", ear=None, mar=None,
                 yaw=None, pitch=None):
    log_event(event, details, ear, mar, yaw, pitch)


# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    face_mesh.close()
    hands.close()
    pygame.mixer.quit()
    raise RuntimeError(
        "Cannot open webcam. Check CAMERA_INDEX and camera permissions."
    )


# =========================================================
# DETECTION STATE
# =========================================================
eye_closed_start = None
yawn_start = None

drowsy = False
alert_reason = ""

gesture_frames = 0
face_missing_frames = 0

last_yawn_logged = 0.0

print("Driver monitoring started. Press Q to quit.")
record_event("SESSION_START", "Monitoring started")


# =========================================================
# MAIN LOOP
# =========================================================
try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Webcam frame could not be read.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        now = time.time()

        ear = None
        mar = None
        yaw = None
        pitch = None

        # ---------------- HANDS ----------------
        thumbs_up = False

        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                if is_thumbs_up(hand):
                    thumbs_up = True

        gesture_frames = gesture_frames + 1 if thumbs_up else 0

        # ---------------- FACE ----------------
        if face_results.multi_face_landmarks:
            face_missing_frames = 0
            face = face_results.multi_face_landmarks[0]

            right_eye = [
                landmark_xy(face, i, width, height)
                for i in RIGHT_EYE
            ]

            left_eye = [
                landmark_xy(face, i, width, height)
                for i in LEFT_EYE
            ]

            draw_right = np.array(right_eye, dtype=np.int32)
            draw_left = np.array(left_eye, dtype=np.int32)

            cv2.polylines(
                frame, [draw_right.reshape((-1, 1, 2))],
                True, (0, 255, 0), 1
            )
            cv2.polylines(
                frame, [draw_left.reshape((-1, 1, 2))],
                True, (0, 255, 0), 1
            )

            ear = (
                calculate_ear(right_eye) +
                calculate_ear(left_eye)
            ) / 2.0

            mar = calculate_mar(face, width, height)
            yaw, pitch, roll = estimate_head_pose(
                face, width, height
            )

            # ---------------- EYE CLOSURE ----------------
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = now
            else:
                eye_closed_start = None

            eyes_closed_too_long = (
                eye_closed_start is not None
                and now - eye_closed_start >= EYE_CLOSED_SECONDS
            )

            # ---------------- YAWNING ----------------
            if mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = now
            else:
                yawn_start = None

            yawn_detected = (
                yawn_start is not None
                and now - yawn_start >= YAWN_SECONDS
            )

            if yawn_detected and now - last_yawn_logged > 5:
                stats["yawns"] += 1
                last_yawn_logged = now

                record_event(
                    "YAWN_DETECTED",
                    f"Mouth aspect ratio={mar:.3f}",
                    ear, mar, yaw, pitch
                )

            # ---------------- TRIGGER ALERT ----------------
            reason = None

            if eyes_closed_too_long:
                reason = "PROLONGED EYE CLOSURE"
            elif yawn_detected:
                reason = "PROLONGED YAWNING"

            if not drowsy and reason is not None:
                drowsy = True
                alert_reason = reason

                if eyes_closed_too_long:
                    stats["drowsiness_alerts"] += 1

                record_event(
                    "ALERT_TRIGGERED",
                    alert_reason,
                    ear, mar, yaw, pitch
                )

        else:
            face_missing_frames += 1

            if face_missing_frames >= 10:
                eye_closed_start = None
                yawn_start = None

        # ---------------- DISMISS ALARM ----------------
        if drowsy and gesture_frames >= GESTURE_CONFIRM_FRAMES:
            drowsy = False

            record_event(
                "ALERT_DISMISSED",
                "Thumbs-up gesture recognized",
                ear, mar, yaw, pitch
            )

            stats["dismissals"] += 1
            alert_reason = ""

            eye_closed_start = None
            yawn_start = None
            gesture_frames = 0

        if drowsy:
            start_alarm()
        else:
            stop_alarm()

        # =================================================
        # DASHBOARD OVERLAY
        # =================================================
        elapsed = int(now - session_start)
        minutes, seconds = divmod(elapsed, 60)

        if drowsy:
            status = "ALERT: " + alert_reason
            status_color = (0, 0, 255)
        elif face_results.multi_face_landmarks:
            status = "MONITORING"
            status_color = (0, 200, 0)
        else:
            status = "NO FACE DETECTED"
            status_color = (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX

        def draw_overlay_text(text, origin, color, scale=0.5, thickness=1):
            cv2.putText(
                frame, text, origin, font, scale,
                (255, 255, 255), thickness + 3, cv2.LINE_AA
            )
            cv2.putText(
                frame, text, origin, font, scale,
                color, thickness, cv2.LINE_AA
            )

        left_stats = [
            (f"Session: {minutes:02d}:{seconds:02d}", 26),
            (f"EAR: {ear:.2f}" if ear is not None else "EAR: N/A", 50),
            (f"MAR: {mar:.2f}" if mar is not None else "MAR: N/A", 74)
        ]
        right_stats = [
            (f"Alerts: {stats['drowsiness_alerts']}", 26),
            (f"Yawns: {stats['yawns']}", 50),
            (f"Dismissals: {stats['dismissals']}", 74)
        ]

        for text, y_position in left_stats:
            draw_overlay_text(text, (12, y_position), (0, 0, 0))

        for text, y_position in right_stats:
            text_width = cv2.getTextSize(text, font, 0.5, 1)[0][0]
            draw_overlay_text(
                text, (width - text_width - 12, y_position),
                (0, 0, 0)
            )

        status_width = cv2.getTextSize(status, font, 0.65, 2)[0][0]
        draw_overlay_text(
            status, ((width - status_width) // 2, 105),
            status_color, scale=0.65, thickness=2
        )

        gesture_hint = "THUMBS UP: dismiss alert | Q: quit"
        hint_width = cv2.getTextSize(gesture_hint, font, 0.5, 1)[0][0]
        draw_overlay_text(
            gesture_hint,
            ((width - hint_width) // 2, height - 16),
            (0, 0, 0)
        )

        display_scale = min(1.5, 1280 / width, 900 / height)
        display_frame = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("AI Driver Monitoring Dashboard", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    record_event("SESSION_END", "Monitoring stopped")

    cap.release()
    face_mesh.close()
    hands.close()
    stop_alarm()
    pygame.mixer.quit()
    cv2.destroyAllWindows()

    print("\nSession summary")
    print("---------------------------")
    print("Duration (seconds):", round(time.time() - session_start, 1))
    print("Drowsiness alerts:", stats["drowsiness_alerts"])
    print("Yawns:", stats["yawns"])
    print("Alarm dismissals:", stats["dismissals"])
    print("CSV log:", os.path.abspath(CSV_FILE))
