import cv2
import time
import random
import datetime
import math
import numpy as np
import mediapipe as mp
import pygame


# -------------------- CONFIGURATION --------------------
EAR_THRESHOLD = 0.23
DROWSINESS_SECONDS = 5
CAMERA_INDEX = 0

ALARM_FILE = "beep.mp3"
WINDOW_NAME = "Driver Drowsiness Detection"

# MediaPipe eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


# -------------------- HELPER FUNCTIONS --------------------
def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio from six eye landmarks."""
    p = np.asarray(eye_points, dtype=np.float64)

    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_eye_points(face_landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = face_landmarks.landmark[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    return points


def draw_eye(img, points):
    points = np.asarray(points, dtype=np.int32)

    cv2.polylines(
        img,
        [points.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 0),
        thickness=1
    )


# -------------------- INITIALIZATION --------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init()

try:
    pygame.mixer.music.load(ALARM_FILE)
    pygame.mixer.music.set_volume(0.7)
    alarm_available = True
except pygame.error as error:
    print(f"Could not load {ALARM_FILE}: {error}")
    print("Place the MP3 file beside main.py.")
    alarm_available = False

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    face_mesh.close()
    pygame.mixer.quit()
    raise SystemExit(1)


# -------------------- STATE VARIABLES --------------------
closed_start_time = None
drowsy = False
face_missing_frames = 0

number_a = 0
number_b = 0
answer_text = ""
wrong_message = ""
challenge_feedback_until = 0.0


def create_challenge():
    global number_a, number_b, answer_text, wrong_message
    number_a = random.randint(10, 30)
    number_b = random.randint(10, 30)
    answer_text = ""
    wrong_message = ""


def stop_alarm():
    if alarm_available:
        pygame.mixer.music.stop()


# -------------------- MAIN LOOP --------------------
try:
    while True:
        success, img = cap.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # Mirror the webcam for a natural preview.
        img = cv2.flip(img, 1)
        height, width = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_time = time.time()
        ear = None

        if results.multi_face_landmarks:
            face_missing_frames = 0
            face = results.multi_face_landmarks[0]

            right_points = get_eye_points(
                face, RIGHT_EYE, width, height
            )
            left_points = get_eye_points(
                face, LEFT_EYE, width, height
            )

            draw_eye(img, right_points)
            draw_eye(img, left_points)

            right_ear = calculate_ear(right_points)
            left_ear = calculate_ear(left_points)
            ear = (right_ear + left_ear) / 2.0

            # Start or reset the eye-closure timer.
            if ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = current_time
            else:
                closed_start_time = None

            # Trigger the alert after sustained eye closure.
            if (
                not drowsy
                and closed_start_time is not None
                and current_time - closed_start_time
                >= DROWSINESS_SECONDS
            ):
                drowsy = True
                create_challenge()

        else:
            face_missing_frames += 1

            # Do not treat a missing face as proof of closed eyes.
            if face_missing_frames >= 10:
                closed_start_time = None

        # Calculate eye-closure duration.
        if closed_start_time is not None:
            closed_duration = current_time - closed_start_time
        else:
            closed_duration = 0.0

        # -------------------- STATUS DISPLAY --------------------
        if drowsy:
            cv2.putText(
                img, "DROWSINESS ALERT!",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 3
            )

            # Keep the warning sound playing until the challenge
            # is answered correctly.
            if alarm_available and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)

            cv2.putText(
                img, f"Solve: {number_a} + {number_b} = ?",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2
            )

            cv2.putText(
                img, f"Answer: {answer_text}_",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2
            )

            cv2.putText(
                img, "Type answer and press ENTER",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            if wrong_message and current_time < challenge_feedback_until:
                cv2.putText(
                    img, wrong_message,
                    (10, 185), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2
                )

        else:
            cv2.putText(
                img, "ACTIVE",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )

            if ear is not None:
                cv2.putText(
                    img, f"EAR: {ear:.2f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                if closed_start_time is not None:
                    cv2.putText(
                        img,
                        f"Eyes closed: {closed_duration:.1f}s",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 165, 255), 2
                    )
            else:
                cv2.putText(
                    img, "No face detected",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2
                )

        # Display date and time.
        date_text = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cv2.putText(
            img, date_text,
            (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        cv2.imshow(WINDOW_NAME, img)

        # -------------------- KEYBOARD INPUT --------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if drowsy:
            # Accept number keys while the challenge is active.
            if ord("0") <= key <= ord("9"):
                if len(answer_text) < 3:
                    answer_text += chr(key)

            elif key in (8, 127):  # Backspace
                answer_text = answer_text[:-1]

            elif key in (10, 13):  # Enter
                if answer_text:
                    if int(answer_text) == number_a + number_b:
                        drowsy = False
                        closed_start_time = None
                        stop_alarm()
                        answer_text = ""
                        wrong_message = ""
                    else:
                        wrong_message = "Incorrect! Try again."
                        challenge_feedback_until = time.time() + 2
                        answer_text = ""

finally:
    cap.release()
    face_mesh.close()
    stop_alarm()
    pygame.mixer.quit()
    cv2.destroyAllWindows()
