import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os

# ---------- Sound helper (non-blocking) ----------
try:
    from playsound import playsound

    def play_sound(file):
        try:
            playsound(file)
        except Exception:
            pass

except Exception:
    # fallback for Windows if playsound not available
    if os.name == "nt":
        import winsound

        def play_sound(file):
            try:
                winsound.Beep(1000, 600)
            except Exception:
                pass
    else:
        def play_sound(file):
            pass

def play_alert_threaded(file="alert.mp3"):
    threading.Thread(target=lambda: play_sound(file), daemon=True).start()


# ---------- Mediapipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Landmark definitions (MediaPipe Face Mesh) ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468]   # sometimes available
RIGHT_IRIS = [473]  # sometimes available
NOSE_TIP = 1
FOREHEAD_POINT = 10

# ---------- Thresholds & parameters (tunable) ----------
EAR_THRESHOLD = 0.22        # lower -> more sensitive to eye closure
EYE_CLOSED_FRAMES = 10      # how many consecutive frames considered closed (~0.33s at 30fps)
HEAD_TURN_THRESHOLD = 0.07  # normalized nose offset from baseline (0-1)
GAZE_THRESHOLD = 0.035      # iris vs eye center deviation (normalized)
ALERT_INTERVAL = 2.0        # seconds between alert sounds while still distracted

# ---------- State variables ----------
eye_closed_counter = 0
last_alert_time = 0
baseline_nose_x = None
baseline_calibrated = False
baseline_frames_count = 30  # calibrate baseline using first N valid frames
baseline_frames_seen = 0
baseline_nose_sum = 0.0

# ---------- Helpers ----------
def calculate_EAR(eye_coords):
    # eye_coords: list of 6 (x,y) points in pixel coords: [p1..p6]
    A = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
    B = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
    C = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
time.sleep(0.5)

if not cap.isOpened():
    print("Cannot open webcam. Exiting.")
    exit()

print("Starting Driver Distraction Alert. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    distraction_detected = False
    debug_head_offset = 0.0
    debug_ear = 0.0
    debug_gaze = 0.0

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        face_landmarks = results.multi_face_landmarks[0].landmark
        # normalized nose x (0..1)
        nose_x = face_landmarks[NOSE_TIP].x

        # ---------- Baseline calibration for nose center ----------
        if not baseline_calibrated:
            # accumulate nose_x for first baseline_frames_count valid frames
            baseline_nose_sum += nose_x
            baseline_frames_seen += 1
            if baseline_frames_seen >= baseline_frames_count:
                baseline_nose_x = baseline_nose_sum / baseline_frames_seen
                baseline_calibrated = True
                print(f"Baseline nose_x calibrated: {baseline_nose_x:.4f}")
        else:
            # head offset normalized relative to baseline
            debug_head_offset = abs(nose_x - baseline_nose_x)

            # head turn detection
            if debug_head_offset > HEAD_TURN_THRESHOLD:
                distraction_detected = True

        # ---------- EAR (eye closure) ----------
        # get pixel coordinates for eyes
        left_eye_coords = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye_coords = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE]
        ear_left = calculate_EAR(left_eye_coords)
        ear_right = calculate_EAR(right_eye_coords)
        ear = (ear_left + ear_right) / 2.0
        debug_ear = ear

        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1
            if eye_closed_counter >= EYE_CLOSED_FRAMES:
                distraction_detected = True
        else:
            eye_closed_counter = 0

        # ---------- Gaze (iris position) - safe check ----------
        gaze_offset = 0.0
        if max(LEFT_IRIS) < len(face_landmarks) and max(RIGHT_IRIS) < len(face_landmarks):
            left_iris_x = face_landmarks[LEFT_IRIS[0]].x
            right_iris_x = face_landmarks[RIGHT_IRIS[0]].x
            left_eye_avg_x = np.mean([face_landmarks[i].x for i in LEFT_EYE])
            right_eye_avg_x = np.mean([face_landmarks[i].x for i in RIGHT_EYE])

            left_gaze = abs(left_iris_x - left_eye_avg_x)
            right_gaze = abs(right_iris_x - right_eye_avg_x)
            gaze_offset = max(left_gaze, right_gaze)
            debug_gaze = gaze_offset

            if gaze_offset > GAZE_THRESHOLD:
                distraction_detected = True

        # ---------- Forehead dot ----------
        forehead_x = int(face_landmarks[FOREHEAD_POINT].x * w)
        forehead_y = int(face_landmarks[FOREHEAD_POINT].y * h)
        dot_color = (0, 0, 255) if distraction_detected else (0, 255, 0)
        cv2.circle(frame, (forehead_x, forehead_y), 6, dot_color, -1)

        # ---------- On-screen labels ----------
        status_text = "DISTRACTED" if distraction_detected else "FOCUSED"
        status_color = (0, 0, 255) if distraction_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Debug info (small)
        cv2.putText(frame, f"EAR:{debug_ear:.3f}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"HeadOff:{debug_head_offset:.3f}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Gaze:{debug_gaze:.3f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # ---------- Alert sound throttle ----------
        if distraction_detected:
            if time.time() - last_alert_time > ALERT_INTERVAL:
                play_alert_threaded("alert.mp3")
                last_alert_time = time.time()
    else:
        # No face detected - show hint
        cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

    cv2.imshow("Driver Distraction Alert", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Optional: allow recalibration by pressing 'c'
    if key == ord('c'):
        baseline_calibrated = False
        baseline_frames_seen = 0
        baseline_nose_sum = 0.0
        print("Calibration reset. Please look forward for a few seconds...")

cap.release()
cv2.destroyAllWindows()
