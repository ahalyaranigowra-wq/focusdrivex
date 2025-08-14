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
    if os.name == "nt":
        import winsound
        def play_sound(file):
            try:
                winsound.Beep(2000, 500)  # fallback beep
            except Exception:
                pass
    else:
        def play_sound(file):
            pass

def play_alert_threaded(file="alert.mp3"):
    threading.Thread(target=lambda: play_sound(file), daemon=True).start()

def trigger_seat_vibration():
    threading.Thread(target=lambda: print("🪑 SEAT VIBRATION ACTIVATED"), daemon=True).start()

def trigger_water_spray():
    threading.Thread(target=lambda: print("💦 WATER SPRAY ACTIVATED"), daemon=True).start()

# ---------- Mediapipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Landmark definitions ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# ---------- Thresholds ----------
EAR_THRESHOLD = 0.22
EYE_CLOSED_FRAMES = 5
HEAD_TURN_THRESHOLD = 0.07
SIDE_MIRROR_MIN = 0.07
SIDE_MIRROR_MAX = 0.10

STAGE_ONE_TIME = 20
STAGE_TWO_TIME = 25
FINAL_ALERT_TIME = 30

# ---------- State variables ----------
eye_closed_counter = 0
baseline_nose_x = None
baseline_calibrated = False
baseline_frames_count = 30
baseline_frames_seen = 0
baseline_nose_sum = 0.0
distraction_start_time = None
current_alert_stage = 0  # 0 = no alert, 1 = stage1, 2 = stage2, 3 = final

# ---------- Helpers ----------
def calculate_EAR(eye_coords):
    A = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
    B = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
    C = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# ---------- Capture ----------
cap = cv2.VideoCapture(0)
time.sleep(0.5)
if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("Driver Monitoring System Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(image_rgb)
    results_hands = hands.process(image_rgb)

    # ---------- Detection flags ----------
    driver_state = "FOCUSED"
    status_color = (0,255,0)
    distraction_detected = False
    checking_side_mirror = False
    mobile_use_detected = False

    # ---------- Mobile use detection ----------
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            hand_y = np.mean([lm.y for lm in hand_landmarks.landmark])
            if hand_y > 0.6:
                mobile_use_detected = True
                driver_state = "USING MOBILE"
                status_color = (0,0,255)

    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        nose_x = face_landmarks[NOSE_TIP].x

        # ---------- Baseline calibration ----------
        if not baseline_calibrated:
            baseline_nose_sum += nose_x
            baseline_frames_seen += 1
            if baseline_frames_seen >= baseline_frames_count:
                baseline_nose_x = baseline_nose_sum / baseline_frames_seen
                baseline_calibrated = True
                print(f"Baseline nose_x calibrated: {baseline_nose_x:.4f}")
        else:
            head_offset = abs(nose_x - baseline_nose_x)
            if SIDE_MIRROR_MIN < head_offset < SIDE_MIRROR_MAX:
                checking_side_mirror = True
                driver_state = "CHECKING SIDE MIRROR"
                status_color = (255,0,0)
            elif head_offset > HEAD_TURN_THRESHOLD:
                distraction_detected = True
                driver_state = "DISTRACTED"
                status_color = (0,0,255)

        # ---------- EAR ----------
        left_eye_coords = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye_coords = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE]
        ear_left = calculate_EAR(left_eye_coords)
        ear_right = calculate_EAR(right_eye_coords)
        ear = (ear_left + ear_right) / 2.0
        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1
            if eye_closed_counter >= EYE_CLOSED_FRAMES:
                distraction_detected = True
                driver_state = "DISTRACTED"
                status_color = (0,0,255)
        else:
            eye_closed_counter = 0

    # ---------- Alert stages ----------
    alert_text = ""
    if driver_state == "DISTRACTED" or driver_state == "USING MOBILE":
        if distraction_start_time is None:
            distraction_start_time = time.time()
        elapsed = time.time() - distraction_start_time

        if elapsed >= FINAL_ALERT_TIME:
            if current_alert_stage != 3:
                current_alert_stage = 3
                trigger_water_spray()  # Stage 3
            alert_text = "FINAL ALERT: WATER SPRAY"
            status_color = (0,0,255)
        elif elapsed >= STAGE_TWO_TIME:
            if current_alert_stage != 2:
                current_alert_stage = 2
                trigger_seat_vibration()  # Stage 2
            alert_text = "STAGE TWO ALERT: SEAT VIBRATION"
            status_color = (0,0,255)
        elif elapsed >= STAGE_ONE_TIME:
            if current_alert_stage != 1:
                current_alert_stage = 1
                play_alert_threaded("alert.mp3")  # Stage 1 plays sound
            alert_text = "STAGE ONE ALERT"
            status_color = (0,0,255)
    else:
        distraction_start_time = None
        current_alert_stage = 0
        if driver_state == "FOCUSED":
            status_color = (0,255,0)
        elif driver_state == "CHECKING SIDE MIRROR":
            status_color = (255,0,0)

    # ---------- Display ----------
    display_text = alert_text if alert_text else driver_state
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    cv2.imshow("Driver Monitoring System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        baseline_calibrated = False
        baseline_frames_seen = 0
        baseline_nose_sum = 0.0
        print("Calibration reset. Look forward for a few seconds...")

cap.release()
cv2.destroyAllWindows()
