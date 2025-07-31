import cv2
import numpy as np
import mediapipe as mp
import time
import os
import platform
from scipy.spatial import distance
from twilio.rest import Client

# Twilio Configuration
TWILIO_ACCOUNT_SID = "YOUR TWILO SID HERE"
TWILIO_AUTH_TOKEN = "YOUR TWILO TOKEN HERE"
TWILIO_PHONE_NUMBER = "TWILO PHN NO"
PARENT_PHONE_NUMBER = "ALERT NOTIFICATION NO"

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# EAR settings
baseline_ear = None
frame_check_awake = 10
awake_flag = 0

# Movement detection
missing_frames = 0
missing_frame_limit = 30

# Safe Zone Box
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAFE_BOX_WIDTH = 300
SAFE_BOX_HEIGHT = 300
SAFE_BOX_X_MIN = (FRAME_WIDTH // 2) - (SAFE_BOX_WIDTH // 2)
SAFE_BOX_Y_MIN = (FRAME_HEIGHT // 2) - (SAFE_BOX_HEIGHT // 2)
SAFE_BOX_X_MAX = SAFE_BOX_X_MIN + SAFE_BOX_WIDTH
SAFE_BOX_Y_MAX = SAFE_BOX_Y_MIN + SAFE_BOX_HEIGHT

def send_sos_alert(message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=PARENT_PHONE_NUMBER
        )
        print("📱 SOS Alert Sent!")
    except Exception as e:
        print(f"[ERROR] SMS failed: {e}")

def play_beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1500, 100)
        else:
            os.system('echo -e "\a"')
    except Exception as e:
        print(f"[ERROR] Beep failed: {e}")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_frame(cap):
    global awake_flag, baseline_ear, missing_frames
    ret, frame = cap.read()
    if not ret:
        missing_frames += 1
        if missing_frames >= missing_frame_limit:
            print("🚨 Baby missing!")
            play_beep()
            send_sos_alert("🚨 Baby is missing from view!")
            missing_frames = 0
        return None

    missing_frames = 0
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    cv2.rectangle(frame, (SAFE_BOX_X_MIN, SAFE_BOX_Y_MIN), (SAFE_BOX_X_MAX, SAFE_BOX_Y_MAX), (255, 0, 0), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            x_coords = [int(l.x * FRAME_WIDTH) for l in landmarks]
            y_coords = [int(l.y * FRAME_HEIGHT) for l in landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if x_min < SAFE_BOX_X_MIN or x_max > SAFE_BOX_X_MAX or y_min < SAFE_BOX_Y_MIN or y_max > SAFE_BOX_Y_MAX:
                print("🚨 Baby left safe zone!")
                cv2.putText(frame, "DANGER! BABY MOVED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_beep()
                send_sos_alert("🚨 Baby moved out of safe zone!")

            # Eye detection
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            left_eye_coords = [(int(p.x * FRAME_WIDTH), int(p.y * FRAME_HEIGHT)) for p in left_eye]
            right_eye_coords = [(int(p.x * FRAME_WIDTH), int(p.y * FRAME_HEIGHT)) for p in right_eye]
            leftEAR = eye_aspect_ratio(left_eye_coords)
            rightEAR = eye_aspect_ratio(right_eye_coords)
            ear = (leftEAR + rightEAR) / 2.0

            # Initialize baseline EAR
            if baseline_ear is None:
                baseline_ear = ear
                print(f"[INFO] Baseline EAR: {baseline_ear:.3f}")

            # Wake up detection
            if baseline_ear is not None and ear > baseline_ear - 0.03:
                awake_flag += 1
                if awake_flag >= frame_check_awake:
                    print("🚨 Baby Woke Up!")
                    cv2.putText(frame, "BABY WOKE UP!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    play_beep()
                    send_sos_alert("🚨 Baby is awake!")
                    awake_flag = 0
            else:
                awake_flag = 0

    return frame

def start_monitoring():
    cap = cv2.VideoCapture(0)
    print("🚀 Baby Monitoring Started")
    while True:
        frame = process_frame(cap)
        if frame is not None:
            cv2.imshow("Baby Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Monitoring Ended")

if __name__ == "__main__":
    start_monitoring()
