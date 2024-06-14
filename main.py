import os
import cv2
import numpy as np
from deepface import DeepFace
import webbrowser
import threading
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
gesture_detected = False
recognized_gesture = None
processing_face = False
processing_gesture = False
recognition_done = False
registration_in_progress = False

registered_faces_dir = 'registered_faces'
registered_gestures_dir = 'registered_gestures'

if not os.path.exists(registered_faces_dir):
    os.makedirs(registered_faces_dir)
if not os.path.exists(registered_gestures_dir):
    os.makedirs(registered_gestures_dir)

def log_message(message):
    print(f"[LOG] {message}")

def register_face(person_id):
    global registration_in_progress
    registration_in_progress = True
    ret, frame = cap.read()
    if ret:
        face_img_path = os.path.join(registered_faces_dir, f"person_{person_id}.jpg")
        cv2.imwrite(face_img_path, frame)
        log_message(f"New face registered for person {person_id}: {face_img_path}")
    registration_in_progress = False

def register_gesture(gesture_name):
    global registration_in_progress
    registration_in_progress = True
    ret, frame = cap.read()
    if ret:
        gesture_img_path = os.path.join(registered_gestures_dir, f"{gesture_name}.jpg")
        cv2.imwrite(gesture_img_path, frame)
        log_message(f"New gesture registered: {gesture_name}")
    registration_in_progress = False

def load_registered_faces():
    registered_faces = []
    for filename in os.listdir(registered_faces_dir):
        if filename.endswith(".jpg"):
            registered_faces.append(cv2.imread(os.path.join(registered_faces_dir, filename)))
    return registered_faces

def load_registered_gestures():
    registered_gestures = {}
    for filename in os.listdir(registered_gestures_dir):
        if filename.endswith(".jpg"):
            gesture_name = os.path.splitext(filename)[0]
            registered_gestures[gesture_name] = cv2.imread(os.path.join(registered_gestures_dir, filename), cv2.IMREAD_GRAYSCALE)
    return registered_gestures

def check_face(frame):
    global face_match, processing_face, recognition_done
    processing_face = True
    log_message("Starting face verification")
    try:
        registered_faces = load_registered_faces()
        for reference_img in registered_faces:
            result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
            log_message(f"Comparing with registered face, Result: {result['verified']}")
            if result['verified']:
                face_match = True
                recognition_done = True
                log_message("Face match found")
                break
            else:
                face_match = False
    except ValueError as e:
        face_match = False
        log_message(f"ValueError in face verification: {e}")

    processing_face = False

def check_gesture(frame, gesture_reference_images=None):
    global gesture_detected, recognized_gesture, processing_gesture, recognition_done
    processing_gesture = True
    gesture_detected = False
    recognized_gesture = None

    log_message("Starting gesture detection")
    try:
        fg_mask = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=55, detectShadows=False).apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            log_message(f"Found contour with area: {area}")

            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y + h, x:x + w]

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                for gesture_name, gesture_img in gesture_reference_images.items():
                    resized_gesture_img = cv2.resize(gesture_img, (w, h))

                    diff = cv2.absdiff(roi_gray, resized_gesture_img)

                    mean_diff = np.mean(diff)

                    log_message(f"Comparing with {gesture_name}, Mean Diff: {mean_diff}")

                    if mean_diff < 47:
                        recognized_gesture = gesture_name
                        gesture_detected = True
                        recognition_done = True
                        log_message(f"Gesture recognized: {recognized_gesture}")
                        break
                if gesture_detected:
                    break
    except Exception as e:
        log_message(f"Error detecting gesture: {e}")
        gesture_detected = False

    processing_gesture = False

def perform_action():
    global face_match, gesture_detected, recognized_gesture
    if face_match:
        log_message("Face matched. Performing action after 3 second cooldown.")
        time.sleep(3)
        webbrowser.open("https://www.google.com/")
    elif gesture_detected:
        log_message(f"Gesture detected: {recognized_gesture}")
        if recognized_gesture == "fist":
            log_message("Fist gesture detected! Performing action...")
            webbrowser.open("https://www.wikipedia.org/")

try:
    while True:
        ret, frame = cap.read()
        if ret:
            if counter % 100 == 0 and not recognition_done and not registration_in_progress:
                if not processing_face:
                    log_message("Spawning thread for face check")
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()

            if counter % 100 == 0 and not recognition_done and not registration_in_progress:
                if not processing_gesture:
                    log_message("Spawning thread for gesture check")
                    gesture_images = load_registered_gestures()
                    threading.Thread(target=check_gesture, args=(frame.copy(), gesture_images)).start()

            counter += 1

            if recognition_done:
                perform_action()
                break

            if face_match:
                cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif gesture_detected:
                cv2.putText(frame, f"Gesture Detected: {recognized_gesture}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == ord("q") or cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1:
            log_message("Quitting the application")
            break
        elif key == ord("r"):
            if not face_match and not gesture_detected and not recognition_done:
                input_type = input("Do you want to register a face or a gesture? (face/gesture): ").strip().lower()
                if input_type == "face":
                    person_id = input("Enter person ID to register new face: ")
                    register_face(person_id)
                elif input_type == "gesture":
                    gesture_name = input("Enter gesture name to register new gesture: ")
                    register_gesture(gesture_name)
                else:
                    log_message("Invalid input. Please enter 'face' or 'gesture'.")
            else:
                log_message("Recognition already done. No registration needed.")
except KeyboardInterrupt:
    log_message("KeyboardInterrupt detected. Exiting gracefully.")
finally:
    cv2.destroyAllWindows()
    cap.release()
    log_message("Resources released. Application closed.")
