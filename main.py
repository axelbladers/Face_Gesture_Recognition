import threading
import cv2
import numpy as np
from deepface import DeepFace
import webbrowser
import os

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set video frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize counters and flags
counter = 0
face_match = False
gesture_detected = False
recognized_gesture = None

# Create directories for storing registered faces if they don't exist
if not os.path.exists('registered_faces'):
    os.makedirs('registered_faces')

# Helper function to log messages
def log_message(message):
    print(f"[LOG] {message}")

# Function to register new faces
def register_face(person_id):
    ret, frame = cap.read()
    if ret:
        face_img_path = f'registered_faces/person_{person_id}.jpg'
        cv2.imwrite(face_img_path, frame)
        log_message(f"New face registered for person {person_id}: {face_img_path}")

# Function to load registered faces
def load_registered_faces():
    registered_faces = []
    for filename in os.listdir('registered_faces'):
        if filename.endswith(".jpg"):
            registered_faces.append(cv2.imread(os.path.join('registered_faces', filename)))
    return registered_faces

# Function to check for face match
def check_face(frame):
    global face_match
    log_message("Starting face verification")
    try:
        registered_faces = load_registered_faces()
        for reference_img in registered_faces:
            result = DeepFace.verify(frame, reference_img.copy())
            log_message(f"Comparing with registered face, Result: {result['verified']}")
            if result['verified']:
                face_match = True
                log_message("Face match found")
                break
            else:
                face_match = False
    except ValueError:
        face_match = False
        log_message("ValueError in face verification")

# Function to check for gesture
def check_gesture(frame):
    global gesture_detected, recognized_gesture
    gesture_detected = False
    recognized_gesture = None

    log_message("Starting gesture detection")
    try:
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            log_message(f"Found contour with area: {area}")

            if area > min_gesture_area:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y + h, x:x + w]

                for gesture_name, gesture_img_path in gesture_reference_images.items():
                    gesture_img = cv2.imread(gesture_img_path, cv2.IMREAD_GRAYSCALE)
                    resized_gesture_img = cv2.resize(gesture_img, (w, h))
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    score, _ = cv2.quality.QualitySSIM_compute(resized_gesture_img, roi_gray)
                    similarity = np.mean(score)

                    log_message(f"Comparing with {gesture_img_path}, Similarity: {similarity}")

                    if similarity > similarity_threshold:
                        recognized_gesture = gesture_name
                        gesture_detected = True
                        log_message(f"Gesture recognized: {recognized_gesture}")
                        break
            if gesture_detected:
                break
    except Exception as e:
        log_message(f"Error detecting gesture: {e}")
        gesture_detected = False

# Function to prompt user for actions
def prompt_user():
    global face_match, gesture_detected, recognized_gesture
    if face_match:
        log_message("Face matched. Prompting user for choice.")
        choice = input("Face matched! Do you want to open YouTube (y) or Spotify (s)? ")
        if choice.lower() == 'y':
            webbrowser.open("https://www.youtube.com/")
        elif choice.lower() == 's':
            webbrowser.open("https://www.spotify.com/")
    elif gesture_detected:
        log_message(f"Gesture detected: {recognized_gesture}")
        if recognized_gesture == "open_hand":
            log_message("Open hand gesture detected! Performing action...")
            webbrowser.open("https://www.google.com/")
        elif recognized_gesture == "thumbs_up":
            log_message("Thumbs up gesture detected! Performing action...")
            webbrowser.open("https://www.youtube.com/")
        elif recognized_gesture == "fist":
            log_message("Fist gesture detected! Performing action...")
            webbrowser.open("https://www.spotify.com/")

# Background subtractor and thresholds for gesture detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
min_gesture_area = 1000
similarity_threshold = 0.7

# Main loop to capture frames and process them
while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                log_message("Spawning threads for face and gesture checks")
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
                threading.Thread(target=check_gesture, args=(frame.copy(),)).start()
            except ValueError:
                log_message("Error starting threads")

        counter += 1
        prompt_user()

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif gesture_detected:
            cv2.putText(frame, f"Gesture Detected: {recognized_gesture}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        log_message("Quitting the application")
        break
    elif key == ord("r"):
        person_id = input("Enter person ID to register new face: ")
        register_face(person_id)

cv2.destroyAllWindows()
