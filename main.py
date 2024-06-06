import threading
import cv2
import numpy as np
from deepface import DeepFace
import webbrowser

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False
gesture_detected = False
recognized_gesture = None

reference_images = ["reference1.jpg", "reference2.jpg", "reference3.jpg"]  # Face reference images
gesture_reference_images = {
    "open_hand": "gesture_open_hand.jpg",  # Replace with paths to your gesture reference images
    "thumbs_up": "gesture_thumbs_up.jpg",
    "fist": "gesture_fist.jpg"
}


def check_face(frame):
    global face_match
    try:
        for reference_img_path in reference_images:
            reference_img = cv2.imread(reference_img_path)
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                face_match = True
                break
            else:
                face_match = False
    except ValueError:
        face_match = False


def check_gesture(frame):
    global gesture_detected, recognized_gesture
    gesture_detected = False
    recognized_gesture = None

    try:
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the area of each contour
            area = cv2.contourArea(contour)

            # If the contour area is above a certain threshold, consider it as a gesture
            if area > min_gesture_area:
                # Extract the region of interest (ROI)
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y + h, x:x + w]

                # Compare ROI with each gesture reference image
                for gesture_name, gesture_img_path in gesture_reference_images.items():
                    gesture_img = cv2.imread(gesture_img_path, cv2.IMREAD_GRAYSCALE)
                    resized_gesture_img = cv2.resize(gesture_img, (w, h))
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Calculate similarity (using Structural Similarity Index as an example)
                    score, _ = cv2.quality.QualitySSIM_compute(resized_gesture_img, roi_gray)
                    similarity = np.mean(score)

                    if similarity > similarity_threshold:
                        recognized_gesture = gesture_name
                        gesture_detected = True
                        break
            if gesture_detected:
                break
    except Exception as e:
        print("Error detecting gesture:", e)
        gesture_detected = False


def prompt_user():
    global face_match, gesture_detected, recognized_gesture
    if face_match:
        choice = input("Face matched! Do you want to open YouTube (y) or Spotify (s)? ")
        if choice.lower() == 'y':
            webbrowser.open("https://www.youtube.com/")
        elif choice.lower() == 's':
            webbrowser.open("https://www.spotify.com/")
    elif gesture_detected:
        if recognized_gesture == "open_hand":
            print("Open hand gesture detected! Performing action...")
            # Define the action for open hand gesture
            webbrowser.open("https://www.google.com/")
        elif recognized_gesture == "thumbs_up":
            print("Thumbs up gesture detected! Performing action...")
            # Define the action for thumbs up gesture
            webbrowser.open("https://www.youtube.com/")
        elif recognized_gesture == "fist":
            print("Fist gesture detected! Performing action...")
            # Define the action for fist gesture
            webbrowser.open("https://www.spotify.com/")


# Background subtractor for gesture detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Minimum area threshold for gesture detection
min_gesture_area = 1000

# Similarity threshold for gesture recognition
similarity_threshold = 0.7  # Adjust this value based on your tests

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
                threading.Thread(target=check_gesture, args=(frame.copy(),)).start()
            except ValueError:
                pass

        counter += 1

        prompt_user()

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif gesture_detected:
            cv2.putText(frame, f"Gesture Detected: {recognized_gesture}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
