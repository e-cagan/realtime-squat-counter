"""
Module for main application.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Mediapipe pose API settings
model_path = 'pose_landmarker_full.task' # Ensure this file is downloaded locally
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Webcam
cap = cv2.VideoCapture(0)

# Pose indexes
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # head, throat, shoulder
    (0, 4), (4, 5), (5, 6), (6, 8),   # other shoulder
    (9, 10),                          # mouth
    (11, 12),                         # between shoulders
    (11, 13), (13, 15),               # left arm
    (12, 14), (14, 16),               # right arm
    (15, 17), (15, 19), (15, 21),     # left hand
    (16, 18), (16, 20), (16, 22),     # right hand
    (11, 23), (12, 24),               # torso sides
    (23, 24),                         # between hips
    (23, 25), (25, 27),               # left leg
    (24, 26), (26, 28),               # right leg
    (27, 29), (27, 31), (29, 31),     # left foot
    (28, 30), (28, 32), (30, 32)      # right foot
]

# State management
SQUAT_DOWN_THRESHOLD = 80
SQUAT_UP_THRESHOLD = 150
current_state = "UP"
repetitions = 0


def convert_angle(v1, v2):
    """
    A helper function that takes cosine angle from given two vectors
    """
    # Calculate the cosine theta then convert it to radian formatted angle
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip values to handle floating-point inaccuracies outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


if __name__ == '__main__':
    with PoseLandmarker.create_from_options(options) as landmarker:
        # Inference loop
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape

            # Convert OpenCV BGR frame to MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Compute exact timestamp in milliseconds
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Run detection
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Access coordinates if landmarks are present
            if detection_result.pose_landmarks:
                for pose_landmarks in detection_result.pose_landmarks:
                    # Single person
                    landmarks = detection_result.pose_landmarks[0]

                    # Take the body parts for squad
                    left_hip, right_hip = landmarks[23], landmarks[24]
                    left_knee, right_knee = landmarks[25], landmarks[26]
                    left_ankle, right_ankle = landmarks[27], landmarks[28]
                    
                    # Vector for hip to knee
                    v1 = np.array([left_hip.x - left_knee.x, left_hip.y - left_knee.y])

                    # Vector for ankle to knee
                    v2 = np.array([left_ankle.x - left_knee.x, left_ankle.y - left_knee.y])

                    # Calculate angle and displat it on the frame
                    angle = convert_angle(v1, v2)
                    degree = np.degrees(angle)

                    # Draw the points
                    for idx, lm in enumerate(landmarks):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)
                    
                    # Draw the connections between points using POSE_CONNECTIONS
                    for start_idx, end_idx in POSE_CONNECTIONS:
                        x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
                        x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Manage the state
                    if degree < SQUAT_DOWN_THRESHOLD and current_state == "UP":
                        # Down detected
                        current_state = "DOWN"
                    elif degree > SQUAT_UP_THRESHOLD and current_state == "DOWN":
                        # Up detected (+1 Repeats)
                        current_state = "UP"
                        repetitions += 1

                    cv2.putText(img=frame, text=f'Degree: {degree:.2f} | State: {current_state} | Repeats: {repetitions}', 
                                org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), thickness=2, fontScale=0.6)

            # Display the frame and add quit option
            cv2.imshow('MediaPipe Pose Tasks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the cap pointer and destroy the window
        cap.release()
        cv2.destroyAllWindows()