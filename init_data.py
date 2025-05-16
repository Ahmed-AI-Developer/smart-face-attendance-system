import cv2
import os
from deepface import DeepFace
from utils import mark_attendance

# Path to known faces
KNOWN_FACES_DIR = "database/faces"

# Load all known images
known_faces = []
names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        known_faces.append(path)
        names.append(os.path.splitext(filename)[0])

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        for i, face_path in enumerate(known_faces):
            result = DeepFace.verify(frame, face_path, enforce_detection=False)
            if result["verified"]:
                name = names[i]
                mark_attendance(name)

                # Draw box & label
                cv2.putText(frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break  # Avoid marking multiple times for one frame

    except Exception as e:
        print("Detection error:", e)

    cv2.imshow("Smart Attendance - DeepFace", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
