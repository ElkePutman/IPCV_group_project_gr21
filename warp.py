import cv2
import numpy as np
from skimage.transform import swirl
import mediapipe as mp

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = face_detection.process(rgb)

    output = frame.copy()

    if face.detections:
        for detection in face.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, bw, bh = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Get a roi for the face
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            fh, fw = face_roi.shape[:2]
            cx, cy = fw // 2, fh // 2

            # Get the face roi to swirl
            warped = swirl(face_roi, center=(cx, cy), strength=5, radius=min(fh, fw) // 1.3, preserve_range=True)
            warped = np.uint8(warped)

            output[y1:y2, x1:x2] = warped

    cv2.imshow("Press 'q' to exit", output)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
