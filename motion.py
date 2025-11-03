import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_mesh
import numpy as np
import os
from skimage.transform import swirl


BASE_INPUT_PATH = os.path.dirname(os.path.abspath(__file__))
HAT_PATH = os.path.join(BASE_INPUT_PATH, 'hat.png')
CIGAR_PATH = os.path.join(BASE_INPUT_PATH, 'Cigar.png')
hat = cv2.imread(HAT_PATH, cv2.IMREAD_UNCHANGED)
cigar = cv2.imread(CIGAR_PATH, cv2.IMREAD_UNCHANGED)



def get_face_mesh_bounds(frame, face_mesh):
    """
    Detects facial landmarks using MediaPipe FaceMesh.

    Args:
        frame: (np.ndarray) The input image (BGR format).
        face_mesh: Initialized MediaPipe FaceMesh object.

    Returns:
        tuple:
            - faces (list of tuples): List of bounding boxes for detected faces, formatted as (x, y, w, h).
            - face_landmarks (list): List of MediaPipe face landmark objects for each detected face.
            - mouth (list of tuples): List of bounding boxes for mouth regions, formatted as (x_min, y_min, x_max, y_max).
    """
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    faces, mouth = [], []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            xf, yf = [], []
            xm, ym = [], []
            
            mouth_indices = [61, 291, 0, 17, 13, 14, 78, 308]

            for i, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                xf.append(x)
                yf.append(y)
                if i in mouth_indices:
                    xm.append(x)
                    ym.append(y)

            if not xf or not yf or not xm or not ym:
                continue

            x_min, x_max = min(xf), max(xf)
            y_min, y_max = min(yf), max(yf)
            x_min_m, x_max_m = min(xm), max(xm)
            y_min_m, y_max_m = min(ym), max(ym)

            faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
            mouth.append((x_min_m, y_min_m, x_max_m, y_max_m))

    return faces, results.multi_face_landmarks if results.multi_face_landmarks else None, mouth

def overlay_transparent_image(bg, overlay, pos=(0, 0), size=(200, 150)):
    """
    Overlays a transparent image onto a background frame.

    Args:
        bg (np.ndarray): The current frame (BGR format).
        overlay (np.ndarray): The image to overlay.
        pos (tuple): Top-left (x, y) coordinates where the overlay should be placed.
        size (tuple): Desired size (width, height) of the overlay image.

    Returns:
        tuple:
            - combined (np.ndarray): The resulting image with overlay applied.
            - pos (tuple): The actual position used for overlay.
    """
    overlay = cv2.resize(overlay, (size[0], size[1]))
    combined = bg.copy()

    h, w = combined.shape[:2]
    x, y = pos

    
    if x < 0 or y < 0 or x + size[0] > w or y + size[1] > h:        
        return combined, pos

    #create a RGBA instead of RGB on the 3th dimension
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            (overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255),
            axis=2
        )

    alpha = overlay[:, :, 3] / 255.0


    #result=background×(1−α)+overlay×α
    for c in range(3):
        combined[y:y + size[1], x:x + size[0], c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * combined[y:y + size[1], x:x + size[0], c]
        )

    return combined, pos


def warp_face(frame, faces):
    """
    Applies a swirl effect to detected face regions.

    Args:
        frame (np.ndarray): The input video frame.
        faces (list of tuples): List of bounding boxes for faces, formatted as (x, y, w, h).

    Returns:
        output (np.ndarray): The frame with the swirl effect applied to detected faces.
    """
    output = frame.copy()
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        
        cx, cy = w // 2, h // 2
        warped = swirl(face_roi, center=(cx, cy), strength=5, radius=min(h, w)//1.3, preserve_range=True)
        warped = np.uint8(warped)
        output[y:y + h, x:x + w] = warped
    return output

def is_fist(hand_landmarks):
    """
    Detects if the hand gesture represents a closed fist.

    Args:
        hand_landmarks: MediaPipe hand landmarks for one detected hand.

    Returns:
        bool: True if a fist gesture is detected, otherwise False.
    """
    fingers = []
    for tip_id in [8, 12, 16, 20]:
        fingers.append(hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y)
    return sum(fingers) == 4


def is_peace(hand_landmarks):
    """
    Detects if the hand shows a peace sign gesture.

    Args:
        hand_landmarks: MediaPipe hand landmarks for one detected hand.

    Returns:
        bool: True if a peace sign is detected, otherwise False.
    """
    fingers=[]
    for tip_id in [8,12,16,20]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)  
        else:
            fingers.append(0)
    return fingers == [1,1,0,0]

def is_thumb_pink(hand_landmarks):
    """
    Detects if both the thumb and pinky fingers are extended while other fingers are folded.

    Args:
        hand_landmarks: MediaPipe hand landmarks for one detected hand.

    Returns:
        tuple:
            - is_thumb_extended (bool): Whether the thumb is extended.
            - pink_extended (bool): Whether the pinky is extended.
            - all_folded (bool): Whether all other fingers are folded.
    """
    lm = hand_landmarks.landmark
    is_thumb_extended = lm[4].y<lm[3].y<lm[2].y<lm[1].y    
    pink_extended  = lm[20].y<lm[19].y<lm[18].y<lm[17].y
    
    folded = []
    for tip_id in [8, 12, 16]:
        folded.append(lm[tip_id].y > lm[tip_id - 2].y)
    
    all_folded = sum(folded) == 3
    return is_thumb_extended, pink_extended, all_folded


def is_smoker_roi(mouth, hand_landmarks, frame, finger_ids=[8, 12]):
    """
    Detects whether the user's fingers overlap with the mouth region,
    simulating a 'smoking' gesture for overlaying a cigar image.

    Args:
        mouth (list of tuples): Mouth region bounding boxes (x_min, y_min, x_max, y_max).
        hand_landmarks: MediaPipe hand landmarks for one detected hand.
        frame (np.ndarray): The current frame (BGR format).
        finger_ids (list of int): Indices of fingers to check for overlap (default: [8, 12]).        

    Returns:
        bool: True if the finger region overlaps with the mouth region, otherwise False.
    """

    if not mouth or hand_landmarks is None:
        return False

    h, w, _ = frame.shape

    for fid in finger_ids:
        
        x1 = int(hand_landmarks.landmark[fid].x * w)
        y1 = int(hand_landmarks.landmark[fid].y * h)
        x2 = int(hand_landmarks.landmark[fid - 2].x * w)
        y2 = int(hand_landmarks.landmark[fid - 2].y * h)

        
        x_min_f = min(x1, x2)
        y_min_f = min(y1, y2)
        x_max_f = max(x1, x2)
        y_max_f = max(y1, y2)
        finger_box = (x_min_f, y_min_f, x_max_f, y_max_f)



        # Check overlap with mouth ROI
        for (xmin, ymin, xmax, ymax) in mouth:
            overlap_x = max(finger_box[0], xmin) < min(finger_box[2], xmax)
            overlap_y = max(finger_box[1], ymin) < min(finger_box[3], ymax)
            if overlap_x and overlap_y:
                return True

    return False


hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)


cam = cv2.VideoCapture(0)
print("Press 'q' to exit")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera frame not available.")
        continue

    faces, face_landmarks_list, mouth  = get_face_mesh_bounds(frame, face_mesh)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_detected = hands.process(frame_rgb)

    for idx, (x, y, w, h) in enumerate(faces):        
        if face_landmarks_list:
            top_forehead = face_landmarks_list[idx].landmark[10]
            h_img, w_img = frame.shape[:2]
            hat_center_x = int(top_forehead.x * w_img)
            hat_top_y = int(top_forehead.y * h_img) -int(h * 0.5)  # shift hat up
        # else:
        #     hat_center_x = x + w // 2
        #     hat_top_y = y - int(h * 0.5)

        hat_width = int(w * 1.2) # 1.2* facewidth
        hat_height = int(hat_width * hat.shape[0] / hat.shape[1])
        hat_x = int(hat_center_x - hat_width / 2)
        hat_y = hat_top_y

        if not hands_detected.multi_hand_landmarks:
            frame, _ = overlay_transparent_image(frame, hat, pos=(hat_x, hat_y), size=(hat_width, hat_height))
        else:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                if all (is_thumb_pink(hand_landmarks)):
                    frame = warp_face(frame, faces)
                    frame, _ = overlay_transparent_image(frame, hat, pos=(hat_x, hat_y), size=(hat_width, hat_height))

                elif is_peace(hand_landmarks):
                    if is_smoker_roi(mouth, hand_landmarks, frame):
                        indexf = hand_landmarks.landmark[8]
                        middlef = hand_landmarks.landmark[12]
                        cw,ch = cigar.shape[0],cigar.shape[1]
                        xpos, ypos = int(min(indexf.x,middlef.x)*w_img), int(min(indexf.y,middlef.y)*h_img)
                        ypos = ypos - 20

                        
                        cigw = int(w * 0.8)
                        cigh = int(cigar.shape[0] * cigw / cigar.shape[1])
                        

                        frame, _ = overlay_transparent_image(frame, cigar, pos=(xpos, ypos), size=(cigw, cigh))
                        frame, _ = overlay_transparent_image(frame, hat, pos=(hat_x, hat_y), size=(hat_width, hat_height))                  

                        
                    else:
                        frame, _ = overlay_transparent_image(frame, hat, pos=(hat_x, hat_y), size=(hat_width, hat_height))

                elif is_fist(hand_landmarks):
                    new_hat_x, new_hat_y = hat_x+50, hat_y-30

                    
                    frame, _ = overlay_transparent_image(frame, hat, pos=(new_hat_x, new_hat_y), size=(hat_width, hat_height))
     

                else:
                    frame, _ = overlay_transparent_image(frame, hat, pos=(hat_x, hat_y), size=(hat_width, hat_height))

    cv2.imshow("Press q to exit", cv2.flip(frame, 1))

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
