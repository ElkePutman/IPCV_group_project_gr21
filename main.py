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
    Detect faces using MediaPipe FaceMesh.
    Returns list of bounding boxes (x, y, w, h) per face and mouth regions.
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
    Add overlay image to the frame
    Inputs:
    - bg input frame
    - image to be overlayed
    - position of overlay
    - size of overlayed image

    Returns the frame with the overlay added and the position of the overlay
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
    Creates a swirl on the face
    Inputs: - current frame
    - Face landmarks

    Returns the frame with the warped face

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
    Detect if hand shows a fist
    Inputs: 
    - Handlandmarks
    """
    fingers = []
    for tip_id in [8, 12, 16, 20]:
        fingers.append(hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y)
    return sum(fingers) == 4

# def is_hand(hand_landmarks):
#     fingers = []
#     for tip_id in [4,8,12,16,20]:
#         if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
#             fingers.append(0)  
#         else:
#             fingers.append(1)
#     return sum(fingers) == 0

# def is_thumb_down(hand_landmarks):
    
#     lm = hand_landmarks.landmark    
#     thumb_tip = lm[4]
#     thumb_low = lm[2]    
#     is_thumb_extended = thumb_tip.y>thumb_low.y

    
#     folded = []
#     for tip_id in [8, 12, 16, 20]:
#         folded.append(lm[tip_id].y > lm[tip_id - 2].y)
    
#     all_folded = sum(folded) == 4

#     return is_thumb_extended and all_folded



def is_peace(hand_landmarks):
    """
    Detect if hand shows a peace sign
    Inputs: 
    - Handlandmarks
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
    Detect if both a thumb an pinky is extended
    Inputs: 
    - Handlandmarks
    """
    lm = hand_landmarks.landmark
    is_thumb_extended = lm[4].y<lm[3].y<lm[2].y<lm[1].y    
    pink_extended  = lm[20].y<lm[19].y<lm[18].y<lm[17].y
    
    folded = []
    for tip_id in [8, 12, 16]:
        folded.append(lm[tip_id].y > lm[tip_id - 2].y)
    
    all_folded = sum(folded) == 3
    return is_thumb_extended, pink_extended, all_folded


def is_smoker_roi(mouth, hand_landmarks, frame, finger_ids=[8, 12], debug=True):
    """
    Detect if fingers overlap with mouth and overlay a cigar
    Inputs: 
    - Hand landmarks
    - Mouth landmarks
    - input frame
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

        # if debug:
        #     # Teken de finger box (blauw) en de tip (rood)
        #     cv2.rectangle(frame, (x_min_f, y_min_f), (x_max_f, y_max_f), (255, 0, 0), 2)
        #     # cv2.circle(frame, (x1, y1), 4, (0, 0, 255), -1)

        # Check overlap met de mond-ROI
        for (xmin, ymin, xmax, ymax) in mouth:
            overlap_x = max(finger_box[0], xmin) < min(finger_box[2], xmax)
            overlap_y = max(finger_box[1], ymin) < min(finger_box[3], ymax)
            if overlap_x and overlap_y:
                # if debug:
                #     cv2.putText(frame, "Overlap!", (xmin, ymin - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                return True

    return False


hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)


cam = cv2.VideoCapture(0)
print("Druk op 'q' om af te sluiten.")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera frame niet beschikbaar.")
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
        else:
            hat_center_x = x + w // 2
            hat_top_y = y - int(h * 0.5)

        hat_width = int(w * 1.2) # 1.2* facewidth
        hat_height = int(hat_width * hat.shape[0] / hat.shape[1])
        hat_x = int(hat_center_x - hat_width / 2)
        hat_y = hat_top_y

        # for (x_min_m, y_min_m, x_max_m, y_max_m) in mouth:
        #     cv2.rectangle(frame, (x_min_m, y_min_m), (x_max_m, y_max_m), (0, 255, 0), 2)

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
