# Import libraries
import cv2
import numpy as np
import os

# Access the webcam video
cap = cv2.VideoCapture(0)


BASE_INPUT_PATH = os.path.dirname(os.path.abspath(__file__))
HAT_PATH = os.path.join(BASE_INPUT_PATH, 'hat.png')
hat = cv2.imread(HAT_PATH, cv2.IMREAD_UNCHANGED)

def recog_face_feat(fr):
    """
    Detects faces in a given frame using a Haar Cascade Classifier.

    The function converts the input frame to grayscale and applies
    OpenCVs Haar cascade classifier to recognize the placement and size of the face

    Args:
        fr (np.ndarray): Input image frame (BGR or grayscale).

    Returns:
        list of tuples: Detected faces represented as (x, y, w, h) bounding boxes.

    Raises:
        IndexError: If the input image does not have at least two dimensions.

    References:
        - https://www.datacamp.com/tutorial/face-detection-python-opencv
        - https://www.geeksforgeeks.org/python/face-detection-using-cascade-classifier-using-opencv-python/
    """
    if fr.ndim > 2:
        fr_grey = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    elif fr.ndim == 2:
        fr_grey == fr
    else:
        IndexError("Wrong size image, make sure the input array for face detection is 2D or higher")
    
    # detect face
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(
        fr_grey, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    return faces

def draw_box(frame,tup,c=(0,255,0),thickness=4):
    """
    Draws rectangular bounding boxes on a frame.

    Args:
        fr (np.ndarray): Input image frame (BGR).
        tup (list of tuples): List of bounding boxes, each formatted as (x, y, w, h).
        c (tuple): Color of the rectangle in BGR format. Default is green (0, 255, 0).
        thickness (int or float): Line thickness of the rectangle borders.

    Returns:
        None: The function modifies the frame in place.
    """
    for (x,y,w,h) in tup:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

def overlay_transparent_image(bg,overlay,pos=(0,0),size =(200,150)):
    """
    Overlays a transparent image onto a background frame.

    Args:
        bg (np.ndarray): The current frame (BGR format).
        overlay (np.ndarray): The image to overlay.
        pos (tuple): Top-left (x, y) coordinates where the overlay should be placed.
        size (tuple): Desired size (width, height) of the overlay image.

    Returns:
        combined (np.ndarray): The resulting image with overlay applied.
            
    """
    overlay = cv2.resize(overlay,(size[0],size[1]))
    bg_width,bg_height,bg_depth = bg.shape
    combined = bg.copy()
    overlay[:,:,3] = overlay[:,:,3]/255
    widthl = 0
    widthr = size[0]
    height1 = 0

    # allow hat to move partly off screen
    if pos[1] < size[1]: # height
        height1 = size[1]-pos[1]

    overlay_im = overlay[height1::,widthl:widthr]
    o_height,o_width,o_depth = overlay_im.shape

    # combine the overlay with background using alpha compositing
    combined[pos[1]-o_height:pos[1],pos[0]:pos[0]+o_width] = \
        bg[pos[1]-o_height:pos[1],pos[0]:pos[0]+o_width]*\
            (np.transpose([[1-overlay_im[:,:,3]]*3][0],(1,2,0)))+\
                np.transpose(overlay_im[:,:,:3],axes=(0,1,2))*np.transpose([overlay_im[:,:,3],overlay_im[:,:,3],overlay_im[:,:,3]],axes=(1,2,0))

    return combined

while True:
    # Read the next frame from the webcam
    status, photo = cap.read()
    frame = cv2.flip(photo,1)

    # face recognition
    faces = recog_face_feat(frame)

    # overlay image
    for (x,y,w,h) in faces:
        frame = overlay_transparent_image(frame,hat,pos=[x,y],size=(w,150))
    
    # draw_box(frame,faces) # draw box around faces

    cv2.imshow("Press 'q' to exit", frame)
    old_frame = frame
    # Wait for 50 milliseconds and check if the 'Enter' key (key code 13) is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()