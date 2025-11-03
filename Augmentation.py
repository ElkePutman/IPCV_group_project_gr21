# Import libraries
import cv2
import numpy as np

# Access the webcam video
cap = cv2.VideoCapture(0)

hat = cv2.imread(r"C:\Users\ejput\OneDrive - University of Twente\BME\252601-Kwartiel 1 2025\Image processing\Group_Project\hat.png",cv2.IMREAD_UNCHANGED)

def recog_face_feat(fr):
    # recognize face and finds face, eyes, mouth rectangle
        # input: frame 3D ARRAY with INT
        # output: cropped frame 3D ARRAY with INT
        # from: https://www.datacamp.com/tutorial/face-detection-python-opencv, 
        #       https://www.geeksforgeeks.org/python/face-detection-using-cascade-classifier-using-opencv-python/
    if fr.ndim > 2:
        fr_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    elif fr.ndim == 2:
        fr_grey == fr
    else:
        IndexError("Wrong size image, make sure the input array for face detection is 2D or higher")
    
    # detect face
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(
        fr_grey, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # detect eyes
    eye_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_classifier.detectMultiScale(
        fr_grey, scaleFactor=1.2, minNeighbors=5)
    
    return faces, eyes

def draw_box(fr,tup,c=(0,255,0),thickness=4):
    # draws boxes
        # input tup TUPLE (x,y,w,h)
        # input fr 3D ARRAY with INT
        # input c TUPLE (B,G,R)
        # input thickness INT or FLOAT
        # ouput NONE, draws rectangle on frame
    for (x,y,w,h) in tup:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

def overlay_transparent_image(bg,overlay,pos=(0,0),size =(200,150)):
    # puts image over other image, preserving alpha information
        # input bg 3D ARRAY with INT
        # input overlay 4D ARRAY with INT, 4th dim is for alpha
        # input pos TUPLE(1x3) with INT
        # input size TUPLE(1x2) with INT wxh
        # output combined 3D ARRAY with INT
    overlay = cv2.resize(overlay,(size[0],size[1]))
    bg_width,bg_height,bg_depth = bg.shape
    combined = bg.copy()
    overlay[:,:,3] = overlay[:,:,3]/255
    widthl = 0
    widthr = size[0]
    height1 = 0
    # allow hat to move partly off screen
    if pos[1] < size[1]: # height
        # overlay = overlay[size[1]-pos[1]::,:]
        height1 = size[1]-pos[1]

    overlay_im = overlay[height1::,widthl:widthr]
    o_height,o_width,o_depth = overlay_im.shape

    # combine the overlay with background using alpha compositing
    combined[pos[1]-o_height:pos[1],pos[0]:pos[0]+o_width] = \
        bg[pos[1]-o_height:pos[1],pos[0]:pos[0]+o_width]*\
            (np.transpose([[1-overlay_im[:,:,3]]*3][0],(1,2,0)))+\
                np.transpose(overlay_im[:,:,:3],axes=(0,1,2))*np.transpose([overlay_im[:,:,3],overlay_im[:,:,3],overlay_im[:,:,3]],axes=(1,2,0))

    return combined, pos





while True:
    # Read the next frame from the webcam
    status, photo = cap.read()
    frame = cv2.flip(photo,1)

    # face recognition
    faces, eyes = recog_face_feat(frame)

    # overlay image
    for (x,y,w,h) in faces:
        # overlay physics
            frame, currentpos = overlay_transparent_image(frame,hat,pos=[x,y],size=(w,150))
    # draw box around faces
    draw_box(frame,faces)
    draw_box(frame,eyes)

    # cv2.rectangle(frame,(0,0),(10,10),(0,0,255),5)
    cv2.imshow("Press 'q' to close", frame)
    old_frame = frame
    # Wait for 50 milliseconds and check if the 'Enter' key (key code 13) is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()