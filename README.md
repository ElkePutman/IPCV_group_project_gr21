# IPCV Group project

This project is the **group assignment** for the *Image Processing & Computer Vision (IPCV)* course.  
It consists of three independent Python programs, each implementing a different exercise of the assignment using OpenCV and MediaPipe.

---

## Overview

### `face_augmentation.py`
Implements the **face augmentation** exercise.  
The script detects a person’s face in real time using a webcam and overlays a hat image on top of the head.

### `warp.py`
Implements the **face warp** exercise.  
It detects the person’s face and applies a swirl effect to distort it using the `skimage.transform.swirl` function.

### `motion.py`
Implements the **motion tracking & interaction** exercise.  
This script detects hand gestures using MediaPipe and triggers different actions:
- **Closed fist** → moves the hat.  
- **Thumb and pinky extended** → applies a swirl (warp) effect to the face.  
- **Peace sign near the mouth** → overlays a cigarimage between the fingers.

---

## Requirements

- **Python:** 3.8 or higher  
- **Required packages:**
  - `opencv-python`
  - `numpy`
  - `matplotlib`
  - `mediapipe`
  - `scikit-image`
  - `os`

## Usage
To run the individual exercises, execute one of the following commands from the terminal:
```bash
# Run the face augmentation exercise
python face_augmentation.py

# Run the face warp exercise
python warp.py

# Run the motion tracking & interaction exercise
python motion.py
```
Press `q` to close the camera. 


## File structure
The expected file structure is shown below:

```bash
IPCV_Group_project_group_21/
├── face_augmentation.py
├── warp.py
├── motion.py
├── Cigar.png
├── hat.png
└── README.md



