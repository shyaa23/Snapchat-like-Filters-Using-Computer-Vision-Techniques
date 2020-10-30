import cv2
import time 
import numpy as np
from cnn_model import *

def apply_filters(face_points, image_copy_1,image_name):

    '''
    Apply animal filters to a person's face

    Parameters:
    --------------------
    face_points: The predicted facial keypoints from the camera
    image_copy_1: Copy of original image

    Returns:
    -------------
    image_copy_1: Animals filters applied to copy of original image
    '''

    animal_filter = cv2.imread("images/"+image_name, cv2.IMREAD_UNCHANGED)

    for i in range(len(face_points)):
        # Get the width of filter depending on left and right eye brow point
        # Adjust the size of the filter slightly above eyebrow points 
        filter_width = 1.1*(face_points[i][14]+15 - face_points[i][18]+15)
        scale_factor = filter_width/animal_filter.shape[1]
        sg = cv2.resize(animal_filter,None, fx=scale_factor, fy = scale_factor, interpolation=cv2.INTER_AREA)
        
        width = sg.shape[1]
        height = sg.shape[0]
        
        # top left corner of animal_filter: x coordinate = average x coordinate of eyes - width/2
        # y coordinate = average y coordinate of eyes - height/2
        x1 = int((face_points[i][2]+5 + face_points[i][0]+5)/2 - width/2)
        x2 = x1 + width

        y1 = int((face_points[i][3]-65 + face_points[i][1]-65)/2 - height/3)
        y2 = y1 + height

        # Create an alpha mask based on the transparency values
        alpha_fil = np.expand_dims(sg[:, :, 3]/255.0, axis=-1)
        alpha_face = 1.0 - alpha_fil
        
        # Take a weighted sum of the image and the animal filter using the alpha values and (1- alpha)
        image_copy_1[y1:y2, x1:x2] = (alpha_fil * sg[:, :, :3] + alpha_face * image_copy_1[y1:y2, x1:x2])
    
    return image_copy_1

# Load the model built in the previous step
model = load_model('models/final_model')

# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Get webcam
camera = cv2.VideoCapture(0)

while True:
    # Read data from the webcam
    _, image = camera.read() 
    image_copy = np.copy(image)
    image_copy_1 = np.copy(image)
    image_copy_2 = np.copy(image)      
    
    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    # Identify faces in the webcam using haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
    faces_keypoints = []
    
    # Loop through faces
    for (x,y,w,h) in faces:
        
        # Crop Faces
        face = gray[y:y+h, x:x+w]
     
        # Scale Faces to 96x96
        scaled_face = cv2.resize(face, (96,96), 0, 0, interpolation=cv2.INTER_AREA)

        # Normalize images to be between 0 and 1
        input_image = scaled_face / 255

        # Format image to be the correct shape for the model
        input_image = np.expand_dims(input_image, axis = 0)
        input_image = np.expand_dims(input_image, axis = -1)

        # Use model to predict keypoints on image
        face_points = model.predict(input_image)[0]

        # Adjust keypoints to coordinates of original image
        face_points[0::2] = face_points[0::2] * w/2 + w/2 + x
        face_points[1::2] = face_points[1::2] * h/2 + h/2 + y
        faces_keypoints.append(face_points)
        
        # Plot facial keypoints on image
        for point in range(15):
            cv2.circle(image_copy, (face_points[2*point], face_points[2*point + 1]), 2, (255, 255, 0), -1)

        cat = apply_filters(faces_keypoints, image_copy_1,"cat.png")
        dog = apply_filters(faces_keypoints, image_copy_2,"custom3.png")
        
        # Screen with the filter
        cv2.imshow('Screen with filter',cat)  
        cv2.imshow('Screen with filter dog',dog)        
        # Screen with facial keypoints   
        cv2.imshow('Screen with facial Keypoints predicted',image_copy)        
           
    if cv2.waitKey(1) & 0xFF == ord("q"):   
        break
   
