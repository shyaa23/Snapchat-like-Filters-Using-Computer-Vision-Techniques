import pandas as pd
import numpy as np
import cv2
from cnn_model import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def prepare_data(df):

    '''
    Prepare data (image and target variables) for training

    Parameters:
    --------------------
    df: Training dataframe

    Returns:
    -------------
    X: Image (Feature) column
    y: Target column (Facial points)
    '''

    # Create numpy array for pixel values in image column that are separated by space
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

    # Drop all rows that have missing values in them
    df = df.dropna()  

    # Normalize the pixel values, scale values between 0 and 1
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)
    # return each images as 96 x 96 x 1
    X = X.reshape(-1, 96, 96, 1) 

    y = df[df.columns[:-1]].values #(30 columns)
    # Normalize the target value, scale values between 0 and 1
    y = (y - 48) / 48 
    # shuffle train data 
    X, y = shuffle(X, y, random_state=42)  
    y = y.astype(np.float32)

    return X,y

def plot_data(img, face_points):

    '''
    Plot image and facial keypoints

    Parameters:
    --------------------
    img: Image column value
    face_point: Target column value
    '''

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(121)
    # Plot the image
    ax.imshow(np.squeeze(img), cmap='gray')
    face_points = face_points * 48 + 48 
    # Plot the keypoints
    ax.scatter(face_points[0::2], face_points[1::2], marker='o', c='c', s=10)
    plt.show()
    
# Load training data
df = pd.read_csv('data/training.csv')

X_train, y_train =prepare_data(df)

# Plot image and facial points for train dataset
plot_data(X_train[200], y_train[200])

# Create the model architecture
my_model = create_model()

# Compile the model with an appropriate optimizer and loss and metrics
compile_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Train the model
hist = train_model(my_model, X_train, y_train)

# Save the model
save_model(my_model, 'models/mm')


