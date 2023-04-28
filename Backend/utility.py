import os
import cv2
import numpy as np
import mediapipe as mp

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Constants
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data') 

# Actions that we try to detect
actions = np.array(['hello', 'mynameis', 'thankyou', 'C', 'E', 'J', 'K', 'S'])

# Thirty videos worth of data
# no_sequences = 30
no_of_videos = 30

# Videos are going to be 30 frames in length
# sequence_length = 30
no_of_frames_in_a_video = 30


# directory for storing logs while training the data
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


def create_dir():
    """
    Creates directories for each action and each video in that action, within a specified range. 

    The function creates a directory for each action in the `actions` array and numbered folders inside each action folder for the 
    number of videos specified in `no_of_videos`.

    Parameters:
        None

    Returns:
        None

    Note:
        The `actions` array stores names of different actions that the deep learning model will be trained on. The`no_of_videos` 
        variable is the number of videos for each action that will be used for training the model.
    """

    for action in actions:
        for curr_video in range(1, no_of_videos + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(curr_video)))
            except:
                print("\nError while generating directories for actions.")


def mediapipe_detection(image, model):
    """
    Detects the face, pose, right and left hand landmarks in a video frame using the Mediapipe Holistic model.
    
    Args:
        image: Input video frame in BGR format.
        model (Mediapipe Holistic model): Trained model for detection.
    
    Returns:
        tuple (numpy array): A tuple containing the processed image in BGR format and a results object with detected data about the face, pose, and hand landmarks.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    """
    Draws the detected landmarks on the original video frame using different colors and styles for face, pose, and hands.

    Args:
        image: Input video frame in BGR format.
        results (Mediapipe results object): Results object containing detected data about the face, pose, and hand landmarks.

    Returns:
        None: The function does not return anything. It modifies the input image directly.
    """

    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    """
    This function takes the `results` object returned by the `mediapipe_detection` function and extracts the keypoint coordinates of the detected face, pose, and hand landmarks. The keypoint coordinates are flattened and concatenated into a numpy array.

    Parameters:
    - results: The results object from `mediapipe_detection` function.

    Returns:
    - numpy.ndarray: A flattened and concatenated numpy array of keypoint coordinates, where the first 132 elements correspond to pose landmarks, the next 1404 elements correspond to face landmarks, the next 63 elements correspond to left-hand landmarks, and the last 63 elements correspond to right-hand landmarks. If a particular landmark (face or hand) is not detected, the corresponding part is filled with a fixed size array of zeros.

    Note: 
    - The dimensions of the returned array are fixed, regardless of the number of detected keypoints in each part.
    """

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def create_model():
    """
    Creates a deep learning model for hand action recognition using LSTM and dense layers.

    Returns:
    -------
    model : keras Sequential object
        The deep learning model for hand action recognition.
    """

    # defining the model
    model = Sequential()
    model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape = (30, 1662)))
    model.add(LSTM(128, return_sequences = True, activation = 'relu'))
    model.add(LSTM(64, return_sequences = False, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(actions.shape[0], activation = 'softmax'))

    return model


def compile_model(model):
    """
    Compiles the deep learning model for hand action recognition.

    Parameters:
    ----------
    model : keras Sequential object
        The deep learning model to be compiled.

    Returns:
    -------
    None
    """

    # compile the defined model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def train_model(model, X_train, y_train):
    """
    Trains the deep learning model for hand action recognition using the training data and labels.

    Parameters:
    ----------
    model : keras Sequential object
        The deep learning model to be trained.
    X_train : numpy.ndarray
        The training data.
    y_train : numpy.ndarray
        The training labels.
    tb_callback : keras.callbacks.TensorBoard object
        The TensorBoard callback object for logging training data.

    Returns:
    -------
    None
    """
    

    # fit the defined model the the training data and labels
    model.fit(X_train, y_train, epochs = 2000, callbacks = [tb_callback])

    # get summary of the trained model
    # model.summary()