# Sign Insight

Sign Insight is a program for detecting hand actions in Indian Sign Language (ISL) using a set of frames for detection of an action, rather than a single frame. This program generates output in the form of captions on live video.

## STEPS for running the program:

1. Clone the repository to your local machine
2. Create a virtual environment and activate it
3. Install the necessary dependencies by running `pip install -r requirements.txt` in your terminal or command prompt
4. Open the terminal or command prompt in the project directory
5. Run the app.py file using the command `python app.py`

Note: The utility.py file contains all the utility functions used by the program, such as functions for data collection, preprocessing, and training the model.


## About the files in the program:

<hr/>

### A. For Data Collection (data_collection.py)
- It creates directories for storing data as per the actions array defined in the utility.py file
- It loops through the total number of videos that are to be captured for each action and stores the videos as .avi files and each frame of each video in a numbered folder in .npy format

<hr/>

### B. For Preprocessing the collected data (data_preprocessing.py)
- It creates labels for each action listed in the utility.py file
- It loads each frame from folder 'Data/action_name/video_number' eg. 'Data/hello/1' in a single numpy array containing all videos data
- It creates a numpy array for storing all the labels in categorical format
- It performs splitting of data and label sets into testing and training parts

<hr/>

### C. For training the model on collected data (model_trainer.py)
- It defines a neural network model
- It compiles the model
- It trains the model on the X_train and y_train dataset
- It saves the architecture of the model in action_recog.json file format and the model itself in action_recog.h5 file format
- It tests the model on the training dataset


<hr/>

### D. For testing the model on collected data (model_tester.py)
- It predicts the output on training dataset
- It calculates and prints multilabel_confusion_matrix
- It calculates the accuracy score of the trained model


<hr/>

### E. For getting utility functions (utility.py)
- It stores constants like path to data files, actions on which model is to be trained, total videos for each actions, total number of frames for each video for each action, path to log files generated while training the model
- It has different functions like:
    - create_dir()
    - mediapipe_detection(image, model)
    - draw_styled_landmarks(image, results)
    - extract_keypoints(results)
    - create_model()
    - compile_model(model)
    - train_model(model, X_train, y_train)

## License

This project is licensed under the MIT License. See the LICENSE file for more information.