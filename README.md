
## STEPS for training the model:

<hr/>

### A. For Data Collection
#### 1. Run the data_collection.py file
#### - It creates directores for storing data as per the actions array defined in the utility.py file
#### - It loops through total number of videos that are to be captured for each action and stores the videos as .avi file and each frame of each video in a numbered folder in .npy format

<hr/>

### B. For Preprocessing the collected data
#### 1. Run the data_preprocessing.py file
#### - It creates lables for each action in listed in the utility.py file
#### - It loads each frame from folder 'Data/action_name/video_number' eg. 'Data/hello/1' in a single numpy array containing all videos data
#### - It creates a numpy array for storing all the lables in categorical format
#### - It preforms splitting of data and labels sets into testing and training parts

<hr/>

### C. For training the model on collected data
#### 1. Run the model_trainer.py file
#### - It defines a neural network model
#### - It compiles the model
#### - It trains the model on the X_train and y_train dataset
#### - It saves the architecture of model in action_recog.json file format and the model itself in action_recog.h5 file format
#### - It tests the model on the training dataset



## STEPS for using the model for prediction:

<hr/>

### C. For training the model on collected data
#### 1. Run the model_trainer.py file
#### - It defines a neural network model
#### - It compiles the model
#### - It trains the model on the X_train and y_train dataset
#### - It saves the architecture of model in action_recog.json file format and the model itself in action_recog.h5 file format
#### - It tests the model on the training dataset
