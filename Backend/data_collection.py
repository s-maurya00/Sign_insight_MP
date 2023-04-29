import os
import numpy as np
import cv2

# My utility functions/constants
from utility import *


def start_data_collection():
    """
    Starts the process of collecting training data for the gesture recognition model.

    This function initializes the camera video capture, sets up a directory structure for storing training data for different actions, and loops through each action and video number to record training data for the model. For each video, it captures frames from the camera feed and uses the Mediapipe library to detect and extract keypoints for face, hands, and pose. It then saves each frame as a video and saves the extracted keypoints as a numpy array.

    Args:
        None

    Returns:
        None
    """

    cap = cv2.VideoCapture(0)

    # store the camera video while recording the gestures
    if(cap.isOpened()):
        vid_width = int(cap.get(3))
        vid_height = int(cap.get(4))
    else:
        print("Error reading the video feed")


    # Generate directories for each action
    create_dir()


    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:

        should_break = False

        # Loop through actions
        for action in actions:

            # Loop through each video
            for video_number in range(1, no_of_videos + 1):

                # stop capturing data if user presses 'q'
                if should_break:
                    break

                try:
                    vid_path = str(os.path.join(DATA_PATH, action, str(video_number)))
                    vid_out = cv2.VideoWriter(vid_path + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (vid_width, vid_height))
                except Exception as e:
                    print("Could not save video")

                # Loop through frames within each video
                for frame_num in range(1, no_of_frames_in_a_video + 1):

                    success, frame = cap.read()
                    
                    try:
                        # Store the video frame
                        vid_out.write(frame)
                    except:
                        pass

                    # Detect face, hands, pose
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks on face, hands, pose
                    draw_styled_landmarks(image, results)
                    
                    # Logic for telling user that a new video is being captured and waiting for 1.5 seconds for user to set his pose  
                    if frame_num == 1:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, video_number), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1500)

                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, video_number), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # get the concatinated keypoints after detection 
                    keypoints = extract_keypoints(results)

                    # export keypoints as np.array
                    npy_path = str(os.path.join(DATA_PATH, action, str(video_number), str(frame_num)))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        should_break = True
                        break

        try:                    
            cap.release()
            vid_out.release()
            cv2.destroyAllWindows()
        except:
            pass

if(__name__ == '__main__'):
    start_data_collection()