import cv2

# My utility functions/constants
from utility import *

# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 140, 0)]
# colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame):

    colors = [(0, 0, 255), (255, 0, 255)]

    output_frame = input_frame.copy()

    max_prob_idx = np.argmax(res)
    second_max_prob_idx = np.argsort(res)[-2]

    max_prob = res[max_prob_idx]
    second_max_prob = res[second_max_prob_idx]

    # Draw rectangle for max and second max action
    cv2.rectangle(output_frame, (0,60), (int(max_prob*100), 90), colors[0], -1)
    cv2.rectangle(output_frame, (0,100), (int(second_max_prob*100), 130), colors[1], -1)

    # Draw action labels for max and second max action
    cv2.putText(output_frame, actions[max_prob_idx], (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.putText(output_frame, actions[second_max_prob_idx], (0, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    return output_frame


def model_predictor(model):

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.9

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        while cap.isOpened():

            # Read feed
            success, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            # 3. Visualization logic
                if np.unique(predictions[-5:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if (__name__ == '__main__'):

    model = create_model()

    compile_model(model)

    model.load_weights('./Model/my_model_KJSCE.h5')

    model_predictor(model)