import cv2 
import mediapipe as mp
import numpy as np 
import matplotlib.pyplot as plt

colors = [(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output_frame, f"{int(prob*100)}", (130, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
    return output_frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for idx, res in enumerate(results.pose_landmarks.landmark) if idx>10]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose


# 함수로 만들어서 사용합시다.
def dl_history_plot(history):
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.plot(history['loss'], label='loss', marker = '.')
    plt.plot(history['val_loss'], label='val_loss', marker = '.')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(history['acc'], label='acc', marker = '.')
    plt.plot(history['val_acc'], label='val_acc', marker = '.')
    plt.ylabel('ACC')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('training_process.png')

    # plt.show()