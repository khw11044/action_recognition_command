import cv2
import requests
import numpy as np
import threading
import queue
import socket
import time 
import os 

from actiontools.network import get_model
from actiontools.mediapipetools import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

threshold = 0.90
actions = np.array(['nothing', 'ready', 'stop'])
sequence_length = 15
model = get_model(actions, sequence_length, 88)
model.load_weights('action.h5')
# 1. New detection variables
sequence = []
sentence = []

frame_save_folder = './frames'
if not os.path.exists(frame_save_folder):
    os.makedirs(frame_save_folder)

# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_feed"

# 모바일 로봇의 IP 주소와 포트 설정
ROBOT_IP = "192.168.0.123"  # 모바일 로봇의 IP 주소 (변경 필요)
ROBOT_PORT = 5000  # 모바일 로봇의 수신 포트


# 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정


frame_queue = queue.Queue(maxsize=5)  # 프레임 큐

running = True
cur_action = 'nothing'
frame_counter = 1  # 저장된 이미지의 파일 이름에 사용할 카운터
start_time = None  # 'ready' 상태 시작 시간을 저장

# 실시간 스트림을 수신하는 스레드
def stream_frames():
    global running
    print("스트리밍 시작...")
    stream = requests.get(url, stream=True, timeout=5)
    if stream.status_code == 200:
        byte_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
            if not running:
                break
            byte_data += chunk
            a = byte_data.find(b'\xff\xd8')
            b = byte_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = byte_data[a:b+2]
                byte_data = byte_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                if not frame_queue.full():
                    frame_queue.put(frame)
    else:
        print(f"스트리밍 실패: {stream.status_code}")



# 2D HPE and Action Recognition 
def process_frames():
    global running, sequence, sentence, cur_action, frame_counter, start_time
    print("프레임 처리 시작...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                height, width = frame.shape[:2]
                

                frame, results = mediapipe_detection(frame, holistic)
                # draw_styled_landmarks(frame, results, mp_drawing, mp_holistic)
                keypoints = extract_keypoints(results)
                
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
                
                if len(sequence) == sequence_length and cur_action != 'ready':
                    try:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=None)[0]
                        select = np.argmax(res)
                        cur_action = actions[select]
                        print('0. cur_action: ', cur_action)
                    except:
                        sequence = []
                        continue
                    
            
                    # Viz probabilities
                    frame = prob_viz(res, actions, frame)
        

                if cur_action == 'ready':
                    # 'ready' 상태일 때 1분 동안 프레임 저장
                    if start_time is None:
                        start_time = time.time()  # 'ready' 상태 시작 시간 기록
                        print('1. 촬영 하겠습니다')

                    elapsed_time = time.time() - start_time
                    if elapsed_time < 10:  # 10초 동안
                        # 프레임을 폴더에 저장
                        frame_filename = os.path.join(frame_save_folder, f'{frame_counter:04d}.jpg')
                        cv2.imwrite(frame_filename, frame)
                        frame_counter += 1  # 다음 파일 이름으로 카운터 증가
                        print(f'2. 촬영 중 : {elapsed_time} : {frame_counter}')
                        time.sleep(0.01)
                    else:
                        # 10초가 지나면 'ready' 상태를 종료
                        cur_action = 'nothing'
                        start_time = None  # 'ready' 상태 종료 시 시간 초기화
                        print('3. 촬영이 끝났습니다.')
                    

                
                # 최종 프레임 출력
                cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                
                cv2.imshow("Action", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break


# 스레드 시작
stream_thread = threading.Thread(target=stream_frames)
process_thread = threading.Thread(target=process_frames)

stream_thread.start()
process_thread.start()


# 스레드 종료 대기
stream_thread.join()
process_thread.join()


# 리소스 정리
cv2.destroyAllWindows()
print("프로그램 종료.")
