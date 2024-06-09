import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# 통신 설정
# port는 본인 컴퓨터에 맞는 COM으로 설정하면 됨, buadrate는 아두이노와 같게 설정
py_serial=serial.Serial(

    port='COM3',
    baudrate=9600,
)

# mediapipe 각종 변수들 선언
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 카메라 화면 출력
cap = cv2.VideoCapture(0)
cap.set(3, 680)
cap.set(4, 480)


#세 개의 점 (a, b, c)을 받아들여, b를 중심으로 하는 두 벡터 사이의 각도를 계산
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # 각도를 음수가 아닌 양수로 변환
    angle = np.abs(radians * 180.0 / np.pi)
    
    #각도 0~180사이값으로 만듬
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

    
# 카메라가 켜졌을 때 while문 실행
while cap.isOpened():
    ret, frame = cap.read()
    
    # RGB 이미지로 변환하기
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  
    
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # 각 관절의 위치점 찾기
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
        hip1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        hip2 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


        shoulder_pos = tuple(np.multiply(shoulder, [640, 480]).astype(int))
        elbow_pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
        wrist_pos = tuple(np.multiply(wrist, [640, 480]).astype(int))
        index_pos = tuple(np.multiply(index, [640, 480]).astype(int))
        hip1_pos = tuple(np.multiply(hip1, [640, 480]).astype(int))
        hip2_pos = tuple(np.multiply(hip2, [640, 480]).astype(int))

        # 포인트 비주얼 처리
        cv2.circle(image, shoulder_pos, 10, (255,0,0), -1)
        cv2.circle(image, elbow_pos, 10, (0,255,0), -1)
        cv2.circle(image, wrist_pos, 10, (0,0,255), -1)  
        cv2.circle(image, index_pos, 10, (0,255,255), -1)
        cv2.circle(image, hip1_pos, 10, (255,255,0), -1) 
        cv2.circle(image, hip2_pos, 10, (255,255,0), -1)            
        cv2.line(image, shoulder_pos, elbow_pos,   (255,255,0), 3)
        cv2.line(image, elbow_pos, wrist_pos, (0,255,255), 3)
        cv2.line(image, wrist_pos, index_pos, (0,255,255), 3)
        cv2.line(image, hip1_pos, shoulder_pos, (0,255,255), 3)
        cv2.line(image, hip1_pos, hip2_pos, (0,255,255), 3)


        # 각도 계산하기
        angle = calculate_angle(shoulder, elbow, wrist) # upperarm
        angle1 = calculate_angle(elbow, wrist, index) # forearm
        angle2 = calculate_angle(hip1, shoulder, elbow) # shoulder
        angle3 = calculate_angle(hip2, hip1, shoulder) # base

        # 각도 표시하기
        cv2.putText(image, f'Angle: {round(angle, 2)}', tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'Angle1: {round(angle1, 2)}', tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f'Angle2: {round(angle2, 2)}', tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, f'Angle3: {round(angle3, 2)}', tuple(np.multiply(hip1, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # print(counter)
    
        cv2.putText(image, f'SimpleKalmanFilter(2, 2, 0.01)', 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        #아두이노로 각각의 angle값 byte형으로 전송
        py_serial.write(int(angle).to_bytes(1,'big'))
        py_serial.write(int(angle1).to_bytes(1,'big'))
        py_serial.write(int(angle2).to_bytes(1,'big'))
        py_serial.write(int(angle3).to_bytes(1,'big'))

    # 디버깅을 위해 아두이노와의 통신이 끊어졌을때 문구 출력
    except Exception as e:
        print(f"Error in read_arduino: {e}")
        py_serial.close()

    cv2.imshow('Mediapipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()