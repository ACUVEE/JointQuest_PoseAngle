import cv2
import mediapipe as mp
import math
import numpy as np

# 미디어파이프의 Pose 모듈을 사용하기 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 웹캠 초기화
cap = cv2.VideoCapture(0)

def get_mid_coor(landmarks, dot1, dot2):
    x = (landmarks[dot1][1]["x"]+landmarks[dot2][1]["x"])/2
    y = (landmarks[dot1][1]["y"]+landmarks[dot2][1]["y"])/2
    z = (landmarks[dot1][1]["z"]+landmarks[dot2][1]["z"])/2

    return {'x' : x,'y' : y,'z' : z}

# 두점간의 벡터 구하는 함수 vec(dot1 -> dot2)
def get_vector(dot1, dot2):
    x = dot2['x'] - dot1['x']
    y = dot2['y'] - dot1['y']
    z = dot2['z'] - dot1['z']
    return {'x' : x,'y' : y,'z' : z}

# Pose 모델 로드
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation = True) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Pose 검출 수행
        results = pose.process(image)
        
        # BGR로 다시 변환하여 화면에 출력
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 검출된 관절 표시
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(results.pose_landmarks.landmark)
            ]
        if results.pose_world_landmarks:
            world_landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(results.pose_world_landmarks.landmark)
            ]
            
            # 고관절 중앙 좌표
            hip_coor = get_mid_coor(world_landmarks,24,23)
            # 어께 중앙 좌표
            shoulder_coor = get_mid_coor(world_landmarks,11,12)
            # 척추 벡터
            spine_vec = get_vector(hip_coor,shoulder_coor)
            
            print("hip")
            print(hip_coor)
            print("mid_shoulder")
            print(shoulder_coor)
            print("spine_vector")
            print(spine_vec)

        # 화면에 출력
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()