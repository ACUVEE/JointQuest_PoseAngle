import cv2
import mediapipe as mp
import math
import numpy as np
from physical_vector import P_vec
import json

# 미디어파이프의 Pose 모듈을 사용하기 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 웹캠 초기화
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 바디크기 json
with open('./2024_04_15.json', 'r') as json_file:
    body_size = json.load(json_file)

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
        
        if results.pose_world_landmarks:
            # 라벨을 붙여서 JSON 형식으로 변환
            landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(results.pose_world_landmarks.landmark)
            ]
            landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}

            # P_vec 클래스 할당
            p_vec = P_vec(landmarks_dict)

            # 벡터 z 값 조정
            # p_vec.correct_vec(body_size,'l_side')
            # p_vec.correct_vec(body_size,'l_u_leg')

            # 텍스트를 표기할 타겟 관절
            target_nums = [12, 14]

            # for target_num in target_nums:

            #     # 텍스트를 표기할 좌표를 위한 스케일링
            #     target_landmark = results.pose_landmarks.landmark[target_num]
            #     x = int(target_landmark.x * frame_width)
            #     y = int(target_landmark.y * frame_height)
                
            #     # 표기할 텍스트(좌표 값)
            #     text = str(int(round(landmarks_dict[f"landmark_{target_num}"]['x'],2)*100))+","+str(int(round(landmarks_dict[f"landmark_{target_num}"]['y'],2)*100))+","+str(int(round(landmarks_dict[f"landmark_{target_num}"]['z'],2)*100))
                
            #     # 텍스트를 이미지 위에 표시
            #     cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 255, 0), 2)

            target_num = 23
            target_landmark = results.pose_landmarks.landmark[target_num]
            x = int(target_landmark.x * frame_width)
            y = int(target_landmark.y * frame_height)

            # 표기할 텍스트(좌표 값)
            text = str(p_vec.angle_between_vectors('l_side','l_u_leg'))
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 255, 0), 2)

            


        if results.pose_world_landmarks:
            world_landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(results.pose_world_landmarks.landmark)
            ]

        # 화면에 출력
        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()