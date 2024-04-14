import cv2
import mediapipe as mp
import math
import numpy as np
from physical_measure import P_vec
# 벡터간 각도 구하는 함수
def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    angle_rad = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    angle_deg = round(np.degrees(angle_rad),1)
    return angle_deg

# 두 점 사이의 거리
def distance_between_points(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

#
def print_landmarks(landmarks):
    for landmark in landmarks:
            if landmark[0] == 16 :
                if landmark[1]["visibility"] >= 0.5:
                    print(landmark)

#
def print_distance(landmarks,dot1, dot2):
    x1, y1, z1 = [landmarks[dot1][1]["x"], landmarks[dot1][1]["y"], landmarks[dot1][1]["z"]]
    x2 = landmarks[dot2][1]["x"]
    y2 = landmarks[dot2][1]["y"]
    z2 = landmarks[dot2][1]["z"]
    print(distance_between_points(x1, y1, z1, x2, y2, z2))

# 미디어파이프의 Pose 모듈을 사용하기 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 웹캠 초기화
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

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
            
            # 관절 좌표를 추출(소수점 아래 4번째에서 반올림)
            joints = []
            for landmark in results.pose_world_landmarks.landmark:
                joints.append((round(landmark.x, 3), round(landmark.y,3), round(landmark.z,3)))

            # 타겟 관절의 각도 표기
            target_num = 14
            
            # 텍스트를 표기할 좌표를 위한 스케일링
            landmark_25 = results.pose_landmarks.landmark[target_num]
            x_25 = int(landmark_25.x * frame_width)
            y_25 = int(landmark_25.y * frame_height)

            # 각도를 구할 타겟 관절과 벡터
            upper_joint = joints[12]
            target_joint = joints[target_num]
            lower_joint = joints[16]
            upper_vec = [upper_joint[0] - target_joint[0], upper_joint[1] - target_joint[1], upper_joint[2] - target_joint[2]]
            lower_vec = [lower_joint[0] - target_joint[0], lower_joint[1] - target_joint[1], lower_joint[2] - target_joint[2]]

            # 텍스트를 이미지 위에 표시
            cv2.putText(image, f"{angle_between_vectors(upper_vec,lower_vec)}", (x_25, y_25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 2)

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