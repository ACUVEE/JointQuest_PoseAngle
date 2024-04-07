import cv2
import mediapipe as mp
import math
import numpy as np

# 각도를 계산하는 함수
def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
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

#
def print_angle(landmarks,dot1, dot2, dot3):
    point1 = [landmarks[dot1][1]["x"], landmarks[dot1][1]["y"], landmarks[dot1][1]["z"]]
    point2 = [landmarks[dot2][1]["x"], landmarks[dot2][1]["y"], landmarks[dot2][1]["z"]]
    point3 = [landmarks[dot3][1]["x"], landmarks[dot3][1]["y"], landmarks[dot3][1]["z"]]

    angle = calculate_angle(point1, point2, point3)
    print("두 직선 사이의 각도:", angle)

# 미디어파이프의 Pose 모듈을 사용하기 위한 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 웹캠 초기화
cap = cv2.VideoCapture(1)

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

            print_angle(world_landmarks,11, 13, 15)
            # print_distance(world_landmarks,12, 11)
            # print_landmarks(world_landmarks)

        # 화면에 출력
        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()