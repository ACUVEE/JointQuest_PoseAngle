import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tkinter import filedialog

def visualize_3d_joints(landmarks, elev_val, azim_val):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # 좌표계 세팅
    ax.view_init(elev=elev_val, azim=azim_val)
    ax.set_xlabel('z Label')
    ax.set_ylabel('x Label')
    ax.set_zlabel('y Label')
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.7])

    # 관절 좌표를 추출
    joints = []
    for landmark in landmarks.landmark:
        joints.append((landmark.x, landmark.y, landmark.z))
    
    # 각 관절 좌표 표시
    xs, ys, zs = zip(*joints)
    ax.scatter(zs, xs, ys, c='r', marker='o')

    # 각 관절을 잇는 선분 그림
    for connection in mp_pose.POSE_CONNECTIONS:
        joint1, joint2 = connection
        x = [joints[joint1][0], joints[joint2][0]]
        y = [joints[joint1][1], joints[joint2][1]]
        z = [joints[joint1][2], joints[joint2][2]]
        ax.plot(z, x, y, color='g')

    plt.show()
    plt.close()

# 미디어파이프 pose 모듈 로드
mp_pose = mp.solutions.pose

# 입력 파일을 읽기
file_path = filedialog.askopenfilename()
image = cv2.imread(file_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 미디어파이프 pose 인식을 위한 인스턴스를 생성
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    elev_val = -170
    azim_val = 40
    if image is not None:

        # 미디어파이프 pose 모델을 이용하여 pose 인식
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 인식된 pose를 3D 그래프로 시각화
        if results.pose_landmarks:
            visualize_3d_joints(results.pose_world_landmarks, elev_val, azim_val)