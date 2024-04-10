import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tkinter import filedialog

# 벡터간 각도 구하는 함수
def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    angle_rad = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

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

    # 관절 좌표를 추출(소수점 아래 4번째에서 반올림)
    joints = []
    for landmark in landmarks.landmark:
        joints.append((round(landmark.x, 3), round(landmark.y,3), round(landmark.z,3)))
    
    # 각도를 구할 타겟 관절과 벡터
    upper_joint = joints[24]
    target_joint = joints[26]
    lower_joint = joints[28]
    upper_vec = [upper_joint[0] - target_joint[0], upper_joint[1] - target_joint[1], upper_joint[2] - target_joint[2]]
    lower_vec = [lower_joint[0] - target_joint[0], lower_joint[1] - target_joint[1], lower_joint[2] - target_joint[2]]

    # 각 관절 좌표 표시 (plot의 카메라 각도 문제를 해결하기 위해 x -> z, y -> x, z -> y 값을 넣어줌)
    xs, ys, zs = zip(*joints)
    ax.scatter(zs, xs, ys, c='r', marker='o')
    # 타겟 관절의 각도 표기
    ax.text(target_joint[2],target_joint[0],target_joint[1],f"({angle_between_vectors(upper_vec, lower_vec)})",fontsize = 10, ha = 'center',va = 'center')

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
        if results.pose_world_landmarks:
            visualize_3d_joints(results.pose_world_landmarks, elev_val, azim_val)