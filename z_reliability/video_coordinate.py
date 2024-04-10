import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tkinter import filedialog
import os

def process():
    # 미디어파이프 pose 모듈을 로드합니다.
    mp_pose = mp.solutions.pose

    # 입력할 파일 요청
    input_file_path = filedialog.askopenfilename()
    file_name, file_extension = os.path.splitext(os.path.basename(input_file_path))
    if(file_extension != ".mp4"):
        print("extension error")
        return 0
    # 출력될 mp4 경로
    output_path = "./output/"+file_name+".mp4"
    
    # 입력 파일 읽기
    cap = cv2.VideoCapture(input_file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 영상의 관절을 추적여 프레임별 png 파일 저장
    track_joint(mp_pose, cap)

    # png파일로 mp4 영상화
    image_path = './output/temp/'
    png2mp4(output_path,image_path ,fps)
    
    # 릴리즈

def track_joint(mp_pose,cap):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        # 카메라 angle setting
        elev_val = -160
        azim_val = 40
        # 임시 이미지 index
        index = 0

        while cap.isOpened():
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 미디어파이프 pose 모델을 이용하여 pose 인식
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 인식된 pose를 3D 그래프로 시각화
            if results.pose_world_landmarks:
                visualize_3d_joints(results.pose_world_landmarks, elev_val, azim_val, index, mp_pose)
                index += 1
                azim_val += 2
        
        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()

def visualize_3d_joints( landmarks, elev_val, azim_val, index, mp_pose):
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

    # 각 관절 좌표 표시
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

    # 임시 이미지 경로
    tmp_path = './output/temp/'+str(index)+'.png'

    # 그래프 이미지 파일 임시 저장
    plt.savefig(tmp_path)
    plt.close(tmp_path)

# 벡터간 각도 구하는 함수
def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    angle_rad = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    angle_deg = round (np.degrees(angle_rad),1)
    return angle_deg

# 파일 정렬을 위한 자연 정렬 키 생성 함수
def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# png 파일셋으로 mp4파일 생성
def png2mp4(output_path,image_folder, fps):
    # 이미지 폴더의 파일들 리스트화
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key = natural_sort_key)
    # 첫번째 이미지의 넓이 높이 추출
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # writer 모듈 로드
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 이미지를 비디오 프레임에 삽입
    for image in images:
        png_path = os.path.join(image_folder, image)
        video.write(cv2.imread(png_path))
        # 삽입한 이미지는 삭제
        os.remove(png_path)

    cv2.destroyAllWindows()
    video.release()
#####################################################

process()