import os
import tkinter as tk
from tkinter import filedialog

import cv2
import mediapipe as mp
import numpy as np


# 벡터간 각도 구하는 함수
def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    angle_rad = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    angle_deg = round(np.degrees(angle_rad), 1)
    return angle_deg


def track_joints_and_save(input_file_path):
    # 미디어파이프 pose 모듈을 로드
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 입력 파일 경로에서 파일명과 확장자를 추출
    input_filename, input_file_extension = os.path.splitext(os.path.basename(input_file_path))

    # 출력 폴더를 생성
    output_directory = "./output/"
    os.makedirs(output_directory, exist_ok=True)

    # 출력 동영상 파일 경로를 설정
    output_file_path = os.path.join(output_directory, f"{input_filename}_output.mp4")

    # 입력 파일을 읽어옵니다.
    cap = cv2.VideoCapture(input_file_path)

    # 동영상 프레임 크기와 FPS 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 동영상 writer 생성
    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 미디어파이프 pose 인식을 위한 인스턴스를 생성
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # 프레임을 읽어옴
            ret, frame = cap.read()
            if not ret:
                break

            # 읽어온 프레임을 BGR에서 RGB로 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 미디어파이프 pose 모델을 이용하여 pose를 인식
            results = pose.process(image_rgb)

            # 인식된 pose를 프레임에 그림
            if results.pose_world_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 타겟 관절의 각도 표기
                target_num = 25

                # 텍스트를 표기할 좌표를 위한 스케일링
                landmark_25 = results.pose_landmarks.landmark[target_num]
                x_25 = int(landmark_25.x * frame_width)
                y_25 = int(landmark_25.y * frame_height)

                # 관절 좌표를 추출(소수점 아래 4번째에서 반올림)
                joints = []
                for landmark in results.pose_world_landmarks.landmark:
                    joints.append((round(landmark.x, 3), round(landmark.y, 3), round(landmark.z, 3)))

                # 각도를 구할 타겟 관절과 벡터
                upper_joint = joints[23]
                target_joint = joints[target_num]
                lower_joint = joints[27]
                upper_vec = [upper_joint[0] - target_joint[0],
                             upper_joint[1] - target_joint[1],
                             upper_joint[2] - target_joint[2]]
                lower_vec = [lower_joint[0] - target_joint[0],
                             lower_joint[1] - target_joint[1],
                             lower_joint[2] - target_joint[2]]

                # 텍스트를 이미지 위에 표시
                cv2.putText(frame, f"{angle_between_vectors(upper_vec, lower_vec)}", (x_25, y_25),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 동영상 파일에 프레임 추가
            out.write(frame)

        print(f"Pose detection video saved to: {output_file_path}")

    # 사용한 자원을 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def select_input_file_and_track():
    # 입력 파일 경로를 선택
    root = tk.Tk()
    root.withdraw()
    input_file_path = filedialog.askopenfilename()

    if input_file_path:
        # 선택한 파일로 관절 추적하여 결과를 저장
        track_joints_and_save(input_file_path)
    else:
        print("No file selected.")


# 함수를 호출하여 파일을 선택하고 관절을 추적하여 결과를 저장
select_input_file_and_track()
