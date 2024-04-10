import tkinter as tk
from tkinter import filedialog
import os
import cv2
import mediapipe as mp

def track_joints_and_save(input_file_path):
    # 미디어파이프 pose 모듈을 로드합니다.
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 입력 파일 경로에서 파일명과 확장자를 추출합니다.
    input_filename, input_file_extension = os.path.splitext(os.path.basename(input_file_path))
    
    # 출력 폴더를 생성합니다.
    output_directory = "./output/"
    os.makedirs(output_directory, exist_ok=True)

    # 출력 동영상 파일 경로를 설정합니다.
    output_file_path = os.path.join(output_directory, f"{input_filename}_output.mp4")

    # 입력 파일을 읽어옵니다.
    cap = cv2.VideoCapture(input_file_path)

    # 동영상 프레임 크기와 FPS 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 동영상 writer 생성
    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 미디어파이프 pose 인식을 위한 인스턴스를 생성합니다.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # 프레임을 읽어옵니다.
            ret, frame = cap.read()
            if not ret:
                break

            # 읽어온 프레임을 BGR에서 RGB로 변환합니다.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 미디어파이프 pose 모델을 이용하여 pose를 인식합니다.
            results = pose.process(image_rgb)

            # 인식된 pose를 프레임에 그립니다.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                landmark_25 = results.pose_landmarks.landmark[25]
                x_25 = int(landmark_25.x * frame_width)
                y_25 = int(landmark_25.y * frame_height)
                # 텍스트를 이미지 위에 표시합니다.
                cv2.putText(frame, "25th joint", (x_25, y_25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 동영상 파일에 프레임을 추가합니다.
            out.write(frame)

        print(f"Pose detection video saved to: {output_file_path}")

    # 사용한 자원을 해제합니다.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def select_input_file_and_track():
    # 파일 대화상자를 통해 입력 파일 경로를 선택합니다.
    root = tk.Tk()
    root.withdraw()  # Tk root 창을 숨깁니다.
    input_file_path = filedialog.askopenfilename()

    if input_file_path:
        # 선택한 파일로 관절 추적하여 결과를 저장합니다.
        track_joints_and_save(input_file_path)
    else:
        print("No file selected.")

# 함수를 호출하여 파일을 선택하고 관절을 추적하여 결과를 저장합니다.
select_input_file_and_track()