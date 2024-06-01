import json
import math
import socket
import sys
import threading
import time
from typing import Dict, List

import numpy as np

from data_form import data_form
from poseprocessor import PoseProcessor
from webcam_handler import WebcamHandler

"""
신체측정자세,준비자세 데이터 수정 및 경로 분리
웹캠 변경 명령(자원 해제 코드도 필요함) 
"""

"""
송신 데이터
Json
{
    "image":None,
    "angle": dict(Json),           {"hip_angle": 20, "knee_angle": 10, "hip_horizon_angle": 12}
    "incorrect_joint": List<str>,
    "body_length": dict(json),
    "error": str
}

수신 데이터
Json
{
    "command":
}
"""

# 상수
WEBCAM_INDEX = 1
MAX_CONNECTION_ATTEMPTS = 5
CONNECTION_TEST_SLEEP = 1  # in seconds
# POSE_PATH = "output/20240415_031753.json"
POSE_PATH = "data/pose_data/measuring_pose.json"
IDLE_CHECK_NODES = ["11", "12", "13", "14", "15", "16", "23", "24", "25", "26", "27", "28"]
CHECK_NODES = ["11", "12"]
MARGIN = 0.1
OUTLIER_RATIO = 0.1

class ClientSocket:
    def __init__(self, ip: str, port: int):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        # 스레드 관리
        self.shutdown_thread_event = threading.Event()
        self.shutdown_system_event = threading.Event()
        self.threads = []
        self.data_form = data_form.copy()
        self.webcam_handler = WebcamHandler(WEBCAM_INDEX)
        # 소켓 연결
        self.connection_attempts = 0
        self.sock = None
        self.connect_server()

    # 서버 연결 펑션
    def connect_server(self):
        while self.connection_attempts < MAX_CONNECTION_ATTEMPTS:
            try:
                # 연결시도
                self.sock = socket.socket()
                self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
                print(f"Connected to {self.TCP_SERVER_IP}:{self.TCP_SERVER_PORT}")
                # 서버로부터 커맨드 입력 대기
                self.receive_commands()
            except Exception as e:
                print(e)
                self.connection_attempts += 1
                print(f"Attempt {self.connection_attempts} failed.")
        # 연결 시도 5회 넘어가면 프로세스 종료
        print(f"Failed to connect after {MAX_CONNECTION_ATTEMPTS}. Exiting program.")
        

    def send_data(self, data: Dict, print_msg: bool = True):
        """
        데이터 전송 함수
        
        parameters:
            data(dict): 전송할 데이터
        """
        json_data = json.dumps(data)
        length = str(len(json_data))

        self.sock.sendall(length.encode("utf-8").ljust(64))
        self.sock.send(json_data.encode("utf-8"))

    def send_error(self, err_msg: str, print_msg: bool = True):
        """
        에러를 서버에게 전달하는 함수

        parameters:
            err_msg(str): 발생한 에러의 메시지
        """
        if print_msg:
            print(err_msg)
        data = self.data_form.copy()
        data["error"] = err_msg
        json_data = json.dumps(data)
        length = str(len(json_data))

        self.sock.sendall(length.encode("utf-8").ljust(64))
        self.sock.send(json_data.encode("utf-8"))

    def receive_commands(self):
        """
        서버로부터 요청을 받고 별도의 쓰레드를 수행시키는 함수

        요청 목록:
            ‘shutdown_connection’ - 소켓 클라이언트 프로세스 종료
            ‘stop_thread’ - 현재 실행중인 서브 쓰레드 모두 종료
            ‘connection_test’ - 1씩 증가하는 angle 값을 가진 데이터를 1초에 한번씩 전송
            ‘measuring_body’ - 길이측정
            4. 무릎 운동
        """
        self.webcam_handler.setup_webcam()
        try:
            while not self.shutdown_system_event.is_set():
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    break
                json_data = json.loads(data)
                command = json_data["command"]
                self.handle_command(command, json_data)
        except Exception as e:
            self.send_error(f"Error in receiving command: {e}")
        finally:
            self.cleanup()
            sys.exit()

    def handle_command(self, command: str, json_data: Dict):
        if command == "shutdown_connection":
            self.shutdown_system_event.set()
        elif command == "stop_thread":
            self.stop_threads()
        elif command == "connection_test":
            self.start_thread(self.connection_test)
        elif command == "measuring_body":
            self.start_thread(self.measure_body_length)
        elif command == "knee_game":
            if json_data["body_length"]:
                body_length = json_data["body_length"]
                self.start_thread(self.knee_game, body_length)
            else:
                self.start_thread(self.knee_game)
        else:
            raise Exception("Wrong command")

    def start_thread(self, target, *args):
        """
        특정 함수를 서브 쓰레드로 수행시키는 함수

        parameters:
            target : 수행시킬 함수명
        """
        self.shutdown_thread_event.clear()
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        self.threads.append(thread)

    def stop_threads(self):
        """
        수행중인 모든 서브 쓰레드를 종료시키는 함수
        """
        self.shutdown_thread_event.set()
        for thread in self.threads:
            thread.join()
        print("All threads have stopped.")

    def cleanup(self):
        """
        소켓 연결을 종료시키는 함수
        """
        self.webcam_handler.release_webcam()
        if self.sock:
            self.sock.close()
        print("Socket closed and resources cleaned up.")

    def connection_test(self):
        """
        소켓 연결을 테스트 하기 위한 함수
        """
        print("Testing socket connection...")
        cnt = 0
        while not self.shutdown_thread_event.is_set():
            data = self.data_form.copy()
            data["angle"] = cnt
            cnt += 1
            self.send_data(data)
            time.sleep(CONNECTION_TEST_SLEEP)

    def measure_body_length(self):
        """
        사용자 신체 길이 측정 함수
        """
        print("Measuring body length...")
        # 포즈 추적을 위한 PoseProcessor할당(mediapipe)
        poseprocessor = PoseProcessor()



        # 신체측정 파라미터
        cnt = 0
        size_dict = {key: [] for key in [
            "lr_shoulder", "rl_shoulder", "l_u_arm", "r_u_arm",
            "l_f_arm", "r_f_arm", "l_side", "r_side", "lr_hip",
            "rl_hip", "l_u_leg", "r_u_leg", "l_l_leg", "r_l_leg"
        ]}
        try:
            while self.webcam_handler.capture.isOpened() and not self.shutdown_thread_event.is_set():
                ret, frame = self.webcam_handler.read_frame()
                if not ret:
                    raise Exception("Failed to read frame from webcam.")

                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(self.load_pose_data(), IDLE_CHECK_NODES, MARGIN)
                frame = poseprocessor.draw_incorrect_joints(frame, self.webcam_handler.frame_width,
                                                            self.webcam_handler.frame_height)
                encoded_image = self.webcam_handler.finalize_image(frame)

                data = self.data_form.copy()
                data["image"] = encoded_image

                # 자세 일치 60프레임 이상 지속 -> 신체 길이 담은 데이터 포함하여 전송
                if pose_check_result and pose_check_result["passed"]:
                    size_dict = poseprocessor.get_body_length(size_dict)
                    cnt += 1
                    if cnt >= 60:
                        data["body_length"] = json.dumps(self.filter_outliers(size_dict))
                        self.shutdown_thread_event.set()
                else:
                    data["incorrect_joint"] = pose_check_result.get("failed_nodes", [])
                    size_dict = {key: [] for key in size_dict.keys()}
                    cnt = 0

                self.send_data(data)
                print(f"count: {cnt}")

        except Exception as e:
            self.send_error(f"Error in measuring_body_length: {e}")

    @staticmethod
    def filter_outliers(size_dict: Dict[str, List[float]]) -> Dict[str, float]:
        filtered_dict = {}
        for key, values in size_dict.items():
            values.sort()
            length = len(values)
            remove_amount = int(length * OUTLIER_RATIO)
            trimmed_values = values[remove_amount:(length - remove_amount)]
            filtered_dict[key] = sum(trimmed_values) / len(trimmed_values)
        return filtered_dict

    @staticmethod
    def load_pose_data() -> Dict:
        with open(POSE_PATH, "r") as file:
            return json.load(file)

    # 준비자세의 일정 시간 유지 판정 후 (충분한 준비자세 유지 시, 준비자세 동안의 타겟 관절의 각도를 전달) 실시간 각도 값 전송
    def knee_game(self, body_length: Dict = None):
        """
        사용자의 준비자세 체크 후(준비자세를 2초가량 유지)
        사용자의 고관절 각도 (무릎 각도)를 넘겨줌

        parameters:
            body_length(dict): 사용자의 신체 길이가 담긴 dict 데이터
        """
        # 자세판단 파라미터
        print("Starting knee game...")
        poseprocessor = PoseProcessor()
        idle_cnt = 0
        idle_angles = []
        angles = []
        cnt = 0
        angle_dict={}

        try:
            # 준비자세 체크
            while self.webcam_handler.capture.isOpened() and not self.shutdown_thread_event.is_set():
                ret, frame = self.webcam_handler.capture.read()
                if not ret:
                    raise Exception("Failed to read frame from webcam.")
                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(self.load_pose_data(), IDLE_CHECK_NODES, MARGIN)
                frame = poseprocessor.draw_incorrect_joints(frame, self.webcam_handler.frame_width,
                                                            self.webcam_handler.frame_height)
                encoded_image = self.webcam_handler.finalize_image(frame)
                data = self.data_form.copy()
                data["image"] = encoded_image

                if pose_check_result and pose_check_result["passed"]:
                    angle = poseprocessor.get_angle_between_joints("l_vertical_hip", body_length)
                    idle_cnt += 1
                    if idle_cnt >= 60:
                        data["angle"]["l_vertical_hip"] = np.median(idle_angles)
                        self.send_data(data)
                        break
                    else:
                        idle_angles.append(angle)
                else:
                    data["incorrect_joint"] = pose_check_result.get("failed_nodes", [])
                    idle_angles.clear()
                    idle_cnt = 0

                self.send_data(data)

            # 각도 추적 시작
            while self.webcam_handler.capture.isOpened() and not self.shutdown_thread_event.is_set():
                ret, frame = self.webcam_handler.capture.read()
                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(self.load_pose_data(), CHECK_NODES, MARGIN)
                frame = poseprocessor.draw_incorrect_joints(frame, self.webcam_handler.frame_width,
                                                            self.webcam_handler.frame_height)
                
                angle = poseprocessor.get_angle_between_joints("l_vertical_hip", body_length)
                frame = poseprocessor.draw_angle(frame, 23, self.webcam_handler.frame_width, self.webcam_handler.frame_height, angle)
                
                lateral_hip_angle = poseprocessor.get_angle_between_joints("l_lateral_hip", body_length)
                frame = poseprocessor.draw_angle(frame, 24, self.webcam_handler.frame_width, self.webcam_handler.frame_height, lateral_hip_angle)

                vertical_knee_angle = poseprocessor.get_angle_between_joints("l_vertical_knee", body_length)
                frame = poseprocessor.draw_angle(frame, 25, self.webcam_handler.frame_width, self.webcam_handler.frame_height, vertical_knee_angle)
                
                encoded_image = self.webcam_handler.finalize_image(frame)
                data = self.data_form.copy()
                data["image"] = encoded_image

                if pose_check_result and pose_check_result.get("failed_nodes"):
                    data["incorrect_joint"] = pose_check_result["failed_nodes"]
                

                # angles Median Filter
                angles.append(angle)
                if len(angles) > 5:
                    angles.pop(0)
                if not cnt % 5 == 0:
                    angle_dict["l_vertical_hip"] = None if math.isnan(np.median(angles)) else np.median(angles)
                    angle_dict["l_lateral_hip"] = None if math.isnan(lateral_hip_angle) else lateral_hip_angle
                    angle_dict["l_vertical_knee"] = None if math.isnan(vertical_knee_angle) else vertical_knee_angle
                else:
                    data["angle"] = None

                data["angle"] = angle_dict

                self.send_data(data)
                print(f"send images {cnt}")
                cnt += 1
        except Exception as e:
            self.send_error(f"Error in knee_game: {e}")


def main():
    client = ClientSocket("localhost", 8081)


if __name__ == "__main__":
    main()
