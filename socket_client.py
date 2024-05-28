import cv2
import sys
import socket
import numpy as np
import time
import base64
import threading
from poseprocessor import PoseProcessor
import json
import math

"""
신체측정자세,준비자세 데이터 수정 및 경로 분리
웹캠 변경 명령(자원 해제 코드도 필요함)


"""

"""
송신 데이터
Json
{
    "image":None,
    "angle": dict(Jsont),           {"hip_angle": 20, "knee_angle": 10, "hip_horizon_angle": 12}
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
class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        # 스레드 관리
        self.shutdown_thread_event = threading.Event()
        self.shutdown_system_event = threading.Event()
        self.threads = []
        self.data_form = {
            "image":None,
            "angle":None,
            "incorrect_joint": None,
            "body_length": None,
            "error": None
        }
        self.webcam_index = 0
        # 소켓 연결
        self.connection_attempts = 0
        self.sock = None
        self.connect_server()

    # 서버 연결 펑션
    def connect_server(self):
        
        try:
            # 연결시도
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connection_attempts = 0
            # 서버로부터 커맨드 입력 대기
            self.receive_commands()
        except Exception as e:
            print(e)
            self.connection_attempts += 1
            # 연결 시도 5회 넘어가면 프로세스 종료
            if self.connection_attempts == 5:
                print(u'Connect fail %d times. exit program'%(self.connection_attempts))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connection_attempts))
            self.connect_server()

    def send_data(self, data):
        """
        데이터 전송 함수
        
        parameters:
            data(dict): 전송할 데이터
        """
        json_data = json.dumps(data)
        length = str(len(json_data))
        
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(json_data.encode('utf-8'))

    def send_error(self,err_msg):
        """
        에러를 서버에게 전달하는 함수

        parameters:
            err_msg(str): 발생한 에러의 메시지
        """
        data = self.data_form.copy()
        data["error"] = err_msg
        json_data = json.dumps(data)
        length = str(len(json_data))
        
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(json_data.encode('utf-8'))

    def receive_commands(self):
        """
        서버로부터 요청을 받고 별도의 쓰레드를 수행시키는 함수

        요청 목록:
            0. 시스템 종료 - 소켓 클라이언트 프로세스 종료
            1. 쓰레드 종료 - 현재 실행중인 서브 쓰레드 모두 종료
            2. 통신 테스트 - 1씩 증가하는 angle 값을 가진 데이터를 1초에 한번씩 전송
            3. 신체길이 측정 - 길이측정
            4. 무릎 운동
        """
        self.setup_webcam(self.webcam_index)
        try:
            while not self.shutdown_system_event.is_set():
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    break
                json_data = json.loads(data)
                command = json_data["command"]
                if command == '0':
                    self.shutdown_system_event.set()
                elif command == '1':
                    self.stop_threads()
                elif command == '2':
                    self.start_thread(self.connection_test)
                elif command == '3':
                    self.start_thread(self.measure_body_length)
                elif command == '4':
                    self.start_thread(self.knee_game)
        except Exception as e:
            err_msg = f'Error in receiving command: {e}'
            print(err_msg)
            self.send_error(err_msg)
            
        finally:
            self.cleanup()

    def setup_webcam(self, webcam_index):
        """
        웹캠 자원 할당 및 정보 가져오는 함수

        parmeters:
            webcam_index(int): 사용할 웹캠의 인덱스

        modifies:
            self.capture
            self.frame_width
            self.frame_height
            self.fps
        """
        try:
            print("importing webcam resources")
            self.capture = cv2.VideoCapture(webcam_index)
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))

            # self.display_webcam()

        except Exception as e:
            err_msg = f"Error with importing webcam resources : {e}"
            print(err_msg)
            self.send_error(err_msg)

    def display_webcam(self):
        """
        웹캠 화면을 띄우는 함수(테스트용)
        """
        
        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Webcam is open. Press 'q' to quit.")
        
        while True:
            ret, frame = self.capture.read()
            
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            # 화면에 프레임 표시
            cv2.imshow("Webcam", frame)
            
            # 'q' 키를 누르면 루프를 빠져나옴
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 자원 해제
        self.capture.release()
        cv2.destroyAllWindows()

    def start_thread(self, target):
        """
        특정 함수를 서브 쓰레드로 수행시키는 함수

        parameters:
            target : 수행시킬 함수명
        """
        self.shutdown_thread_event.clear()
        thread = threading.Thread(target = target, daemon=True)
        thread.start()
        self.threads.append(thread)

    def stop_threads(self):
        """
        수행중인 모든 서브 쓰레드를 종료시키는 함수
        """
        self.shutdown_thread_event.set()
        for thread in self.threads:
            thread.join()
        print("All threads have been stopped.")

    def cleanup(self):
        """
        소켓 연결을 종료시키는 함수
        """
        self.sock.close()
        print("Socket closed and resources cleaned up.")

    def finalize_image(self,image):
        """
        이미지 처리 최종 단계(crop, compress, encode) 함수

        parameters:
            image(numpy.ndarray) : 이미지
        """
        # 프레임 가공 데이터
        aspect_ratio = 1        # 프레임 crop 비율 x:y
        crop_width = min(self.frame_width, int(self.frame_height * aspect_ratio))
        crop_height = min(self.frame_height, int(self.frame_width / aspect_ratio))
        start_x = int((self.frame_width - crop_width)/ 2)
        start_y = int((self.frame_height - crop_height)/ 2)
        cropped_frame = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),40]     # 이미지 압축품질
        result, imgencode = cv2.imencode('.jpg', cropped_frame, encode_param)
        encoded_image = base64.b64encode(imgencode).decode('utf-8')

        return encoded_image

    def connection_test(self):
        """
        소켓 연결을 테스트 하기 위한 함수
        """
        print("testing socket connection...")
        cnt = 0
        while not self.shutdown_thread_event.is_set():
            data = self.data_form.copy()
            data["angle"] = cnt
            cnt += 1
            print(cnt)
            json_data = json.dumps(data)
            length = str(len(json_data))

            self.sock.sendall(length.encode('utf-8').ljust(64))
            self.sock.send(json_data.encode('utf-8'))
            time.sleep(1)


    def measure_body_length(self):
        """
        사용자 신체 길이 측정 함수
        """
        print("preparing for measuring BodyLength")
        # 자세판단 파라미터
        pose_path = "output/20240415_031753.json"
        with open(pose_path, "r") as file:
            correct_pose = json.load(file)
        check_nodes = ["11", "12", "13", "14", "15", "16", "23", "24", "25", "26", "27", "28"]
        margin = 0.1

        # 신체측정 파라미터
        cnt = 0
        outlier_ratio = 0.1
        size_dict = {"lr_shoulder" : [], "rl_shoulder" : [], "l_u_arm" : [], "r_u_arm" : [], "l_f_arm" : [], "r_f_arm" : [], "l_side" : [], "r_side" : [], "lr_hip" : [], "rl_hip" : [], "l_u_leg" : [], "r_u_leg" : [], "l_l_leg" : [], "r_l_leg" : [] }

        # 포즈 추적을 위한 PoseProcessor할당(mediapipe)
        poseprocessor = PoseProcessor()
        try:
            while self.capture.isOpened() and not self.shutdown_thread_event.is_set():
                data = self.data_form.copy()
                ret, frame = self.capture.read()
                if not ret:
                    raise Exception("Failed to read frame from webcam.")
                
                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(correct_pose, check_nodes, margin)
                frame = poseprocessor.draw_incorrect_joints(frame,self.frame_width,self.frame_height)
                encoded_image = self.finalize_image(frame)
                
                data["image"] = encoded_image
                
                # 자세 일치 60프레임 이상 지속 -> 신체 길이 담은 데이터 포함하여 전송
                if pose_check_result and pose_check_result["passed"]:
                    size_dict = poseprocessor.get_body_length(size_dict)
                    cnt +=1
                    if(cnt >= 60):
                        for key in size_dict.keys():
                            size_dict[key].sort()
                            length = len(size_dict[key])
                            remove_amount = int(length*outlier_ratio)
                            tmp = size_dict[key][remove_amount:(length - remove_amount)]
                            size_dict[key] = sum(tmp)/len(tmp)

                        data["body_length"] = size_dict

                        self.shutdown_thread_event.set()
                else:
                    if pose_check_result and pose_check_result["failed_nodes"]:
                        data["incorrect_joint"] = pose_check_result["failed_nodes"]
                    for key in size_dict.keys():
                        size_dict[key] = []
                    cnt = 0

                json_data = json.dumps(data)
                length = str(len(json_data))

                # 데이터 전송
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(json_data.encode('utf-8'))
                print(u'count: %d'%(cnt))
        except Exception as e:
            err_msg = f"Error in measuring_body_length: {e}"
            print(err_msg)
            self.send_error(err_msg)

# 준비자세의 일정 시간 유지 판정 후 (충분한 준비자세 유지 시, 준비자세 동안의 타겟 관절의 각도를 전달) 실시간 각도 값 전송
    def knee_game(self,):
        """
        사용자의 준비자세 체크 후(2초)
        사용자의 고관절 각도 (무릎 각도)를 넘겨줌
        """
        # 자세판단 파라미터
        pose_path = "output/20240415_031753.json"
        with open(pose_path, "r") as file:
            correct_pose = json.load(file)
        idle_check_nodes = ["11", "12", "13", "14", "15", "16", "23", "24", "25", "26", "27", "28"]
        check_nodes = ["11", "12",]
        margin = 0.1
        idle_cnt = 0
        idle_angles = []
        angles = []
        cnt = 0
        poseprocessor = PoseProcessor()

        try:
            while self.capture.isOpened() and not self.shutdown_thread_event.is_set():
                data = self.data_form.copy()
                ret, frame = self.capture.read()
                if not ret:
                    raise Exception("Failed to read frame from webcam.")
                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(correct_pose, idle_check_nodes, margin)
                frame = poseprocessor.draw_incorrect_joints(frame,self.frame_width,self.frame_height)
                encoded_image = self.finalize_image(frame)
                data["image"] = encoded_image

                if pose_check_result and pose_check_result["passed"]:
                    angle = poseprocessor.get_angle_between_joints(target_movement="l_vertical_hip")
                    idle_cnt +=1
                    if(idle_cnt >= 60):
                        data["angle"] = np.median(idle_angles)
                        json_data = json.dumps(data)
                        length = str(len(json_data))

                        # 데이터 전송 (데이터 크기, 데이터)
                        self.sock.sendall(length.encode('utf-8').ljust(64))
                        self.sock.send(json_data.encode('utf-8'))
                        break

                    else:
                        idle_angles.append(angle)
                else:
                    if pose_check_result and pose_check_result["failed_nodes"]:
                        data["incorrect_joint"] = pose_check_result["failed_nodes"]
                    idle_angles.clear

                json_data = json.dumps(data)
                length = str(len(json_data))

                # 데이터 전송 (데이터 크기, 데이터)
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(json_data.encode('utf-8'))
            
            while self.capture.isOpened() and not self.shutdown_thread_event.is_set():
                data = self.data_form.copy()
                ret, frame = self.capture.read()
                frame = poseprocessor.initialize(frame)
                pose_check_result = poseprocessor.check_pose(correct_pose, check_nodes, margin)
                frame = poseprocessor.draw_incorrect_joints(frame,self.frame_width,self.frame_height)
                encoded_image = self.finalize_image(frame)
                data["image"] = encoded_image
                if pose_check_result and pose_check_result["failed_nodes"]:
                    data["incorrect_joint"] = pose_check_result["failed_nodes"]
                angle = poseprocessor.get_angle_between_joints(target_movement="l_vertical_hip")
                
                # angles Median Filter
                angles.append(angle)
                if len(angles) > 5:
                    del angles[0]
                if not cnt%5 == 0:
                    if math.isnan(np.median(angles)):
                        data["angle"] = None
                    else:
                        data["angle"] = np.median(angles)
                else:
                    data["angle"] = None

                json_data = json.dumps(data)
                length = str(len(json_data))

                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(json_data.encode('utf-8'))
                print(u'send images %d'%(cnt))
                cnt+=1
        except Exception as e:
            err_msg = f"Error in knee_game: {e}"
            print(err_msg)
            self.send_error(err_msg)

def main():
    client = ClientSocket('localhost', 8081)

if __name__ == "__main__":
    main()
