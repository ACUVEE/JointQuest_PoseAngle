import cv2
import sys
import socket
import numpy as np
import time
import base64
import threading
from poseprocessor import PoseProcessor
import json


class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connection_attempts = 0
        self.sock = None
        self.webcam_index = 0
        # 스레드 관리
        self.shutdown_thread_event = threading.Event()
        self.shutdown_system_event = threading.Event()
        self.threads = []
        self.data_form = {

            "imgae":None,
            "angle":None,
            "incorrect_joint": None,
            "body_length": None,
            "error": None
        }
        # 서버 연결 펑션 호출
        self.connectServer()

    def connectServer(self):
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
            self.connectServer()

    def send_error(self,e):
        data = self.data_form.copy()
        data["error"] = e
        json_data = json.dumps(data)
        length = str(len(json_data))
        
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(json_data.encode('utf-8'))

    def receive_commands(self):
        try:
            while not self.shutdown_system_event.is_set():
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    break
                json_data = json.loads(data)
                command = json_data["command"]
                if command == '0':
                    self.stop_threads()
                elif command == '1':
                    self.start_thread(self.sendData)
                elif command == '2':
                    self.start_thread(self.measureBodyLength)
                elif command == '3':
                    self.shutdown_system_event.set()
                elif command == '4':
                    self.start_thread(self.connection_test)
        except Exception as e:
            err_msg = f'Error in receiving command: {e}'
            print(err_msg)
            self.send_error(err_msg)
            
        finally:
            self.cleanup()

    def start_thread(self, target):
        self.shutdown_thread_event.clear()
        thread = threading.Thread(target = target, daemon=True)
        thread.start()
        self.threads.append(thread)

    def stop_threads(self):
        self.shutdown_thread_event.set()
        for thread in self.threads:
            thread.join()
        print("All threads have been stopped.")

    def cleanup(self):
        self.sock.close()
        print("Socket closed and resources cleaned up.")

    def finalize_image(self,image,target = -1, angle = 0):
        # 프레임 가공 데이터
        aspect_ratio = 3 / 4        # 프레임 crop 비율 x:y
        crop_width = min(self.frame_width, int(self.frame_height * aspect_ratio))
        crop_height = min(self.frame_height, int(self.frame_width / aspect_ratio))
        start_x = (self.frame_width - crop_width) // 2
        start_y = (self.frame_height - crop_height) // 2
        cropped_frame = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),40]     # 이미지 압축품질
        result, imgencode = cv2.imencode('.jpg', cropped_frame, encode_param)
        encoded_image = base64.b64encode(imgencode).decode('utf-8')

        return encoded_image

    def connection_test(self):
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


    def measureBodyLength(self):
        print("preparing for measuring BodyLength")
        # 자세판단 파라미터
        pose_path = "output/20240415_031753.json"
        with open(pose_path, "r") as file:
            correct_pose = json.load(file)
        check_nodes = ["11", "12", "13", "14", "15", "16", "23", "24", "25", "26", "27", "28"]
        margin = 0.08

        # 신체측정 파라미터
        cnt = 0
        outlier_ratio = 0.1
        size_dict = {"lr_shoulder" : [], "rl_shoulder" : [], "l_u_arm" : [], "r_u_arm" : [], "l_f_arm" : [], "r_f_arm" : [], "l_side" : [], "r_side" : [], "lr_hip" : [], "rl_hip" : [], "l_u_leg" : [], "r_u_leg" : [], "l_l_leg" : [], "r_l_leg" : [] }
        
        # 웹캠 자원 할당
        capture = cv2.VideoCapture(self.webcam_index)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(capture.get(cv2.CAP_PROP_FPS))

        # 포즈 추적을 위한 PoseProcessor할당(mediapipe)
        poseprocessor = PoseProcessor()
        try:
            while capture.isOpened() and not self.shutdown_thread_event.is_set():
                data = self.data_form.copy()
                ret, frame = capture.read()
                frame = poseprocessor.process(frame,self.frame_width,self.frame_height)
                pose_check_result = poseprocessor.check_pose(correct_pose, check_nodes, margin)
                frame = poseprocessor.draw_incorrect_joints(frame,self.frame_width,self.frame_height)
                encoded_image = self.finalize_image(frame)
                
                data["image"] = encoded_image
                
                # 자세 일치 60프레임 이상 지속 -> 신체 길이 담은 데이터 포함하여 전송
                if pose_check_result and pose_check_result["passed"]:
                    poseprocessor.get_body_length(size_dict)
                    cnt +=1
                    if(cnt >= 60):
                        for key in size_dict.keys():
                            size_dict[key].sort()
                            length = len(size_dict[key])
                            remove_amount = int(length*outlier_ratio)
                            tmp = size_dict[key][remove_amount:(length - remove_amount)]
                            size_dict[key] = sum(tmp)/len(tmp)

                        data["body_length"] = size_dict
                else:
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

    def sendData(self):
        print("preparing for sending images")
        pose_path = "output/20240415_031753.json"
        with open(pose_path, "r") as file:
            correct_pose = json.load(file)
        check_nodes = ["11", "12"]
        margin = 0.08
        cnt = 0
        angles = []
        capture = cv2.VideoCapture(self.webcam_index)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(capture.get(cv2.CAP_PROP_FPS))        # 웹캠 주사율
        
        poseprocessor = PoseProcessor()

        try:
            while capture.isOpened() and not self.shutdown_thread_event.is_set():  # 이미지 전송 플래그 확인
                data = self.data_form.copy()
                ret, frame = capture.read()
                frame = poseprocessor.process(frame,self.frame_width,self.frame_height)

                pose_check_result = poseprocessor.check_pose(correct_pose, check_nodes, margin)
                frame = poseprocessor.draw_incorrect_joints(frame,self.frame_width,self.frame_height)
                if pose_check_result and pose_check_result["failed_nodes"]:
                    data["incorrect_joint"] = pose_check_result["failed_nodes"]
                angle = poseprocessor.get_angle_between_joints()
                # angles Median Filter (변화의 정도를 가늠하기 어려움)
                angles.append(angle)
                if len(angles) > 5:
                    del angles[0]
                if not cnt%5 == 0:
                    data["angle"] = np.median(angles)
                else:
                    data["angle"] = None

                encoded_image = self.finalize_image(frame)
                data["image"] = encoded_image
                json_data = json.dumps(data)
                length = str(len(json_data))

                # 데이터 전송 (데이터 크기, 데이터)
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(json_data.encode('utf-8'))
                print(u'send images %d'%(cnt))
                cnt+=1
        except Exception as e:
            err_msg = f"Error in sending data: {e}"
            print(err_msg)
            self.send_error(err_msg)
        finally:
            capture.release()  # 캡처 해제

def main():
    client = ClientSocket('localhost', 8081)

if __name__ == "__main__":
    main()
