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
        self.connectCount = 0
        self.connectServer()
        self.sendData_flag = True

    def connectServer(self):
        try:
            # 연결시도
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            # 이미지 전송 시작
            self.connectionThread()
        except Exception as e:
            print(e)
            self.connectCount += 1
            # 연결 시도 5회 넘어가면 프로세스 종료
            if self.connectCount == 5:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def connectionThread(self):
        while True:
            try:
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    break
                if data =='0':
                    # 이미지 전송 쓰레드
                    sendImage_thread = threading.Thread(target=self.sendData)
                    sendImage_thread.start()
                elif data == '1':
                    if self.sendData_flag:
                        self.sendData_flag = False
                        print("sendImage thread's terminated")
                        sendImage_thread.join()
                    else:
                        print("sendImage thread's aleady been terminated")
            except Exception as e:
                print("error: ",e)
                break  # 오류 발생 시 반복문 종료
            # finally:
            #     sendImage_thread.join()

    def imageProcessing(self,image,target = -1, angle = 0):
        # 프레임 가공 데이터
        aspect_ratio = 3 / 4        # 프레임 crop 비율 x:y
        crop_width = min(self.frame_width, int(self.frame_height * aspect_ratio))
        crop_height = min(self.frame_height, int(self.frame_width / aspect_ratio))
        start_x = (self.frame_width - crop_width) // 2
        start_y = (self.frame_height - crop_height) // 2
        cropped_frame = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

        return cropped_frame

    def sendData(self):
        print("preparing for sending images")
        self.sendData_flag = True
        cnt = 0
        data = {}
        angles = []
        capture = cv2.VideoCapture(1)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(capture.get(cv2.CAP_PROP_FPS))        # 웹캠 주사율
        
        poseprocessor = PoseProcessor()

        try:
            while capture.isOpened() and self.sendData_flag:  # 이미지 전송 플래그 확인
                ret, frame = capture.read()
                frame = poseprocessor.process(frame,self.frame_width,self.frame_height)
                angle = poseprocessor.get_angle_between_joints()
                # angles Median Filter (변화의 정도를 가늠하기 어려움)
                angles.append(angle)
                if len(angles) > 5:
                    del angles[0]
                if not cnt%5 == 0:
                    data["angle"] = np.median(angles)
                else:
                    data["angle"] = None
                # crop image
                cropped_frame = self.imageProcessing(frame)
                # 데이터 인코딩
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),40]     # 이미지 압축품질
                result, imgencode = cv2.imencode('.jpg', cropped_frame, encode_param)
                encoded_image = base64.b64encode(imgencode).decode('utf-8')  # Base64 인코딩 및 문자열로 디코딩
                data["image"] = encoded_image
                json_data = json.dumps(data)
                length = str(len(json_data))

                # 데이터 전송 (데이터 크기, 데이터)
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(json_data.encode('utf-8'))
                print(u'send images %d'%(cnt))
                cnt+=1
        except Exception as e:
            print(e)
        finally:
            capture.release()  # 캡처 해제

def main():
    client = ClientSocket('localhost', 8080)

if __name__ == "__main__":
    main()
