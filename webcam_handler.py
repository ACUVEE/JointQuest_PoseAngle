import base64
from typing import Tuple

import cv2
import numpy as np

# 상수
WEBCAM_QUALITY = 40


class WebcamHandler:
    def __init__(self, index: int):
        self.index = index
        self.capture = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

    def setup_webcam(self):
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
            print("Importing webcam resources...")
            self.capture = cv2.VideoCapture(self.index)
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            # self.display_webcam()
            print("Webcam successfully initialized.")

        except Exception as e:
            raise RuntimeError(f"Error importing webcam: {e}")

    def release_webcam(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()

    def display_webcam(self):
        """
        웹캠 화면을 띄우는 함수(테스트용)
        """

        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        print("Webcam is open. Press 'q' to quit.")

        while True:
            ret, frame = self.read_frame()
            if not ret:
                raise RuntimeError("Error: Failed to grab frame.")
                break

            # 화면에 프레임 표시
            cv2.imshow("Webcam", frame)

            # 'q' 키를 누르면 루프를 빠져나옴
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 자원 해제
        self.release_webcam()
        cv2.destroyAllWindows()

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        if not self.capture:
            raise RuntimeError("Webcam is not initialized.")
        return self.capture.read()

    def finalize_image(self, image: np.ndarray, quality: int = WEBCAM_QUALITY) -> str:
        crop_width = min(self.frame_width, int(self.frame_height))
        crop_height = min(self.frame_height, int(self.frame_width))
        start_x = int((self.frame_width - crop_width) / 2)
        start_y = int((self.frame_height - crop_height) / 2)
        cropped_frame = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, imgencode = cv2.imencode(".jpg", cropped_frame, encode_param)
        return base64.b64encode(imgencode).decode("utf-8")
