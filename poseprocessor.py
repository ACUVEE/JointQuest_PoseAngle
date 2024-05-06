import cv2
import mediapipe as mp
import math
from physical_vector import P_vec

class PoseProcessor():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation = True)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_joint(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.joints = results

    def draw_landmark(self, image):
        if self.joints.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, self.joints.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image
    
    def get_angle_between_joints(self):
        if self.joints.pose_world_landmarks:
            p_vec = P_vec(self.joints.pose_world_landmarks.landmark)
            angle = p_vec.angle_between_vectors('unit_vector_y','l_u_leg')
            return angle
        return math.nan
    
    def draw_angle(self, image, target_joint, frame_width, frame_height, angle):
        if self.joints.pose_world_landmarks:
            target_landmark = self.joints.pose_landmarks.landmark[target_joint]
            x = int(target_landmark.x * frame_width)
            y = int(target_landmark.y * frame_height)
            result = cv2.putText(image, str(angle), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
            return result
        return image
    
    def process(self, frame, frame_width, frame_height):
        self.detect_joint(frame)
        angle = self.get_angle_between_joints()
        image = self.draw_landmark(frame)
        image = self.draw_angle(image, 23, frame_width, frame_height, angle)

        return image

# # 캡처 객체 및 프레임 크기
# cap = cv2.VideoCapture(1)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # 미디어 파이프 클래스 객체 생성
# asd = qwe()

# while cap.isOpened():
#     ret, frame = cap.read()
#     # 프레임 처리
#     frame = asd.process(frame, frame_width, frame_height)

#     # 프레임 출력
#     cv2.imshow("asd", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 자원 해제
# cap.release()
# cv2.destroyAllWindows()
