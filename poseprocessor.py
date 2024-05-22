import cv2
import mediapipe as mp
import math
import json
from pose_compare import compare_poses
from physical_vector import P_vec

class PoseProcessor():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation = True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_count = 0

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
        
    def get_body_length(self, size_dict:dict):
        if self.joints.pose_world_landmarks:
            p_vec = P_vec(self.joints.pose_world_landmarks.landmark)
            for key in size_dict.keys():
                    size_dict[key].append(p_vec.get_3d_vetcor_size(key))
        return size_dict
    
    def check_pose(self, correct_pose:dict = {}, check_nodes:list = [], margin:float = 0.1):
        passed ={}
        if self.joints.pose_world_landmarks:
            # 라벨을 붙여서 JSON 형식으로 변환
            landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(self.joints.pose_world_landmarks.landmark)
            ]
            landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}
            # 채점 수행
            passed = compare_poses(correct_pose, landmarks_dict, margin, len(check_nodes), *check_nodes)
            self.incorrect_joints = passed["failed_nodes"]
        else:
            self.incorrect_joints = []
        return passed
    
    def draw_incorrect_joints(self, frame, frame_width, frame_height):
        radius = 30
        color = (0, 0, 255)
        thickness = 4
        
        if self.joints.pose_world_landmarks:
            if self.incorrect_joints:
                for joint in self.incorrect_joints:
                    target_landmark = self.joints.pose_landmarks.landmark[int(joint)]
                    if target_landmark:
                        x = int(target_landmark.x * frame_width)
                        y = int(target_landmark.y * frame_height)
                        center_coordinates = (x, y)
                        cv2.circle(frame, center_coordinates, radius ,color, thickness)

        return frame