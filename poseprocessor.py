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
        """
        이미지의 사람 관절을 추적하는 함수
        (다른 함수를 사용하기 위해 먼저 수행되어야 함)

        parameters:
            frame(이미지)

        modifies: 
            self.joints
            self.p_vec
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.joints = results
        self.p_vec = P_vec(self.joints.pose_world_landmarks.landmark)

    def draw_landmark(self, image):
        """
        감지된 관절의 landmark와 이를 잇는 선을 이미지위에 draw하는 함수

        parameters:
            image(이미지)
        
        preconditoins:
            self.joints가 detect_joint 함수 호출로 초기화 된 상태여야함

        return:
            image(이미지)
        """
        if self.joints.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, self.joints.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image
    
    def get_angle_between_joints(self, target_movement:str)->float:
        """
        특정 동작에 대한 각도를 구하는 함수

        parameters:
            taeget_movement(문자열) : 추적할 동작명으로 좌우_움직임방향_부위 형식을 가짐    
                ex) r_vertical_knee
        
        preconditoins:
            self.joints가 detect_joint 함수 호출로 초기화 된 상태여야함
        
        return:
            각도(float)
            각도를 구할 수 없는 경우 nan 반환
        """
        #외회전과 내회전 등 관절의 회전은 감지할 수 없음 
        if self.joints.pose_world_landmarks:
            # 고관절 굴곡 신전 (굴곡 180 ~ 신전 0)
            if target_movement == "r_vertical_hip":
                angle = self.p_vec.angle_between_vectors('unit_vector_y','r_u_leg')
            elif target_movement == "l_vertical_hip":
                angle = self.p_vec.angle_between_vectors('unit_vector_y','l_u_leg')
            # 고관절 외전 내전 (내전 0 ~ 외전180 )
            elif target_movement == "r_lateral_hip":
                angle = self.p_vec.angle_between_vectors('unit_vector_x','r_u_leg')
            elif target_movement == "l_lateral_hip":
                angle = self.p_vec.angle_between_vectors('unit_vector_-x','l_u_leg')
            # 무릎 굴곡 신전 (굴곡 180 ~ 신전 0)
            elif target_movement == "r_vertical_knee":
                angle = self.p_vec.angle_between_vectors('r_u_leg','r_l_leg')
            elif target_movement == "l_vertical_knee":
                angle = self.p_vec.angle_between_vectors('l_u_leg','l_l_leg')
            return angle
        return math.nan
    
    def draw_angle(self, image, target_joint:int, frame_width:int, frame_height:int, angle:float):
        """
        각도 값을 이미지에 그려주는 함수

        parameters:
            image(MathLike) : 각도를 표시할 이미지
            target_joint(int) : 각도가 표시될 관절 넘버
            frame_width(int) : 이미지 너비
            frame_height(int) : 이미지 높이
            angle(float) : 표시할 각도 값

        preconditoins:
            self.joints가 detect_joint 함수 호출로 초기화 된 상태여야함

        return:
            image(MathLike) : 각도가 표시된 이미지
         """
        if self.joints.pose_world_landmarks:
            target_landmark = self.joints.pose_landmarks.landmark[target_joint]
            x = int(target_landmark.x * frame_width)
            y = int(target_landmark.y * frame_height)
            result = cv2.putText(image, str(angle), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
            return result
        return image
    
    def initialize(self, frame,):
        """
        필수요소 세팅을 위한 함수 수행하는 함수
        """
        self.detect_joint(frame)    # 필수
        image = self.draw_landmark(frame)   # 필수 X

        return image
        
    def get_body_length(self, size_dict:dict):
        """
        추적된 관절 값을 이용하여 신체부위의 (3D상의)길이를 측정하는 함수

        parameters:
            size_dict(dict): 사용자의 신체부위를 담을 dict로 밸류는 array형태를 가져야함

        preconditoins:
            self.joints가 detect_joint 함수 호출로 초기화 된 상태여야함

        return
            사용자의 신체부위가 (append)누적된 dict
        """
        if self.joints.pose_world_landmarks:
            for key in size_dict.keys():
                    size_dict[key].append(self.p_vec.get_3d_vetcor_size(key))
        return size_dict
    
    def check_pose(self, correct_pose:dict = {}, check_nodes:list = [], margin:float = 0.1)->dict:
        """
        관절의 위치가 올바른 위치에 있는지 확인하는 함수

        parameters:
            correct_pose(dict): 올바른 자세의 관절 위치가 담긴 딕셔너리
            check_nodes(list): 체크할 관절의 넘버가 담긴 리스트
            margin(float): 올바른 자세로 판단할 여유 margin (m 단위)

        preconditoins:
            self.joints가 detect_joint 함수 호출로 초기화 된 상태여야함

        Modifies:
            self.incoorect_joints: 잘못된 위치에 있는 관절의 리스트로 초기화

        return:
            passed(dict): 관절의 위치가 모두 올바른 위치에 있는지에 대한 여부
        """
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
    
    def draw_incorrect_joints(self, frame, frame_width:int, frame_height:int):
        """
        잘못된 위치에 있는 관절을 강조하는 원을 이미지위에 그리는 함수
        
        parmeters:
            frame(MathLike): draw할 이미지
            frame_width(int): 이미지의 너비
            frame_height(int): 이미지의 높이

        preconditions:
            self.incorrect_joints: self.check_pose() 함수를 통해 초기화 된 상태여야함

        return
            frame(MathLike): 강조된 원이 그려진 이미지
        """
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
    
