import math
import numpy as np
import json
class P_vec:
    # 벡터의 표현에 문제가 있는 듯 하다.
    # 엉덩이 중앙을 (0,0,0) 좌표로 두는 3차원 좌표계로 생각하였는데 다시 생각해보자
    # world_landmarks가 아닌 일반 랜드마크를 고려하자
    # 월드 랜드마크를 인자로 받아, 각 부위별 벡터로 변환하여 저장
    def __init__(self, landmarks_dict):
        #각 관절의 좌표를 [x,y,z]순서의 배열로 담고 있음
        self.joint_dot = {
            'l_shoulder' : [landmarks_dict["landmark_11"]['x'],landmarks_dict["landmark_11"]['y'],landmarks_dict["landmark_11"]['z']],
            'r_shoulder' : [landmarks_dict["landmark_12"]['x'],landmarks_dict["landmark_12"]['y'],landmarks_dict["landmark_12"]['z']],

            'l_elbow' : [landmarks_dict["landmark_13"]['x'],landmarks_dict["landmark_13"]['y'],landmarks_dict["landmark_13"]['z']],
            'r_elbow' : [landmarks_dict["landmark_14"]['x'],landmarks_dict["landmark_14"]['y'],landmarks_dict["landmark_14"]['z']],

            'l_wrist' : [landmarks_dict["landmark_15"]['x'],landmarks_dict["landmark_15"]['y'],landmarks_dict["landmark_15"]['z']],
            'r_wrist' : [landmarks_dict["landmark_16"]['x'],landmarks_dict["landmark_16"]['y'],landmarks_dict["landmark_16"]['z']],

            'l_hip' : [landmarks_dict["landmark_23"]['x'],landmarks_dict["landmark_23"]['y'],landmarks_dict["landmark_23"]['z']],
            'r_hip' : [landmarks_dict["landmark_24"]['x'],landmarks_dict["landmark_24"]['y'],landmarks_dict["landmark_24"]['z']],

            'l_knee' : [landmarks_dict["landmark_25"]['x'],landmarks_dict["landmark_25"]['y'],landmarks_dict["landmark_25"]['z']],
            'r_knee' : [landmarks_dict["landmark_26"]['x'],landmarks_dict["landmark_26"]['y'],landmarks_dict["landmark_26"]['z']],

            'l_ankle' : [landmarks_dict["landmark_27"]['x'],landmarks_dict["landmark_27"]['y'],landmarks_dict["landmark_27"]['z']],
            'r_ankle' : [landmarks_dict["landmark_28"]['x'],landmarks_dict["landmark_28"]['y'],landmarks_dict["landmark_28"]['z']]
        }
        # 부위별 벡터를 담은 딕셔너리
        self.b_vectors = {"rl_shoulder" : self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_11"]), 
                          "lr_shoulder" : self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_12"]), 
                          "l_u_arm" : self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_13"]), 
                          "r_u_arm" : self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_14"]), 
                          "l_f_arm" : self.get_3d_vec(landmarks_dict["landmark_13"],landmarks_dict["landmark_15"]), 
                          "r_f_arm" : self.get_3d_vec(landmarks_dict["landmark_14"],landmarks_dict["landmark_16"]), 
                          "l_side" : self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_23"]), 
                          "r_side" : self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_24"]), 
                          "lr_hip" : self.get_3d_vec(landmarks_dict["landmark_23"],landmarks_dict["landmark_24"]), 
                          "rl_hip" : self.get_3d_vec(landmarks_dict["landmark_24"],landmarks_dict["landmark_23"]), 
                          "l_u_leg" : self.get_3d_vec(landmarks_dict["landmark_23"],landmarks_dict["landmark_25"]), 
                          "r_u_leg" : self.get_3d_vec(landmarks_dict["landmark_24"],landmarks_dict["landmark_26"]), 
                          "l_l_leg" : self.get_3d_vec(landmarks_dict["landmark_25"],landmarks_dict["landmark_27"]), 
                          "r_l_leg" : self.get_3d_vec(landmarks_dict["landmark_26"],landmarks_dict["landmark_28"]) }

    # 두 점을 받아 딕셔너리 형태의 벡터 반환
    def get_3d_vec(self, dot1, dot2):
        x = dot2['x'] - dot1['x']
        y = dot2['y'] - dot1['y']
        z = dot2['z'] - dot1['z']
        return {'x' : x,'y' : y,'z' : z}
    
    # 특정 부위의 벡터를 corrct
    def correct_vec(self, partsize:dict, partname:str):
        tmp_size = partsize[partname]**2 - self.get_2d_vetcor_size(partname)**2
        if(tmp_size > 0):
            z = math.sqrt(tmp_size)
            # print(z)
            if(self.b_vectors[partname]['z'] >= 0):
                self.b_vectors[partname]['z'] = z
                # print("+")
            else:
                self.b_vectors[partname]['z'] = -z
                # print("-")
        
    # z값 제외한 벡터의 크기 반환
    def get_2d_vetcor_size(self, partname:str):
        return math.sqrt(self.b_vectors[partname]['x'] ** 2 + self.b_vectors[partname]['y'] ** 2)

    # 벡터 크기 반환
    def get_3d_vetcor_size(self, partname:str):
        return math.sqrt(self.b_vectors[partname]['x'] ** 2 + self.b_vectors[partname]['y'] ** 2 + self.b_vectors[partname]['z'] ** 2)

    # 두 벡터간 각도 반환
    def angle_between_vectors(self, body_part1:str, body_part2:str):
        b_vector1 = self.b_vectors[body_part1]
        b_vector2 = self.b_vectors[body_part2]
        v1 = [b_vector1['x'], b_vector1['y'], b_vector1['z']]
        v2 = [b_vector2['x'], b_vector2['y'], b_vector2['z']]
        # 벡터의 크기 계산
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        # 벡터의 내적 계산
        dot_product = np.dot(v1, v2)
        
        # 각도 계산
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        angle_in_radians = np.arccos(cos_theta)
        angle_in_degrees = np.degrees(angle_in_radians)
        
        return angle_in_degrees
    
    def angle_between_2d_vectors(self, body_part1:str, body_part2:str):
        b_vector1 = self.b_vectors[body_part1]
        b_vector2 = self.b_vectors[body_part2]
        v1 = [b_vector1['x'], b_vector1['y']]
        v2 = [b_vector2['x'], b_vector2['y']]

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        v1_magnitude = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        v2_magnitude = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        cos_theta = dot_product / (v1_magnitude * v2_magnitude)
        theta = math.acos(cos_theta)  # 두 벡터 사이의 각도(라디안)
        # 라디안을 도로 변환하여 반환
        return math.degrees(theta)
    
    def angle_between_vectors_2d(self, body_part1:str, body_part2:str):

        b_vector1 = self.b_vectors[body_part1]
        b_vector2 = self.b_vectors[body_part2]
        v1 = [b_vector1['x'], b_vector1['y']]
        v2 = [b_vector2['x'], b_vector2['y']]

        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        cos_theta = dot_product / (v1_norm * v2_norm)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle_rad)

    # 2d벡터 크기를 json으로 저장
    def store_2d_body_size(self):
        body_size = {}
        for key in self.b_vectors.keys():
            body_size[key] = self.get_2d_vetcor_size(key)
        
        with open('test.json','w') as json_file:
            json.dump(body_size,json_file, indent=4, separators=(',', ': '))

        return True

