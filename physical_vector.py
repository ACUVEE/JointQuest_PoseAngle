import math
import numpy as np
import json
class P_vec:
    # world_landmarks가 아닌 일반 랜드마크를 고려하자
    # 월드 랜드마크를 인자로 받아, 각 부위별 벡터로 변환하여 저장
    def __init__(self, landmarks):

        self.landmarks = [
            {"coordinate": [math.nan,math.nan,math.nan] if lm.visibility < 0.5 else [lm.x, lm.y, lm.z],"visibility": lm.visibility}
            for i, lm in enumerate(landmarks)
        ]
        #각 관절의 좌표를 [x,y,z]순서의 배열로 담고 있음
        self.joint_coordinates = {
            'l_shoulder' : self.landmarks[11]["coordinate"],
            'r_shoulder' : self.landmarks[12]["coordinate"],

            'l_elbow' : self.landmarks[13]["coordinate"],
            'r_elbow' : self.landmarks[14]["coordinate"],

            'l_wrist' : self.landmarks[15]["coordinate"],
            'r_wrist' : self.landmarks[16]["coordinate"],

            'l_hip' : self.landmarks[23]["coordinate"],
            'r_hip' : self.landmarks[24]["coordinate"],

            'l_knee' : self.landmarks[25]["coordinate"],
            'r_knee' : self.landmarks[26]["coordinate"],

            'l_ankle' : self.landmarks[27]["coordinate"],
            'r_ankle' : self.landmarks[28]["coordinate"]
        }
        # 부위별 벡터를 담은 딕셔너리
        self.body_part_vectors = {
                          "lr_shoulder" : self.get_3d_vector(self.joint_coordinates['l_shoulder'],self.joint_coordinates['r_shoulder']), 
                          "rl_shoulder" : self.get_3d_vector(self.joint_coordinates['r_shoulder'],self.joint_coordinates['l_shoulder']), 
                          
                          "l_u_arm" : self.get_3d_vector(self.joint_coordinates['l_shoulder'],self.joint_coordinates['l_elbow']), 
                          "r_u_arm" : self.get_3d_vector(self.joint_coordinates['r_shoulder'],self.joint_coordinates['r_elbow']), 
                          
                          "l_f_arm" : self.get_3d_vector(self.joint_coordinates['l_elbow'],self.joint_coordinates['l_wrist']), 
                          "r_f_arm" : self.get_3d_vector(self.joint_coordinates['r_elbow'],self.joint_coordinates['r_wrist']), 
                          
                          "l_side" : self.get_3d_vector(self.joint_coordinates['l_shoulder'],self.joint_coordinates['l_hip']), 
                          "r_side" : self.get_3d_vector(self.joint_coordinates['r_shoulder'],self.joint_coordinates['r_hip']), 
                          
                          "lr_hip" : self.get_3d_vector(self.joint_coordinates['l_hip'],self.joint_coordinates['r_hip']), 
                          "rl_hip" : self.get_3d_vector(self.joint_coordinates['r_hip'],self.joint_coordinates['l_hip']), 
                          
                          "l_u_leg" : self.get_3d_vector(self.joint_coordinates['l_hip'],self.joint_coordinates['l_knee']), 
                          "r_u_leg" : self.get_3d_vector(self.joint_coordinates['r_hip'],self.joint_coordinates['r_knee']), 
                          
                          "l_l_leg" : self.get_3d_vector(self.joint_coordinates['l_knee'],self.joint_coordinates['l_ankle']), 
                          "r_l_leg" : self.get_3d_vector(self.joint_coordinates['r_knee'],self.joint_coordinates['r_ankle']) 
                          }

    # 두 점을 받아 딕셔너리 형태의 벡터 반환
    def get_3d_vector(self, dot1, dot2):
        x = dot2[0] - dot1[0]
        y = dot2[1] - dot1[1]
        z = dot2[2] - dot1[2]
        return [x,y,z]
    
    # 벡터 크기 반환
    def get_3d_vetcor_size(self, partname:str):
        return math.sqrt(self.body_part_vectors[partname][0] ** 2 + self.body_part_vectors[partname][1] ** 2 + self.body_part_vectors[partname][2] ** 2)

    # z값 제외한 벡터의 크기 반환
    def get_2d_vetcor_size(self, partname:str):
        return math.sqrt(self.body_part_vectors[partname][0] ** 2 + self.body_part_vectors[partname][1] ** 2)

    # 특정 부위의 벡터를 corrct
    def correct_vector(self, body_part_length:dict, partname:str):
        length_gap = abs(body_part_length[partname]**2 - self.get_3d_vetcor_size(partname)**2)
        if(length_gap > 2):
            print("z coord is corrected")
            z = math.sqrt(abs(body_part_length[partname]**2 - self.get_2d_vetcor_size(partname)**2))

            if(self.body_part_vectors[partname][2] >= 0):
                self.body_part_vectors[partname][2] = z
            else:
                self.body_part_vectors[partname][2] = -z

    # 두 벡터간 각도 반환
    def angle_between_vectors(self, body_part1:str, body_part2:str, body_part_length:dict = {}):
        # 몸 길이 입력이 있을 시, 벡터 값 수정
        if body_part_length:
            print("z coord is loaded")
            self.correct_vector(body_part_length,body_part1)
            self.correct_vector(body_part_length,body_part2)

        # 벡터 계산
        v1 = self.body_part_vectors[body_part1]
        v2 = self.body_part_vectors[body_part2]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)

        cos_theta = dot_product / (v1_norm * v2_norm)
        angle_in_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return np.degrees(angle_in_rad)
    
    # 두 벡터간의 2D상 각도
    def angle_between_vectors_2d(self, body_part1:str, body_part2:str):
        v1 = self.body_part_vectors[body_part1][0:2]
        v2 = self.body_part_vectors[body_part2][0:2]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)

        cos_theta = dot_product / (v1_norm * v2_norm)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle_rad)

    # 2d벡터 크기를 json으로 저장
    def store_2d_body_size(self):
        body_size = {}
        for key in self.body_part_vectors.keys():
            body_size[key] = self.get_2d_vetcor_size(key)
        
        with open('test.json','w') as json_file:
            json.dump(body_size,json_file, indent=4, separators=(',', ': '))

        return True

