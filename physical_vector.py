import math
import numpy as np
import json
class P_vec:

    # 월드 랜드마크를 인자로 받아, 각 부위별 벡터로 변환하여 저장
    def __init__(self, landmarks_dict):
        # 수정필요
        self.lr_shoulder = self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_12"])
        self.rl_shoulder = self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_11"])

        self.l_u_arm = self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_13"])
        self.r_u_arm = self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_14"])
        
        self.l_f_arm = self.get_3d_vec(landmarks_dict["landmark_13"],landmarks_dict["landmark_15"])
        self.r_f_arm = self.get_3d_vec(landmarks_dict["landmark_14"],landmarks_dict["landmark_16"])
        
        self.l_side = self.get_3d_vec(landmarks_dict["landmark_11"],landmarks_dict["landmark_23"])
        self.r_side = self.get_3d_vec(landmarks_dict["landmark_12"],landmarks_dict["landmark_24"])
        
        self.lr_hip = self.get_3d_vec(landmarks_dict["landmark_23"],landmarks_dict["landmark_24"])
        self.rl_hip = self.get_3d_vec(landmarks_dict["landmark_24"],landmarks_dict["landmark_23"])
        
        self.l_u_leg = self.get_3d_vec(landmarks_dict["landmark_23"],landmarks_dict["landmark_25"])
        self.r_u_leg = self.get_3d_vec(landmarks_dict["landmark_24"],landmarks_dict["landmark_26"])
        
        self.l_l_leg = self.get_3d_vec(landmarks_dict["landmark_25"],landmarks_dict["landmark_27"])
        self.r_l_leg = self.get_3d_vec(landmarks_dict["landmark_26"],landmarks_dict["landmark_28"])
        # 부위별 벡터를 담은 딕셔너리
        self.b_vectors = {"rl_shoulder" : self.rl_shoulder, 
                          "lr_shoulder" : self.lr_shoulder, 
                          "l_u_arm" : self.l_u_arm, 
                          "r_u_arm" : self.r_u_arm, 
                          "l_f_arm" : self.l_f_arm, 
                          "r_f_arm" : self.r_f_arm, 
                          "l_side" : self.l_side, 
                          "r_side" : self.r_side, 
                          "rl_hip" : self.rl_hip, 
                          "lr_hip" : self.lr_hip, 
                          "l_u_leg" : self.l_u_leg, 
                          "r_u_leg" : self.r_u_leg, 
                          "l_l_leg" : self.l_l_leg, 
                          "r_l_leg" : self.r_l_leg }

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
        if(self.b_vectors[partname]['z'] >= 0):
            self.b_vectors[partname]['z'] = z
        else:
            self.b_vectors[partname]['z'] = -z
        
    # z값 제외한 벡터의 크기 반환
    def get_2d_vetcor_size(self, partname:str):
        return math.sqrt(self.b_vectors[partname]['x'] ** 2 + self.b_vectors[partname]['y'] ** 2)

    # 두 벡터간 각도 반환
    def angle_between_vectors(self, body_part1:str, body_part2:str):
        b_vector1 = self.b_vectors[body_part1]
        b_vector2 = self.b_vectors[body_part2]
        vec1 = [b_vector1['x'], b_vector1['y'], b_vector1['z']]
        vec2 = [b_vector2['x'], b_vector2['y'], b_vector2['z']]
        dot_product = np.dot(vec1, vec2)
        magnitude_vec1 = np.linalg.norm(vec1)
        magnitude_vec2 = np.linalg.norm(vec2)
        angle_rad = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
        angle_deg = round(np.degrees(angle_rad),1)
        return angle_deg
    
    # 2d벡터 크기를 json으로 저장
    def store_2d_body_size(self):
        body_size = {}
        for key in self.b_vectors.keys():
            body_size[key] = self.get_2d_vetcor_size(key)
        
        with open('test.json','w') as json_file:
            json.dump(body_size,json_file, indent=4, separators=(',', ': '))

        return True

