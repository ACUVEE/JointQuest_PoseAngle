import random
from physical_vector import P_vec
class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

# 임의의 이미지에서 랜드마크를 추적하는 함수
def track_landmarks():
    # 이미지 처리 후 추적된 랜드마크 생성
    landmarks = [
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)),
        Landmark(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1))
    ]
    return landmarks

def test_p_vec():
    landmarks = track_landmarks()  # 이미지에서 추적된 랜드마크를 얻음
    p_vec = P_vec(landmarks)  # P_vec 클래스 인스턴스 생성

    # 테스트할 몸 길이 정보
    body_part_length = {
        'lr_shoulder': 50, 'rl_shoulder': 50,
        'l_u_arm': 30, 'r_u_arm': 30,
        'l_f_arm': 25, 'r_f_arm': 25,
        'l_side': 40, 'r_side': 40,
        'lr_hip': 50, 'rl_hip': 50,
        'l_u_leg': 40, 'r_u_leg': 40,
        'l_l_leg': 40, 'r_l_leg': 40
    }

    # 두 벡터 간의 각도 테스트
    angle1 = p_vec.angle_between_vectors('lr_shoulder', 'l_u_arm', body_part_length)
    angle2 = p_vec.angle_between_vectors('lr_shoulder', 'l_side', body_part_length)
    print("lr_shoulder와 l_u_arm 사이의 각도:", angle1)
    print("lr_shoulder와 l_side 사이의 각도:", angle2)

    # 2D 벡터 간의 각도 테스트
    angle3 = p_vec.angle_between_2d_vectors('lr_shoulder', 'l_u_arm')
    angle4 = p_vec.angle_between_2d_vectors('lr_shoulder', 'l_side')
    print("lr_shoulder와 l_u_arm 사이의 2D 각도:", angle3)
    print("lr_shoulder와 l_side 사이의 2D 각도:", angle4)

    # 2D 벡터 크기 저장
    p_vec.store_2d_body_size()

    # 3D 벡터 크기 저장
    p_vec.store_3d_body_size()

    p_vec.store_joint_coordinates()

if __name__ == "__main__":
    test_p_vec()