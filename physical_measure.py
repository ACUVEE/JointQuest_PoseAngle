from physical_vector import P_vec
from pose_compare import compare_poses
import cv2
import mediapipe as mp
import json

def measure_body_size(pose,cap,uid):
    # 스탠딩 자세 로드
    with open("output/20240415_031753.json", "r") as file:
        correct_pose = json.load(file)
        print("correct_pose is loaded")
    # 채점할 관절 넘버 리스트
    check_nodes = ["11", "12", "13", "14", "15", "16", "23", "24", "25", "26", "27", "28"]
    # 오차 범위 (m)
    margin = 0.1
    # 이상치 제거를 위한 잘라낼 비율
    outlier_ratio = 0.1
    # output_path = f"./{uid}.json"

    # correct frame count
    count = 0

    size_dict = {"lr_shoulder" : [], "rl_shoulder" : [], "l_u_arm" : [], "r_u_arm" : [], "l_f_arm" : [], "r_f_arm" : [], "l_side" : [], "r_side" : [], "lr_hip" : [], "rl_hip" : [], "l_u_leg" : [], "r_u_leg" : [], "l_l_leg" : [], "r_l_leg" : [] }

# 웹캠에서 넘어오는 각 프레임을 분석
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 BGR (opencv 기본 형식) -> RGB (미디어파이프 형식)로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 프레임에서 포즈 감지
        results = pose.process(image)

        if results.pose_world_landmarks:
            # 라벨을 붙여서 JSON 형식으로 변환
            landmarks = [
                (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
                for i, lm in enumerate(results.pose_world_landmarks.landmark)
            ]
            landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}

            # 결과 프레임 출력
            cv2.imshow("Mediapipe Pose Detection", frame)

            # 채점 수행
            passed = compare_poses(correct_pose, landmarks_dict, margin, 12, *check_nodes)
            # 입력된 좌표로 바디 벡터 생성
            b_vec = P_vec(results.pose_world_landmarks.landmark)
            # 자세 일치한다면 count 증가시키고, 부위별 벡터의 크기를 누적하며 저장
            if(passed['passed']):
                print(count)
                for key in b_vec.body_part_vectors.keys():
                    size_dict[key].append(b_vec.get_3d_vetcor_size(key))
                    
                count += 1
                # 자세 일치가 30 프레임을 넘어가면, 누적된 데이터로 평균 값 구하여 json으로 저장후 종료
                if(count >= 60):
                    print("success")
                    # 이상치를 제거하고 평균 값을 구하여 저장
                    for key in size_dict.keys():
                        size_dict[key].sort()
                        length = len(size_dict[key])
                        remove_amount = int(length*outlier_ratio)
                        tmp = size_dict[key][remove_amount:(length - remove_amount)]
                        size_dict[key] = sum(tmp)/len(tmp)
                    # 결과 output_path에 저장
                    cv2.destroyAllWindows()
                    return size_dict
            else:
                for key in b_vec.body_part_vectors.keys():
                    size_dict[key] = []
                count = 0

        # Q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return {}

    # 리소스 release
    cap.release()
