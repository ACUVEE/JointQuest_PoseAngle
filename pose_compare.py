# 두 포즈를 비교해서 노드들이 비슷한 위치에 있는지 검사
def compare_poses(
        ref_pose: dict, comp_pose: dict, margin: float, pass_threshold: int, *nodes: list
) -> dict:
    """
    파라미터:
    ref_pose, comp_pose (dict): 기준 포즈와 비교 포즈의 딕셔너리 객체
    margin (float): 각 노드를 중심으로 얼마나 큰 인식 영역을 둘지 지정
    pass_threshold (int): 몇 개의 노드가 영역 내에 들어와야 성공으로 판별할지 지정
    nodes (list): 비교를 수행할 노드의 리스트

    리턴값:
    pass_threshold 중 몇 개의 노드가 통과하였고, 몇 번 노드가 통과했는지 딕셔너리로 반환
    """

    def is_within_margin(ref_pos, comp_pos, margin):
        return abs(ref_pos - comp_pos) <= margin

    def get_node_index(node_name):
        if node_name.isdigit():
            return int(node_name)
        else:
            return int(node_name.split("_")[-1])

    def get_node_position(pose, node_name):
        index = get_node_index(node_name)
        return pose[f"landmark_{index}"]["x"], pose[f"landmark_{index}"]["y"]

    # 리스트의 각 노드에 대해 비교 수행
    count = 0
    passed_nodes = []  # 통과한 노드 리스트
    failed_nodes = []  # 통과 못한 노드 리스트
    for node in nodes:
        # 기준 노드와 비교 노드의 위치 가져오기
        ref_x, ref_y = get_node_position(ref_pose, node)
        comp_x, comp_y = get_node_position(comp_pose, node)

        # 비교 노드가 기준 노드 근처에 있을 때 카운터 증가
        if is_within_margin(ref_x, comp_x, margin) and is_within_margin(
                ref_y, comp_y, margin
        ):
            count += 1
            passed_nodes.append(node)
        else:
            failed_nodes.append(node)

    # pass_threshold에 비교해 통과한 노드 수와 그의 리스트, 그리고 통과 여부 리턴
    return {
        "passed": count >= pass_threshold,
        "threshold": pass_threshold,
        "count": count,
        "passed_nodes": passed_nodes,
        "failed_nodes": failed_nodes,
    }


# 예시 데이터 (딕셔너리 형태)
ref_pose = {
    "landmark_0": {"x": 0.464, "y": 0.270, "z": -0.241, "visibility": 0.999},
    "landmark_1": {"x": 0.460, "y": 0.257, "z": -0.210, "visibility": 0.999},
}
comp_pose_right = {
    "landmark_0": {"x": 0.454, "y": 0.275, "z": -0.240, "visibility": 0.999},
    "landmark_1": {"x": 0.100, "y": 0.100, "z": -0.210, "visibility": 0.999},
}
comp_pose_wrong = {
    "landmark_0": {"x": 0.100, "y": 0.100, "z": -0.001, "visibility": 0.999},
    "landmark_1": {"x": 0.100, "y": 0.100, "z": -0.001, "visibility": 0.999},
}

result = compare_poses(ref_pose, comp_pose_wrong, 0.05, 1, "landmark_0", "landmark_1")
print(result)  # Output: True or False