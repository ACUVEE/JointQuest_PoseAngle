import json
import os

def get_user_info():
    uid = input("uid 입력:\n")
    json_path = os.path.join("./user",uid+".json")

    user_data = read_json(json_path,uid)

    if not user_data['body_part_size']:
        flag = input("바디사이즈 입력 여부 확인")
        if flag:
            print("loading for measuring module")
            body_length_data = physical_measure()
            with open(json_path,'w') as file:
                json.dump(body_length_data, file)
    else:
        with open(json_path,'r') as file:
            result = json.load(file)
            return result


def read_json(json_path,uid):
    user_dir = "./user"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    if not os.path.exists(json_path):
        new_data = {
            "uid" : f"{uid}",
            "body_part_size": {},
        }

        with open(json_path, 'w') as file:
            json.dump(new_data, file)
        
    with open(json_path,'r') as file:
        data = json.load(file)
        return data
    

print(get_user_info())