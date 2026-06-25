import json

def load_all_breed_info():
    """读取外部JSON犬种资料库，替代代码内硬编码字典"""
    file_path = "backend/config/dog_info.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_breed_translate_dict():
    """获取中英名称互查字典"""
    info = load_all_breed_info()
    trans = {}
    for cn_name, detail in info.items():
        trans[cn_name] = detail["name_en"]
        trans[detail["name_en"]] = cn_name
    return trans