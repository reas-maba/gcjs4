import json
import os
from typing import Dict, List, Optional, Union

INFO_JSON_PATH = "backend/config/dog_info.json"

class DogInfoManager:
    """犬种百科信息统一管理类，支持增删改查、搜索、批量导出"""
    def __init__(self):
        self.data: Dict[str, dict] = {}
        self.load_data()

    def load_data(self):
        """从JSON文件加载全部犬种信息，文件不存在则初始化空库"""
        if os.path.exists(INFO_JSON_PATH):
            with open(INFO_JSON_PATH, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}
            self.save_data()

    def save_data(self):
        """持久化写入JSON文件"""
        with open(INFO_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def get_all_breeds(self) -> List[str]:
        """获取所有犬种中文名称列表"""
        return list(self.data.keys())

    def get_all_info(self) -> Dict:
        """返回完整犬种信息字典"""
        return self.data

    def get_single_info(self, breed_cn: str) -> Optional[dict]:
        """根据中文名称查询单犬种详情"""
        return self.data.get(breed_cn, None)

    def search_by_keyword(self, keyword: str) -> List[str]:
        """模糊搜索：中文名/英文名/产地/性格匹配"""
        result = []
        kw = keyword.lower()
        for name, info in self.data.items():
            match1 = kw in name.lower()
            match2 = kw in info["name_en"].lower()
            match3 = kw in info["origin"].lower()
            match4 = any(kw in t.lower() for t in info["temperament"])
            if match1 or match2 or match3 or match4:
                result.append(name)
        return result

    def add_breed(self, breed_data: dict) -> bool:
        """新增犬种信息，校验必填字段"""
        required_fields = [
            "name", "name_en", "origin", "size", "weight", "height",
            "lifespan", "temperament", "description", "coat", "color",
            "activity_level", "trainability", "family_friendly", "apartment_friendly"
        ]
        for field in required_fields:
            if field not in breed_data:
                print(f"新增失败：缺少必填字段 {field}")
                return False
        breed_cn = breed_data["name"]
        self.data[breed_cn] = breed_data
        self.save_data()
        return True

    def update_breed(self, breed_cn: str, update_dict: dict) -> bool:
        """更新指定犬种部分字段"""
        if breed_cn not in self.data:
            return False
        self.data[breed_cn].update(update_dict)
        self.save_data()
        return True

    def delete_breed(self, breed_cn: str) -> bool:
        """删除犬种记录"""
        if breed_cn not in self.data:
            return False
        del self.data[breed_cn]
        self.save_data()
        return True

    def filter_by_size(self, size_type: str) -> List[str]:
        """按体型筛选：小型/中小型/中型/大型"""
        res = []
        for name, info in self.data.items():
            if info["size"] == size_type:
                res.append(name)
        return res

    def filter_family_friendly(self) -> List[str]:
        """筛选适合家庭饲养的犬种"""
        res = []
        for name, info in self.data.items():
            if info["family_friendly"] is True:
                res.append(name)
        return res

# 初始化内置基础犬种数据（示例）
def init_default_data():
    manager = DogInfoManager()
    if len(manager.data) > 0:
        return
    sample_data = [
        {
            "name": "柴犬",
            "name_en": "Shiba Inu",
            "origin": "日本",
            "size": "中小型",
            "weight": "8-11kg",
            "height": "35-41cm",
            "lifespan": "12-15年",
            "temperament": ["忠诚", "活泼", "独立", "警觉"],
            "description": "柴犬是日本原生古老犬种，性格独立干净，运动量中等，网红热门伴侣犬。",
            "coat": "双层短毛",
            "color": ["赤色", "黑褐色", "芝麻色", "白色"],
            "activity_level": "中等",
            "trainability": "中等",
            "family_friendly": True,
            "apartment_friendly": True
        },
        {
            "name": "金毛寻回犬",
            "name_en": "Golden Retriever",
            "origin": "英国",
            "size": "大型",
            "weight": "25-34kg",
            "height": "51-61cm",
            "lifespan": "10-12年",
            "temperament": ["温顺", "亲人", "聪明", "友善"],
            "description": "金毛性格极其温和，导盲犬常用品种，对小孩包容，运动需求高。",
            "coat": "双层长毛",
            "color": ["浅金色", "深金色"],
            "activity_level": "高",
            "trainability": "高",
            "family_friendly": True,
            "apartment_friendly": False
        }
    ]
    for item in sample_data:
        manager.add_breed(item)
    print("默认犬种数据初始化完成")

if __name__ == "__main__":
    init_default_data()
    mgr = DogInfoManager()
    print("全部犬种：", mgr.get_all_breeds())
    print("搜索关键词「日本」：", mgr.search_by_keyword("日本"))