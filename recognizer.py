import io
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os

class DogRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "backend/models/dog_breed_model.pth"
        self.class_map = self._load_class_mapping()
        self.num_classes = len(self.class_map)
        self.model = self._init_model()
        self.transform = self._get_transform_pipeline()

    # 加载类别中英文映射文件
    def _load_class_mapping(self):
        with open("backend/config/breed_mapping.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # 初始化ResNet50模型
    def _init_model(self):
        net = models.resnet50(weights=None)
        in_feature = net.fc.in_features
        net.fc = torch.nn.Linear(in_feature, self.num_classes)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            net.load_state_dict(checkpoint)
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            net.fc = torch.nn.Linear(net.fc.in_features, self.num_classes)

        net.to(self.device)
        net.eval()
        return net

    # 图像预处理流水线
    def _get_transform_pipeline(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 字节流图片预处理
    def preprocess(self, img_byte):
        img = Image.open(io.BytesIO(img_byte)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    # 推理并返回Top5结果
    def get_top5_prediction(self, img_byte):
        tensor_img = self.preprocess(img_byte)
        with torch.no_grad():
            output = self.model(tensor_img)
            prob = torch.softmax(output, dim=1)[0]

        top5_vals, top5_idx = torch.topk(prob, k=5)
        res_list = []
        for val, idx in zip(top5_vals, top5_idx):
            cls_id = str(idx.item())
            info = self.class_map[cls_id]
            res_list.append({
                "breed_cn": info["cn"],
                "breed_en": info["en"],
                "confidence": round(val.item() * 100, 2)
            })
        return res_list