import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from tqdm import tqdm

# 日志配置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("DogTrain")

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "backend/models/dog_breed_model.pth"
DATA_ROOT = "backend/data/low-resolution"
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)

class StanfordDogDataset(Dataset):
    """自定义斯坦福犬类数据集加载器"""
    def __init__(self, file_list, label_map, transform=None):
        self.file_list = file_list
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        folder_name = os.path.basename(os.path.dirname(img_path))
        label = self.label_map[folder_name]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def build_data_transforms():
    """构建训练集/验证集两种数据增强流水线"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_transform

def scan_all_images(root_dir):
    """遍历数据集所有图片路径，生成标签映射"""
    all_image_paths = []
    folder_to_label = {}
    label_id = 0

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        folder_to_label[folder] = label_id
        label_id += 1
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(("jpg", "jpeg", "png")):
                full_path = os.path.join(folder_path, img_name)
                all_image_paths.append(full_path)
    logger.info(f"扫描完成，共 {len(all_image_paths)} 张图片，{label_id} 个犬种类别")
    return all_image_paths, folder_to_label, label_id

def create_resnet50_model(num_classes):
    """迁移学习构建ResNet50分类网络"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # 冻结主干卷积层
    for param in model.parameters():
        param.requires_grad = False
    # 解冻最后两层残差块
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train Epoch")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val Epoch")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def main():
    # 1. 加载数据集
    all_imgs, folder_label_map, class_num = scan_all_images(DATA_ROOT)
    train_paths, val_paths = train_test_split(all_imgs, test_size=0.2, random_state=42)
    train_trans, val_trans = build_data_transforms()

    # 2. 构建数据集与加载器
    train_ds = StanfordDogDataset(train_paths, folder_label_map, train_trans)
    val_ds = StanfordDogDataset(val_paths, folder_label_map, val_trans)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 模型、损失、优化器
    model = create_resnet50_model(class_num)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    best_acc = 0.0

    # 4. 迭代训练
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)

        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"最优模型已保存，当前最佳验证准确率：{best_acc:.4f}")

    logger.info(f"训练全部完成！最高验证准确率：{best_acc:.4f}")

    # 保存类别映射文件，推理时使用
    with open("backend/config/folder_label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(folder_label_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()