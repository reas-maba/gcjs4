[README.md](https://github.com/user-attachments/files/28945721/README.md)
# DogBreeds Recognizer V1.0

🐕 狗品种识别系统

## 功能特点

- 用户可上传狗的图片，系统自动完成品种识别
- 输出识别结果，包含狗品种名称与置信度（Top 5）
- 支持120种常见犬种识别
- 界面简洁、操作简单

## 技术栈

- **前端**: HTML5 + CSS3 + JavaScript
- **后端**: Flask + Python
- **深度学习**: PyTorch + ResNet50
- **数据集**: Stanford Dogs Dataset

## 快速开始

### 1. 安装依赖

```bash
pip install -r backend/requirements.txt
```

### 2. 准备数据集

```bash
cd backend/src
python download_dataset.py
```

或手动下载：[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar)

### 3. 训练模型（可选）

```bash
cd backend/src
python train.py
```

### 4. 启动服务

```bash
python start_backend.py
```

### 5. 访问界面

打开 `frontend/public/index.html` 即可使用

## 项目结构

```
DogBreeds Recognizer/
├── backend/
│   ├── src/
│   │   ├── app.py          # Flask后端服务
│   │   ├── model.py        # 模型类
│   │   ├── train.py        # 训练脚本
│   │   └── download_dataset.py  # 数据集下载脚本
│   ├── models/             # 训练好的模型
│   ├── data/               # 数据集目录
│   └── requirements.txt    # 依赖配置
├── frontend/
│   └── public/
│       └── index.html      # 前端界面
├── start_backend.py        # 启动脚本
├── DATASET_README.md       # 数据集说明
└── README.md               # 项目说明
```

## API接口

### POST /api/predict

上传图片进行识别

**请求**:
```
Content-Type: multipart/form-data
Body: file=<图片文件>
```

**响应**:
```json
{
    "predictions": [
        {"breed": "golden retriever", "confidence": 95.23},
        {"breed": "labrador retriever", "confidence": 3.12},
        ...
    ]
}
```

### GET /api/breeds

获取支持的犬种列表

**响应**:
```json
{
    "breeds": ["affenpinscher", "afghan hound", ...]
}
```

## 支持的犬种

系统支持120种犬种识别，包括：
- Golden Retriever（金毛寻回犬）
- Labrador Retriever（拉布拉多）
- German Shepherd（德国牧羊犬）
- French Bulldog（法国斗牛犬）
- Pug（哈巴狗）
- 等...
