import os
import sys

print("=" * 50)
print("🐕 DogBreeds Recognizer - 系统诊断")
print("=" * 50)

print("\n📊 环境信息")
print("-" * 30)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

try:
    import torch
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("⚠️  CUDA 不可用")
except ImportError as e:
    print(f"❌ 无法导入 torch: {e}")

print("\n📁 项目结构")
print("-" * 30)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"项目根目录: {project_root}")

files_to_check = [
    "backend/src/train_fast.py",
    "backend/src/app.py",
    "backend/src/model.py",
    "backend/data/low-resolution",
    "frontend/public/index.html"
]

for f in files_to_check:
    path = os.path.join(project_root, f)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {f}")

print("\n🚀 启动命令")
print("-" * 30)
print("激活环境: conda activate py39_dev")
print("训练模型: python backend/src/train_fast.py")
print("启动服务: python run_app.py")
print("打开前端: frontend/public/index.html")

print("\n" + "=" * 50)