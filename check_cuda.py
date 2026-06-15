import torch

print("=" * 30)
print("PyTorch CUDA 状态检查")
print("=" * 30)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA 不可用，请安装 GPU 版本的 PyTorch")
print("=" * 30)