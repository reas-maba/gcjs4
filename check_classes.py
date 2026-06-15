import os

data_dir = 'backend/data/low-resolution'

classes = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]

print(f"找到 {len(classes)} 个类别:\n")
for i, cls in enumerate(classes):
    print(f"{i:3d}: {cls}")

print("\n将结果保存到 classes.txt...")
with open('classes.txt', 'w', encoding='utf-8') as f:
    for i, cls in enumerate(classes):
        f.write(f"{i:3d}: {cls}\n")