# DogBreeds Recognizer - 数据集说明

## 数据集来源

本项目使用 **Stanford Dogs Dataset**，包含120个犬种，约20,580张图片。

## 数据集下载

### 方法一：自动下载（推荐）

运行以下命令自动下载并整理数据集：

```bash
cd backend/src
python download_dataset.py
```

### 方法二：手动下载

1. 访问 [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
2. 下载 `images.tar` 文件
3. 解压到 `backend/data/` 目录
4. 运行整理脚本：

```bash
cd backend/src
python -c "from download_dataset import organize_dataset; organize_dataset('../data')"
```

## 数据组织结构

```
backend/data/
├── train/
│   ├── affenpinscher/
│   ├── afghan_hound/
│   ├── ... (120个犬种文件夹)
├── val/
│   ├── affenpinscher/
│   ├── afghan_hound/
│   ├── ... (120个犬种文件夹)
└── Images/ (原始数据集，可选保留)
```

## 数据集分割

- 训练集：80% 的图片
- 验证集：20% 的图片

## 支持的犬种列表

affenpinscher, afghan_hound, african_hunting_dog, airedale,
american_staffordshire_terrier, appenzeller, australian_terrier,
basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog,
black-and-tan_coonhound, blenheim_spaniel, bloodhound, bluetick,
border_collie, border_terrier, borzoi, boston_bull, bouvier_des_flandres,
boxer, brabancon_griffon, briard, brittany_spaniel, bull_mastiff,
cairn, cardigan, chesapeake_bay_retriever, chihuahua, chow,
clumber, cocker_spaniel, collie, curly-coated_retriever, dandie_dinmont,
dhole, dingo, doberman, english_foxhound, english_setter,
english_springer, entlebucher, eskimo_dog, flat-coated_retriever,
french_bulldog, german_shepherd, german_short-haired_pointer,
giant_schnauzer, golden_retriever, gordon_setter, great_dane,
great_pyrenees, greater_swiss_mountain_dog, groenendael, ibizan_hound,
irish_setter, irish_terrier, irish_water_spaniel, irish_wolfhound,
italian_greyhound, japanese_spaniel, keeshond, kelpie, kerry_blue_terrier,
komondor, kuvasz, labrador_retriever, lakeland_terrier, leonberg,
lhasa, malamute, malinois, maltese_dog, mastiff, miniature_pinscher,
miniature_poodle, miniature_schnauzer, newfoundland, norfolk_terrier,
norwegian_elkhound, norwich_terrier, old_english_sheepdog, otterhound,
papillon, pekinese, pembroke, pomeranian, pug, redbone,
rhodesian_ridgeback, rottweiler, saint_bernard, saluki, samoyed,
schipperke, scotch_terrier, scottish_deerhound, sealyham_terrier,
shetland_sheepdog, shih-tzu, siberian_husky, silky_terrier,
soft-coated_wheaten_terrier, staffordshire_bullterrier, standard_poodle,
standard_schnauzer, sussex_spaniel, tibetan_mastiff, tibetan_terrier,
toy_poodle, toy_terrier, vizsla, walker_hound, weimaraner,
welsh_springer_spaniel, west_highland_white_terrier, whippet,
wire-haired_fox_terrier, yorkshire_terrier