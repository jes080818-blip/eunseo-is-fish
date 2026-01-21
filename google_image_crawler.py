import os
from PIL import Image
from torchvision import transforms

# 경로 설정
input_dir = "dataset/original"
output_dir = "dataset/augmented"
os.makedirs(output_dir, exist_ok=True)

# 증강 파이프라인
transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0)
    )
])

# 한 이미지당 생성할 개수
AUG_PER_IMAGE = 10  # 필요에 따라 조절

count = 0

for filename in os.listdir(input_dir):
    if not filename.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert("RGB")

    name, ext = os.path.splitext(filename)

    for i in range(AUG_PER_IMAGE):
        aug_img = transform(img)
        save_name = f"{name}_aug{i}{ext}"
        aug_img.save(os.path.join(output_dir, save_name))
        count += 1

print(f"✅ 총 생성된 이미지 수: {count}")


import os
from PIL import Image
from torchvision import transforms

input_dir = "/home/jetson/2026_youth/ai26/wash_data"
output_dir = "/home/jetson/2026_youth/ai26/wash_data_aug"

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

AUG_PER_IMAGE = 10  # ← 이 숫자 조절해서 총 1000장 맞추기

total = 0

for label in ["clean", "dirty"]:
    in_dir = os.path.join(input_dir, label)
    out_dir = os.path.join(output_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(in_dir):
        if not f.lower().endswith(".png"):
            continue

        img = Image.open(os.path.join(in_dir, f)).convert("RGB")
        name = f.replace(".png", "")

        for i in range(AUG_PER_IMAGE):
            aug = transform(img)
            aug.save(os.path.join(out_dir, f"{name}_aug{i}.png"))
            total += 1

print(f"✅ 총 생성된 이미지 수: {total}")



from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random

# 설정
root = Path("/home/jetson/2026_youth/ai26/wash_date")
clear_dir = root / "clear"
dirty_dir = root / "dirty"

# 출력 디렉토리 (YOLO 분류 형식)
output_root = root / "organized"
output_root.mkdir(exist_ok=True)

# 이미지 수집
clear_images = list(clear_dir.glob("*.png"))
dirty_images = list(dirty_dir.glob("*.png"))

print(f"✅ clear 이미지: {len(clear_images)}개")
print(f"✅ dirty 이미지: {len(dirty_images)}개")

# train/val/test 비율 설정 (70/15/15)
def split_data(images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]
    
    return train, val, test

# 각 클래스별로 분할
clear_train, clear_val, clear_test = split_data(clear_images)
dirty_train, dirty_val, dirty_test = split_data(dirty_images)

# 디렉토리 생성 및 파일 복사
def copy_images(image_list, split, class_name):
    dest_dir = output_root / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for img in image_list:
        shutil.copy2(img, dest_dir / img.name)
    
    return len(image_list)

# 복사 실행
print("\n📁 파일 복사 중...")
splits = {
    "train": (clear_train, dirty_train),
    "val": (clear_val, dirty_val),
    "test": (clear_test, dirty_test)
}

for split, (clear_imgs, dirty_imgs) in splits.items():
    clear_count = copy_images(clear_imgs, split, "clear")
    dirty_count = copy_images(dirty_imgs, split, "dirty")
    print(f"✅ {split:5s}: clear={clear_count:3d}, dirty={dirty_count:3d}")

print(f"\n✅ 완료! 결과: {output_root}")
print("\n📂 최종 구조:")
print("""
organized/
├── train/
│   ├── clear/  (png files)
│   └── dirty/  (png files)
├── val/
│   ├── clear/
│   └── dirty/
└── test/
    ├── clear/
    └── dirty/
""")



