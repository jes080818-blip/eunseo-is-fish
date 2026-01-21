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
import random

# =====================
# 경로 설정
# =====================
root = Path("/home/jetson/2026_youth/ai26/wash_data")        # 원본 데이터
aug_root = Path("/home/jetson/2026_youth/ai26/wash_data_aug")  # 증강 데이터

clear_dir = root / "clean"
dirty_dir = root / "dirty"

aug_clear_dir = aug_root / "clean"
aug_dirty_dir = aug_root / "dirty"

# 출력 디렉토리
output_root = root.parent / "organized"
output_root.mkdir(exist_ok=True)

# =====================
# 이미지 수집
# =====================
clear_images = list(clear_dir.glob("*.png"))
dirty_images = list(dirty_dir.glob("*.png"))

aug_clear_images = list(aug_clear_dir.glob("*.png"))
aug_dirty_images = list(aug_dirty_dir.glob("*.png"))

print(f"✅ 원본 clear 이미지: {len(clear_images)}개")
print(f"✅ 원본 dirty 이미지: {len(dirty_images)}개")
print(f"🔥 증강 clear 이미지: {len(aug_clear_images)}개")
print(f"🔥 증강 dirty 이미지: {len(aug_dirty_images)}개")

# =====================
# val / test 분할 (원본만 사용)
# =====================
def split_val_test(images, val_ratio=0.5):
    random.shuffle(images)
    mid = int(len(images) * val_ratio)
    return images[:mid], images[mid:]

clear_val, clear_test = split_val_test(clear_images)
dirty_val, dirty_test = split_val_test(dirty_images)

# =====================
# 파일 복사 함수
# =====================
def copy_images(image_list, split, class_name):
    dest_dir = output_root / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for img in image_list:
        shutil.copy2(img, dest_dir / img.name)

    return len(image_list)

# =====================
# 복사 실행
# =====================
print("\n📁 파일 복사 중...")

# train → 증강 데이터
train_clear = copy_images(aug_clear_images, "train", "clean")
train_dirty = copy_images(aug_dirty_images, "train", "dirty")
print(f"✅ train : clean={train_clear}, dirty={train_dirty}")

# val / test → 원본 데이터
val_clear = copy_images(clear_val, "val", "clean")
val_dirty = copy_images(dirty_val, "val", "dirty")
print(f"✅ val   : clean={val_clear}, dirty={val_dirty}")

test_clear = copy_images(clear_test, "test", "clean")
test_dirty = copy_images(dirty_test, "test", "dirty")
print(f"✅ test  : clean={test_clear}, dirty={test_dirty}")

print(f"\n🎉 완료! 결과 폴더: {output_root}")
