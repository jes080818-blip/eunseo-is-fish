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
