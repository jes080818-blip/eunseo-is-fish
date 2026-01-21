import os
import re
import nltk
from nltk.corpus import movie_reviews

# ===============================
# 1. NLTK 데이터 경로 설정
# ===============================
DATA_PATH = "/home/jetson/2026_youth/ai26/wash_data"
nltk.data.path.append(DATA_PATH)

# (이미 있으면 다시 안 받음)
nltk.download("movie_reviews", download_dir=DATA_PATH)

# ===============================
# 2. 저장할 증강 데이터 경로
# ===============================
SAVE_BASE = "/home/jetson/2026_youth/ai26/wash_data/augmented_reviews"

for label in ["pos", "neg"]:
    os.makedirs(os.path.join(SAVE_BASE, label), exist_ok=True)

# ===============================
# 3. 전처리 함수 (증강용)
# ===============================
def preprocess_text(words):
    """
    - 소문자 변환
    - 특수문자 제거
    """
    text = " ".join(words).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# 4. 데이터 증강 + 저장
# ===============================
count = 0

for fileid in movie_reviews.fileids():
    label = movie_reviews.categories(fileid)[0]
    words = movie_reviews.words(fileid)

    # (1) 원본 텍스트
    original_text = " ".join(words)

    # (2) 전처리된 텍스트 (증강)
    cleaned_text = preprocess_text(words)

    # 파일 이름 (슬래시 제거)
    base_name = fileid.replace("/", "_")

    # 저장 경로
    orig_path  = os.path.join(SAVE_BASE, label, f"{base_name}_orig.txt")
    clean_path = os.path.join(SAVE_BASE, label, f"{base_name}_clean.txt")

    # 파일 저장
    with open(orig_path, "w", encoding="utf-8") as f:
        f.write(original_text)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    count += 2

print("✅ 데이터 증강 완료")
print(f"총 생성된 리뷰 파일 수: {count}")
print(f"저장 위치: {SAVE_BASE}")
