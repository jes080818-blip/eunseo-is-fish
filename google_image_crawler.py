# ===============================
# 0. 라이브러리
# ===============================
from mpi4py import MPI
import nltk
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ===============================
# 1. MPI 초기화
# ===============================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()     # 프로세스 번호
size = comm.Get_size()    # 전체 프로세스 수

# ===============================
# 2. NLTK 데이터 경로 설정
# ===============================
DATA_PATH = "/home/jetson/2026_youth/ai26/wash_data"

nltk.data.path.append(DATA_PATH)
nltk.download("movie_reviews", download_dir=DATA_PATH)

from nltk.corpus import movie_reviews

# ===============================
# 3. 데이터 로드 (rank 0에서만)
# ===============================
if rank == 0:
    data = []
    for fileid in movie_reviews.fileids():
        text = movie_reviews.words(fileid)
        label = movie_reviews.categories(fileid)[0]
        data.append((" ".join(text), label))

    random.shuffle(data)

    train_data = data[:1500]
    test_data  = data[1500:]

else:
    train_data = None
    test_data  = None

# ===============================
# 4. 데이터 브로드캐스트
# ===============================
train_data = comm.bcast(train_data, root=0)
test_data  = comm.bcast(test_data,  root=0)

# ===============================
# 5. Vocabulary 생성 (rank 0 → broadcast)
# ===============================
if rank == 0:
    all_tokens = [
        word.lower()
        for text, _ in train_data
        for word in text.split()
    ]
    vocab = Counter(all_tokens)
    vocab_size = 20000
    most_common = vocab.most_common(vocab_size)
    itos = [w for w, _ in most_common]
    stoi = {w: i + 1 for i, w in enumerate(itos)}
else:
    stoi = None

stoi = comm.bcast(stoi, root=0)
vocab_size = len(stoi)

# ===============================
# 6. 문장 인코딩 함수
# ===============================
def encode_sentence(sent, max_len=200):
    tokens = sent.lower().split()
    encoded = [stoi.get(w, 0) for w in tokens[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    return torch.tensor(padded)

# ===============================
# 7. Dataset 정의
# ===============================
class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        x = encode_sentence(text)
        y = 1 if label == "pos" else 0
        return x, torch.tensor(y)

# ===============================
# 8. MPI용 데이터 분할
# ===============================
local_train_data = train_data[rank::size]

train_loader = DataLoader(
    MovieDataset(local_train_data),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    MovieDataset(test_data),
    batch_size=32
)

# ===============================
# 9. 모델 정의
# ===============================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        out = out[:, -1]
        out = self.fc(out)
        return self.act(out)

model = SentimentRNN(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_size=128
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# ===============================
# 10. 학습 (MPI 병렬)
# ===============================
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred.squeeze(), y.float())

        optimizer.zero_grad()
        loss.backward()

        # ===== MPI Gradient 평균 =====
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data.numpy()
                avg_grad = comm.allreduce(grad, op=MPI.SUM)
                param.grad.data = torch.tensor(avg_grad / size)
        # =============================

        optimizer.step()
        total_loss += loss.item()

    # Loss 집계
    avg_loss = comm.reduce(total_loss, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"[Epoch {epoch+1}] Loss: {avg_loss / size / len(train_loader):.4f}")

# ===============================
# 11. 테스트 (rank 0만)
# ===============================
if rank == 0:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predicted = (pred.squeeze() > 0.5).long()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    print("Test Accuracy:", correct / total)

# ===============================
# 12. 예측 함수 (rank 0)
# ===============================
def predict_sent(sent):
    x = encode_sentence(sent).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).item()
    return "Positive" if pred > 0.5 else "Negative"

if rank == 0:
    print(predict_sent("I love this movie!"))
    print(predict_sent("this was the worst film ever!"))
