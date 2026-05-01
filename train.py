import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, re

# =========================
# 1. LOAD DATA
# =========================
def extract_label(line):
    m = re.search(r'"(.)"', line)
    return m.group(1) if m else None

def load_uji(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    chars, labels = [], []
    cur_char, cur_stroke = [], []
    recording = False
    cur_label = None

    for line in lines:
        line = line.strip()

        if line.startswith(".SEGMENT"):
            if cur_char:
                chars.append(cur_char)
                labels.append(cur_label)
            cur_char = []
            cur_label = extract_label(line)

        elif line == ".PEN_DOWN":
            cur_stroke = []
            recording = True

        elif line == ".PEN_UP":
            if cur_stroke:
                cur_char.append(cur_stroke)
            recording = False

        elif recording:
            parts = line.split()
            if len(parts) == 2:
                cur_stroke.append((float(parts[0]), float(parts[1])))

    if cur_char:
        chars.append(cur_char)
        labels.append(cur_label)

    return chars, labels


base = "uji+pen+characters"
all_chars, all_labels = [], []

for f in os.listdir(base):
    if f.startswith("UJIpenchars"):
        c, l = load_uji(os.path.join(base, f))
        all_chars.extend(c)
        all_labels.extend(l)

print("Total samples:", len(all_chars))


# =========================
# 2. PREPROCESS
# =========================
def flatten(c):
    pts = []
    for s in c:
        pts.extend(s)
    return np.array(pts)

def normalize(seq):
    seq = seq - np.mean(seq, axis=0)
    seq = seq / (np.std(seq, axis=0) + 1e-8)
    return seq

SEQ_LEN = 40


# =========================
# 3. MODEL
# =========================
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# =========================
# 4. TRAIN + SAVE
# =========================
os.makedirs("models", exist_ok=True)

def train_and_save(char):
    data = [c for c,l in zip(all_chars, all_labels) if l == char]

    if len(data) < 10:
        print(f"Skipping {char} (too few samples)")
        return

    X, Y = [], []

    for c in data[:40]:   # slightly more data
        seq = normalize(flatten(c))
        if len(seq) < SEQ_LEN + 1:
            continue

        for i in range(len(seq) - SEQ_LEN):
            X.append(seq[i:i+SEQ_LEN])
            Y.append(seq[i+1:i+SEQ_LEN+1])

    if len(X) == 0:
        print(f"Skipping {char} (no usable sequences)")
        return

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    model = LSTM()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=64,
        shuffle=True
    )

    print(f"\nTraining {char}...")

    for epoch in range(40):
        total = 0
        for bx, by in loader:
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"{char} Epoch {epoch}, Loss: {total/len(loader):.4f}")

    # 🔥 SAVE MODEL
    path = f"models/{char}.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved {char} → {path}")


# =========================
# 5. TRAIN ALL LETTERS
# =========================
letters = sorted(set(all_labels))

for ch in letters:
    if ch is None:
        continue
    train_and_save(ch)

print("\nAll models saved in /models")