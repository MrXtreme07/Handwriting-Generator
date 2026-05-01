import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import re

# =========================
# 1. LOAD DATA
# =========================
def extract_label_from_segment(line):
    match = re.search(r'"(.)"', line)
    return match.group(1) if match else None

def load_uji(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    chars, labels = [], []
    current_char, current_stroke = [], []
    recording = False
    current_label = None

    for line in lines:
        line = line.strip()

        if line.startswith(".SEGMENT"):
            if current_char:
                chars.append(current_char)
                labels.append(current_label)

            current_char = []
            current_label = extract_label_from_segment(line)

        elif line == ".PEN_DOWN":
            current_stroke = []
            recording = True

        elif line == ".PEN_UP":
            if current_stroke:
                current_char.append(current_stroke)
            recording = False

        elif recording:
            parts = line.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                current_stroke.append((x, y))

    if current_char:
        chars.append(current_char)
        labels.append(current_label)

    return chars, labels

# Load all data
base = "uji+pen+characters"
all_chars, all_labels = [], []

for f in os.listdir(base):
    if f.startswith("UJIpenchars"):
        c, l = load_uji(os.path.join(base, f))
        all_chars.extend(c)
        all_labels.extend(l)

print("Total:", len(all_chars))

# =========================
# 2. FILTER + REDUCE VARIATION
# =========================
TARGET = 'A'
chars = [c for c, l in zip(all_chars, all_labels) if l == TARGET][:20]

print("Using samples:", len(chars))

# =========================
# 3. PREPROCESS (ABSOLUTE XY)
# =========================
def flatten(c):
    seq = []
    for s in c:
        seq.extend(s)
    return np.array(seq)

def normalize(seq):
    seq = seq - np.mean(seq, axis=0)
    seq = seq / (np.std(seq, axis=0) + 1e-8)
    return seq

SEQ_LEN = 40
X, Y = [], []

for c in chars:
    seq = normalize(flatten(c))
    if len(seq) < SEQ_LEN + 1:
        continue

    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        Y.append(seq[i+1:i+SEQ_LEN+1])

X = np.array(X)
Y = np.array(Y)

print("Dataset:", X.shape)

# =========================
# 4. TORCH
# =========================
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# =========================
# 5. MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

model = LSTMModel()

# =========================
# 6. TRAIN (SHORT + STABLE)
# =========================
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, Y),
    batch_size=64,
    shuffle=True
)

opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(100):   # 🔥 NOT 500/2000
    total = 0
    for bx, by in loader:
        opt.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        opt.step()
        total += loss.item()

    print(f"Epoch {epoch}, Loss: {total/len(loader):.4f}")

# =========================
# 7. SHORT-HORIZON GENERATION (KEY FIX)
# =========================
model.eval()

seed = X[0:1].clone()
generated = []

steps = 40   # 🔥 SHORT ONLY

for _ in range(steps):
    with torch.no_grad():
        pred = model(seed)

    next_step = pred[:, -1:, :]

    # small noise for realism
    next_step += torch.randn_like(next_step) * 0.02

    generated.append(next_step.numpy()[0,0])

    seed = torch.cat([seed[:,1:,:], next_step], dim=1)

generated = np.array(generated)

# =========================
# 8. VISUALIZE (BEST DEMO)
# =========================
seed_np = X[0].numpy()

plt.figure(figsize=(6,4))

# real
plt.plot(seed_np[:,0], seed_np[:,1], label="Real Stroke")

# predicted continuation
plt.plot(generated[:,0], generated[:,1], label="Predicted Continuation")

plt.legend()
plt.gca().invert_yaxis()
plt.title("LSTM Handwriting Continuation")
plt.show()