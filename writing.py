import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, re, random

# =========================
# 1. MODEL
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
# 2. LOAD MODELS
# =========================
def load_models():
    models = {}
    for file in os.listdir("models"):
        if file.endswith(".pt"):
            ch = file[0]
            model = LSTM()
            model.load_state_dict(torch.load(f"models/{file}", map_location="cpu"))
            model.eval()
            models[ch] = model
    return models

models = load_models()
print("Loaded models:", list(models.keys()))

# =========================
# 3. LOAD DATA (for seeds)
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

# =========================
# 4. PREPROCESS
# =========================
def flatten_with_pen(c):
    pts = []
    pen = []

    for stroke in c:
        for p in stroke:
            pts.append(p)
            pen.append(1)   # pen down

        pts.append(stroke[-1])
        pen.append(0)       # pen up

    return np.array(pts), np.array(pen)

def normalize(seq):
    seq = seq - np.mean(seq, axis=0)
    seq = seq / (np.std(seq, axis=0) + 1e-8)
    return seq

SEQ_LEN = 40

# =========================
# 5. GENERATION
# =========================
def continue_stroke(model, seed, steps=30):
    seed = seed.clone()
    out = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(seed)

        nxt = pred[:, -1:, :]
        nxt += torch.randn_like(nxt) * 0.01
        out.append(nxt.numpy()[0,0])

        seed = torch.cat([seed[:,1:,:], nxt], dim=1)

    return np.array(out)

def generate_letter(model, char):
    candidates = [c for c,l in zip(all_chars, all_labels) if l == char]

    if len(candidates) == 0:
        return None, None

    sample = random.choice(candidates)

    seq, pen = flatten_with_pen(sample)
    seq = normalize(seq)

    seed_np = seq[:SEQ_LEN]
    seed = torch.tensor(seed_np, dtype=torch.float32).unsqueeze(0)

    gen = continue_stroke(model, seed)

    full = np.vstack([seed_np, gen])

    # alignment
    full[:,1] -= np.min(full[:,1])
    full[:,1] /= (np.max(full[:,1]) + 1e-8)
    full[:,0] -= np.mean(full[:,0])

    return full, pen[:SEQ_LEN]

# =========================
# 6. WORD GENERATION
# =========================
def generate_word(word, spacing=1.5, space_gap=3.0):
    full_traj = []
    full_pen = []
    offset = np.array([0.0, 0.0])

    for ch in word:
        # HANDLE SPACE
        if ch == " ":
            offset = offset + np.array([space_gap, 0])
            continue

        model = models.get(ch)
        if model is None:
            print(f"Skipping '{ch}'")
            continue

        letter, pen = generate_letter(model, ch)
        if letter is None:
            continue

        letter = letter + offset

        width = np.max(letter[:,0]) - np.min(letter[:,0])
        offset = offset + np.array([width + spacing, 0])

        full_traj.append(letter)
        full_pen.append(pen)

    traj = np.vstack(full_traj)
    pen = np.concatenate(full_pen)

    return traj, pen

# =========================
# 7. PLOT WITH PEN-UP
# =========================
def plot_with_pen(traj, pen):
    plt.figure(figsize=(8,4))

    start = 0
    for i in range(1, len(traj)):
        if i < len(pen) and pen[i] == 0:
            plt.plot(traj[start:i,0], traj[start:i,1])
            start = i

    plt.plot(traj[start:,0], traj[start:,1])

    plt.gca().invert_yaxis()
    plt.title("Handwriting with Pen-Up")
    plt.show()

# =========================
# 8. ANIMATION
# =========================
def animate(traj):
    plt.figure(figsize=(8,4))

    for i in range(0, len(traj), 2):
        plt.cla()
        plt.plot(traj[:i,0], traj[:i,1])
        plt.gca().invert_yaxis()
        plt.pause(0.01)

    plt.show()

# =========================
# 9. RUN
# =========================
word = input("Enter word: ").upper()

traj, pen = generate_word(word)

plot_with_pen(traj, pen)
animate(traj)