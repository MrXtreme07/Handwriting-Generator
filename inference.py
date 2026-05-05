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
        self.lstm = nn.LSTM(2, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# =========================
# 2. LOAD MODELS
# =========================
def load_models():
    models = {}
    for f in os.listdir("models"):
        if f.endswith(".pt"):
            ch = f[0]
            m = LSTM()
            m.load_state_dict(torch.load(f"models/{f}", map_location="cpu"))
            m.eval()
            models[ch] = m
    return models

models = load_models()

# =========================
# 3. LOAD DATA (for real comparison)
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
# 4. UTILS
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
# 5. GENERATION
# =========================
def generate(model, seed, steps=50):
    seed = seed.clone()
    out = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(seed)

        nxt = pred[:, -1:, :]
        out.append(nxt.numpy()[0,0])
        seed = torch.cat([seed[:,1:,:], nxt], dim=1)

    return np.array(out)

# =========================
# 6. CREATE OUTPUT FOLDER
# =========================
os.makedirs("report_images", exist_ok=True)

# =========================
# 7. PLOT 1 — GENERATED LETTERS
# =========================
for ch, model in list(models.items())[:3]:
    seed = torch.randn(1, SEQ_LEN, 2)
    traj = generate(model, seed)

    plt.figure()
    plt.plot(traj[:,0], traj[:,1])
    plt.gca().invert_yaxis()
    plt.title(f"Generated Trajectory ({ch})")
    plt.savefig(f"report_images/generated_{ch}.png")
    plt.close()

# =========================
# 8. PLOT 2 — REAL vs PREDICTED
# =========================
sample_char = all_chars[0]
seq = normalize(flatten(sample_char))

seed_np = seq[:SEQ_LEN]
target = seq[SEQ_LEN:SEQ_LEN+50]

seed = torch.tensor(seed_np, dtype=torch.float32).unsqueeze(0)

model = list(models.values())[0]
pred = generate(model, seed)

plt.figure()
plt.plot(target[:,0], target[:,1], label="Real")
plt.plot(pred[:,0], pred[:,1], label="Predicted")
plt.legend()
plt.gca().invert_yaxis()
plt.title("Real vs Predicted Trajectory")
plt.savefig("report_images/real_vs_pred.png")
plt.close()

# =========================
# 9. PLOT 3 — ERROR (MSE over time)
# =========================
errors = np.mean((target[:len(pred)] - pred)**2, axis=1)

plt.figure()
plt.plot(errors)
plt.title("MSE Error over Time")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.savefig("report_images/error_curve.png")
plt.close()

# =========================
# 10. PLOT 4 — WEIGHT DISTRIBUTION
# =========================
weights = []

for param in model.parameters():
    weights.extend(param.detach().numpy().flatten())

plt.figure()
plt.hist(weights, bins=50)
plt.title("Model Weight Distribution")
plt.savefig("report_images/weights.png")
plt.close()

# =========================
# 11. PLOT 5 — TRAJECTORY FLOW
# =========================
plt.figure()
plt.quiver(pred[:-1,0], pred[:-1,1],
           pred[1:,0]-pred[:-1,0],
           pred[1:,1]-pred[:-1,1],
           angles='xy', scale_units='xy', scale=1)
plt.gca().invert_yaxis()
plt.title("Trajectory Direction Field")
plt.savefig("report_images/vector_field.png")
plt.close()

print("All plots saved in report_images/")