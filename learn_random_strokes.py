import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

def load_uji_characters(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    characters = []
    current_char = []
    current_stroke = []
    recording = False

    for line in lines:
        line = line.strip()

        if line.startswith(".SEGMENT"):
            # save previous character
            if current_char:
                characters.append(current_char)
            current_char = []

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

    # append last character
    if current_char:
        characters.append(current_char)

    return characters

all_chars = []
base_path = "uji+pen+characters"
for file in os.listdir(base_path):
    if file.startswith("UJIpenchars"):
        path = os.path.join(base_path, file)
        chars = load_uji_characters(path)
        all_chars.extend(chars)

characters = all_chars

def flatten_character(character):
    """Flattens a character (list of strokes) into a single sequence of points."""
    sequence = []
    for stroke in character:
        sequence.extend(stroke)
    return sequence

def to_dxdy(sequence):
    """Converts a sequence of (x, y) points into (dx, dy) motion vectors."""
    sequence = np.array(sequence)
    dx = np.diff(sequence[:,0])
    dy = np.diff(sequence[:,1])
    return np.stack([dx, dy], axis=1)

# --- Building the dataset ---

X = []
Y = []

SEQ_LEN = 20

for char in characters:
    seq = flatten_character(char)

    if len(seq) < SEQ_LEN + 1:
        continue

    motion = to_dxdy(seq)

    for i in range(len(motion) - SEQ_LEN):
        X.append(motion[i:i+SEQ_LEN])
        Y.append(motion[i+1:i+SEQ_LEN+1])

X = np.array(X)
Y = np.array(Y)

print("Dataset Shapes: ", X.shape)

# --- Normalization ---

mean = np.mean(X, axis=(0,1))
std  = np.std(X, axis=(0,1)) + 1e-8

X = (X - mean) /  std
Y = (Y - mean) / std

# --- Converting to PyTorch ---

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- LSTM Model ---
class  HandwritingLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first = True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

model = HandwritingLSTM()

# --- Training Loop ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    total_loss = 0

    for batch_X, batch_Y in loader:
        optimizer.zero_grad()

        output = model(batch_X)
        loss = criterion(output, batch_Y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

# --- Generating New Handwriting ---
model.eval()

input_seq = X_tensor[0:1]
generated = []

for _ in range(200):
    with torch.no_grad():
        pred = model(input_seq)

    next_step = pred[:, -1:, :]  # FIXED
    generated.append(next_step.numpy()[0,0,:])  # FIXED
    input_seq = torch.cat([input_seq[:, 1:, :], next_step], dim=1)

generated = np.array(generated)

# Denormalize
generated = generated * std + mean

# Reconstruct
x_gen = np.cumsum(generated[:,0])
y_gen = np.cumsum(generated[:,1])

# --- Plotting the generated handwriting ---
plt.figure(figsize=(5,5))
plt.plot(x_gen, y_gen)
plt.gca().invert_yaxis()
plt.title("Generated Handwriting")
plt.show()