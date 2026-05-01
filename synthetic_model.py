# lstm_handwriting_synthetic.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. Generate Better Data
# =========================
t = np.linspace(0, 4*np.pi, 200)

# circular motion (better than straight sine)
x = np.sin(t)
y = np.cos(t)

# convert to motion (dx, dy)
dx = np.diff(x)
dy = np.diff(y)

# =========================
# 2. Create Sequences
# =========================
seq_len = 20

X = []
Y = []

for i in range(len(dx) - seq_len):
    X.append(np.stack([dx[i:i+seq_len], dy[i:i+seq_len]], axis=1))
    Y.append(np.stack([dx[i+1:i+seq_len+1], dy[i+1:i+seq_len+1]], axis=1))

X = np.array(X)
Y = np.array(Y)

# =========================
# 3. Normalize (CRITICAL)
# =========================
mean = np.mean(X, axis=(0,1))
std = np.std(X, axis=(0,1)) + 1e-8

X = (X - mean) / std
Y = (Y - mean) / std

# =========================
# 4. Add Noise (helps learning)
# =========================
noise = np.random.normal(0, 0.01, X.shape)
X = X + noise

# =========================
# 5. Convert to Torch
# =========================
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# =========================
# 6. Model
# =========================
class HandwritingLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = HandwritingLSTM()

# =========================
# 7. Training Setup
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 8. Training Loop
# =========================
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    
    output = model(X_tensor)
    loss = criterion(output, Y_tensor)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =========================
# 9. Generate New Sequence
# =========================
model.eval()

input_seq = X_tensor[0:1]  # start from real sequence
generated = []

for _ in range(150):
    with torch.no_grad():
        pred = model(input_seq)
    
    next_step = pred[:, -1:, :]
    generated.append(next_step.numpy()[0,0])
    
    input_seq = torch.cat([input_seq[:, 1:, :], next_step], dim=1)

generated = np.array(generated)

# =========================
# 10. Denormalize
# =========================
generated = generated * std + mean

# =========================
# 11. Reconstruct (x, y)
# =========================
x_gen = np.cumsum(generated[:,0])
y_gen = np.cumsum(generated[:,1])

# =========================
# 12. Plot
# =========================
plt.figure(figsize=(6,6))
plt.plot(x_gen, y_gen)
plt.gca().invert_yaxis()
plt.title("Generated Trajectory (LSTM)")
plt.show()

plt.ion()
fig, ax = plt.subplots()

for i in range(len(x_gen)):
    ax.clear()
    ax.plot(x_gen[:i], y_gen[:i])
    ax.set_title("LSTM Drawing")
    ax.invert_yaxis()
    plt.pause(0.01)

plt.ioff()
plt.show()