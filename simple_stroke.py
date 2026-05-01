import numpy as np
import matplotlib.pyplot as plt

# generate a curve (like handwriting)
t = np.linspace(0, 4*np.pi, 100)

x = t
y = np.sin(t)

plt.plot(x, y)
plt.gca().invert_yaxis()
plt.title("Sample Stroke")
plt.show()

dx = np.diff(x)
dy = np.diff(y)

plt.plot(dx, dy)
plt.title("Motion (dx, dy)")
plt.show()

seq_len = 20

X = []
Y = []

for i in range(len(dx) - seq_len):
    X.append(np.stack([dx[i:i+seq_len], dy[i:i+seq_len]], axis=1))
    Y.append(np.stack([dx[i+1:i+seq_len+1], dy[i+1:i+seq_len+1]], axis=1))

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

plt.plot(np.cumsum(X[0][:,0]), np.cumsum(X[0][:,1]))
plt.gca().invert_yaxis()
plt.title("Reconstructed Stroke")
plt.show()