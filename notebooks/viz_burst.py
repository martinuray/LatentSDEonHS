import matplotlib.pyplot as plt
import numpy as np

from utils.anomaly_detection import create_random_burst_mask


def create_random_mask(n_features, x_len):
    return np.random.rand(n_features, x_len) > 0.05


x_len = 100
n_features = 50
masked_ratio = 0.1

x = create_random_burst_mask(n_features, x_len, masked_ratio=masked_ratio)
print((~x).sum() / (x_len * n_features))

plt.figure(figsize=(8, 8))
plt.imshow(x, cmap='gray', aspect='equal')



plt.show()
