import matplotlib.pyplot as plt
import numpy as np

x_train_full = np.load("/tmp/x_train.npy")
x_train_sparse = np.load("/tmp/x_interp.npy")
keep_mask = np.load("/tmp/keep_mask.npy").T

i = 2
k = 3

x_train_full, x_train_sparse = x_train_full[2000*i:2000*(i+1), 10*k:10*(k+1)], x_train_sparse[2000*i:2000*(i+1), 10*k:10*(k+1)]
keep_mask = keep_mask[2000*i:2000*(i+1), 10*k:10*(k+1)]

fig, axs = plt.subplots(nrows=10, figsize=(20,20))

for i in range(10):
    axs[i].plot(x_train_full[:, i])
    axs[i].plot(x_train_sparse[:, i])
    axs[i].plot(keep_mask[:, i])
    axs[i].grid()

plt.show()
