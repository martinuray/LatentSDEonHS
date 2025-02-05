##%
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# stuff for signals
base_frequency = 1.0   # Hz
fs = 1 / (10*10**-3)   # Hz; every 10ms; as the idea with the testbed
step_size = 1/fs
start_s = 0
end_s = 1/base_frequency

# stuff for the dataset
num_periods = 1
num_features = 2

#%%
x = np.arange(0, end_s, step_size)
Y = np.zeros((num_periods, len(x), num_features))


def norm_data(y_):
    min_vals = y_.min(axis=1, keepdims=True)
    max_vals = y_.max(axis=1, keepdims=True)
    n_arr = (y_ - min_vals) / (max_vals - min_vals)
    # in the basic sine example the data is in [-0.8, 0.8]
    return n_arr * 0.8


Y[:, :, 0] = 1.00 * np.sin(2*np.pi*base_frequency*1.*x) #+ 0.21 * np.sin(2*np.pi*base_frequency*2*x)

if num_features > 1:
    Y[:, :, 1] = 0.30 * np.cos(2*np.pi*base_frequency*4.*x) + 0.21 * np.sin(2*np.pi*base_frequency*2*x)
if num_features > 2:
    Y[:, :, 2] = 0.45 * np.cos(2*np.pi*base_frequency*16/4*x) + 0.21 * np.sin(2*np.pi*base_frequency*4/4*x)
if num_features > 3:
    Y[:, :, 3] = 0.40 * np.cos(2*np.pi*base_frequency*18/4*x) + 0.21 * np.sin(2*np.pi*base_frequency*8/4*x)

# normalize Y, but over the second axis only
Y = norm_data(Y)
Y_masked = Y.copy()

# roughly 30 % of the samples are non-nan
ratio_of_not_sampling = 1 - 30/100 # as with the sine example
number_of_samples = int(x.shape[0] * ratio_of_not_sampling)

mask = np.random.rand(num_periods, x.shape[0], num_features) < ratio_of_not_sampling
Y_masked[mask] = 0.

#%%
plt.figure()
for idx in range(num_features):
    y_m = Y_masked[0, :, idx][~mask[0, :, idx]]
    x_m = x[~mask[0, :, idx]]
    plt.plot(x, Y[0, :, idx], '.', label=f"GT Ft. {idx}")
    plt.plot(x_m, y_m, '.', label=f"Subsampled Ft. {idx}")
plt.grid()
plt.legend()
plt.show()

#%% convert to tensor structure
periods_to_store = []

for per in range(num_periods):
    vals = Y[per]
    m = ~mask[per]
    tt = x[m.any(axis=1)]
    vals = vals[m.any(axis=1)]
    m = m[m.any(axis=1)] * 1

    if tt[0] != 0.0:
        tt = np.insert(tt, 0, 0.0)
        vals = np.insert(vals, 0, np.zeros(vals.shape[1]), axis=0)
        m = np.insert(m, 0, np.zeros(m.shape[1]), axis=0)

    vals = torch.tensor(vals)
    tt = torch.tensor(tt)
    m = torch.tensor(m)

    periods_to_store.append((per, tt, vals, m))

# testing the shapes before storing
for _, tt_, vals_, m_ in periods_to_store:
    assert tt_.shape[0] == vals_.shape[0]
    assert m_.shape == vals_.shape

torch.save(
    periods_to_store,
    os.path.join(f'data_dir/BasicData/processed/basic_data_num-ft_{num_features}.pt')
)
