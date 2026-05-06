import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_1.txt")
data = data.drop(columns=["Enable"])
labels = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_label_1.txt")

start_idx = 100000
end_idx = 125000
subsample = 10

col_idx = np.arange(0, data.shape[1])
col_idx = [0, 4, 12, 14]

data = data.iloc[start_idx:end_idx:subsample, col_idx] # downsample to 10Hz
labels = labels.iloc[start_idx:end_idx:subsample, 0].to_numpy() == 1

if len(labels) != len(data):
    raise ValueError(f"Label/data length mismatch: {len(labels)} labels for {len(data)} samples")

num_samples = data.shape[0]
dt = subsample / 100.0
t = np.arange(num_samples) * dt

label_edges = np.diff(np.pad(labels.astype(np.int8), (1, 1)))
anomaly_starts = np.flatnonzero(label_edges == 1)
anomaly_ends = np.flatnonzero(label_edges == -1)
anomaly_spans = [
    (t[start], t[min(end, num_samples - 1)] + dt)
    for start, end in zip(anomaly_starts, anomaly_ends)
]

#%%
fig, axs = plt.subplots(nrows=data.shape[1], ncols=1, figsize=(10, 1.25*data.shape[1]), sharex=True)
axs = np.atleast_1d(axs)

for i in range(data.shape[1]):
    axs[i].plot(t, data.iloc[:, i], label=data.columns[i])
    for span_start, span_end in anomaly_spans:
        axs[i].axvspan(span_start, span_end, color="red", alpha=0.15, linewidth=0)
    axs[i].set_title(data.columns[i])

axs[-1].set_xlabel("Time (s)")
axs[-1].set_xlim(t.min(), t.max()+1)
plt.tight_layout()
plt.show()
