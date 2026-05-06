import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_1.txt")
data = data.drop(columns=["Enable"])
labels = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_label_1.txt")

start_idx = 100000
end_idx = 125000
subsample = 10

col_idx = [0, 14, 12]

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
    (float(t[start]), float(t[min(end, num_samples - 1)] + dt))
    for start, end in zip(anomaly_starts, anomaly_ends)
]

#%%
# Plot styling chosen to keep anomalies salient without overpowering traces.
LINE_COLOR = "0.2"
ANOMALY_COLOR = "red"
ANOMALY_ALPHA = 0.1
GRID_ALPHA = 0.25
LINE_WIDTH = 1.0
LABEL_MA_WINDOW = 50
SEPARATOR_COLOR = "0.55"

label_kernel = np.ones(LABEL_MA_WINDOW, dtype=float) / LABEL_MA_WINDOW
labels_ma = -np.log(np.convolve(np.abs(labels.astype(float) - 1 - 1e-5), label_kernel, mode="same"))

n_channels = data.shape[1] + 1
fig = plt.figure(figsize=(15, 1.0 * n_channels + 1), constrained_layout=True)

spacer_height = 0.5                                    # shrink spacer too
height_ratios = [1.0] * (n_channels - 1) + [spacer_height, 1.0]

gs = fig.add_gridspec(
    nrows=n_channels + 1,
    ncols=1,
    height_ratios=height_ratios,
    hspace=0.04,
)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.02)

axs = []
for i in range(n_channels - 1):
    share_ax = axs[0] if axs else None
    axs.append(fig.add_subplot(gs[i, 0], sharex=share_ax))

# Draw a visual separator in the dedicated spacer row.
sep_ax = fig.add_subplot(gs[n_channels - 1, 0])
sep_ax.set_axis_off()
sep_ax.axhline(0.5, color=SEPARATOR_COLOR, linewidth=1.2, alpha=0.9)

# Add last channel after the spacer row to create a larger final gap.
axs.append(fig.add_subplot(gs[n_channels, 0], sharex=axs[0]))

for i in range(data.shape[1]):
    ax = axs[i]
    ax.plot(t, data.iloc[:, i], color=LINE_COLOR, linewidth=LINE_WIDTH)
    for span_start, span_end in anomaly_spans:
        ax.axvspan(span_start, span_end, color=ANOMALY_COLOR, alpha=ANOMALY_ALPHA, linewidth=0)
        ax.axvline(span_start, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
        ax.axvline(span_end, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
    ax.set_ylabel(str(data.columns[i]), rotation=90, va="center")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)

r = 2
labels_ma_bengign, labels_ma_anom = labels_ma.copy(), labels_ma.copy()
labels_ma_bengign[labels_ma > r] = np.nan
labels_ma_anom[labels_ma <= r] = np.nan

axs[-1].plot(t, labels_ma_bengign, markersize=1, color=LINE_COLOR,
             linewidth=LINE_WIDTH, label="Score $\\log p_\\theta(\\mathbf{x}^i \\mid \\mathbf{z})$")
axs[-1].plot(t, labels_ma_anom, markersize=2, color='red',
             linewidth=LINE_WIDTH*2, )
axs[-1].axhline(r, color="purple", linestyle="--", label="Threshold $r$")
axs[-1].set_ylabel("$\\log p_\\theta(\\mathbf{x}^i \\mid \\mathbf{z})$", rotation=90, va="center")
axs[-1].yaxis.set_label_coords(-0.08, 0.5)
axs[-1].set_xlabel("Time in window (s)")
axs[-2].set_xlabel("Time in window (s)")
axs[-1].set_xlim(t.min(), t.max()+1)
axs[-1].set_title("Anomaly Scores")
axs[0].set_title("Sensory Data")

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


window_start_s = start_idx / 100.0
window_end_s = end_idx / 100.0

plt.savefig("out/motivational_figure.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close('all')