from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from anomaly_detection import build_modules_and_optim, calculate_z_normalization_values
from data.qad_provider import QADProvider


def _select_checkpoint_path() -> Path:
    pattern = "checkpoint_AD_QAD_*_1_60.h5"
    candidates = sorted(Path("checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in checkpoints/ matching '{pattern}'")
    return candidates[-1]


def _score_trace_with_checkpoint(checkpoint_path: Path, trace_id: int = 1) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint["args"]
    args.device = device

    provider = QADProvider(
        data_dir=getattr(args, "data_dir", "data_dir"),
        dataset_number=trace_id,
        window_length=args.data_window_length,
        window_overlap=getattr(args, "data_window_overlap", 0.0),
        data_normalization_strategy=getattr(args, "data_normalization_strategy", "none"),
        subsample=args.subsample,
        seed=getattr(args, "seed", -1),
        fixed_subsample_mask=getattr(args, "fixed_subsample_mask", False),
        raw_subdir="qad_clean_txt_100Hz",
    )

    desired_t = checkpoint["desired_t"].to(device)
    modules, _, _, _ = build_modules_and_optim(args, provider.input_dim, desired_t)
    modules.load_state_dict(checkpoint["modules"])
    modules.eval()

    batch_size = max(1, int(getattr(args, "batch_size", 256)))
    dl_tst = provider.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=0,
        pin_memory=False,
    )

    normalization_stats = None
    if getattr(args, "normalize_score", False):
        dl_trn = provider.get_train_loader(
            batch_size=batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        normalization_stats = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)

    n_time = int(dl_tst.dataset.indcs.max().item()) + 1
    all_scores = np.zeros((n_time, dl_tst.dataset.input_dim), dtype=np.float64)
    normalize_counts = np.zeros(n_time, dtype=np.float64)

    with torch.no_grad():
        for batch in dl_tst:
            parts = {key: val.to(device) for key, val in batch.items()}
            indcs = parts["inp_indcs"].cpu().numpy().astype(int)
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            h = modules["recog_net"](inp)
            qzx, _ = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_eval_samples,))
            pxz = modules["pxz_net"](zis)

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]

            if normalization_stats is not None:
                denom = (normalization_stats["max"] - normalization_stats["min"]).clamp_min(1e-8)
                aux_log_prob = (aux_log_prob - normalization_stats["min"]) / denom

            if aux_log_prob.dim() == 4:
                aux_log_prob = aux_log_prob.mean(axis=0)

            for idx in range(aux_log_prob.shape[0]):
                all_scores[indcs[idx, :], :] += aux_log_prob[idx, :, :].cpu().numpy()

            values, counts = np.unique(indcs, return_counts=True)
            for key, value in zip(values, counts):
                normalize_counts[key] += value

    provider.cleanup()

    all_scores = np.divide(
        all_scores,
        normalize_counts[:, None],
        out=np.zeros_like(all_scores),
        where=normalize_counts[:, None] > 0,
    )
    return np.linalg.norm(all_scores, ord=1, axis=1)


#%%
data = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_1.txt")
data = data.drop(columns=["Enable"])
labels = pd.read_csv("data_dir/QAD/raw/qad_clean_txt_100Hz/test_label_1.txt")

start_idx = 130000 #115000
end_idx = 200000 #122500
subsample = 10
col_idx = [0, 14, 12]
SCORE_MA_WINDOW = 30
q = 99.0

window_length = 5000
max_signal_len = data.shape[0] // window_length * window_length
data = data[:max_signal_len]
labels = labels[:max_signal_len]

data = data.iloc[start_idx:end_idx:subsample, col_idx]
labels = labels.iloc[start_idx:end_idx:subsample, 0].to_numpy() == 1

#%%
checkpoint_path = _select_checkpoint_path()
scores_full = _score_trace_with_checkpoint(checkpoint_path, trace_id=1)
r = float(np.nanpercentile(scores_full, q))
scores = scores_full[start_idx//10:end_idx//10]

#%%
if len(labels) != len(data) or len(scores) != len(data):
    raise ValueError(
        f"Length mismatch: labels={len(labels)}, scores={len(scores)}, data={len(data)}"
    )

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

LINE_COLOR = "0.2"
ANOMALY_COLOR = "red"
ANOMALY_ALPHA = 0.1
GRID_ALPHA = 0.25
LINE_WIDTH = 1.0
SEPARATOR_COLOR = "0.55"
YLABEL_X = -0.05

score_kernel = np.ones(SCORE_MA_WINDOW, dtype=float) / SCORE_MA_WINDOW
scores_ma = np.convolve(scores, score_kernel, mode="same")

n_channels = data.shape[1] + 1
spacer_height = 0.6

fig = plt.figure(figsize=(15, 1.2 * n_channels + 1.0), constrained_layout=False)


height_ratios = [1.2] * (n_channels - 1) + [spacer_height, 1.0]
gs = fig.add_gridspec(nrows=n_channels + 1, ncols=1, height_ratios=height_ratios, hspace=0.06)

axs = []
for i in range(n_channels - 1):
    share_ax = axs[0] if axs else None
    axs.append(fig.add_subplot(gs[i, 0], sharex=share_ax))

sep_ax = fig.add_subplot(gs[n_channels - 1, 0])
sep_ax.set_axis_off()
#sep_ax.axhline(0.5, color=SEPARATOR_COLOR, linewidth=1.2, alpha=0.9)

axs.append(fig.add_subplot(gs[n_channels, 0], sharex=axs[0]))

for i in range(data.shape[1]):
    ax = axs[i]
    ax.plot(t, data.iloc[:, i], color=LINE_COLOR, linewidth=LINE_WIDTH)
    for span_start, span_end in anomaly_spans:
        ax.axvspan(span_start, span_end, color=ANOMALY_COLOR, alpha=ANOMALY_ALPHA, linewidth=0)
        ax.axvline(span_start, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
        ax.axvline(span_end, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
    ax.set_ylabel(str(data.columns[i]), rotation=90, va="center")
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)

scores_benign = scores_ma.copy()
scores_anom = scores_ma.copy()
scores_benign[scores_ma > r] = np.nan
scores_anom[scores_ma <= r] = np.nan

axs[-1].plot(
    t,
    scores_benign,
    color=LINE_COLOR,
    linewidth=LINE_WIDTH,
    label="Score from stored model",
)
axs[-1].plot(t, scores_anom, color="red", linewidth=LINE_WIDTH * 1.5)
axs[-1].axhline(r, color="purple", linestyle="--", label="95th percentile threshold")
axs[-1].set_ylabel("Anomaly score", rotation=90, va="center")
axs[-1].yaxis.set_label_coords(YLABEL_X, 0.5)
axs[-1].set_xlabel("Time in window (s)")
axs[-1].set_xlim(t.min(), t.max() + 1)
axs[-1].set_title(f"Anomaly Scores")#, fontsize=10)
axs[0].set_title("Sensory Data")

# Keep time ticks only on the bottom subplot to avoid repeated labels.
for ax in axs[:-1]:
    ax.tick_params(axis="x", which="both", labelbottom=False)

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

Path("out").mkdir(exist_ok=True)
fig.tight_layout(pad=0.2, h_pad=0.1)
plt.savefig("out/motivational_figure.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close("all")
