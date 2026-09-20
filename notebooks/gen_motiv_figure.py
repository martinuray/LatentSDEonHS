import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from anomaly_detection import (
    build_modules_and_optim,
    calculate_feature_reconstruction_weights,
    calculate_z_normalization_values,
)
from data.qad_provider import QADProvider, load_qad_pkl


def _select_checkpoint_path() -> Path:
    pattern = "checkpoint_qad_*_best_Rn.h5"
    candidates = sorted(Path("checkpoints").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in checkpoints/ matching '{pattern}'")
    return candidates[-1]


def _resolve_trace_id(args, default: int = 1) -> int:
    for attr in ("trace_id", "dataset_number"):
        value = getattr(args, attr, None)
        if value is not None:
            return int(value)

    trace_ids = getattr(args, "trace_ids", None)
    if trace_ids:
        first = trace_ids[0]
        if isinstance(first, (list, tuple)):
            first = first[0]
        if isinstance(first, str) and "," in first:
            first = first.split(",", 1)[0]
        try:
            return int(first)
        except (TypeError, ValueError):
            pass

    return int(default)


def _load_qad_raw_trace(data_dir: str, trace_id: int, decimation_factor: int):
    raw_dir = Path(data_dir)
    if not (raw_dir / f"test_{trace_id}.pkl").exists():
        raw_dir = raw_dir / "QAD" / "raw"
    data = load_qad_pkl(raw_dir / f"test_{trace_id}.pkl")
    labels = load_qad_pkl(raw_dir / f"test_label_{trace_id}.pkl", is_label=True)

    if "Enable" in data.columns:
        data = data.drop(columns=["Enable"])

    data = data.iloc[::decimation_factor].reset_index(drop=True)
    labels = labels.iloc[::decimation_factor].reset_index(drop=True)

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] > 1:
            labels = labels.iloc[:, 0:1]
    else:
        labels = labels.to_frame(name="labels")

    aligned_len = min(len(data), len(labels))
    data = data.iloc[:aligned_len].reset_index(drop=True)
    labels = labels.iloc[:aligned_len].reset_index(drop=True)

    return data, labels


def _score_trace_with_checkpoint(checkpoint_path: Path, trace_id: int = 1) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint["args"]
    args.device = device
    args.mc_eval_samples = 5
    decimation_factor = max(1, int(getattr(args, "data_decimation_factor", 10)))

    provider = QADProvider(
        data_dir=getattr(args, "data_dir", "data_dir"),
        dataset_number=trace_id,
        window_length=args.data_window_length,
        window_overlap=getattr(args, "data_window_overlap", 0.0),
        data_normalization_strategy=getattr(args, "data_normalization_strategy", "none"),
        subsample=args.subsample,
        seed=getattr(args, "seed", -1),
        fixed_subsample_mask=getattr(args, "fixed_subsample_mask", False),
        decimation_factor=decimation_factor,
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

    dl_trn = provider.get_train_loader(
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    normalization_stats = None
    if getattr(args, "normalize_score", False):
        normalization_stats = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)

    # Weighted-mean channel aggregation: weight each channel's -log p_theta by
    # its inverse training-reconstruction MSE (see calculate_feature_reconstruction_weights),
    # so channels the model reconstructs faithfully on nominal training data
    # dominate the score, instead of the previous plain per-channel sum.
    weighting = "exp-inverse" if getattr(args, "score_aggregation", None) == "weighted-mse-exp" else "inverse"
    feature_weights = calculate_feature_reconstruction_weights(
        args, dl_trn, modules, desired_t, device, weighting=weighting
    )["feature_weights"]

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
    return (all_scores * feature_weights.reshape(1, -1)).sum(axis=1)



#%%
checkpoint_path = _select_checkpoint_path()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
checkpoint_args = checkpoint["args"]
trace_id = _resolve_trace_id(checkpoint_args, default=1)
data_dir = getattr(checkpoint_args, "data_dir", "data_dir")
decimation_factor = max(1, int(getattr(checkpoint_args, "data_decimation_factor", 10)))

data, labels = _load_qad_raw_trace(data_dir, trace_id, decimation_factor)

start_idx = 0 #115000
end_idx = -1 #122500
subsample = 1 #decimation_factor*10
col_idx = [0, 1, 3]
SCORE_MA_WINDOW = 30
q = 99.0

window_length = 5000
scores_full = _score_trace_with_checkpoint(checkpoint_path, trace_id=trace_id)
r = float(np.nanpercentile(scores_full, q))

data = data.iloc[start_idx // subsample:end_idx // subsample -1, col_idx]
labels = labels.iloc[start_idx // subsample:end_idx // subsample -1, 0].to_numpy() == 1
scores = scores_full[start_idx // subsample:end_idx // subsample]

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


#%% Fig 1. Motivational Figure
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
axs[-1].set_ylabel("$-\\log p_\theta$", rotation=90, va="center")
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

import sys; sys.exit(0)

#%% Content for Fig 2. ML Flow
data_len = 650
fig, axs = plt.subplots(nrows=3, ncols = 1, figsize=(5, 4 ))
for i in range(data.shape[1]):
    ax = axs[i]
    ax.plot(t[:data_len], data.iloc[:data_len, i], color=LINE_COLOR, linewidth=LINE_WIDTH)
    #for span_start, span_end in anomaly_spans:
    #    ax.axvspan(span_start, span_end, color=ANOMALY_COLOR, alpha=ANOMALY_ALPHA, linewidth=0)
    #    ax.axvline(span_start, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
    #    ax.axvline(span_end, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
    #ax.set_ylabel(str(data.columns[i]), rotation=90, va="center")
    #ax.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax.set_xlim(t[:data_len].min(), t[:data_len].max())
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)

plt.savefig(f'/tmp/motiv_fig/train.pdf', dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.show()

starts = [1000, 2900]

for idx, start in enumerate(starts):
    fig, axs = plt.subplots(nrows=3, ncols = 1, figsize=(5, 4 ))
    for i in range(data.shape[1]):
        ax = axs[i]
        ax.plot(t[start:start+data_len], data.iloc[start:start+data_len, i], color=LINE_COLOR, linewidth=LINE_WIDTH)
        for span_start, span_end in anomaly_spans:
            ax.axvspan(span_start, span_end, color=ANOMALY_COLOR, alpha=ANOMALY_ALPHA, linewidth=0)
            ax.axvline(span_start, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
            ax.axvline(span_end, color=ANOMALY_COLOR, alpha=0.5, linewidth=0.8, linestyle="--")
        #ax.set_ylabel(str(data.columns[i]), rotation=90, va="center")
        #ax.yaxis.set_label_coords(YLABEL_X, 0.5)
        ax.set_xlim(t[start:start+data_len].min(), t[start:start+data_len].max())
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        #ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)

    plt.tight_layout()
    plt.savefig(f'/tmp/motiv_fig/test_{idx}.pdf', dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4/3))
    axs.plot(t[start:start+data_len], scores_benign[start:start+data_len],
             color=LINE_COLOR, linewidth=LINE_WIDTH)
    axs.plot(t[start:start + data_len], scores_anom[start:start + data_len],
             color='red', linewidth=LINE_WIDTH*2)
    axs.set_ylim(0, 50)
    axs.set_yticklabels([])
    axs.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(f'/tmp/motiv_fig/score_{idx}.pdf', dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

plt.close('all')


#%% Zip all PDF files
import zipfile
import os

motiv_fig_dir = Path('/tmp/motiv_fig')
zip_path = motiv_fig_dir / 'motiv_figures.zip'

pdf_patterns = ['train*.pdf', 'test_*.pdf', 'score_*.pdf']
pdf_files = []
for pattern in pdf_patterns:
    pdf_files.extend(motiv_fig_dir.glob(pattern))

if pdf_files:
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pdf_file in sorted(pdf_files):
            zipf.write(pdf_file, arcname=pdf_file.name)
    print(f"Created archive: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"Files in archive: {len(pdf_files)}")
else:
    print("No PDF files found matching patterns: train*.pdf, test_*.pdf, score_*.pdf")
