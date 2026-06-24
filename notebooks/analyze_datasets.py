import ast
import glob
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from anomaly_detection import eval_scores
from scipy.stats import iqr

def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _to_2d_numpy(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def _align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if len(common_cols) == 0:
        raise ValueError("No common feature columns between train and test")
    return train_df[common_cols], test_df[common_cols], common_cols


def _scale_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df, test_df, cols = _align_columns(train_df, test_df)
    train_df = _numeric_clean(train_df)
    test_df = _numeric_clean(test_df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_np = scaler.fit_transform(train_df)
    test_np = scaler.transform(test_df)
    return train_np, test_np, cols


def load_swat(data_dir: str = "data_dir") -> dict:
    root = os.path.join(data_dir, "SWaT", "raw")
    train_df = _safe_read_csv(os.path.join(root, "train.csv"))
    test_df = _safe_read_csv(os.path.join(root, "test.csv"))
    labels = _safe_read_csv(os.path.join(root, "labels.csv")).to_numpy().squeeze()

    train_np, test_np, cols = _scale_train_test(train_df, test_df)
    return {"train": train_np, "test": test_np, "labels": labels, "columns": cols}


def load_wadi(data_dir: str = "data_dir") -> dict:
    root = os.path.join(data_dir, "WaDi", "raw", "v2")
    train_df = _safe_read_csv(os.path.join(root, "WADI_14days.csv"), index_col=0)
    test_df = _safe_read_csv(
        os.path.join(root, "WADI_attackdata_labelled.csv"),
        index_col=0,
        skiprows=1,
    )

    label_col = "Attack LABLE (1:No Attack, -1:Attack)"
    if label_col not in test_df.columns:
        for candidate in test_df.columns:
            uc = str(candidate).upper()
            if "LABLE" in uc and "ATTACK" in uc:
                label_col = candidate
                break

    labels_raw = pd.to_numeric(test_df[label_col], errors="coerce").fillna(1)
    labels = (labels_raw != 1).astype(int).to_numpy()
    test_df = test_df.drop(columns=[label_col], errors="ignore")

    date_like_cols = [c for c in test_df.columns if ("DATE" in str(c).upper() or "TIME" in str(c).upper())]
    if len(date_like_cols) > 0:
        train_df = train_df.drop(columns=date_like_cols, errors="ignore")
        test_df = test_df.drop(columns=date_like_cols, errors="ignore")

    train_np, test_np, cols = _scale_train_test(train_df, test_df)
    return {"train": train_np, "test": test_np, "labels": labels, "columns": cols}


def get_all_smd(mode="train"):
    all_files = glob.glob(os.path.join('data_dir/SMD/raw', mode, '*.txt'))

    data = {}

    for file_ in all_files:
        machine = file_.split('/')[-1].replace('.txt', '')
        data[machine] = {}
        data[machine]['train'] = None
        data[machine]['test'] = None
        data[machine]['labels'] = None
        if mode == 'train':
            data[machine]['train'] = pd.read_csv(file_, delimiter=',',
                                                 header=None)
        elif mode == 'test':
            data[machine]['test'] = pd.read_csv(file_, delimiter=',',
                                                header=None)
            label_file = file_.replace('test', 'test_label')
            data[machine]['labels'] = pd.read_csv(label_file, delimiter=',',
                                                  header=None)

    assert all(
        df[mode].shape[1] == next(iter(data.values()))[mode].shape[1]
        for df in
        data.values()), "Not all DataFrames have the same number of columns"

    return data




def minmax_scale_traces(smd_train, smd_test):
    for machine_key, _ in smd_train.items():
        scaler = MinMaxScaler(feature_range=(0, 1))
        smd_train[machine_key]["train"] = scaler.fit_transform(_numeric_clean(smd_train[machine_key]["train"]))
        smd_test[machine_key]["test"] = scaler.transform(_numeric_clean(smd_test[machine_key]["test"]))


def load_smd(data_dir: str = "data_dir") -> dict:
    train_data = get_all_smd("train")
    test_data = get_all_smd("test")
    minmax_scale_traces(train_data, test_data)

    merged = {}
    for machine in sorted(train_data.keys()):
        labels = test_data[machine]["labels"].to_numpy().squeeze().astype(float)
        merged[machine] = {
            "train": _to_2d_numpy(train_data[machine]["train"]),
            "test": _to_2d_numpy(test_data[machine]["test"]),
            "labels": labels,
            "columns": list(range(_to_2d_numpy(train_data[machine]["train"]).shape[1])),
        }
    return merged


def _load_qad_txt(path: str, is_label: bool = False):
    kwargs = {}
    if not is_label:
        kwargs["sep"] = None
        kwargs["engine"] = "python"

    df = _safe_read_csv(path, **kwargs)
    if is_label and isinstance(df, pd.DataFrame) and len(df.columns) == 1 and "labels" not in df.columns:
        df.columns = ["labels"]
    return df


def load_qad(data_dir: str = "data_dir", raw_subdir: str = "qad_clean_txt_100Hz") -> dict:
    root = os.path.join(data_dir, "QAD", "raw", raw_subdir)
    if not os.path.isdir(root):
        fallback = os.path.join(data_dir, "QAD", "raw", "qad_clean_txt_100Hz")
        root = fallback if os.path.isdir(fallback) else root

    out = {}
    for train_file in sorted(glob.glob(os.path.join(root, "train_*.txt"))):
        dsid = os.path.basename(train_file).replace("train_", "").replace(".txt", "")
        test_file = os.path.join(root, f"test_{dsid}.txt")
        label_file = os.path.join(root, f"test_label_{dsid}.txt")
        if not (os.path.isfile(test_file) and os.path.isfile(label_file)):
            continue

        train_df = _load_qad_txt(train_file, is_label=False)
        test_df = _load_qad_txt(test_file, is_label=False)
        labels = _load_qad_txt(label_file, is_label=True)
        labels = pd.to_numeric(labels.iloc[:, 0], errors="coerce").fillna(0).to_numpy().squeeze()

        train_np, test_np, cols = _scale_train_test(train_df, test_df)
        aligned_len = min(len(labels), test_np.shape[0])
        out[f"qad_{dsid}"] = {
            "train": train_np,
            "test": test_np[:aligned_len],
            "labels": labels[:aligned_len],
            "columns": cols,
        }

    return out


def load_psm(data_dir: str = "data_dir") -> dict:
    root = os.path.join(data_dir, "PSM", "raw")
    train_df = _safe_read_csv(os.path.join(root, "train.csv"))
    test_df = _safe_read_csv(os.path.join(root, "test.csv"))
    label_df = _safe_read_csv(os.path.join(root, "test_label.csv"))

    drop_cols = [c for c in train_df.columns if "timestamp" in str(c).lower()]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    labels = label_df["label"].to_numpy().squeeze() if "label" in label_df.columns else label_df.to_numpy().squeeze()
    train_np, test_np, cols = _scale_train_test(train_df, test_df)
    aligned_len = min(len(labels), test_np.shape[0])

    return {
        "train": train_np,
        "test": test_np[:aligned_len],
        "labels": labels[:aligned_len],
        "columns": cols,
    }


def _anomaly_mask_from_sequences(num_values: int, anomaly_sequences: str) -> np.ndarray:
    y = np.zeros(int(num_values), dtype=float)
    for start_idx, stop_idx in ast.literal_eval(anomaly_sequences):
        start = max(0, min(int(start_idx), len(y)))
        stop = max(start, min(int(stop_idx), len(y)))
        y[start:stop] = 1.0
    return y


def load_nasa(data_dir: str = "data_dir", spacecraft: str = "SMAP") -> dict:
    root = os.path.join(data_dir, "nasa", "raw")
    labels_csv = _safe_read_csv(os.path.join(root, "labeled_anomalies.csv"))
    labels_csv = labels_csv[labels_csv["spacecraft"] == spacecraft]

    out = {}
    for _, row in labels_csv.iterrows():
        chan_id = str(row["chan_id"])
        train_path = os.path.join(root, "train", f"{chan_id}.npy")
        test_path = os.path.join(root, "test", f"{chan_id}.npy")
        if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
            continue

        train_np = _to_2d_numpy(np.load(train_path))
        test_np = _to_2d_numpy(np.load(test_path))
        y = _anomaly_mask_from_sequences(int(row["num_values"]), row["anomaly_sequences"])

        train_df = pd.DataFrame(train_np)
        test_df = pd.DataFrame(test_np)
        train_scaled, test_scaled, cols = _scale_train_test(train_df, test_df)

        aligned_len = min(len(y), test_scaled.shape[0])
        out[chan_id] = {
            "train": train_scaled,
            "test": test_scaled[:aligned_len],
            "labels": y[:aligned_len],
            "columns": cols,
        }

    return out


def _safe_load(label: str, fn, *args, **kwargs) -> Optional[dict]:
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        print(f"[WARN] {label}: {exc}")
    except Exception as exc:
        print(f"[WARN] {label}: {type(exc).__name__}: {exc}")
    return None


def load_all_datasets(data_dir: str = "data_dir", qad_subdir: str = "qad_clean_txt_100Hz") -> Dict[str, Optional[dict]]:
    return {
        "SWaT": _safe_load("SWaT", load_swat, data_dir),
        "WaDi": _safe_load("WaDi", load_wadi, data_dir),
        "SMD": _safe_load("SMD", load_smd, data_dir),
        "QAD": _safe_load("QAD", load_qad, data_dir, qad_subdir),
        "PSM": _safe_load("PSM", load_psm, data_dir),
        "SMAP": _safe_load("SMAP", load_nasa, data_dir, "SMAP"),
        "MSL": _safe_load("MSL", load_nasa, data_dir, "MSL"),
    }


# Notebook-friendly object that exposes all benchmark datasets in one place.
all_datasets = load_all_datasets()

