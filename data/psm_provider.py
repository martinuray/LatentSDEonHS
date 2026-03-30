import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, normalize_masked_data
from data.dataset_provider import DatasetProvider


def _load_psm_csv(path: str) -> np.ndarray:
    """Load a PSM feature CSV, dropping the leading timestamp column."""
    df = pd.read_csv(path)
    ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
    df = df.drop(columns=ts_cols, errors="ignore")
    arr = df.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _load_psm_labels(path: str) -> np.ndarray:
    """Load PSM test_label.csv, returning a 1-D float32 label array."""
    df = pd.read_csv(path)
    return df["label"].to_numpy(dtype=np.float32)


def _windowize(data: np.ndarray, window_length: int, window_overlap: float):
    """Return (starts, windows) with drop-tail behaviour."""
    if data.ndim == 1:
        data = data[:, None]

    stride = window_length if window_overlap <= 0.0 else max(1, int(window_length * (1.0 - window_overlap)))

    starts = np.arange(0, data.shape[0] - window_length + 1, stride, dtype=np.int64)
    if starts.size == 0:
        return starts, np.zeros((0, window_length, data.shape[1]), dtype=data.dtype)

    windows = np.stack([data[s:s + window_length] for s in starts], axis=0)
    return starts, windows


class PSMData:
    params = None
    labels = ["Anomaly"]
    labels_dict = {"Anomaly": 0}

    def __init__(
        self,
        root_path,
        mode: str = "train",
        window_length: int = 100,
        window_overlap: float = 0.0,
        normalizer=None,
        data_normalization_strategy: str = "none",
    ):
        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root_path = root_path
        self.mode = mode
        self.window_length = window_length
        self.window_overlap = window_overlap

        if not self._check_exists():
            if not self._check_exist_raw_data():
                raise RuntimeError(f"PSM raw data not found in {self.raw_folder}")
            self._process_psm_data()

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)
        if self.mode == "test":
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, "PSM", "raw")

    @property
    def processed_folder(self):
        ov = str(self.window_overlap).replace(".", "p")
        return os.path.join(self.root_path, "PSM", "processed", f"wl{self.window_length}_ov{ov}")

    @property
    def training_file(self):
        return "train_psm.pt"

    @property
    def test_file(self):
        return "test_psm.pt"

    @property
    def val_file(self):
        return "val_psm.pt"

    @property
    def label_file(self):
        return "labels_psm.pt"

    @property
    def destination_file(self):
        return {"train": self.training_file, "test": self.test_file, "val": self.val_file}[self.mode]

    def _check_exist_raw_data(self):
        os.makedirs(self.processed_folder, exist_ok=True)
        required = [
            os.path.join(self.raw_folder, "train.csv"),
            os.path.join(self.raw_folder, "test.csv"),
            os.path.join(self.raw_folder, "test_label.csv"),
        ]
        return all(os.path.isfile(f) for f in required)

    def _check_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _apply_optional_scaler(self, arr: np.ndarray) -> np.ndarray:
        if self.scaler is None and self.data_normalization_strategy != "none":
            if self.data_normalization_strategy not in ["std", "min-max"]:
                raise ValueError(f"Invalid normalization strategy: {self.data_normalization_strategy}")
            self.scaler = StandardScaler() if self.data_normalization_strategy == "std" else MinMaxScaler()
            self.scaler.fit(arr)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr

    def _build_data_tuples(self, starts: np.ndarray, windows: np.ndarray):
        data_tensor = torch.tensor(windows, dtype=torch.float32)
        mask = torch.ones_like(data_tensor)
        mask[data_tensor.isnan()] = 0
        data_tensor[data_tensor.isnan()] = 0

        local_t = np.arange(self.window_length)
        return [
            (part_idx, torch.tensor(local_t + int(s), dtype=torch.long),
             data_tensor[part_idx], mask[part_idx])
            for part_idx, s in enumerate(starts.tolist())
        ]

    def _process_psm_data(self):
        logging.warning(f"Processing PSM data (mode={self.mode})")

        if self.mode in ["train", "val"]:
            arr = _load_psm_csv(os.path.join(self.raw_folder, "train.csv"))
            arr = self._apply_optional_scaler(arr)

            starts, windows = _windowize(arr, self.window_length, self.window_overlap)
            data = self._build_data_tuples(starts, windows)

            n = len(data)
            split_idx = int(n * 0.9)
            idxs = np.random.permutation(n)
            torch.save([data[i] for i in idxs[:split_idx]], os.path.join(self.processed_folder, self.training_file))
            torch.save([data[i] for i in idxs[split_idx:]], os.path.join(self.processed_folder, self.val_file))

        elif self.mode == "test":
            arr = _load_psm_csv(os.path.join(self.raw_folder, "test.csv"))
            arr = self._apply_optional_scaler(arr)

            starts, windows = _windowize(arr, self.window_length, self.window_overlap)
            data = self._build_data_tuples(starts, windows)

            labels = _load_psm_labels(os.path.join(self.raw_folder, "test_label.csv"))
            _, label_windows = _windowize(labels[:, None], self.window_length, self.window_overlap)
            label_windows = label_windows.squeeze(-1)

            torch.save(data, os.path.join(self.processed_folder, self.test_file))
            torch.save(label_windows, os.path.join(self.processed_folder, self.label_file))


class PSMDataset(Dataset):
    input_dim = None

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        window_length: int = 100,
        window_overlap: float = 0.0,
        subsample: float = 1.0,
        data_normalization_strategy: str = "none",
    ):
        self.mode = mode
        self.subsample = subsample

        self.datasets = []
        self._lengths = []
        self._cumulative = []

        train_data = PSMData(
            data_dir,
            mode="train",
            window_length=window_length,
            window_overlap=window_overlap,
            data_normalization_strategy=data_normalization_strategy,
        )

        objs = {
            "train": train_data,
            "test": PSMData(
                data_dir, mode="test",
                window_length=window_length, window_overlap=window_overlap,
                normalizer=train_data.scaler,
            ),
            "val": PSMData(
                data_dir, mode="val",
                window_length=window_length, window_overlap=window_overlap,
                normalizer=train_data.scaler,
            ),
        }

        data = objs[mode]
        raw = data.data

        if len(raw) == 0:
            logging.warning("PSM dataset is empty.")
            return

        data_min, data_max = get_data_min_max(objs["train"][:])

        tps_base = raw[0][1].float()
        tps_max = tps_base.max()
        if tps_max > 0:
            tps_base = tps_base / tps_max

        indcs = torch.stack([raw[i][1] for i in range(len(raw))])
        obs = torch.stack([raw[i][2] for i in range(len(raw))]).float()
        msk = torch.stack([raw[i][3] for i in range(len(raw))]).float()
        tps = tps_base[None, :].repeat(obs.shape[0], 1).float()

        if mode == "test":
            tgt = torch.tensor(data.targets)
        else:
            tgt = torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.long)

        obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)

        n_samples = obs.shape[0]
        n_time = tps.shape[1]
        self._lengths.append(n_samples)

        self.datasets.append({
            "inp_obs": (obs * msk).float(),
            "inp_msk": msk.long(),
            "inp_tps": tps,
            "inp_tid": torch.arange(n_time).repeat(n_samples, 1).long(),
            "indcs": indcs,
            "evd_obs": obs.float(),
            "evd_msk": torch.ones_like(msk).long(),
            "evd_tid": torch.arange(n_time).repeat(n_samples, 1).long(),
            "evd_tps": tps,
            "aux_tgt": tgt.long(),
            "data_min": data_min,
            "data_max": data_max,
            "input_dim": obs.shape[-1],
            "num_timepoints": n_time,
            "dataset_id": "PSM",
        })

        self._cumulative.append(0)

        ds0 = self.datasets[0]
        PSMDataset.input_dim = ds0["input_dim"]
        self.input_dim = ds0["input_dim"]
        self.num_timepoints = ds0["num_timepoints"]
        self.data_min = ds0["data_min"]
        self.data_max = ds0["data_max"]
        self.indcs = ds0["indcs"]

    @property
    def has_aux(self):
        return False

    @property
    def num_datasets(self) -> int:
        return len(self.datasets)

    def get_dataset(self, ds_idx: int) -> dict:
        return self.datasets[ds_idx]

    @property
    def input_dims(self) -> list:
        return [ds["input_dim"] for ds in self.datasets]

    @property
    def num_timepoints_list(self) -> list:
        return [ds["num_timepoints"] for ds in self.datasets]

    def __len__(self):
        return sum(self._lengths)

    def _resolve_index(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        for ds_idx in range(len(self._lengths) - 1, -1, -1):
            if idx >= self._cumulative[ds_idx]:
                return ds_idx, idx - self._cumulative[ds_idx]
        raise IndexError(f"Index {idx} could not be resolved")

    def __getitem__(self, idx):
        ds_idx, local_idx = self._resolve_index(idx)
        ds = self.datasets[ds_idx]

        if self.mode == "train":
            msk = (torch.rand(ds["inp_msk"][local_idx].shape) < self.subsample).to(torch.int).long()
        else:
            msk = ds["inp_msk"][local_idx].long()

        return {
            "inp_obs": ds["inp_obs"][local_idx].float(),
            "inp_msk": msk,
            "inp_tid": ds["inp_tid"][local_idx].long(),
            "inp_tps": ds["inp_tps"][local_idx].float(),
            "evd_obs": ds["evd_obs"][local_idx].float(),
            "evd_msk": ds["evd_msk"][local_idx].long(),
            "evd_tid": ds["evd_tid"][local_idx].long(),
            "evd_tps": ds["evd_tps"][local_idx].float(),
            "aux_tgt": ds["aux_tgt"][local_idx].long(),
            "inp_indcs": ds["indcs"][local_idx],
            "dataset_idx": ds_idx,
        }


class PSMProvider(DatasetProvider):
    def __init__(
        self,
        data_dir=None,
        window_length: int = 100,
        window_overlap: float = 0.0,
        data_normalization_strategy: str = "none",
        subsample: float = 1.0,
    ):
        super().__init__()

        common_kwargs = {
            "window_length": window_length,
            "window_overlap": window_overlap,
            "data_normalization_strategy": data_normalization_strategy,
        }

        self._ds_trn = PSMDataset(data_dir, "train", subsample=subsample, **common_kwargs)
        self._ds_tst = PSMDataset(data_dir, "test", **common_kwargs)
        self._ds_val = PSMDataset(data_dir, "val", subsample=subsample, **common_kwargs)

    @property
    def input_dim(self):
        return PSMDataset.input_dim

    @property
    def input_dims(self) -> list:
        return self._ds_trn.input_dims

    @property
    def num_datasets(self) -> int:
        return self._ds_trn.num_datasets

    @property
    def num_timepoints_list(self) -> list:
        return self._ds_trn.num_timepoints_list

    @property
    def data_min(self):
        return self._ds_trn.data_min

    @property
    def data_max(self):
        return self._ds_trn.data_max

    @property
    def num_timepoints(self):
        return self._ds_trn.num_timepoints

    @property
    def num_train_samples(self) -> int:
        return len(self._ds_trn)

    @property
    def num_test_samples(self) -> int:
        return len(self._ds_tst)

    @property
    def num_val_samples(self) -> int:
        return len(self._ds_val)

    def get_train_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_trn, **kwargs)

    def get_test_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_tst, **kwargs)

    def get_val_loader(self, **kwargs):
        return DataLoader(self._ds_val, **kwargs)

