import glob
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, normalize_masked_data
from data.dataset_provider import DatasetProvider
from utils.anomaly_detection import create_random_burst_mask


FILENAME_RE = re.compile(
    r"^(?P<index>\d+)_(?P<dataset_name>.+?)_id_(?P<dataset_id>\d+)_(?P<domain>.+?)_tr_(?P<train_index>\d+)_1st_(?P<first_anomaly_index>\d+)\.csv$"
)


def _parse_filename(path: str | Path) -> dict:
    match = FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"TSB-AD-M filename does not match expected pattern: {Path(path).name}")
    info = match.groupdict()
    return {
        "file_index": int(info["index"]),
        "dataset_name": info["dataset_name"],
        "dataset_id": int(info["dataset_id"]),
        "domain": info["domain"],
        "train_index": int(info["train_index"]),
        "first_anomaly_index": int(info["first_anomaly_index"]),
    }


def _load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df


def _pick_label_column(df: pd.DataFrame) -> str:
    for col in reversed(list(df.columns)):
        if "label" in str(col).strip().lower():
            return str(col)
    return str(df.columns[-1])


def _windowize(data: np.ndarray, window_length: int, window_overlap: float):
    if data.ndim == 1:
        data = data[:, None]

    stride = window_length if window_overlap <= 0.0 else max(1, int(window_length * (1.0 - window_overlap)))
    n_steps = data.shape[0]
    starts = np.arange(0, n_steps - window_length + 1, stride, dtype=np.int64)
    if starts.size == 0:
        return starts, np.zeros((0, window_length, data.shape[1]), dtype=data.dtype)

    windows = np.stack([data[s:s + window_length] for s in starts], axis=0)
    return starts, windows


class TSBADMData:
    params = None
    labels = ["Anomaly"]
    labels_dict = {"Anomaly": 0}

    def __init__(
        self,
        root_path,
        dataset_file: str | Path,
        mode: str = "train",
        window_length: int = 100,
        window_overlap: float = 0.0,
        normalizer=None,
        data_normalization_strategy: str = "none",
        processed_root: str = None,
    ):
        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root_path = root_path
        self.dataset_file = Path(dataset_file)
        self.mode = mode
        self.window_length = window_length
        self.window_overlap = window_overlap
        self._processed_root = processed_root


        self.meta = _parse_filename(self.dataset_file)
        self.dataset_stem = self.dataset_file.stem

        if not self._check_exists():
            self._process_tsb_ad_m_data()

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)
        if self.mode == "test":
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, "TSB-AD-M", "raw")

    @property
    def processed_folder(self):
        if self._processed_root is not None:
            return self._processed_root
        return os.path.join(self.root_path, "TSB-AD-M", "processed")

    @property
    def training_file(self):
        return f"train_{self.dataset_stem}.pt"

    @property
    def test_file(self):
        return f"test_{self.dataset_stem}.pt"

    @property
    def val_file(self):
        return f"val_{self.dataset_stem}.pt"

    @property
    def label_file(self):
        return f"labels_{self.dataset_stem}.pt"

    @property
    def destination_file(self):
        return {
            "train": self.training_file,
            "test": self.test_file,
            "val": self.val_file,
        }[self.mode]

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

        tuples = []
        local_t = np.arange(self.window_length)
        for part_idx, s in enumerate(starts.tolist()):
            indcs = torch.tensor(local_t + int(s), dtype=torch.long)
            tuples.append((part_idx, indcs, data_tensor[part_idx], mask[part_idx]))
        return tuples

    def _process_tsb_ad_m_data(self):
        logging.warning(
            "Processing TSB-AD-M file %s (mode=%s, train_index=%d, first_anomaly_index=%d)",
            self.dataset_stem,
            self.mode,
            self.meta["train_index"],
            self.meta["first_anomaly_index"],
        )

        os.makedirs(self.processed_folder, exist_ok=True)
        df = _load_csv(self.dataset_file)
        if df.empty:
            torch.save([], os.path.join(self.processed_folder, self.destination_file))
            if self.mode == "test":
                torch.save(np.zeros(0, dtype=np.float32), os.path.join(self.processed_folder, self.label_file))
            return

        label_col = _pick_label_column(df)
        feature_df = df.drop(columns=[label_col], errors="ignore")
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        labels = (labels > 0).astype(np.float32)

        total_len = len(df)
        train_index = int(self.meta["train_index"])
        train_index = max(0, min(train_index, total_len))

        if self.mode == "test":
            feature_arr = feature_df.iloc[train_index:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            labels_arr = labels[train_index:]
        else:
            feature_arr = feature_df.iloc[:train_index].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            labels_arr = np.zeros(feature_arr.shape[0], dtype=np.float32)

        feature_arr = self._apply_optional_scaler(feature_arr)
        starts, windows = _windowize(feature_arr, self.window_length, self.window_overlap)
        data = self._build_data_tuples(starts, windows)

        if self.mode == "train":
            data_len = len(data)
            if data_len == 0:
                torch.save([], os.path.join(self.processed_folder, self.training_file))
                torch.save([], os.path.join(self.processed_folder, self.val_file))
                return
            indices = np.random.permutation(data_len)
            split_idx = int(data_len * 0.9)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            torch.save([data[i] for i in train_indices], os.path.join(self.processed_folder, self.training_file))
            torch.save([data[i] for i in val_indices], os.path.join(self.processed_folder, self.val_file))
        elif self.mode == "val":
            data_len = len(data)
            if data_len == 0:
                torch.save([], os.path.join(self.processed_folder, self.val_file))
                return
            indices = np.random.permutation(data_len)
            split_idx = int(data_len * 0.9)
            val_indices = indices[split_idx:]
            torch.save([data[i] for i in val_indices], os.path.join(self.processed_folder, self.val_file))
        elif self.mode == "test":
            _, label_windows = _windowize(labels_arr[:, None], self.window_length, self.window_overlap)
            label_windows = label_windows.squeeze(-1)
            torch.save(data, os.path.join(self.processed_folder, self.test_file))
            torch.save(label_windows, os.path.join(self.processed_folder, self.label_file))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


class TSBADMDataset(Dataset):
    input_dim = None

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        dataset_number=None,
        window_length: int = 100,
        window_overlap: float = 0.0,
        subsample: float = 1.0,
        seed: int = -1,
        data_normalization_strategy: str = "none",
        fixed_subsample_mask: bool = False,
        processed_root: str = None,
    ):
        self.mode = mode
        self.subsample = subsample
        self.fixed_subsample_mask = fixed_subsample_mask
        self.seed = seed

        self.datasets = []
        self._lengths = []
        self._cumulative = []

        dataset_paths = self._resolve_dataset_paths(data_dir, dataset_number)

        for dataset_path in dataset_paths:
            train_data = TSBADMData(
                data_dir,
                mode="train",
                dataset_file=dataset_path,
                window_length=window_length,
                window_overlap=window_overlap,
                data_normalization_strategy=data_normalization_strategy,
                processed_root=processed_root,
            )

            objs = {
                "train": train_data,
                "test": TSBADMData(
                    data_dir,
                    mode="test",
                    dataset_file=dataset_path,
                    window_length=window_length,
                    window_overlap=window_overlap,
                    normalizer=train_data.scaler,
                    processed_root=processed_root,
                ),
                "val": TSBADMData(
                    data_dir,
                    mode="val",
                    dataset_file=dataset_path,
                    window_length=window_length,
                    window_overlap=window_overlap,
                    normalizer=train_data.scaler,
                    processed_root=processed_root,
                ),
            }

            data = objs[mode]
            raw = data.data
            if len(raw) == 0:
                logging.warning("Skipping empty TSB-AD-M dataset %s (mode=%s)", dataset_path.name, mode)
                continue

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

            meta = train_data.meta
            self.datasets.append(
                {
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
                    "dataset_id": train_data.dataset_stem,
                    "file_index": meta["file_index"],
                    "dataset_name": meta["dataset_name"],
                    "domain": meta["domain"],
                    "train_index": meta["train_index"],
                    "first_anomaly_index": meta["first_anomaly_index"],
                }
            )

            if self.fixed_subsample_mask:
                masked_ratio = 1.0 - self.subsample
                n_features_data = obs.shape[-1]
                indcs_long = indcs.long()
                full_len = int(indcs_long.max().item()) + 1
                burst_mask = create_random_burst_mask(
                    n_features=n_features_data,
                    x_len=full_len,
                    masked_ratio=masked_ratio,
                    seed=self.seed,
                )
                full_mask = torch.from_numpy(burst_mask.T.astype(np.int64)).long()
                self.datasets[-1]["fixed_inp_msk"] = full_mask[indcs_long]

        csum = 0
        for length in self._lengths:
            self._cumulative.append(csum)
            csum += length

        if len(self.datasets) > 0:
            ds0 = self.datasets[0]
            TSBADMDataset.input_dim = ds0["input_dim"]
            self.input_dim = ds0["input_dim"]
            self.num_timepoints = ds0["num_timepoints"]
            self.data_min = ds0["data_min"]
            self.data_max = ds0["data_max"]
            self.indcs = ds0["indcs"]

    @staticmethod
    def _resolve_dataset_paths(data_dir: str, dataset_number):
        raw_folder = os.path.join(data_dir, "TSB-AD-M", "raw")
        files = sorted(glob.glob(os.path.join(raw_folder, "*.csv")))
        if len(files) == 0:
            raise RuntimeError(f"No TSB-AD-M datasets found in {raw_folder}")

        parsed = []
        for file_ in files:
            try:
                meta = _parse_filename(file_)
            except ValueError:
                continue
            parsed.append((meta["file_index"], Path(file_)))

        parsed = sorted(parsed, key=lambda item: item[0])
        paths = [path for _, path in parsed]

        if dataset_number is None:
            return paths

        if isinstance(dataset_number, (list, tuple, set)):
            wanted = {int(x) for x in dataset_number}
            return [path for path in paths if _parse_filename(path)["file_index"] in wanted]

        wanted = int(dataset_number)
        for path in paths:
            if int(_parse_filename(path)["file_index"]) == wanted:
                return [path]
        return []

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

        if self.fixed_subsample_mask:
            msk = ds["fixed_inp_msk"][local_idx].long()
        else:
            msk = (torch.rand(ds["inp_msk"][local_idx].shape) < self.subsample).to(torch.int).long()

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


class TSBADMProvider(DatasetProvider):
    def __init__(
        self,
        data_dir=None,
        window_length: int = 100,
        window_overlap: float = 0.0,
        data_normalization_strategy: str = "none",
        subsample: float = 1.0,
        seed: int = -1,
        fixed_subsample_mask: bool = False,
        dataset_number=None,
    ):
        super().__init__()
        self._processed_root = tempfile.mkdtemp(prefix="LatentSDEonHS_TSB_AD_M_processed_")

        common_kwargs = {
            "window_length": window_length,
            "window_overlap": window_overlap,
            "data_normalization_strategy": data_normalization_strategy,
            "processed_root": self._processed_root,
        }

        self._ds_trn = TSBADMDataset(
            data_dir,
            "train",
            dataset_number=dataset_number,
            subsample=subsample,
            seed=seed,
            fixed_subsample_mask=fixed_subsample_mask,
            **common_kwargs,
        )
        self._ds_tst = TSBADMDataset(
            data_dir,
            "test",
            dataset_number=dataset_number,
            subsample=subsample,
            seed=seed,
            fixed_subsample_mask=fixed_subsample_mask,
            **common_kwargs,
        )
        self._ds_val = TSBADMDataset(
            data_dir,
            "val",
            dataset_number=dataset_number,
            subsample=subsample,
            seed=seed,
            fixed_subsample_mask=fixed_subsample_mask,
            **common_kwargs,
        )

    @property
    def has_aux(self):
        return False

    @property
    def input_dim(self):
        return TSBADMDataset.input_dim

    @property
    def input_dims(self) -> list:
        return self._ds_trn.input_dims

    @property
    def num_datasets(self) -> int:
        return self._ds_trn.num_datasets

    def get_dataset(self, ds_idx: int) -> dict:
        return self._ds_trn.get_dataset(ds_idx)

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

    def cleanup(self):
        shutil.rmtree(self._processed_root, ignore_errors=True)



