import glob
import logging
import os
import shutil
import tempfile

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, normalize_masked_data
from data.dataset_provider import DatasetProvider
from utils.anomaly_detection import create_random_burst_mask


def _load_txt(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[:, None]
    return data


def _windowize(data: np.ndarray, window_length: int, window_overlap: float):
    """Return (starts, windows) with drop-tail behavior."""
    if data.ndim == 1:
        data = data[:, None]

    stride = max(1, int(window_length * (1.0 - window_overlap)))
    if window_overlap <= 0.0:
        stride = window_length

    n_steps = data.shape[0]
    starts = np.arange(0, n_steps - window_length + 1, stride, dtype=np.int64)
    if starts.size == 0:
        return starts, np.zeros((0, window_length, data.shape[1]), dtype=data.dtype)

    windows = np.stack([data[s:s + window_length] for s in starts], axis=0)
    return starts, windows


class SMDData(object):
    params = None
    labels = ["Anomaly"]
    labels_dict = {"Anomaly": 0}

    def __init__(
        self,
        root_path,
        machine_id="machine-1-1",
        mode="train",
        window_length: int = 100,
        window_overlap: float = 0.0,
        normalizer=None,
        data_normalization_strategy: str = "none",
        processed_root: str = None,
    ):
        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root_path = root_path
        self.machine_id = machine_id
        self.mode = mode
        self.window_length = window_length
        self.window_overlap = window_overlap
        self._processed_root = processed_root

        if not self._check_exists():
            if not self._check_exist_raw_data():
                raise RuntimeError(f"SMD machine not found: {self.machine_id}")
            self._process_smd_data()

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)
        if self.mode == "test":
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, "SMD", "raw")

    @property
    def processed_folder(self):
        if self._processed_root is not None:
            return self._processed_root
        ov = str(self.window_overlap).replace(".", "p")
        return os.path.join(self.root_path, "SMD", "processed", f"wl{self.window_length}_ov{ov}")

    @property
    def training_file(self):
        return f"train_{self.machine_id}.pt"

    @property
    def test_file(self):
        return f"test_{self.machine_id}.pt"

    @property
    def val_file(self):
        return f"val_{self.machine_id}.pt"

    @property
    def label_file(self):
        return f"labels_{self.machine_id}.pt"

    @property
    def destination_file(self):
        return {
            "train": self.training_file,
            "test": self.test_file,
            "val": self.val_file,
        }[self.mode]

    def _check_exist_raw_data(self):
        os.makedirs(self.processed_folder, exist_ok=True)
        required_files = [
            os.path.join(self.raw_folder, "train", f"{self.machine_id}.txt"),
            os.path.join(self.raw_folder, "test", f"{self.machine_id}.txt"),
            os.path.join(self.raw_folder, "test_label", f"{self.machine_id}.txt"),
        ]
        return all(os.path.isfile(f) for f in required_files)

    def _check_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

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
            indcs = torch.tensor(local_t + s, dtype=torch.long)
            tuples.append((part_idx, indcs, data_tensor[part_idx], mask[part_idx]))
        return tuples

    def _process_smd_data(self):
        logging.warning(f"Processing SMD machine {self.machine_id} (mode={self.mode})")

        if self.mode in ["train", "val"]:
            train_arr = _load_txt(os.path.join(self.raw_folder, "train", f"{self.machine_id}.txt"))
            train_arr = self._apply_optional_scaler(train_arr)

            starts, windows = _windowize(train_arr, self.window_length, self.window_overlap)
            data = self._build_data_tuples(starts, windows)

            data_len = len(data)
            split_idx = int(data_len * 0.9)
            indices = np.random.permutation(data_len)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            torch.save([data[i] for i in train_indices], os.path.join(self.processed_folder, self.training_file))
            torch.save([data[i] for i in val_indices], os.path.join(self.processed_folder, self.val_file))

        elif self.mode == "test":
            test_arr = _load_txt(os.path.join(self.raw_folder, "test", f"{self.machine_id}.txt"))
            test_arr = self._apply_optional_scaler(test_arr)

            starts, windows = _windowize(test_arr, self.window_length, self.window_overlap)
            data = self._build_data_tuples(starts, windows)

            labels = _load_txt(os.path.join(self.raw_folder, "test_label", f"{self.machine_id}.txt")).squeeze()
            labels = labels.astype(np.float32)
            _, label_windows = _windowize(labels[:, None], self.window_length, self.window_overlap)
            label_windows = label_windows.squeeze(-1)

            torch.save(data, os.path.join(self.processed_folder, self.test_file))
            torch.save(label_windows, os.path.join(self.processed_folder, self.label_file))


class SMDDataset(Dataset):
    input_dim = None

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        machine_ids=None,
        window_length: int = 100,
        window_overlap: float = 0.0,
        subsample: float = 1.0,
        data_normalization_strategy: str = "none",
        fixed_subsample_mask: bool = False,
        processed_root: str = None,
    ):
        self.mode = mode
        self.subsample = subsample
        self.fixed_subsample_mask = fixed_subsample_mask

        self.datasets = []
        self._lengths = []
        self._cumulative = []

        machine_ids = self._resolve_machine_ids(data_dir, machine_ids)

        for machine_id in machine_ids:
            train_data = SMDData(
                data_dir,
                mode="train",
                machine_id=machine_id,
                window_length=window_length,
                window_overlap=window_overlap,
                data_normalization_strategy=data_normalization_strategy,
                processed_root=processed_root,
            )

            objs = {
                "train": train_data,
                "test": SMDData(
                    data_dir,
                    mode="test",
                    machine_id=machine_id,
                    window_length=window_length,
                    window_overlap=window_overlap,
                    normalizer=train_data.scaler,
                    processed_root=processed_root,
                ),
                "val": SMDData(
                    data_dir,
                    mode="val",
                    machine_id=machine_id,
                    window_length=window_length,
                    window_overlap=window_overlap,
                    normalizer=train_data.scaler,
                    processed_root=processed_root,
                ),
            }

            data = objs[mode]
            raw = data.data
            if len(raw) == 0:
                logging.warning(f"Skipping empty SMD machine {machine_id} (mode={mode})")
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
                    "dataset_id": machine_id,
                }
            )

            if self.fixed_subsample_mask:
                if self.mode == "train":
                    masked_ratio = 1.0 - self.subsample
                    n_features_data = obs.shape[-1]
                    burst_mask = create_random_burst_mask(
                        n_features=n_samples * n_features_data,
                        x_len=n_time,
                        masked_ratio=masked_ratio,
                    )  # (n_samples * n_features_data, n_time)
                    burst_arr = burst_mask.reshape(n_samples, n_features_data, n_time).transpose(0, 2, 1)
                    self.datasets[-1]["fixed_inp_msk"] = torch.from_numpy(
                        burst_arr.astype(np.int64)
                    ).long()
                else:  # val — all observations available
                    self.datasets[-1]["fixed_inp_msk"] = torch.ones_like(msk).long()

        csum = 0
        for length in self._lengths:
            self._cumulative.append(csum)
            csum += length

        if len(self.datasets) > 0:
            ds0 = self.datasets[0]
            SMDDataset.input_dim = ds0["input_dim"]
            self.input_dim = ds0["input_dim"]
            self.num_timepoints = ds0["num_timepoints"]
            self.data_min = ds0["data_min"]
            self.data_max = ds0["data_max"]
            self.indcs = ds0["indcs"]

    @staticmethod
    def _resolve_machine_ids(data_dir: str, machine_ids):
        if machine_ids is None:
            train_dir = os.path.join(data_dir, "SMD", "raw", "train")
            files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))
            ids = [os.path.basename(f).replace(".txt", "") for f in files]
            if len(ids) == 0:
                raise RuntimeError(f"No SMD machines found in {train_dir}")
            return ids

        if isinstance(machine_ids, str):
            return [machine_ids]

        return list(machine_ids)

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

        if self.mode in ("train", "val"):
            if self.fixed_subsample_mask:
                msk = ds["fixed_inp_msk"][local_idx].long()
            else:
                masked_ratio = 1.0 - self.subsample
                n_time_s, n_feat_s = ds["inp_msk"][local_idx].shape  # (n_time, n_features)
                burst_mask = create_random_burst_mask(
                    n_features=n_feat_s,
                    x_len=n_time_s,
                    masked_ratio=masked_ratio,
                )  # (n_feat_s, n_time_s)
                msk = torch.from_numpy(burst_mask.T.astype(np.int64)).long()  # (n_time_s, n_feat_s)
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


class SMDProvider(DatasetProvider):
    def __init__(
        self,
        data_dir=None,
        machine_ids=None,
        window_length: int = 100,
        window_overlap: float = 0.0,
        data_normalization_strategy: str = "none",
        subsample: float = 1.0,
        fixed_subsample_mask: bool = False,
    ):
        super().__init__()
        self._processed_root = tempfile.mkdtemp(prefix="LatentSDEonHS_SMD_processed_")

        common_kwargs = {
            "machine_ids": machine_ids,
            "window_length": window_length,
            "window_overlap": window_overlap,
            "data_normalization_strategy": data_normalization_strategy,
            "processed_root": self._processed_root,
        }

        self._ds_trn = SMDDataset(
            data_dir, "train", subsample=subsample,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)
        self._ds_tst = SMDDataset(data_dir, "test", **common_kwargs)
        self._ds_val = SMDDataset(
            data_dir, "val", subsample=subsample,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)

    @property
    def input_dim(self):
        return SMDDataset.input_dim

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

    def cleanup(self):
        shutil.rmtree(self._processed_root, ignore_errors=True)

