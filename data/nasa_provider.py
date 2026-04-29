"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""
import ast
import glob
import logging
import os
import shutil
import tempfile
from typing import DefaultDict

import numpy as np
import pandas as pd

import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider
from data.process_water_treatment_datasets import reshape_data
from utils.anomaly_detection import create_random_burst_mask


class NASAData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(
            self, root, data_kind="MSL", mode='train', dataset_id="A-1",
            window_length:int = 100, window_overlap: float = 0.0,
            normalizer=None, data_normalization_strategy:str="none",
            processed_root: str = None,
    ):

        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root = root
        self.data_kind = data_kind
        self.mode = mode
        self.window_length = window_length

        self.overlapping_windows = window_overlap
        self._processed_root = processed_root

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        self.dataset_id = dataset_id

        if not self._check_exists(): # does raw exist?
            if not self._check_exist_raw_data():
                raise RuntimeError('Dataset not found. You can use download=True to download it')

            if self.data_kind in ["MSL", "SMAP"]:
                if not self._check_exists():
                    self._process_nasa_data()
            else:
                raise NotImplementedError

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)

        if self.mode == 'test':
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)


    def _check_exist_raw_data(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        required_files = [
            os.path.join(self.raw_folder, 'train', f'{self.dataset_id}.npy'),
            os.path.join(self.raw_folder, 'test', f'{self.dataset_id}.npy'),
            os.path.join(self.raw_folder, f'labeled_anomalies.csv')
        ]

        return all(os.path.isfile(f) for f in required_files)

    def _check_exists(self):
        return os.path.isfile(os.path.join(
            self.processed_folder, f'{self.mode}_{self.dataset_id}.pt')
        )

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'nasa', 'raw')

    @property
    def processed_folder(self):
        if self._processed_root is not None:
            return self._processed_root
        return os.path.join(self.root, 'nasa', 'processed', self.data_kind)

    @property
    def training_file(self):
        return f'train_{self.dataset_id}.pt'

    @property
    def test_file(self):
        return f'test_{self.dataset_id}.pt'

    @property
    def val_file(self):
        return f'val_{self.dataset_id}.pt'

    @property
    def label_file(self):
        return f'labels_{self.dataset_id}.pt'

    @property
    def destination_file(self):
        mode_to_file = {
            'train': self.training_file,
            'test': self.test_file,
            'val': self.val_file
        }
        return mode_to_file.get(self.mode, self.val_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def normalize_data(self, data_):
        if self.scaler is None:
            if self.data_normalization_strategy != "none":
                assert self.data_normalization_strategy in ["std", "min-max"]
                if self.data_normalization_strategy == "std":
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(data_)

        if self.scaler is not None:
            data_ = self.scaler.transform(data_)
        return data_

    def _process_nasa_data(self):
        logging.warning(f"Processing NASA Data, Dataset {self.dataset_id} (mode={self.mode})")
        if self.mode == 'test' and self.dataset_id == 'P-10':
            pass
        raw_data = np.load(os.path.join(self.raw_folder, f'{self.mode}', f'{self.dataset_id}.npy'))
        raw_data = reshape_data(raw_data, self.window_length, remove_zero_column=False)


        if self.mode == 'test':
            target_info = pd.read_csv(os.path.join(self.raw_folder, f'labeled_anomalies.csv'))


            target_info['anomaly_sequences'] = target_info['anomaly_sequences'].apply(ast.literal_eval)
            row = target_info[target_info['chan_id'] == self.dataset_id].iloc[0]

            self.targets = np.zeros((row['num_values']))
            for start_idx, stop_idx in row['anomaly_sequences']:
                    self.targets[start_idx:stop_idx] = 1

            self.targets = reshape_data(self.targets, self.window_length, remove_zero_column=False)
        else:
            self.targets = np.zeros(raw_data.shape[0:2])

        indcs = torch.arange(raw_data.shape[1])
        data_tensor = torch.Tensor(raw_data)
        mask = torch.ones_like(data_tensor)

        # handle nan values in the dataset
        mask[data_tensor.isnan()] = 0
        data_tensor[data_tensor.isnan()] = 0

        assert data_tensor.shape[0] == mask.shape[0]

        data = [(part_idx, indcs+(indcs.shape[0]*part_idx), data_tensor[part_idx, :, :],
                 mask[part_idx, :, :]) for part_idx in range(mask.shape[0])]

        if self.mode == 'train':
            data_len = len(data)
            indices = np.random.permutation(data_len)
            split_idx = int(data_len * 0.9)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            torch.save([data[i] for i in train_indices], os.path.join(self.processed_folder, self.destination_file))
            torch.save([data[i] for i in val_indices], os.path.join(self.processed_folder, self.val_file))

        elif self.mode == 'test':
            torch.save(data, os.path.join(self.processed_folder, self.destination_file))
            torch.save(self.targets, os.path.join(self.processed_folder, self.label_file))


class NASADataset(Dataset):
    """Dataset that holds *multiple* NASA sub-datasets (one per channel /
    dataset_id) as separate list entries so that datasets of different
    lengths and feature dimensions can coexist.

    Attributes
    ----------
    datasets : list[dict]
        Each element is a dict with the tensors for one dataset_id:
        ``inp_obs, inp_msk, inp_tps, inp_tid, indcs,
          evd_obs, evd_msk, evd_tid, evd_tps, aux_tgt,
          data_min, data_max, input_dim, num_timepoints, dataset_id``
    """

    def __init__(self, data_dir: str, mode: str = 'train', data_kind: str = None,
                 dataset: str = None, window_length: int = 100,
                 window_overlap: float = 0.0, subsample: float = 1.0, seed: int =-1,
                 data_normalization_strategy: str = "none",
                 fixed_subsample_mask: bool = False,
                 processed_root: str = None):

        self.mode = mode
        self.subsample = subsample
        self.fixed_subsample_mask = fixed_subsample_mask
        self.seed = seed

        label_data = pd.read_csv(
            os.path.join(data_dir, 'nasa', 'raw', 'labeled_anomalies.csv'))
        dataset_ids = label_data[
            label_data['spacecraft'] == dataset]['chan_id'].values

        # ------------------------------------------------------------------
        # Load every NASAData object (train is always needed for normalisation)
        # ------------------------------------------------------------------
        objs = {'train': [], 'test': [], 'val': []}

        for dataset_id in dataset_ids:
            train_data = NASAData(
                data_dir, mode='train', dataset_id=dataset_id, data_kind=dataset,
                window_length=window_length, window_overlap=window_overlap,
                data_normalization_strategy=data_normalization_strategy,
                processed_root=processed_root)

            objs['train'].append(train_data)
            objs['test'].append(NASAData(
                data_dir, mode='test', dataset_id=dataset_id, data_kind=dataset,
                window_length=window_length, window_overlap=window_overlap,
                normalizer=train_data.scaler, processed_root=processed_root))
            objs['val'].append(NASAData(
                data_dir, mode='val', dataset_id=dataset_id, data_kind=dataset,
                window_length=window_length, window_overlap=window_overlap,
                normalizer=train_data.scaler, processed_root=processed_root))

        # ------------------------------------------------------------------
        # Process each dataset_id *separately* — keep as list entries
        # ------------------------------------------------------------------
        self.datasets = []       # list of dicts, one per dataset_id
        self._lengths = []       # sample count per dataset
        self._cumulative = []    # cumulative offsets for flat indexing

        for ds_idx, dataset_id in enumerate(dataset_ids):
            data_obj = objs[mode][ds_idx]
            train_obj = objs['train'][ds_idx]

            # Per-dataset normalisation bounds (always from training split)
            data_min, data_max = get_data_min_max(train_obj[:])

            raw = data_obj.data  # list of (part_idx, indcs, obs, msk)
            if len(raw) == 0:
                logging.warning(f"Skipping empty dataset: {dataset_id} (mode={mode})")
                continue

            # Build tensors for this sub-dataset
            tps_base = raw[0][1].float()

            tps_max = tps_base.max()
            if tps_max > 0:
                tps_base = tps_base / tps_max

            indcs = torch.stack([raw[i][1] for i in range(len(raw))])
            obs   = torch.stack([raw[i][2] for i in range(len(raw))]).float()
            msk   = torch.stack([raw[i][3] for i in range(len(raw))]).float()
            tps   = tps_base[None, :].repeat(obs.shape[0], 1).float()

            if mode == 'test':
                tgt = torch.Tensor(data_obj.targets)
            else:
                tgt = torch.zeros((obs.shape[0], obs.shape[1])).long()

            obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)

            n_samples = obs.shape[0]
            n_time    = tps.shape[1]
            self._lengths.append(n_samples)

            self.datasets.append({
                'inp_obs':  (obs * msk).float(),
                'inp_msk':  msk.long(),
                'inp_tps':  tps,
                'inp_tid':  torch.arange(n_time).repeat(n_samples, 1).long(),
                'indcs':    indcs,
                'evd_obs':  obs.float(),
                'evd_msk':  torch.ones_like(msk).long(),
                'evd_tid':  torch.arange(n_time).repeat(n_samples, 1).long(),
                'evd_tps':  tps,
                'aux_tgt':  tgt.long(),
                'data_min': data_min,
                'data_max': data_max,
                'input_dim':      obs.shape[-1],
                'num_timepoints': n_time,
                'dataset_id':     dataset_id,
            })

            if self.fixed_subsample_mask:
                masked_ratio = 1.0 - self.subsample
                n_features_data = obs.shape[-1]
                indcs_long = indcs.long()
                full_len = int(indcs_long.max().item()) + 1
                burst_mask = create_random_burst_mask(
                    n_features=n_features_data,
                    x_len=full_len,
                    masked_ratio=masked_ratio,
                    seed=self.seed
                )  # (n_features, full_len)
                full_mask = torch.from_numpy(burst_mask.T.astype(np.int64)).long()  # (full_len, n_features)
                self.datasets[-1]['fixed_inp_msk'] = full_mask[indcs_long]  # (n_samples, n_time, n_features)

        # Build cumulative offsets for flat indexing
        cumsum = 0
        for length in self._lengths:
            self._cumulative.append(cumsum)
            cumsum += length

        self.feature_names = NASAData.params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def num_datasets(self) -> int:
        """Number of separate sub-datasets (channel ids)."""
        return len(self.datasets)

    def get_dataset(self, ds_idx: int) -> dict:
        """Return the full tensor dict for sub-dataset *ds_idx*."""
        return self.datasets[ds_idx]

    @property
    def input_dims(self) -> list:
        """Per-dataset feature dimensions."""
        return [ds['input_dim'] for ds in self.datasets]

    @property
    def num_timepoints_list(self) -> list:
        """Per-dataset number of time-points."""
        return [ds['num_timepoints'] for ds in self.datasets]

    @property
    def has_aux(self):
        return False

    # ------------------------------------------------------------------
    # Dataset interface (flat indexing across all sub-datasets)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return sum(self._lengths)

    def _resolve_index(self, idx: int):
        """Map a flat index to *(dataset_idx, local_idx)*."""
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
            msk = ds['fixed_inp_msk'][local_idx].long()
        else:
            msk = (torch.rand(ds['inp_msk'][local_idx].shape) < self.subsample).to(torch.int).long()

        return {
            'inp_obs':    ds['inp_obs'][local_idx].float(),
            'inp_msk':    msk,
            'inp_tid':    ds['inp_tid'][local_idx].long(),
            'inp_tps':    ds['inp_tps'][local_idx].float(),
            'evd_obs':    ds['evd_obs'][local_idx].float(),
            'evd_msk':    ds['evd_msk'][local_idx].long(),
            'evd_tid':    ds['evd_tid'][local_idx].long(),
            'evd_tps':    ds['evd_tps'][local_idx].float(),
            'aux_tgt':    ds['aux_tgt'][local_idx].long(),
            'inp_indcs':  ds['indcs'][local_idx],
            'dataset_idx': ds_idx,
        }

    # ------------------------------------------------------------------
    # Optional per-dataset normalisation utilities
    # ------------------------------------------------------------------
    def fit_normalizer(self, ds_idx: int = None):
        """Fit a StandardScaler on one (``ds_idx``) or all sub-datasets."""
        logging.info("Fitting Standard Scaler")
        if ds_idx is not None:
            scaler = StandardScaler()
            scaler.fit(self.datasets[ds_idx]['evd_obs'])
            return scaler
        return [self.fit_normalizer(i) for i in range(self.num_datasets)]

    def normalizer_transform(self, scaler, ds_idx: int = None):
        """Apply a fitted scaler to one (``ds_idx``) or all sub-datasets."""
        logging.info("Applying Scaler")
        if ds_idx is not None:
            ds = self.datasets[ds_idx]
            ds['evd_obs'] = torch.Tensor(scaler.transform(ds['evd_obs']))
            ds['inp_obs'] = torch.Tensor(scaler.transform(ds['inp_obs']))
        else:
            for i, sc in enumerate(scaler):
                self.normalizer_transform(sc, ds_idx=i)


class NASAProvider(DatasetProvider):
    def __init__(self, data_dir=None, dataset=None,
                 window_length: int = 100, window_overlap: float = 0.0,
                 data_normalization_strategy: str = "none", subsample=1.0, seed=-1,
                 fixed_subsample_mask: bool = False):
        DatasetProvider.__init__(self)

        if dataset not in ["SMAP", "MSL"]:
            raise NotImplementedError

        self._dataset = dataset
        self._processed_root = tempfile.mkdtemp(prefix=f"LatentSDEonHS_{dataset}_processed_")

        common_kwargs = {
            'dataset': dataset,
            'window_length': window_length,
            'window_overlap': window_overlap,
            'data_normalization_strategy': data_normalization_strategy,
            'processed_root': self._processed_root,
        }

        self._ds_trn = NASADataset(
            data_dir, 'train', subsample=subsample, seed=seed,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)
        self._ds_tst = NASADataset(
            data_dir, 'test', subsample=subsample, seed=seed,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)
        self._ds_val = NASADataset(
            data_dir, 'val', subsample=subsample, seed=seed,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)

    @property
    def input_dims(self) -> list:
        """Per-dataset feature dimensions (from training split)."""
        return self._ds_trn.input_dims

    @property
    def num_datasets(self) -> int:
        return self._ds_trn.num_datasets

    @property
    def num_timepoints_list(self) -> list:
        return self._ds_trn.num_timepoints_list

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


def create_win_periods(data_, win_size_, win_stride_):
    """ returns the rolling windows of the given flattened data """
    if win_stride_ < 1:
        win_stride_ = 1

    windows = sliding_window_view(data_, (
        win_size_, data_.shape[1]))
    windows = windows.squeeze()
    indcs = sliding_window_view(np.arange(data_.shape[0]), win_size_)

    return indcs[::win_stride_], windows.squeeze()[::win_stride_, :]

