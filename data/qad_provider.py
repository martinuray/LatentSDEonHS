import logging
import os
import glob
import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, normalize_masked_data
from data.dataset_provider import DatasetProvider
from data.process_water_treatment_datasets import reshape_data


class QADData:

    params = None
    labels = None
    params_dict = None
    labels_dict = None

    def __init__(
            self, root_path, dataset_number=1, mode='train',
            window_length: int = 100, window_overlap: float = 0.0,
            normalizer=None,
            data_normalization_strategy: str = "none",
            raw_subdir: str = "qad_clean_txt_100Hz"
    ):

        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root_path = root_path
        self.dataset_number = dataset_number
        self.mode = mode
        self.window_length = window_length
        self.raw_subdir = raw_subdir

        self.overlapping_windows = window_overlap

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_exists():
            if not self._check_exist_raw_data():
                raise RuntimeError('Dataset not found.')
            self._process_QAD_data()

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)

        if self.mode == 'test':
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)


    def _check_exist_raw_data(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        required_files = [
            os.path.join(self.raw_folder, f'train_{self.dataset_number}.txt'),
            os.path.join(self.raw_folder, f'test_{self.dataset_number}.txt'),
            os.path.join(self.raw_folder, f'test_label_{self.dataset_number}.txt')
        ]

        return all(os.path.isfile(f) for f in required_files)

    def _check_exists(self):
        return os.path.isfile(os.path.join(
            self.processed_folder, f'{self.mode}_{self.dataset_number}.pt')
        )

    def _resolve_raw_subdir(self):
        requested = os.path.join(self.root_path, "QAD", "raw", self.raw_subdir)
        if os.path.isdir(requested):
            return self.raw_subdir

        fallback = "qad_clean_txt_100Hz"
        fallback_path = os.path.join(self.root_path, "QAD", "raw", fallback)
        if os.path.isdir(fallback_path):
            logging.warning(
                f"Requested QAD folder '{self.raw_subdir}' not found. Falling back to '{fallback}'."
            )
            return fallback

        return self.raw_subdir

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, "QAD", "raw", self._resolve_raw_subdir())

    @property
    def processed_folder(self):
        # Keep processed namespace aligned with the actual raw folder name.
        return os.path.join(self.root_path, "QAD", "processed", self._resolve_raw_subdir())

    @property
    def training_file(self):
        return f'train_{self.dataset_number}.pt'

    @property
    def test_file(self):
        return f'test_{self.dataset_number}.pt'

    @property
    def val_file(self):
        return f'val_{self.dataset_number}.pt'

    @property
    def label_file(self):
        return f'labels_{self.dataset_number}.pt'

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
        if self.scaler is None and self.data_normalization_strategy != "none":
            if self.data_normalization_strategy not in ["std", "min-max"]:
                raise ValueError(f"Invalid normalization strategy: {self.data_normalization_strategy}")

            if self.data_normalization_strategy == "std":
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(data_)

        if self.scaler is not None:
            data_ = self.scaler.transform(data_)
        return data_

    def _process_QAD_data(self, n_samples=None):
        logging.warning("Processing QAD Data")
        raw_data = load_qad_txt(os.path.join(self.raw_folder, f'{self.mode}_{self.dataset_number}.txt'))
        raw_data = reshape_data(raw_data, self.window_length)

        if self.mode == 'test':
            self.targets = load_qad_txt(
                os.path.join(self.raw_folder, f'test_label_{self.dataset_number}.txt'),
                is_label=True,
            )
            self.targets = reshape_data(self.targets, self.window_length)
        else:
            self.targets = np.zeros(raw_data.shape[0:2])

        indcs = torch.arange(raw_data.shape[1])
        data_tensor = torch.Tensor(raw_data)
        mask = torch.ones_like(data_tensor)

        # handle nan values in the dataset
        mask[data_tensor.isnan()] = 0
        data_tensor[data_tensor.isnan()] = 0

        if n_samples is not None:
            logging.warning(f"Limiting dataset to {n_samples} samples")
            data_tensor = data_tensor[:n_samples]
            mask = mask[:n_samples]
            indcs = indcs[:n_samples]
            if self.mode == 'test':
                self.targets = self.targets[:n_samples]
        else:
            logging.debug("No limit on the dataset size")

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



class QADDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str = 'train', dataset_number: int = None,
                 window_length: int = 100, window_overlap: float = 0.0, subsample: float = 1.0,
                 data_normalization_strategy: str = "none", raw_subdir: str = "qad_clean_txt_100Hz",
                 fixed_subsample_mask: bool = False):

        self.mode = mode
        self.subsample = subsample
        self.fixed_subsample_mask = fixed_subsample_mask

        self.datasets = []
        self._lengths = []
        self._cumulative = []

        dataset_ids = self._resolve_dataset_ids(data_dir, dataset_number, raw_subdir)

        for dataset_id in dataset_ids:
            train_data = QADData(
                data_dir, mode='train', dataset_number=dataset_id,
                window_length=window_length, window_overlap=window_overlap,
                data_normalization_strategy=data_normalization_strategy,
                raw_subdir=raw_subdir)

            objs = {
                'train': train_data,
                'test': QADData(
                    data_dir, mode='test', dataset_number=dataset_id,
                    window_length=window_length, window_overlap=window_overlap,
                    normalizer=train_data.scaler, raw_subdir=raw_subdir),
                'val': QADData(
                    data_dir, mode='val', dataset_number=dataset_id,
                    window_length=window_length, window_overlap=window_overlap,
                    normalizer=train_data.scaler, raw_subdir=raw_subdir)
            }

            data = objs[mode]
            data_min, data_max = get_data_min_max(objs['train'][:])

            raw = data.data
            if len(raw) == 0:
                logging.warning(f"Skipping empty QAD dataset {dataset_id} (mode={mode})")
                continue

            tps_base = raw[0][1].float()
            tps_max = tps_base.max()
            if tps_max > 0:
                tps_base = tps_base / tps_max

            indcs = torch.stack([raw[i][1] for i in range(len(raw))])
            obs = torch.stack([raw[i][2] for i in range(len(raw))]).float()
            msk = torch.stack([raw[i][3] for i in range(len(raw))]).float()
            tps = tps_base[None, :].repeat(obs.shape[0], 1).float()

            if mode == 'test':
                tgt = torch.Tensor(data.targets)
            else:
                tgt = torch.zeros((obs.shape[0], obs.shape[1])).long()

            obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)

            n_samples = obs.shape[0]
            n_time = tps.shape[1]
            self._lengths.append(n_samples)

            self.datasets.append({
                'inp_obs': (obs * msk).float(),
                'inp_msk': msk.long(),
                'inp_tps': tps,
                'inp_tid': torch.arange(n_time).repeat(n_samples, 1).long(),
                'indcs': indcs,
                'evd_obs': obs.float(),
                'evd_msk': torch.ones_like(msk).long(),
                'evd_tid': torch.arange(n_time).repeat(n_samples, 1).long(),
                'evd_tps': tps,
                'aux_tgt': tgt.long(),
                'data_min': data_min,
                'data_max': data_max,
                'input_dim': obs.shape[-1],
                'num_timepoints': n_time,
                'dataset_id': dataset_id,
            })

            if self.mode in ('train', 'val') and self.fixed_subsample_mask:
                self.datasets[-1]['fixed_inp_msk'] = (
                    torch.rand(self.datasets[-1]['inp_msk'].shape) < self.subsample
                ).long()

        csum = 0
        for length in self._lengths:
            self._cumulative.append(csum)
            csum += length

        self.feature_names = QADData.params

        # Backward-compatible aliases for single-dataset code paths.
        if len(self.datasets) > 0:
            ds0 = self.datasets[0]
            QADDataset.input_dim = ds0['input_dim']
            self.input_dim = ds0['input_dim']
            self.num_timepoints = ds0['num_timepoints']
            self.data_min = ds0['data_min']
            self.data_max = ds0['data_max']
            self.indcs = ds0['indcs']
            self.inp_obs = ds0['inp_obs']
            self.inp_msk = ds0['inp_msk']
            self.inp_tps = ds0['inp_tps']
            self.inp_tid = ds0['inp_tid']
            self.evd_obs = ds0['evd_obs']
            self.evd_msk = ds0['evd_msk']
            self.evd_tid = ds0['evd_tid']
            self.evd_tps = ds0['evd_tps']
            self.aux_tgt = ds0['aux_tgt']

    @staticmethod
    def _resolve_dataset_ids(data_dir: str, dataset_number, raw_subdir: str):
        if dataset_number is None:
            raw_folder = os.path.join(data_dir, "QAD", "raw", raw_subdir)
            if not os.path.isdir(raw_folder):
                fallback = os.path.join(data_dir, "QAD", "raw", "qad_clean_txt_100Hz")
                raw_folder = fallback if os.path.isdir(fallback) else raw_folder

            train_files = glob.glob(os.path.join(raw_folder, "train_*.txt"))
            ids = []
            for file_ in train_files:
                match = re.match(r"^train_(\d+)\.txt$", os.path.basename(file_))
                if match is not None:
                    ids.append(int(match.group(1)))

            ids = sorted(set(ids))
            if len(ids) == 0:
                raise RuntimeError(f"No QAD datasets found in {raw_folder}")
            return ids

        if isinstance(dataset_number, (list, tuple, set)):
            return [int(x) for x in dataset_number]

        return [int(dataset_number)]

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
        return [ds['input_dim'] for ds in self.datasets]

    @property
    def num_timepoints_list(self) -> list:
        return [ds['num_timepoints'] for ds in self.datasets]

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

        if self.mode in ('train', 'val'):
            if self.fixed_subsample_mask:
                msk = ds['fixed_inp_msk'][local_idx].long()
            else:
                msk = (torch.rand(ds['inp_msk'][local_idx].shape) < self.subsample).long()
        else:
            msk = ds['inp_msk'][local_idx].long()

        return {
            'inp_obs': ds['inp_obs'][local_idx].float(),
            'inp_msk': msk,
            'inp_tid': ds['inp_tid'][local_idx].long(),
            'inp_tps': ds['inp_tps'][local_idx].float(),
            'evd_obs': ds['evd_obs'][local_idx].float(),
            'evd_msk': ds['evd_msk'][local_idx].long(),
            'evd_tid': ds['evd_tid'][local_idx].long(),
            'evd_tps': ds['evd_tps'][local_idx].float(),
            'aux_tgt': ds['aux_tgt'][local_idx].long(),
            'inp_indcs': ds['indcs'][local_idx],
            'dataset_idx': ds_idx,
        }

    def fit_normalizer(self):
        logging.info("Fitting Standard Scaler")
        scaler = StandardScaler()
        scaler.fit(self.evd_obs)
        return scaler

    def normalizer_transform(self, scaler):
        logging.info("Applying Scaler")
        self.evd_obs = scaler.transform(self.evd_obs)
        self.inp_obs = scaler.transform(self.inp_obs)


class QADProvider(DatasetProvider):
    def __init__(self, data_dir=None, dataset_number=None, window_length: int = 100,
                 window_overlap: float = 0.0,
                 data_normalization_strategy: str = "none", subsample: float = 1.0,
                 raw_subdir: str = "qad_clean_txt_100Hz",
                 fixed_subsample_mask: bool = False):
        super().__init__()

        self._dataset = dataset_number

        common_kwargs = {
            'dataset_number': dataset_number,
            'window_length': window_length,
            'window_overlap': window_overlap,
            'data_normalization_strategy': data_normalization_strategy,
            'raw_subdir': raw_subdir,
        }

        self._ds_trn = QADDataset(
            data_dir, 'train', subsample=subsample,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)
        self._ds_tst = QADDataset(data_dir, 'test', **common_kwargs)
        self._ds_val = QADDataset(
            data_dir, 'val', subsample=subsample,
            fixed_subsample_mask=fixed_subsample_mask, **common_kwargs)

    @property 
    def input_dim(self):
        return QADDataset.input_dim

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


def load_qad_txt(dataset_path, is_label: bool = False):
    # sep=None lets pandas infer comma/tab separators from converted TXT files.
    kwargs = {}
    if not is_label:
        kwargs["sep"] = None
        kwargs["engine"] = "python"

    data = pd.read_csv(dataset_path, **kwargs)

    if isinstance(data, pd.Series):
        data = data.to_frame(name="labels")

    # Label files should always expose a canonical `labels` column.
    if is_label and isinstance(data, pd.DataFrame) and len(data.columns) == 1 and "labels" not in data.columns:
        data.columns = ["labels"]

    return data
