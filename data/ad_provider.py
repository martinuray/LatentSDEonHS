"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""
import glob
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, normalize_masked_data
from data.dataset_provider import DatasetProvider
from utils.anomaly_detection import create_random_burst_mask

    

class ADData(object):

    WADI_TEST_FILE_CANDIDATES = [
        'WADI_attackdata_labelled.csv',
        'attackdata_labbelled.csv',
    ]
    WADI_LABEL_COLUMN_CANDIDATES = [
        'Arrack LABLE',
        'Attack LABLE',
        'Attack LABLE (1:No Attack, -1:Attack)',
    ]

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(
            self, root, processed_root, data_kind="SWaT", mode='train',
            window_length:int = 100, window_overlap: float = 0.75,
            n_samples = None, columns = None,
            normalizer=None,
            data_normalization_strategy:str="none"
    ):

        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root = root
        self.data_kind = data_kind
        self.mode = mode
        self.max_signal_length = window_length

        self.overlapping_windows = window_overlap

        self._processed_root = processed_root

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_exists(): # does raw exist?
            if not self._check_exist_raw_data():
                raise RuntimeError('Dataset not found. You can use download=True to download it')

            if self.data_kind in ["SWaT", "WaDi"]:
                if not self._check_exists():
                    self._process_water_treatment_data(
                        columns=columns, n_samples=n_samples)
            else:
                raise NotImplementedError

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)

        if self.mode == 'test':
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)


    def _check_exist_raw_data(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        if self.data_kind == "SMD": # TODO: maybe a bit more love
            return os.path.isdir(os.path.join(self.raw_folder, self.mode))

        if self.data_kind == "WaDi":
            train_candidates = [
                os.path.join(self.raw_folder, 'train.csv'),
                os.path.join(self.raw_folder, 'WADI_14days.csv'),
            ]
            test_candidates = [
                os.path.join(self.raw_folder, 'test.csv'),
                *[os.path.join(self.raw_folder, fname) for fname in self.WADI_TEST_FILE_CANDIDATES],
            ]
            label_candidates = [
                os.path.join(self.raw_folder, 'labels.csv'),
                *[os.path.join(self.raw_folder, fname) for fname in self.WADI_TEST_FILE_CANDIDATES],
            ]
            return any(os.path.isfile(f) for f in train_candidates) and \
                   any(os.path.isfile(f) for f in test_candidates) and \
                   any(os.path.isfile(f) for f in label_candidates)

        # Check for CSV files (original raw data)
        csv_exists = os.path.isfile(os.path.join(self.raw_folder, 'train.csv')) and \
                     os.path.isfile(os.path.join(self.raw_folder, 'test.csv')) and \
                     os.path.isfile(os.path.join(self.raw_folder, 'labels.csv'))

        # Check for NPY files (pre-preprocessed)
        npy_exists = os.path.isfile(os.path.join(self.raw_folder, 'train.npy')) and \
                     os.path.isfile(os.path.join(self.raw_folder, 'test.npy')) and \
                     os.path.isfile(os.path.join(self.raw_folder, 'labels.npy'))

        return csv_exists or npy_exists

    def _check_exists(self):
        if self.data_kind == "SMD": # TODO: maybe a bit more love
            return os.path.isfile(os.path.join(self.processed_folder,
                                               f'{self.mode}.pt'))

        return os.path.isfile(os.path.join(self.processed_folder, f'{self.mode}.pt'))

    @property
    def raw_folder(self):
        if self.data_kind == 'WaDi':
            requested = os.path.join(self.root, self.data_kind, 'raw', 'v2')
            if os.path.isdir(requested):
                return requested
        return os.path.join(self.root, self.data_kind, 'raw')

    @property
    def processed_folder(self):
        return self._processed_root

    @property
    def training_file(self):
        return 'train.pt'

    @property
    def test_file(self):
        return 'test.pt'

    @property
    def val_file(self):
        return 'val.pt'

    @property
    def destination_file(self):
        if self.mode == 'train':
            return self.training_file
        if self.mode == 'test':
            return self.test_file
        return self.val_file

    @property
    def label_file(self):
        return 'labels.pt'

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

    def _process_water_treatment_data(self, columns=None, n_samples=None):
        logging.warning("Processing Water Treatment Data")

        # Check if CSV files are available and preprocess them
        csv_candidates = [os.path.join(self.raw_folder, 'train.csv')]
        if self.data_kind == 'WaDi':
            csv_candidates.append(os.path.join(self.raw_folder, 'WADI_14days.csv'))
        csv_path = next((path for path in csv_candidates if os.path.isfile(path)), None)
        test_labels = None

        if csv_path is not None:
            # Load and preprocess from CSV files
            train_df, test_df, test_labels = self._load_and_preprocess_water_treatment_csv()

            # Store column names as params
            if columns is None:
                self.params = train_df.columns
            else:
                self.params = columns

            # Reshape data into windows
            train_np = self._reshape_data_to_windows(train_df, self.max_signal_length)
            test_np = self._reshape_data_to_windows(test_df, self.max_signal_length)

            raw_data = train_np if self.mode == 'train' else test_np
        else:
            raise NotImplementedError("Data missing")

        indcs = np.arange(0, raw_data.shape[1])

        if self.mode == 'test':
            if test_labels is not None:
                # Use preprocessed labels from CSV
                self.targets = self._reshape_data_to_windows(test_labels, self.max_signal_length)
            else:
                raise NotImplementedError("Data Labels missing")
        else:
            self.targets = np.zeros(raw_data.shape[0:2])

        indcs = torch.Tensor(indcs)
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

        torch.save(data, os.path.join(self.processed_folder, self.destination_file))
        if self.mode == 'test':
            torch.save(self.targets, os.path.join(self.processed_folder, self.label_file))

    def _load_and_preprocess_water_treatment_csv(self):
        """Load and preprocess water treatment data from CSV files."""
        logging.warning(f"Loading and preprocessing {self.data_kind} data from CSV files")

        # Load CSV files
        if "WaDi" in self.data_kind:
            train_df, test_df, test_labels = self._load_and_preprocess_wadi_csv()
        else:
            train_df = pd.read_csv(os.path.join(self.raw_folder, 'train.csv'))
            test_df = pd.read_csv(os.path.join(self.raw_folder, 'test.csv'))
            test_labels = pd.read_csv(os.path.join(self.raw_folder, 'labels.csv'))
            test_labels = test_labels.iloc[:, 0]  # Get first column

        # Handle missing values
        train_df = train_df.fillna(train_df.mean())
        test_df = test_df.fillna(test_df.mean())
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        # Trim column names
        train_df = train_df.rename(columns=lambda x: x.strip())
        test_df = test_df.rename(columns=lambda x: x.strip())

        # Remove columns where there is only one unique value
        #train_col_set = set(train_df.columns[train_df.nunique() == 1].tolist())
        #test_col_set = set(test_df.columns[test_df.nunique() == 1].tolist())
        #common_set = list(train_col_set.union(test_col_set))

        #train_df = train_df.drop(common_set, axis=1)
        #test_df = test_df.drop(common_set, axis=1)

        # Normalize data using MinMaxScaler
        normalizer = MinMaxScaler(feature_range=(0, 1))
        normalizer.fit(train_df)

        train_df_normalized = train_df.copy()
        test_df_normalized = test_df.copy()
        train_df_normalized[train_df.columns] = normalizer.transform(train_df)
        test_df_normalized[test_df.columns] = normalizer.transform(test_df)

        # Remove first 2160 samples (stabilization period) for SWaT/WaDi
        train_df_normalized = train_df_normalized.iloc[2160:]

        return train_df_normalized, test_df_normalized, test_labels

    def _load_and_preprocess_wadi_csv(self):
        train_path = next(
            path for path in [
                os.path.join(self.raw_folder, 'WADI_14days.csv'),
                os.path.join(self.raw_folder, 'train.csv'),
            ] if os.path.isfile(path)
        )

        test_path = next(
            path for path in [
                *[os.path.join(self.raw_folder, fname) for fname in self.WADI_TEST_FILE_CANDIDATES],
                os.path.join(self.raw_folder, 'test.csv'),
            ] if os.path.isfile(path)
        )

        train_df = pd.read_csv(train_path, header=0)
        test_df = pd.read_csv(test_path, header=1)

        train_df.columns = [str(col).strip() for col in train_df.columns]
        test_df.columns = [str(col).strip() for col in test_df.columns]

        label_col = None
        label_candidates = {c.strip().upper() for c in self.WADI_LABEL_COLUMN_CANDIDATES}
        for col in test_df.columns:
            col_norm = str(col).strip().upper()
            if col_norm in label_candidates or ("LABLE" in col_norm and "ATTACK" in col_norm):
                label_col = col
                break

        if label_col is None:
            raise ValueError(f"Could not find WaDi label column in {test_path}")

        raw_labels = pd.to_numeric(test_df[label_col], errors='coerce')
        test_labels = (raw_labels != 1).astype(float)

        metadata_cols = {"ROW", "ROW ", "DATE", "DATE ", "TIME", "TIME "}
        train_drop_cols = [c for c in train_df.columns if str(c).strip().upper() in metadata_cols]
        test_drop_cols = [c for c in test_df.columns if str(c).strip().upper() in metadata_cols]
        train_df = train_df.drop(columns=train_drop_cols, errors='ignore')
        test_df = test_df.drop(columns=test_drop_cols + [label_col], errors='ignore')

        common_cols = [c for c in train_df.columns if c in test_df.columns]
        if len(common_cols) == 0:
            raise ValueError("No common WaDi feature columns between train and test")

        train_df = train_df[common_cols].apply(pd.to_numeric, errors='coerce')
        test_df = test_df[common_cols].apply(pd.to_numeric, errors='coerce')
        return train_df, test_df, test_labels

    def _reshape_data_to_windows(self, data, window_length, remove_zero_column=False):
        """Reshape data into windows."""
        if isinstance(data, pd.DataFrame):
            data_np = data.to_numpy()
        else:
            data_np = np.array(data)

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        mod = data_np.shape[0] % window_length

        if mod > 0:
            data_np = data_np[:-mod]

        if remove_zero_column and data_np.ndim > 1:
            data_np = data_np[:, 1:]  # Remove first column (index)

        shaped = data_np.reshape(-1, window_length, data_np.shape[1])
        return shaped


    def get_all_dfs(self):
        all_files = glob.glob(os.path.join(self.raw_folder, self.mode, '*.txt'))

        data = {}

        for file_ in all_files:
            machine = file_.split('/')[-1].replace('.txt', '')
            data[machine] = {}
            data[machine]['train'] = None
            data[machine]['val'] = None
            data[machine]['test'] = None
            data[machine]['labels'] = None
            if self.mode in ['train', 'val']:
                data[machine]['train'] = pd.read_csv(file_, delimiter=',', header=None)
            elif self.mode == 'test':
                data[machine]['test'] = pd.read_csv(file_, delimiter=',', header=None)
                label_file = file_.replace('test', 'test_label')
                data[machine]['labels'] = pd.read_csv(label_file, delimiter=',', header=None)

        assert all(df[self.mode].shape[1] == next(iter(data.values()))[self.mode].shape[1] for df in data.values()), "Not all DataFrames have the same number of columns"
        return data


class ADDataset(Dataset):

    input_dim = None  # nr. of different measurements per time point

    def __init__(self, data_dir: str, processed_root: str, mode: str='train', data_kind: str=None,
                 window_length: int=100, window_overlap:float = 0.75, subsample=1.0,
                 n_samples: int=None, data_normalization_strategy:str="none",
                 fixed_subsample_mask: bool = False):

        self.mode = mode
        self.subsample=subsample
        self.fixed_subsample_mask = fixed_subsample_mask
        self.processed_root = processed_root

        objs = dict()
        objs['train'] = ADData(
            data_dir, processed_root=processed_root, mode='train', data_kind=data_kind,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples,
            data_normalization_strategy=data_normalization_strategy)

        objs['test'] = ADData(
            data_dir, processed_root=processed_root, mode='test', data_kind=data_kind,
            window_length=window_length, window_overlap=window_overlap,
            columns=objs['train'].params, n_samples=n_samples,
            normalizer=objs['train'].scaler)

        objs['val'] = ADData(
            data_dir, processed_root=processed_root, mode='val', data_kind=data_kind,
            window_length=window_length, window_overlap=window_overlap,
            columns=objs['train'].params, n_samples=n_samples,
            normalizer=objs['train'].scaler)

        data = objs[mode]

        data_min, data_max = get_data_min_max(objs['train'][:])

        self.feature_names = ADData.params

        tps = data.data[0][1]
        tps = torch.Tensor(tps / tps.max())
        indcs = torch.vstack([data.data[part_idx][1] for part_idx in
                            range(len(data.data))])
        obs = torch.vstack([data.data[part_idx][2][None, :, :] for part_idx in
                                 range(len(data.data))]).float()

        msk = torch.vstack([data.data[part_idx][3][None, :, :] for part_idx in
                                 range(len(data.data))]).float()
        tps = tps[None, :].repeat(obs.shape[0], 1).float()

        if mode == 'test':
            tgt = torch.Tensor(data.targets)
        else:
            tgt = torch.zeros((obs.shape[0], obs.shape[1])).long()

        obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)

        n_samples = obs.shape[0]
        n_time = tps.shape[1]
        input_dim = obs.shape[-1]
        inp_tid = torch.arange(0, n_time).repeat(n_samples, 1).long()

        # Keep one entry to mirror NASADataset layout.
        self.datasets = [{
            'inp_obs': (obs * msk).float(),
            'inp_msk': msk.long(),
            'inp_tps': tps,
            'inp_tid': inp_tid,
            'indcs': indcs,
            'evd_obs': obs.float(),
            'evd_msk': torch.ones_like(msk).long(),
            'evd_tid': inp_tid.long(),
            'evd_tps': tps,
            'aux_tgt': tgt.long(),
            'data_min': data_min,
            'data_max': data_max,
            'input_dim': input_dim,
            'num_timepoints': n_time,
            'dataset_id': data_kind,
        }]

        if self.fixed_subsample_mask:
            if self.mode == 'train':
                masked_ratio = 1.0 - self.subsample
                # Build one fixed mask on the full (pre-window) timeline per feature,
                # then gather by window indices to match dataset window layout.
                indcs_long = indcs.long()
                full_len = int(indcs_long.max().item()) + 1
                burst_mask = create_random_burst_mask(
                    n_features=input_dim,
                    x_len=full_len,
                    masked_ratio=masked_ratio,
                )  # (n_features, full_len)
                full_mask = torch.from_numpy(burst_mask.T.astype(np.int64)).long()  # (full_len, n_features)
                self.datasets[0]['fixed_inp_msk'] = full_mask[indcs_long]  # (n_samples, n_time, n_features)
            else:  # val
                self.datasets[0]['fixed_inp_msk'] = torch.ones_like(msk).long()

        self._lengths = [n_samples]
        self._cumulative = [0]

        # Backward-compatible aliases used by legacy code.
        ds = self.datasets[0]
        ADDataset.input_dim = ds['input_dim']
        self.input_dim = ds['input_dim']
        self.num_timepoints = ds['num_timepoints']
        self.data_min = ds['data_min']
        self.data_max = ds['data_max']
        self.inp_obs = ds['inp_obs']
        self.inp_msk = ds['inp_msk']
        self.inp_tps = ds['inp_tps']
        self.inp_tid = ds['inp_tid']
        self.indcs = ds['indcs']
        self.evd_obs = ds['evd_obs']
        self.evd_msk = ds['evd_msk']
        self.evd_tid = ds['evd_tid']
        self.evd_tps = ds['evd_tps']
        self.aux_tgt = ds['aux_tgt']

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
        return 0, idx

    def __getitem__(self, idx):
        ds_idx, local_idx = self._resolve_index(idx)
        ds = self.datasets[ds_idx]

        if self.mode in ('train', 'val'):
            if self.fixed_subsample_mask:
                msk = ds['fixed_inp_msk'][local_idx].long()
            else:
                masked_ratio = 1.0 - self.subsample
                n_time_s, n_feat_s = ds['inp_msk'][local_idx].shape  # (n_time, n_features)
                burst_mask = create_random_burst_mask(
                    n_features=n_feat_s,
                    x_len=n_time_s,
                    masked_ratio=masked_ratio,
                )  # (n_feat_s, n_time_s)
                msk = torch.from_numpy(burst_mask.T.astype(np.int64)).long()  # (n_time_s, n_feat_s)
        else:
            msk = ds['inp_msk'][local_idx].long()

        inp_and_evd = {
            'inp_obs' : ds['inp_obs'][local_idx].float(),
            'inp_msk' : msk,
            'inp_tid' : ds['inp_tid'][local_idx].long(),
            'inp_tps' : ds['inp_tps'][local_idx].float(),
            'evd_obs' : ds['evd_obs'][local_idx].float(),
            'evd_msk' : ds['evd_msk'][local_idx].long(),
            'evd_tid' : ds['evd_tid'][local_idx].long(),
            'evd_tps' : ds['evd_tps'][local_idx].float(),
            'aux_tgt' : ds['aux_tgt'][local_idx].long(),
            'inp_indcs' : ds['indcs'][local_idx],
            'dataset_idx': ds_idx,
            }
        return inp_and_evd

    def fit_normalizer(self):
        logging.info("Fitting Standard Scaler")
        scaler = StandardScaler()
        scaler.fit(self.evd_obs)
        return scaler

    def normalizer_transform(self, scaler):
        logging.info("Applying Scaler")
        self.evd_obs = scaler.transform(self.evd_obs)
        self.inp_obs = scaler.transform(self.inp_obs)


class ADProvider(DatasetProvider):
    def __init__(self, data_dir=None, dataset=None, window_length:int = 100,
                 window_overlap:float = 0.75, sample_tp=0.5, n_samples:int=None,
                 data_normalization_strategy:str="none", subsample=1.0,
                 fixed_subsample_mask: bool = False):
        DatasetProvider.__init__(self)

        if dataset not in ["SWaT", "WaDi", "SMD"]:
            raise NotImplementedError

        self._dataset = dataset

        processed_root = tempfile.mkdtemp(prefix=f"LatentSDEonHS_{dataset}_processed_")

        self._sample_tp = sample_tp
        self._ds_trn = ADDataset(
            data_dir, processed_root, 'train', data_kind=dataset,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples, subsample=subsample,
            data_normalization_strategy=data_normalization_strategy,
            fixed_subsample_mask=fixed_subsample_mask)

        self._ds_tst = ADDataset(
            data_dir, processed_root, 'test', data_kind=dataset,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples,
            data_normalization_strategy=data_normalization_strategy)

        self._ds_val = ADDataset(data_dir, processed_root, 'val', data_kind=dataset,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples, subsample=subsample,
            data_normalization_strategy=data_normalization_strategy,
            fixed_subsample_mask=fixed_subsample_mask)

        #scaler = self._ds_trn.fit_normalizer()
        #self._ds_trn.normalizer_transform(scaler)
        #self._ds_tst.normalizer_transform(scaler)

    @property 
    def input_dim(self):
        return ADDataset.input_dim

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
    def sample_tp(self):
        return self._sample_tp

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
        shutil.rmtree(self._ds_trn.processed_root)


def create_win_periods(data_, win_size_, win_stride_):
    """ returns the rolling windows of the given flattened data """
    if win_stride_ < 1:
        win_stride_ = 1

    windows = sliding_window_view(data_, (
        win_size_, data_.shape[1]))
    windows = windows.squeeze()
    indcs = sliding_window_view(np.arange(data_.shape[0]), win_size_)

    return indcs[::win_stride_], windows.squeeze()[::win_stride_, :]

