"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd

import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider
from data.process_water_treatment_datasets import reshape_data


class QADData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(
            self, root_path, dataset_number=1, mode='train',
            window_length:int = 100, window_overlap: float = 0.0,
            normalizer=None,
            data_normalization_strategy:str="none"
    ):

        self.scaler = normalizer
        self.data_normalization_strategy = data_normalization_strategy
        self.root_path = root_path
        self.dataset_number = dataset_number
        self.mode = mode
        self.window_length = window_length
        self.ds_hz = 100

        self.overlapping_windows = window_overlap

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_exists(): # does raw exist?
            if not self._check_exist_raw_data():
                raise RuntimeError('Dataset not found.')

            if not self._check_exists():
                self._process_QAD_data()
            else:
                raise NotImplementedError

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)

        if self.mode == 'test':
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file), weights_only=False)


    def _check_exist_raw_data(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        print(os.path.join(self.raw_folder, f'train_{self.dataset_number}.pkl') ,
            os.path.join(self.raw_folder, f'test_{self.dataset_number}.pkl') ,
            os.path.join(self.raw_folder, f'test_label_{self.dataset_number}.pkl'))

        return os.path.isfile(os.path.join(self.raw_folder, f'train_{self.dataset_number}.pkl')) and \
            os.path.isfile(os.path.join(self.raw_folder, f'test_{self.dataset_number}.pkl')) and \
            os.path.isfile(os.path.join(self.raw_folder, f'test_label_{self.dataset_number}.pkl'))
                
    def _check_exists(self):
        return os.path.isfile(os.path.join(
            self.processed_folder, f'{self.mode}_{self.dataset_number}.pt')
        )

    @property
    def raw_folder(self):
        return os.path.join(self.root_path, "QAD", f'raw/qad_clean_pkl_{self.ds_hz}'
                                                   f'Hz/')

    @property
    def processed_folder(self):
        return os.path.join(self.root_path, "QAD", f'processed/qad_clean_pkl_{self.ds_hz}Hz/')

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
        if self.mode == 'train':
            return self.training_file
        if self.mode == 'test':
            return self.test_file
        return self.val_file

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

    def _process_QAD_data(self, n_samples=None):
        logging.warning("Processing QAD Data")
        raw_data = load_pickl(os.path.join(self.raw_folder, f'{self.mode}_{self.dataset_number}.pkl'))
        raw_data = reshape_data(raw_data, self.window_length)

        if self.mode == 'test':
            self.targets = load_pickl(os.path.join(self.raw_folder, f'test_label_{self.dataset_number}.pkl'))
            self.targets = reshape_data(self.targets, self.window_length)
        else:
            self.targets = np.zeros(raw_data.shape[0:2])

        #self.params_dict = {k: i for i, k in enumerate(self.params)}

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
            val_indices = indices[:int(data_len * 0.9)]
            train_indices = indices[int(data_len * 0.9):]

            torch.save([data[i] for i in train_indices], os.path.join(self.processed_folder, self.destination_file))
            torch.save([data[i] for i in val_indices], os.path.join(self.processed_folder, self.val_file))


        if self.mode == 'test':
            torch.save(data, os.path.join(self.processed_folder, self.destination_file))
            torch.save(self.targets, os.path.join(self.processed_folder, self.label_file))



class QADDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train', dataset_number: int=1,
                 window_length: int=100, window_overlap:float = 0.75, subsample=1.0,
                 data_normalization_strategy:str="none"):

        self.mode = mode
        self.subsample=subsample

        objs = dict()
        objs['train'] = QADData(
            data_dir, mode='train', dataset_number=dataset_number,
            window_length=window_length, window_overlap=window_overlap,
            data_normalization_strategy=data_normalization_strategy)

        objs['test'] = QADData(
            data_dir, mode='test', dataset_number=dataset_number,
            window_length=window_length, window_overlap=window_overlap,
            normalizer=objs['train'].scaler)

        objs['val'] = QADData(
            data_dir, mode='val', dataset_number=dataset_number,
            window_length=window_length, window_overlap=window_overlap,
            normalizer=objs['train'].scaler)

        data = objs[mode]

        data_min, data_max = get_data_min_max(objs['train'][:])

        self.feature_names = QADData.params
        QADDataset.input_dim = data[0][2].shape[1]

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

        self.num_timepoints = tps.shape[1]

        self.inp_obs = obs.float() # obs_new
        self.inp_obs = (obs * msk).float()# obs_new
        self.inp_msk = msk.long() #obs_msk
        self.inp_tps = tps
        self.inp_tid = torch.arange(0, self.inp_tps.shape[1]).repeat(obs.shape[0], 1).long() #tid_new.long()
        self.indcs = indcs

        self.evd_msk = torch.ones_like(self.inp_msk).long()
        self.evd_tid = self.inp_tid.long() #all_idx.repeat(self.tps.shape[0], 1).long()
        self.evd_tps = tps
        self.evd_obs = obs.float()
        self.aux_tgt = tgt.long()

    @property
    def has_aux(self):
        return False

    def __len__(self):
        return len(self.evd_obs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            msk = (torch.rand(self.inp_msk[idx].shape) < self.subsample).to(torch.int).long()
        else:
            msk = self.inp_msk[idx].long()

        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx].float(),
            'inp_msk' : msk,
            'inp_tid' : self.inp_tid[idx].long(),
            'inp_tps' : self.inp_tps[idx].float(),
            'evd_obs' : self.evd_obs[idx].float(),
            'evd_msk' : self.evd_msk[idx].long(),
            'evd_tid' : self.evd_tid[idx].long(),
            'evd_tps' : self.evd_tps[idx].float(),
            'aux_tgt' : self.aux_tgt[idx].long(),
            'inp_indcs' : self.indcs[idx]
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


class QADProvider(DatasetProvider):
    def __init__(self, data_dir=None, dataset_number=None, window_length:int = 100,
                 window_overlap:float = 0.0,
                 data_normalization_strategy:str="none", subsample=1.0):
        DatasetProvider.__init__(self)

        self._dataset = dataset_number

        self._ds_trn = QADDataset(
            data_dir, 'train', dataset_number=dataset_number,
            window_length=window_length, window_overlap=window_overlap,
            subsample=subsample,
            data_normalization_strategy=data_normalization_strategy)

        self._ds_tst = QADDataset(
            data_dir, 'test', dataset_number=dataset_number,
            window_length=window_length, window_overlap=window_overlap,
            data_normalization_strategy=data_normalization_strategy)

        self._ds_val = QADDataset(data_dir, 'val', dataset_number=dataset_number,
                                  window_length=window_length, window_overlap=window_overlap,
                                  subsample=subsample,
                                  data_normalization_strategy=data_normalization_strategy)

    @property 
    def input_dim(self):
        return QADDataset.input_dim

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


def load_pickl(dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, pd.Series):
        data = data.to_frame(name='labels')
    return data