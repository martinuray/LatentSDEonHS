"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from torch.distributions import Categorical
from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split
from .common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from .dataset_provider import DatasetProvider

    

class BasicData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(self, root, data_kind=None, mode='train', n_samples = None, num_features = 1):
        self.data_kind = data_kind
        if self.data_kind is None:
            self.data_kind = self.__class__.__name__

        self.root = root
        self.mode = mode
        self.num_features = num_features

        self.params = np.arange(0, self.num_features)
        self.params_dict = {k: i for i, k in enumerate(self.params)}

        self.labels = []
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        #self.download()

        if not self._check_exists():
            print((os.path.join(self.processed_folder, self.destination_file)))
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # TODO: differentiate somehow on train/test/val data
        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)
        #self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            #self.labels = self.labels[:n_samples]


    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
                
    def _check_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.data_kind, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.data_kind, 'processed')

    @property
    def training_file(self):
        if self.data_kind == 'Oscillator':
            return f'{self.mode}.pt'
        return f'basic_data_train.pt'

    @property
    def test_file(self):
        if self.data_kind == 'Oscillator':
            return f'{self.mode}.pt'
        return f'basic_data_test.pt'

    @property
    def val_file(self):
        if self.data_kind == 'Oscillator':
            return f'{self.mode}.pt'
        return f'basic_data_val.pt'

    @property
    def destination_file(self):
        if self.mode == 'train':
            return self.training_file
        if self.mode == 'test':
            return self.test_file
        return self.val_file

    @property
    def label_file(self):
        return ''

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]


class BasicDataDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train', data_kind: str=None, num_features=1, random_state: int=42):

        objs = {
            'train': BasicData(data_dir, mode='train', data_kind=data_kind, n_samples=8000, num_features=num_features),
            'test': BasicData(data_dir, mode='test', data_kind=data_kind, n_samples=8000, num_features=num_features),
            'validation': BasicData(data_dir, mode='validation', data_kind=data_kind, n_samples=8000, num_features=num_features),
        }
        data = objs[mode]

        # TODO
        data_min, data_max = get_data_min_max(data[:])
        self.data_min = data_min
        self.data_max = data_max

        self.feature_names = BasicData.params

        _, _, vals, _, _ = data[0]
        BasicDataDataset.input_dim = vals.size(-1)

        self.tps = torch.stack([tps for _, tps, _, _, _ in data], dim=0).float()
        self.obs = torch.stack([obs for _, _, obs, _, _ in data], dim=0)
        self.msk = torch.stack([msk for _, _, _, msk, _ in data], dim=0)
        self.tgt = torch.stack([tgt for _, _, _, _, tgt in data], dim=0)
        self.num_timepoints = self.tps.shape[1]

        self.inp_obs = self.obs.float() # obs_new
        self.inp_obs = (self.obs * self.msk).float()# obs_new
        self.inp_msk = self.msk.long() #obs_msk
        self.inp_tps = self.tps.float() #tps_new
        self.inp_tid = torch.arange(0, self.inp_tps.shape[1]).repeat(self.tps.shape[0], 1).long() #tid_new.long()

        self.evd_msk = torch.ones_like(self.inp_msk).long()
        self.evd_tid = self.inp_tid.long() #all_idx.repeat(self.tps.shape[0], 1).long()
        self.evd_tps = self.tps.float()
        self.evd_obs = self.obs.float()
        self.aux_tgt = self.tgt.long()

    @property
    def has_aux(self):
        return False

    def __len__(self):
        return len(self.evd_obs)

    def __getitem__(self, idx):
        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx].float(),
            'inp_msk' : self.inp_msk[idx].long(),
            'inp_tid' : self.inp_tid[idx].long(),
            'inp_tps' : self.inp_tps[idx].float(),
            'evd_obs' : self.evd_obs[idx].float(),
            'evd_msk' : self.evd_msk[idx].long(),
            'evd_tid' : self.evd_tid[idx].long(),
            'evd_tps' : self.evd_tps[idx].float(),
            'aux_tgt' : self.aux_tgt[idx].long()
            }
        return inp_and_evd


class BasicDataProvider(DatasetProvider):
    def __init__(self, data_dir=None, data_kind=None, quantization=0.01, sample_tp=0.5, random_state=42, num_features=1):
        DatasetProvider.__init__(self)
    
        self._sample_tp = sample_tp
        self._ds_trn = BasicDataDataset(data_dir, 'train', data_kind=data_kind, num_features=num_features, random_state=random_state)
        self._ds_tst = BasicDataDataset(data_dir, 'test', data_kind=data_kind, num_features=num_features, random_state=random_state)
        self._ds_val = BasicDataDataset(data_dir, 'validation', data_kind=data_kind, num_features=num_features, random_state=random_state)

        # TODO: necessary?
        assert self._ds_trn.num_timepoints == self._ds_tst.num_timepoints
        #assert torch.all(self._ds_trn.data_min == self._ds_tst.data_min)
        #assert torch.all(self._ds_trn.data_max == self._ds_tst.data_max)
        #assert torch.all(self._ds_trn.data_max == self._ds_val.data_max)

    @property 
    def input_dim(self):
        return BasicDataDataset.input_dim

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
    def quantization(self):
        return self._quantization

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

    def get_val_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_val, **kwargs)
