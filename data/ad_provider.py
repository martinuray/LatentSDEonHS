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
from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider

    

class ADData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(self, root, data_kind="SWaT", mode='train', n_samples = None, subsample = None):

        self.root = root
        self.data_kind = data_kind
        self.mode = mode

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_exists():
            print((os.path.join(self.processed_folder, self.destination_file)))
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # TODO: differentiate somehow on train/test/val data
        raw_data_df = pd.read_csv(os.path.join(self.processed_folder, self.destination_file))
        raw_data_df = raw_data_df.loc[:, (raw_data_df.nunique() > 2)] # TODO: bc of memory issues
        added_ = np.zeros(((3000 - raw_data_df.shape[0]) % 3000, raw_data_df.shape[1]))
        raw_data = np.vstack((raw_data_df, added_))

        raw_data = raw_data.reshape(-1, 3000, raw_data.shape[1])

        if self.mode == 'test':
            self.targets = pd.read_csv(os.path.join(self.processed_folder, self.label_file))
            self.targets = np.array(self.targets)
            self.targets = np.hstack((self.targets.squeeze(), added_[:, 0]))

        else:
            self.targets = np.zeros(raw_data.shape[0:2])

        self.targets = self.targets.reshape(-1, 3000)

        self.params = list(raw_data_df.columns)
        self.params_dict = {k: i for i, k in enumerate(self.params)}

        indcs = torch.arange(0, raw_data.shape[1])
        data_tensor = torch.Tensor(raw_data)
        mask = torch.ones_like(data_tensor)
        mask[-1,-added_.shape[0]:, :] = 0

        if subsample is not None:
            mask = (torch.rand(data_tensor.shape) < subsample) * 1
            #sum = mask.sum(dim=-1)
            #inc_ignore = sum != 0

            #indcs = indcs[inc_ignore]
            #data_tensor = data_tensor[inc_ignore, :]
            #mask = mask[inc_ignore, :]


        if n_samples is not None:
            data_tensor = data_tensor[:n_samples]
            mask = mask[:n_samples]
            indcs = indcs[:n_samples]
            if self.mode == 'test':
                self.targets = self.targets[:n_samples]

        assert data_tensor.shape[0] == mask.shape[0]

        self.data = [(part_idx, indcs, data_tensor[part_idx,:,:], mask[part_idx,:,:]) for part_idx in range(mask.shape[0])]


    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
                
    def _check_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.data_kind)

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.data_kind)

    @property
    def training_file(self):
        return 'train.csv'

    @property
    def test_file(self):
        return 'test.csv'

    @property
    def val_file(self):
        return None

    @property
    def destination_file(self):
        if self.mode == 'train':
            return self.training_file
        if self.mode == 'test':
            return self.test_file
        return self.val_file

    @property
    def label_file(self):
        return 'labels.csv'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]


class ADDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train', data_kind: str=None):

        objs = {
            'train': ADData(data_dir, mode='train', data_kind=data_kind, subsample=0.1, n_samples=6000),
            'test': ADData(data_dir, mode='test', data_kind=data_kind, n_samples=6000),
        }
        data = objs[mode]

        # TODO
        data_min, data_max = get_data_min_max(data[:])
        self.data_min = data_min
        self.data_max = data_max

        self.feature_names = ADData.params
        ADDataset.input_dim = data[0][2].shape[1]

        self.tps = data.data[0][1]
        self.tps = torch.Tensor(self.tps / self.tps.max())
        self.obs = torch.vstack([data.data[part_idx][2][None, :, :] for part_idx in
                                 range(len(data.data))]).float()

        self.msk = torch.vstack([data.data[part_idx][3][None, :, :] for part_idx in
                                 range(len(data.data))]).float()
        self.tps = self.tps[None, :].repeat(self.obs.shape[0], 1).float()

        #self.msk = self.msk[None, :, :]
        self.tgt = torch.Tensor(data.targets)
        #self.tgt = self.tgt[None, :]
        self.num_timepoints = self.tps.shape[1]

        self.inp_obs = self.obs.float() # obs_new
        self.inp_obs = (self.obs * self.msk).float()# obs_new
        self.inp_msk = self.msk.long() #obs_msk
        self.inp_tps = self.tps
        self.inp_tid = torch.arange(0, self.inp_tps.shape[1]).repeat(self.obs.shape[0], 1).long() #tid_new.long()

        self.evd_msk = torch.ones_like(self.inp_msk).long()
        self.evd_tid = self.inp_tid.long() #all_idx.repeat(self.tps.shape[0], 1).long()
        self.evd_tps = self.tps
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


class ADProvider(DatasetProvider):
    def __init__(self, data_dir=None, data_kind=None, sample_tp=0.5):
        DatasetProvider.__init__(self)
    
        self._sample_tp = sample_tp
        self._ds_trn = ADDataset(data_dir, 'train', data_kind=data_kind)
        self._ds_tst = ADDataset(data_dir, 'test', data_kind=data_kind)

        # TODO: necessary?
        #assert self._ds_trn.num_timepoints == self._ds_tst.num_timepoints
        #assert torch.all(self._ds_trn.data_min == self._ds_tst.data_min)
        #assert torch.all(self._ds_trn.data_max == self._ds_tst.data_max)
        #assert torch.all(self._ds_trn.data_max == self._ds_val.data_max)

    @property 
    def input_dim(self):
        return ADDataset.input_dim

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

    def get_train_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_trn, **kwargs)

    def get_test_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_tst, **kwargs)


#%%
ad = ADProvider(data_dir='data_dir', data_kind="SWaT")