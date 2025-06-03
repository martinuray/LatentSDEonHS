"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""

import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from traitlets import observe

from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider


class AeroData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(self, root, mode='train'):

        self.root = root
        self.mode = mode

        self.download()

        if not self._check_exists():
            print((os.path.join(self.processed_folder, self.destination_file)))
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file), weights_only=False)

        self.params = [
            'target profile',
            'u1',
            'i1',
            'w1 (rotation speed)',
            'measured pitch (rad)',
            'est. pitch (rad)',
            'est. pitch speed',
            'pitch angle (from accel)',
        ]
        self.params_dict = {k: i for i, k in enumerate(self.params)}

        if self.mode == 'test':
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))


    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        raw_data = torch.load(os.path.join(self.raw_folder, self.destination_file))
        feature_names = raw_data.keys()

        indcs = torch.arange(2500)
        tps = indcs / 2500
        obs = np.array([raw_data[feature] for feature in feature_names])
        obs = np.transpose(obs, (1, 2, 0))  # Permute to (batch_size x n_features x signal_per_batch)
        obs = torch.Tensor(obs)

        msk = torch.ones_like(obs)
        if self.mode == 'test':
            tgt = torch.load(os.path.join(self.raw_folder, 'labels.pt'))[:-1].reshape(-1, 2500)
            tgt = torch.from_numpy(tgt)
        else:
            tgt = torch.zeros(obs.shape[0:2])

        data = [(part_idx, tps, obs[part_idx, :, :], msk[part_idx, :, :], tgt[part_idx, :]) for part_idx in range(msk.shape[0])]

        torch.save(data, os.path.join(self.processed_folder, self.destination_file))
        if self.mode == 'test':
            torch.save(tgt, os.path.join(self.processed_folder, self.label_file))

    def _check_exists(self):
        if self.mode == 'test':
            return os.path.isfile(os.path.join(self.processed_folder, self.destination_file)) and os.path.isfile(os.path.join(self.processed_folder, self.label_file))
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def training_file(self):
        return f'train_data.pt'

    @property
    def test_file(self):
        return f'test_data.pt'

    @property
    def destination_file(self):
        if self.mode == 'train':
            return self.training_file
        if self.mode == 'test':
            return self.test_file
        return None

    @property
    def label_file(self):
        return 'test_labels.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]


class AeroDataDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train'):

        objs = {
            'train': AeroData(data_dir, mode='train'),
            'test': AeroData(data_dir, mode='test'),
        }
        data = objs[mode]

        # TODO
        data_min, data_max = get_data_min_max(data[:])
        self.data_min = data_min
        self.data_max = data_max

        self.feature_names = AeroData.params

        _, _, vals, _, _ = data[0]
        AeroDataDataset.input_dim = vals.size(-1)

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


class AeroDataProvider(DatasetProvider):
    def __init__(self, data_dir=None, data_kind=None, random_state=42):
        DatasetProvider.__init__(self)
    
        self._ds_trn = AeroDataDataset(data_dir, 'train')
        self._ds_tst = AeroDataDataset(data_dir, 'test')

    @property
    def input_dim(self):
        return AeroDataDataset.input_dim

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

