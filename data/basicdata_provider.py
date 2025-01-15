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

    params = np.arange(0,4)

    params_dict = {k: i for i, k in enumerate(params)}

    labels = [ ]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, n_samples = None):

        self.root = root
        self.train = train
        self.reduce = "average"

        self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

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

        experiment_runs = []

        for csvfile in os.listdir(self.raw_folder):
            if (self.train and 'train' in csvfile) or (not self.train and 'test' in csvfile):
                pass  # for readability
            else:
                continue

            record_id = int(csvfile.split('.')[0].split('_')[-1])

            df_ = pd.read_csv(os.path.join(self.raw_folder, csvfile),
                              index_col=0)

            # drop all rows in df_, where the columns are all nan

            # remove all lines in df_, where there are all columns nan
            df_ = df_.loc[~df_.isnull().apply(lambda x: all(x), axis=1)]
            #df_.drop(, inplace=True)

            tt = [0.]
            vals = [torch.zeros(len(self.params))]
            mask = [torch.zeros(len(self.params))]

            for row in df_.iterrows():
                if row[0] != 0.: # to make sure, that there exists something for time point 0
                    tt.append(row[0])
                    vals.append(torch.zeros(len(self.params)))
                    mask.append(torch.zeros(len(self.params)))

                data = row[1]  # helper
                for param in self.params_dict.keys():
                    if not np.isnan(data.iloc[param]):
                        vals[-1][self.params_dict[param]] = float(data.iloc[param])
                        mask[-1][self.params_dict[param]] = 1

            tt = torch.tensor(tt)
            vals = torch.stack(vals)
            mask = torch.stack(mask)

            experiment_runs.append((record_id, tt, vals, mask))

        torch.save(
            experiment_runs,
            os.path.join(self.processed_folder, self.destination_file),
        )
                
    def _check_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'basic_data_train.pt'

    @property
    def test_file(self):
        return 'basic_data_test.pt'

    @property
    def destination_file(self):
        if self.train:
            return self.training_file
        return self.test_file

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
    
    def __init__(self, data_dir: str, mode: str='train', quantization: float=0.01, random_state: int=42):
        self._quantization = quantization
        
        trn_obj = BasicData(data_dir, train=True, n_samples=8000)
        tst_obj = BasicData(data_dir, train=False, n_samples=8000)
        all_obj = trn_obj[:] + tst_obj[:]
        
        #trn_data, tst_data = train_test_split(all_obj, train_size=0.8, random_state=random_state, shuffle=True)
        data_min, data_max = get_data_min_max(trn_obj)
        _, _, vals, _ = trn_obj[0]
        BasicDataDataset.input_dim = vals.size(-1)

        len_tt = [ex[1].size(0) for ex in trn_obj]
        maxlen = np.max(len_tt) # max. nr. of available timepoints at given quantization
        
        if mode=='train':
            data = trn_obj[:]
        elif mode=='test':
            data = tst_obj[:]
            
        obs = torch.zeros([len(data), maxlen, BasicDataDataset.input_dim])
        msk = torch.zeros([len(data), maxlen, BasicDataDataset.input_dim])
        tps = torch.zeros([len(data), maxlen])

        for b, (_, record_tps, record_obs, record_msk) in enumerate(data):
            currlen = record_tps.size(0)
            obs[b, :currlen] = record_obs
            msk[b, :currlen] = record_msk
            tps[b, :currlen] = record_tps
        
        obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)
        
        tid = (tps/self._quantization).round().long()
        if torch.max(tps) != 0.:
            tps = tps / torch.max(tps)
        
        self.evd_obs = obs
        self.evd_msk = msk.long()
        self.evd_tid = tid.long()
        self.evd_tps = tps
        self.data_min = data_min
        self.data_max = data_max
        self.feature_names = BasicData.params
        
        self.num_timepoints = int(np.round(48./self._quantization))+1

    @property    
    def has_aux(self):
        return False

    def __len__(self):
        return len(self.evd_obs)

    def __getitem__(self, idx):
        inp_and_evd = {
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx],
            'evd_tps' : self.evd_tps[idx]
            }
        return inp_and_evd


class BasicDataProvider(DatasetProvider):
    def __init__(self, data_dir=None, quantization=0.1, sample_tp=0.5, random_state=42):
        DatasetProvider.__init__(self)
    
        self._sample_tp = sample_tp
        self._quantization = quantization
        self._ds_trn = BasicDataDataset(data_dir, 'train', quantization=quantization, random_state=random_state)
        self._ds_tst = BasicDataDataset(data_dir, 'test', quantization=quantization, random_state=random_state)
        
        assert self._ds_trn.num_timepoints == self._ds_tst.num_timepoints
        assert torch.all(self._ds_trn.data_min == self._ds_tst.data_min)
        assert torch.all(self._ds_trn.data_max == self._ds_tst.data_max)
        
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
    def num_train_samples(self):
        return len(self._ds_trn)
    
    @property 
    def num_test_samples(self):
        return len(self._ds_tst)
    
    @property 
    def num_val_samples(self):
        raise NotImplementedError
    
    def _collate(self, data):
        batch = default_collate(data)
        inp_obs, inp_msk, inp_tid = subsample_timepoints(
            batch['evd_obs'].clone(), 
            batch['evd_msk'].clone(),
            batch['evd_tid'].clone(), self.sample_tp)
        batch['inp_obs'] = inp_obs
        batch['inp_tps'] = inp_tid/(self.num_timepoints-1)
        batch['inp_msk'] = inp_msk
        batch['inp_tid'] = inp_tid 
        return batch
        
    def get_train_loader(self, **kwargs):
        return DataLoader(self._ds_trn, collate_fn=self._collate, **kwargs)
    
    def get_test_loader(self, **kwargs):
        return DataLoader(self._ds_tst, collate_fn=self._collate, **kwargs)
    
    
def subsample_timepoints(data, mask, tid, p=1.):
    assert 0. <= p <= 1.
    if p == 1.:
        sub_data, sub_mask, sub_tid = data, mask, tid
    else:
        tp_msk = torch.rand(tid.shape, device=tid.device) <= p # -> [batch_size, num_time_points] 
        sub_tid = tid * tp_msk
        tp_msk = tp_msk.unsqueeze(-1).expand_as(data)
        sub_data, sub_mask = (x * tp_msk for x in [data, mask])
    return sub_data, sub_mask, sub_tid

