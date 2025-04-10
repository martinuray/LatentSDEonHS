"""Dataset provider for the the IMM Data of the JRC ISIA.
Author: Martin Uray
"""
import logging
import os
import pickle
import zipfile

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider


class IMMData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(self, root, mode='train', n_samples = None):

        self.root = root
        self.mode = mode

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_processed_exists():
            logging.info('Dataset not processed yet. Attempting to processes..')

            if not self._check_raw_exists():
                logging.fatal(f'Dataset not found. Make sure it is placed in {self.raw_folder}.')
                raise RuntimeError(f'Dataset not found. Make sure it is placed in {self.raw_folder}.')

            self.preprocess_data()

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file))
        with open(os.path.join(self.processed_folder, 'parameter.pkl'), 'rb') as fp:
            self.params = pickle.load(fp)
        self.params_dict = {k: i for i, k in enumerate(self.params)}

        #if self.mode == 'test':
        #    self.labels = torch.load(
        #        os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            if self.mode == 'test':
                self.labels = self.labels[:n_samples]


    def preprocess_data(self):
            """
            Preprocesses the raw data by extracting and concatenating CSV files
            from a zip archive and storing it into one torch pickle.
            """
            os.makedirs(self.processed_folder, exist_ok=True)

            data = None
            with open(os.path.join(self.raw_folder, 'raw.zip'), 'rb') as zip_file:
                with zipfile.ZipFile(zip_file, 'r') as archive:
                    archive.extractall(self.raw_folder)

            # Load the CSVs into a list of dataframes
            for file_ in os.listdir(self.raw_folder):
                if not file_.endswith(".csv"):
                    continue

                feature_name = file_.replace('.csv', '')

                data_ = pd.read_csv(os.path.join(self.raw_folder, file_), index_col=0, header=None, skiprows=1, names=[feature_name])
                if data is None:
                    data = data_
                else:
                    data = pd.concat([data, data_], axis=1)

                os.remove(os.path.join(self.raw_folder, file_))

            data.index = pd.to_datetime(data.index, unit='s')
            data.sort_index(inplace=True)

            # temp, t
            data = data.loc[:, ['set' not in col for col in data.columns.tolist()]]

            start_index = data[data['Control.imm.status.productionCycleCounter'] == 1.0].index[0]

            # we are accepting losing the last cycle, however dont have noise in the end
            end_index = data[data['Control.imm.status.productionCycleCounter'] == 120.0].index[0]

            start_test_index = data[data['Control.imm.status.productionCycleCounter'] == 21.0].index[0]

            data = data[data.index >= start_index]
            data = data[data.index < end_index]
            data_train = data[data.index < start_test_index]
            data_test = data[data.index >= start_test_index]
            
            def split_on_prod_cycle(df_):
                ind = df_['Control.imm.status.productionCycleCounter'].dropna().index.to_list()
                iind = [df_.index.get_loc(i) for i in ind]
                df_.drop(columns=['Control.imm.status.productionCycleCounter'], inplace=True)
                df_ = df_.dropna(how='all')     # remove empty rows
                dfs_ = np.split(df_, iind, axis=0)
                return dfs_

            dfs_train = split_on_prod_cycle(data_train)
            dfs_test = split_on_prod_cycle(data_test)

            def convert_to_tensor_structure(dfs_):
                data_to_store = []
                for idx, df_ in enumerate(dfs_):
                    # we dont want to store empty dataframess
                    if df_.shape[0] == 0:
                        continue

                    # do something with the time
                    tt = df_.index.to_pydatetime()
                    tt = np.array([t.total_seconds() for t in (tt - tt[0])])
                    tt = torch.tensor(tt/tt.max())
                    d = torch.tensor(df_.to_numpy())
                    msk_ = torch.tensor((~df_.isna()*1).to_numpy())

                    d[torch.isnan(d)] = 0

                    data_to_store.append(
                        (idx, tt, d, msk_)
                    )
                return data_to_store

            train_to_store = convert_to_tensor_structure(dfs_train)
            test_to_store = convert_to_tensor_structure(dfs_test)

            with open(os.path.join(self.processed_folder, 'parameter.pkl'), 'wb') as fp:
                pickle.dump(dfs_train[0].columns.to_list(), fp)

            torch.save(train_to_store, os.path.join(self.processed_folder, self.training_file))
            torch.save(test_to_store, os.path.join(self.processed_folder, self.test_file))

                
    def _check_processed_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, self.destination_file))

    def _check_raw_exists(self):
        return os.path.isfile(os.path.join(self.raw_folder, 'raw.zip'))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'train.pt'

    @property
    def test_file(self):
        return 'test.pt'

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


class IMMDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train'):

        objs = {
            'train': IMMData(data_dir, mode='train', n_samples=6000),
            'test': IMMData(data_dir, mode='test', n_samples=6000),
        }
        data = objs[mode]

        max_signal_length = np.max([
            len(objs[mode_].data[part_idx][2])
             for mode_ in ['train', 'test']
             for part_idx in range(len(objs[mode_].data))
        ])

        data_min, data_max = get_data_min_max(objs['train'][:])

        self.feature_names = IMMData.params
        IMMDataset.input_dim = data[0][2].shape[1]

        tps = [data.data[part_idx][1][:] for part_idx in range(len(data.data))]
        tps = torch.stack(
            [F.pad(tp, (0, max_signal_length - tp.size(0))) for tp in tps],
            dim=0)
        tps = tps / tps.max(dim=1, keepdim=True).values

        obs = [data.data[part_idx][2][:, :] for part_idx in range(len(data.data))]
        obs = pad_list_and_stack(obs, max_signal_length)

        msk = [data.data[part_idx][3][:, :] for part_idx in range(len(data.data))]
        msk = pad_list_and_stack(msk, max_signal_length)

        #tgt = torch.Tensor(data.targets)

        obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)

        self.num_timepoints = tps.shape[1]

        self.inp_obs = obs.float() # obs_new
        self.inp_obs = (obs * msk).float()# obs_new
        self.inp_msk = msk.long() #obs_msk
        self.inp_tps = tps
        self.inp_tid = torch.arange(0, self.inp_tps.shape[1]).repeat(obs.shape[0], 1).long() #tid_new.long()

        self.evd_msk = torch.ones_like(self.inp_msk).long()
        self.evd_tid = self.inp_tid.long() #all_idx.repeat(self.tps.shape[0], 1).long()
        self.evd_tps = tps
        self.evd_obs = obs.float()
        #self.aux_tgt = tgt.long()

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
            #'aux_tgt' : self.aux_tgt[idx].long()
            }
        return inp_and_evd


class IMMProvider(DatasetProvider):
    def __init__(self, data_dir=None,  sample_tp=0.5):
        DatasetProvider.__init__(self)
    
        self._sample_tp = sample_tp
        self._ds_trn = IMMDataset(data_dir, mode='train')
        self._ds_tst = IMMDataset(data_dir, mode='test')

    @property 
    def input_dim(self):
        return IMMDataset.input_dim

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


def pad_list_and_stack(lo, max_signal_length):
    lo = [F.pad(ob, (0, 0, 0, max_signal_length - ob.size(0))) for ob in lo]
    lo = torch.stack(lo, dim=0).float()
    return lo

#%%
a = IMMProvider('/scratch1/muray/LatentSDEonHS/data_dir/')