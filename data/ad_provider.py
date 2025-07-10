"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""
import glob
import logging
import os
from typing import DefaultDict

import numpy as np
import pandas as pd

import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from data.common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from data.dataset_provider import DatasetProvider

    

class ADData(object):

    params = None
    labels = None
    params_dict, labels_dict = None, None

    def __init__(
            self, root, data_kind="SWaT", mode='train',
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

        self.labels = ['Anomaly']
        self.labels_dict = {k: i for i, k in enumerate(self.labels)}

        if not self._check_exists(): # does raw exist?
            if not self._check_exist_raw_data():
                raise RuntimeError('Dataset not found. You can use download=True to download it')

            if self.data_kind in ["SWaT", "WaDi"]:
                if not self._check_exists():
                    self._process_water_treatment_data(
                        columns=columns, n_samples=n_samples)
            elif self.data_kind in ["SMD"]:
                self._process_server_machine_data(
                    columns=columns, n_samples=n_samples)
            else:
                raise NotImplementedError

        self.data = torch.load(os.path.join(self.processed_folder, self.destination_file))

        if self.mode == 'test':
            self.targets = torch.load(os.path.join(self.processed_folder, self.label_file))


    def _check_exist_raw_data(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        if self.data_kind == "SMD": # TODO: maybe a bit more love
            return os.path.isdir(os.path.join(self.raw_folder, self.mode))

        return os.path.isfile(os.path.join(self.raw_folder, 'train.csv')) and \
            os.path.isfile(os.path.join(self.raw_folder, 'test.csv')) and \
            os.path.isfile(os.path.join(self.raw_folder, 'labels.csv'))
                
    def _check_exists(self):
        if self.data_kind == "SMD": # TODO: maybe a bit more love
            return os.path.isfile(os.path.join(self.processed_folder,
                                               f'{self.mode}.pt'))

        return os.path.isfile(os.path.join(self.processed_folder, f'{self.mode}.pt'))

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.data_kind, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.data_kind, 'processed')

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
        raw_data_df = pd.read_csv(
            os.path.join(self.raw_folder, f'{self.mode}.csv'),
            delimiter=',')
        if columns is None:
            # TODO: otherwise issue when normalizing
            #raw_data_df = raw_data_df.loc[:, (raw_data_df.nunique() > 1)]
            self.params = list(raw_data_df.columns)
        else:
            self.params = columns
            raw_data_df = raw_data_df.loc[:, self.params]

        # normalize data
        raw_data_df = self.normalize_data(raw_data_df)
        if "WaDi" in self.data_kind:
            logging.info(f"Limiting WaDi Dataset to {60_000} samples, as per reference")
            # done as with QuoVadis, see
            # https://github.com/ssarfraz/QuoVadisTAD/blob/8e2de5a1574d1f8b2b669e2aa81a34fd92bd5b58/quovadis_tad/model_utils/model_def.py#L55
            raw_data_df = raw_data_df.iloc[:60_000]

        if self.mode == 'test':
            added_ = np.zeros(((self.max_signal_length - raw_data_df.shape[
                0]) % self.max_signal_length, raw_data_df.shape[1]))
            raw_data = np.vstack((raw_data_df, added_)).copy()
            raw_data = raw_data.reshape(-1, self.max_signal_length,
                                        raw_data.shape[1])

            # TODO take care that no overlapping windows are taken
            self.targets = pd.read_csv(
                os.path.join(self.raw_folder, f'labels.csv'))
            self.targets = np.array(self.targets.values)
            self.targets = np.hstack((self.targets.squeeze(), added_[:, 0]))
            self.targets = self.targets.reshape(-1, self.max_signal_length)
        else:
            raw_data = raw_data_df.copy()
            raw_data = create_win_periods(raw_data, self.max_signal_length,
                                          int(self.max_signal_length * (1-self.overlapping_windows)))

            self.targets = np.zeros(raw_data.shape[0:2])

        # temp stuff
#        raw_data = raw_data_df.copy()
#        raw_data = create_win_periods(raw_data, self.max_signal_length,
#                                      int(self.max_signal_length * (1 - self.overlapping_windows)))

#        if self.mode == 'test':
#            self.targets = pd.read_csv(
#                os.path.join(self.raw_folder, f'labels.csv'))
#            self.targets = np.array(self.targets.values)
#            self.targets = create_win_periods(self.targets, self.max_signal_length,
#                                              int(self.max_signal_length * (1 - self.overlapping_windows)))
#        else:
#            self.targets = np.zeros(raw_data.shape[0:2])

        self.params_dict = {k: i for i, k in enumerate(self.params)}
        indcs = torch.arange(0, raw_data.shape[1])
        data_tensor = torch.Tensor(raw_data)
        mask = torch.ones_like(data_tensor)

        # handle nan values in the dataset
        mask[data_tensor.isnan()] = 0
        data_tensor[data_tensor.isnan()] = 0

        if self.mode == 'test':
            mask[-1, -added_.shape[0]:, :] = 0

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

        data = [(part_idx, indcs, data_tensor[part_idx, :, :],
                 mask[part_idx, :, :]) for part_idx in
                range(mask.shape[0])]

        torch.save(data, os.path.join(self.processed_folder, self.destination_file))
        if self.mode == 'test':
            torch.save(self.targets, os.path.join(self.processed_folder, self.label_file))

    def _process_server_machine_data(self, columns=None, n_samples=None):
        logging.warning("Processing Server Machine Data")

        all_dfs = self.get_all_dfs()

        if columns is None:
            self.params = next(iter(all_dfs.values()))[self.mode].columns
        else:
            self.params = columns

        if self.mode == 'test':
            raw_data_list, raw_targets_list, masks_list = [], [], []
            for _, values in all_dfs.items():
                test_data = values['test']
                added_ = np.zeros(((self.max_signal_length - test_data.shape[
                    0]) % self.max_signal_length, test_data.shape[1]))
                raw_data = np.vstack((test_data, added_)).copy()
                raw_data = raw_data.reshape(-1, self.max_signal_length,
                                            raw_data.shape[1])
                raw_data_list.append(raw_data)
                msk = np.ones_like(raw_data)
                msk[-1, -added_.shape[0]:, :] = 0
                masks_list.append(msk)

                # TODO take care that no overlapping windows are taken
                targets = values['labels']
                targets = np.array(targets)
                targets = np.hstack((targets.squeeze(), added_[:, 0]))
                targets = targets.reshape(-1, self.max_signal_length)
                raw_targets_list.append(targets)

            raw_data = np.vstack(raw_data_list)
            self.targets = np.vstack(raw_targets_list)
            data_tensor = torch.Tensor(raw_data)
            mask = torch.Tensor(np.vstack(masks_list))

        else:
            raw_dfs = []
            for _, values in all_dfs.items():
                rd = values['train'].to_numpy().copy()
                raw_dfs.append(create_win_periods(rd, self.max_signal_length,
                                                  int(self.max_signal_length * self.overlapping_windows)))

            raw_data = np.vstack(raw_dfs)
            self.targets = np.zeros(raw_data.shape[0:2])
            data_tensor = torch.Tensor(raw_data)
            mask = torch.ones_like(data_tensor)

        self.params_dict = {k: i for i, k in enumerate(self.params)}
        indcs = torch.arange(0, raw_data.shape[1])

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

        #assert data_tensor.shape[0] == mask.shape[0]

        data = [(part_idx, indcs, data_tensor[part_idx, :, :],
                 mask[part_idx, :, :]) for part_idx in
                range(mask.shape[0])]

        torch.save(data, os.path.join(self.processed_folder, self.destination_file))
        if self.mode == 'test':
            torch.save(self.targets, os.path.join(self.processed_folder, self.label_file))

    def get_all_dfs(self):
        all_files = glob.glob(os.path.join(self.raw_folder, self.mode, '*.txt'))

        data = {}

        for file_ in all_files:
            machine = file_.split('/')[-1].replace('.txt', '')
            data[machine] = {}
            data[machine]['train'] = None
            data[machine]['test'] = None
            data[machine]['labels'] = None
            if self.mode == 'train':
                data[machine]['train'] = pd.read_csv(file_, delimiter=',', header=None)
            elif self.mode == 'test':
                data[machine]['test'] = pd.read_csv(file_, delimiter=',', header=None)
                label_file = file_.replace('test', 'test_label')
                data[machine]['labels'] = pd.read_csv(label_file, delimiter=',', header=None)

        assert all(df[self.mode].shape[1] == next(iter(data.values()))[self.mode].shape[1] for df in data.values()), "Not all DataFrames have the same number of columns"
        return data


class ADDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train', data_kind: str=None,
                 window_length: int=100, window_overlap:float = 0.75,
                 n_samples: int=None, data_normalization_strategy:str="none"):

        objs = dict()
        objs['train'] = ADData(
            data_dir, mode='train', data_kind=data_kind,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples,
            data_normalization_strategy=data_normalization_strategy)

        objs['test'] = ADData(
            data_dir, mode='test', data_kind=data_kind,
            window_length=window_length, window_overlap=window_overlap,
            columns=objs['train'].params, n_samples=n_samples,
            normalizer=objs['train'].scaler)

        data = objs[mode]

        data_min, data_max = get_data_min_max(objs['train'][:])

        self.feature_names = ADData.params
        ADDataset.input_dim = data[0][2].shape[1]

        tps = data.data[0][1]
        tps = torch.Tensor(tps / tps.max())
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
                 data_normalization_strategy:str="none"):
        DatasetProvider.__init__(self)

        if dataset not in ["SWaT", "WaDi", "SMD"]:
            raise NotImplementedError

        self._dataset = dataset

        self._sample_tp = sample_tp
        self._ds_trn = ADDataset(
            data_dir, 'train', data_kind=dataset,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples,
            data_normalization_strategy=data_normalization_strategy)

        self._ds_tst = ADDataset(
            data_dir, 'test', data_kind=dataset,
            window_length=window_length, window_overlap=window_overlap,
            n_samples=n_samples,
            data_normalization_strategy=data_normalization_strategy)

        #scaler = self._ds_trn.fit_normalizer()
        #self._ds_trn.normalizer_transform(scaler)
        #self._ds_tst.normalizer_transform(scaler)

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


def create_win_periods(data_, win_size_, win_stride_):
    """ returns the rolling windows of the given flattened data """
    if win_stride_ < 1:
        win_stride_ = 1

    windows = sliding_window_view(data_, (
        win_size_, data_.shape[1]))
    return windows.squeeze()[::win_stride_, :]

