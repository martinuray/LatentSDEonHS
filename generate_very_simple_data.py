##%
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import types
noglobal = lambda f: types.FunctionType(f.__code__, {})


# paths
data_dir = 'data_dir/BasicData/processed'


def generate_signals(num_features_, num_periods_, tps_, base_frequency_=1.0, noise_level_=0.3):

    Y_ = np.zeros((num_periods_, tps.shape[0], num_features_))
    msk_ = generate_base_mask(Y_)

    Y_[:, :, 0] = 1.00 * np.sin(2 * np.pi * base_frequency_ * 1. * tps_)
    Y_[:, :, 0] += np.random.random(Y_[:, :, 0].shape[0:2]) * noise_level_

    if num_features_ > 1:
        Y_[:, :, 1] = (1 - np.cos(2 * np.pi * base_frequency_ * 1. * tps_))
        Y_[:, :, 1] += np.random.random(Y_[:, :, 1].shape[0:2]) * noise_level_ * 3

    if num_features_ > 2:
        Y_[:, 25:50, 2] = (1 - np.cos(2 * np.pi * 4 * base_frequency_ * 1. * tps[25:50])) + np.random.random() * noise_level_ / 4
        Y_[:, :, 2] += np.random.random(Y_[:, :, 2].shape[0:2]) * noise_level_ * 2
        msk_[:, 25:50, 2] = np.random.rand(25) < 0.5

    if num_features_ > 3:
        Y_[:, 50:75, 3] = (1 - np.cos(2 * np.pi * 4 * base_frequency_ * 1. * tps_[50:75])) + np.random.random() * noise_level_ / 3
        Y_[:, :, 3] += np.random.random(Y_[:, :, 3].shape[0:2]) * noise_level_ * 1.7
        msk_[:, 50:75, 3] = np.random.rand(25) < 0.5

    Y_ = norm_data(Y_)
    return Y_, msk_


def norm_data(y_: np.array) -> np.array:
    min_vals = y_.min(axis=1, keepdims=True)
    max_vals = y_.max(axis=1, keepdims=True)
    n_arr = (y_ - min_vals) / (max_vals - min_vals)
    # in the basic sine example the data is in [-0.8, 0.8]
    return n_arr


def generate_base_mask(Y_, ratio_of_sampling_=0.3):
    s = Y_.shape
    # roughly 30 % of the samples are non-nan
    msk_ = np.random.rand(s[0], s[1], s[2]) > ratio_of_sampling_
    msk_ = msk_ * 1
    return msk_


def convert_to_tensor_structure(Y_, tps_, msk_, aux_tgt=None):

    periods_to_store_ = [
        (per,
         torch.tensor(tps_),
         torch.tensor(Y_[per]),
         torch.tensor(msk_[per]),
         torch.zeros_like(torch.tensor(Y_[per])) if aux_tgt is None else torch.tensor(aux_tgt[per]))

         for per in range(Y_.shape[0])
    ]

    # testing the shapes before storing
    for _, tt_, vals_, m_, tgt_ in periods_to_store_:
        assert tt_.shape[0] == vals_.shape[0]
        assert m_.shape == vals_.shape
        #assert vals_.shape == tgt_.shape

    return periods_to_store_


def store_to_file(data_, data_dir_, filename_):
    os.makedirs(data_dir_, exist_ok=True)
    torch.save(data_, os.path.join(data_dir, f'{filename_}.pt'))


def introduce_anomalies(Y_):
    tgt_ = np.zeros_like(Y_)

    # per 1
    tgt_[0, :, 3][Y_[0, :, 3] > 0.5] = 1
    Y_[0, :, 3][Y_[0, :, 3] > 0.5] = 1

    # per 2
    tgt_[1, 75:85, 2] = 1
    Y_[1, 75:85, 2] = 1

    # per 3
    tgt_[2, 0:15, 2] = 1
    Y_[2, 0:15, 2] = 0.5

    # per 4
    per, feature = 3, 3
    tgt_[per, 10:20, feature] = 1
    Y_[per, 10:20, feature] = Y_[per, 10:20, feature] * -0.01

    # per 4
    per, feature = 4, 1
    tgt_[per, 15:35, feature] = 1
    Y_[per, 15:35, feature] = Y_[per, 15:35, feature] * 0.

    # per 5 - point anomalies
    per, feature = 5, 1
    idxs = np.random.permutation(np.arange(Y_[per,:,feature].shape[0]))[:10]
    tgt_[per, idxs, feature] = 1
    Y_[per, idxs, feature] = 0.

    # per 6 - point anomalies
    per, feature = 6, 3
    idxs = np.random.permutation(np.arange(Y_[per, :, feature].shape[0]))[:10]
    tgt_[per, idxs, feature] = 1
    Y_[per, idxs, feature] = 1

    return Y_, tgt_

def produce_dataset_split(num_features_, num_periods_, tps_, data_dir_, filename_, validation=False):
    Y_, mask_ = generate_signals(num_features_, num_periods_, tps_)
    if validation:
        Y_, tgt_ = introduce_anomalies(Y_)
        datastructure_to_store = convert_to_tensor_structure(Y_, tps_, mask_, aux_tgt=tgt_)
    else:
        datastructure_to_store = convert_to_tensor_structure(Y_, tps_, mask_)
    store_to_file(datastructure_to_store, data_dir_, filename_)

#%%
# stuff for signals
fs = 1 / (10*10**-3)   # Hz; every 10ms; as the idea with the testbed
step_size = 1/fs
num_features = 4

tps = np.arange(0, 1, step_size)

# stuff for the dataset
num_periods_train, num_periods_test, num_periods_val = 20, 10, 7
# train
produce_dataset_split(num_features, num_periods_train, tps, data_dir, filename_='basic_data_train')

# test
produce_dataset_split(num_features, num_periods_test, tps, data_dir, filename_='basic_data_test')

# val
produce_dataset_split(num_features, num_periods_val, tps, data_dir, filename_='basic_data_val', validation=True)



##%
#fig, axs = plt.subplots(num_features,1, sharex=True)
#for idx in range(num_features):
#    axs[idx].plot(tps, Y[0, :, idx], '.', label=f"GT Ft. {idx}")

#    y_m = Y[0, :, idx][mask[0, :, idx] == 0]
#    x_m = tps[mask[0, :, idx] == 0]
#    axs[idx].plot(x_m, y_m, '.', label=f"Subsampled Ft. {idx}")

#    axs[idx].grid()
#    axs[idx].legend()
##axs[0].set_suptitle("Generated Data")
#plt.show()
