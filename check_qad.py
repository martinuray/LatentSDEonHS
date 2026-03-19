import pickle
#import matplotlib.pyplot as plt
import numpy as np

#%%
for i in range(1, 11):
    with open(f'data_dir/QAD/raw/qad_clean_pkl_100Hz/test_label_{i}.pkl', 'rb') as f:
        data = pickle.load(f)

    na_cnt = data[data.isna()].shape[0]
    if na_cnt > 0:
        print(f"idx {i}: {na_cnt}")
del data, f

print("------------------")
for i in range(1, 11):
    d =  np.load(f'data_dir/QAD/raw/qad_clean_pkl_100Hz/test_label_{i}.pkl', allow_pickle=True)
    na_cnt = d[d.isna()].shape[0]
    if na_cnt > 0:
        print(f"idx {i}: {na_cnt}")
    del d