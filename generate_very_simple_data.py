##%
import numpy as np
import matplotlib.pyplot as plt

base_frequency = 0.2   # Hz
fs = 1 / (10*10**-3)   # Hz; every 10ms; as the idea with the testbed
step_size = 1/fs

num_periods = 50

start_s = 0
end_s = 1/base_frequency * num_periods

num_features = 4

#%%
x = np.arange(0, end_s, step_size)
Y = np.zeros((num_features, len(x)))

def norm_data(y_):
    # in the basic sine example the data is in [-0.8, 0.8]
    return (2. * (y_ - np.min(y_)) / np.ptp(y_) - 1) * 0.8

Y[0] = 1.00 * np.sin(2*np.pi*base_frequency*1.*x) + 0.21 * np.sin(2*np.pi*base_frequency*2*x)
Y[1] = 0.30 * np.cos(2*np.pi*base_frequency*4.*x) + 0.21 * np.sin(2*np.pi*base_frequency*2*x)
Y[2] = 0.45 * np.cos(2*np.pi*base_frequency*16/4*x) + 0.21 * np.sin(2*np.pi*base_frequency*4/4*x)
Y[3] = 0.40 * np.cos(2*np.pi*base_frequency*18/4*x) + 0.21 * np.sin(2*np.pi*base_frequency*8/4*x)

Y[0] = norm_data(Y[0])
Y[1] = norm_data(Y[1])
Y[2] = norm_data(Y[2])
Y[3] = norm_data(Y[3])

#%%
plt.figure()
for i in range(num_features):
    plt.plot(x, Y[i])

plt.grid()
plt.show()

#%% verify the periodicity we expect
n_samples_per_periode = int(fs * 1/base_frequency)
for feature in range(num_features):
    a = Y[feature].reshape(n_samples_per_periode, -1)
    plt.plot(np.arange(n_samples_per_periode), a)
    plt.show()

#%%
min_sample = 0.25
max_sample = 0.50

for feature in range(num_features):
    ratio_of_not_sampling = np.random.random_sample() * (max_sample - min_sample) + min_sample

    idxs_to_nan = np.random.choice(Y[feature].shape[0],
                                   int(Y[feature].shape[0] * ratio_of_not_sampling),
                                   replace=False)

    Y[feature][idxs_to_nan] = np.nan

#%% check on "sampled" data
n_samples_per_periode = int(fs * 1/base_frequency)
for feature in range(num_features):
    a = Y[feature].reshape(n_samples_per_periode, -1)
    plt.plot(np.arange(n_samples_per_periode), a[:,0])
    plt.show()

#%%
import pandas as pd
df = pd.DataFrame(data=Y.T,    # values
                  index=x,    # 1st column as index
                  columns=np.arange(0,num_features))  # 1st row as the column names

sp_ = df.shape[0]//2
df_train_, df_test_ = df.iloc[:sp_], df.iloc[sp_:]

def store_data(df_, dataset_label='train'):
    for idx in range(num_periods//2):
        df_i = df_.iloc[idx*n_samples_per_periode:(idx+1)*n_samples_per_periode]
        df_i.index = df_i.index - df_i.index[0]
        df_i.to_csv(f'data_dir/BasicData/raw/sample_{dataset_label}_it_{idx}.csv')

store_data(df_train_, 'train')
store_data(df_test_, 'test')

#%%
