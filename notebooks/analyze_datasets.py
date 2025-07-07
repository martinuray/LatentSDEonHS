import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from basic_data_anomaly_detection import logprob2f1s
from scipy.stats import iqr

from utils.scoring_functions import Evaluator


def normalise_scores(test_delta, norm="median-iqr", smooth=True,
                     smooth_window=5):
    """
    Args:
        norm: None, "mean-std" or "median-iqr"
    """
    if norm == "mean-std":
        err_scores = StandardScaler().fit_transform(test_delta)
    elif norm == "median-iqr":
        n_err_mid = np.median(test_delta, axis=0)
        n_err_iqr = iqr(test_delta, axis=0)
        epsilon = 1e-2

        err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    elif norm is None:
        err_scores = test_delta
    else:
        raise ValueError(
            'specified normalisation not implemented, please use one of {None, "mean-std", "median-iqr"}')

    if smooth:
        smoothed_err_scores = np.zeros(err_scores.shape)

        for i in range(smooth_window, len(err_scores)):
            smoothed_err_scores[i] = np.mean(
                err_scores[i - smooth_window: i + smooth_window - 1], axis=0)
        return smoothed_err_scores
    else:
        return err_scores

def get_results_for_best_score_normalizations(
        scores,
        test_labels,
):
    # get score under all three normalizations
    df_list = []
    f1_scores = []
    normalisations = ["median-iqr", "mean-std", None]
    for n in normalisations:
        r = logprob2f1(
            normalise_scores(scores, norm=n).max(1),
            test_labels
        )
        f1_scores.append(r['f1'])
        df_list.append(r)

    best_score_idx = np.array(f1_scores).argmax()
    print(df_list[best_score_idx]['f1'], df_list)
    return df_list[best_score_idx]

def pca_recon_error(np_train, np_test, n_components=10):
    p = PCA(n_components=n_components, svd_solver='full')
    p.fit(np_train)
    l = p.transform(np_test)
    recon = p.inverse_transform(l)
    return np.abs(np_test - recon)


def viz_pca_testset(test_pca, labels, text_label):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test_pca[:, 0], test_pca[:, 1], test_pca[:, 2],
               c=labels, marker='o')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(text_label)
    plt.show()
    plt.close('all')


def data_transform_pca(df_train, df_test):
    p = PCA(n_components=3, svd_solver='full')
    p.fit(df_train)
    test_pca = p.transform(df_test)
    return test_pca


def get_all_smd(mode='train'):
    all_files = glob.glob(os.path.join('data_dir/SMD/raw', mode, '*.txt'))

    data = {}

    for file_ in all_files:
        machine = file_.split('/')[-1].replace('.txt', '')
        data[machine] = {}
        data[machine]['train'] = None
        data[machine]['test'] = None
        data[machine]['labels'] = None
        if mode == 'train':
            data[machine]['train'] = pd.read_csv(file_, delimiter=',',
                                                 header=None)
        elif mode == 'test':
            data[machine]['test'] = pd.read_csv(file_, delimiter=',',
                                                header=None)
            label_file = file_.replace('test', 'test_label')
            data[machine]['labels'] = pd.read_csv(label_file, delimiter=',',
                                                  header=None)

    assert all(
        df[mode].shape[1] == next(iter(data.values()))[mode].shape[1]
        for df in
        data.values()), "Not all DataFrames have the same number of columns"

    d = [data[key][mode] for key, _ in data.items()]
    #d = pd.concat(d)

    l = None
    if mode != 'train':
        l = [data[key]['labels'].to_numpy().squeeze() for key, _ in data.items()]
        #l = pd.concat(l).to_numpy().squeeze()

    return data


def logprob2f1(scores, true_labels):
    eval = Evaluator()

    if type(scores) is list:
        scores = np.cat(scores, dim=0)

    if type(true_labels) is list:
        true_labels = np.cat(true_labels, dim=0)

    f1 = eval.best_f1_score(true_labels, scores)

    return f1


def minmax_scale_traces(smd_train, smd_test):
    for machine_key, _ in smd_train.items():
        scaler = MinMaxScaler()
        smd_train[machine_key]['train'] = scaler.fit_transform(smd_train[machine_key]['train'])
        smd_test[machine_key]['test'] = scaler.transform(smd_test[machine_key]['test'])


#%% load Swat data
swat_train = pd.read_csv('data_dir/SWaT/raw/train.csv')
swat_test = pd.read_csv('data_dir/SWaT/raw/test.csv')
swat_labels = pd.read_csv('data_dir/SWaT/raw/labels.csv').to_numpy().squeeze()[:,1]

mm_scaler = MinMaxScaler(feature_range=(0,1))
swat_train = mm_scaler.fit_transform(swat_train)
swat_test = mm_scaler.transform(swat_test)

# %% load Wadi Data
wadi_train = pd.read_csv('data_dir/WaDi/raw/train.csv', index_col=0)
wadi_test = pd.read_csv('data_dir/WaDi/raw/test.csv', index_col=0)
wadi_labels = pd.read_csv('data_dir/WaDi/raw/labels.csv').to_numpy().squeeze()
wadi_dim = wadi_train.shape[1]

mm_scaler = MinMaxScaler(feature_range=(0,1))
wadi_train = mm_scaler.fit_transform(wadi_train)
wadi_test = mm_scaler.transform(wadi_test)

#%%
smd_train = get_all_smd('train')
smd_test = get_all_smd('test')
minmax_scale_traces(smd_train, smd_test)



# %% PCA transform
#swat_test_pca = data_transform_pca(swat_train, swat_test)
#wadi_test_pca = data_transform_pca(wadi_train, wadi_test)
#smd_test_pca = data_transform_pca(smd_train, smd_test)

# %% visualizes pca data in a 3d plot, colored by labels
#viz_pca_testset(swat_test_pca, swat_labels, "SWaT")
#viz_pca_testset(wadi_test_pca, wadi_labels, "WaDi")
#viz_pca_testset(smd_test_pca, smd_labels, "SMD")

#%% get recon error and try out
swat_score = pca_recon_error(swat_train, swat_test)
f1_swat = get_results_for_best_score_normalizations(swat_score, swat_labels)

#%%
wadi_score = pca_recon_error(wadi_train, wadi_test, n_components=50)
f1_wadi = get_results_for_best_score_normalizations(wadi_score, wadi_labels, )
#swat_score = swat_score.max(1)
#wadi_score = wadi_score.max(1)

#f1_swat, _ = logprob2f1s(torch.from_numpy(swat_score.squeeze()), torch.from_numpy(swat_labels), including_ts=False)
#f1_wadi, _ = logprob2f1s(torch.from_numpy(wadi_score.squeeze()), torch.from_numpy(wadi_labels), including_ts=False)

#smd_scores = {key: normalise_scores_trace_all(smd_score) for key, smd_score in
#              smd_scores.items()}

# %%
smd_scores_pca = {key: pca_recon_error(smd_train[key]['train'], smd_test[key]['test'])
                  for key, _ in smd_train.items()}

f1_smd_t = [get_results_for_best_score_normalizations(smd_score, smd_test[key]['labels'].values) for key, smd_score in smd_scores_pca.items()]
f1_smd = pd.DataFrame(f1_smd_t).mean()


# %%
console = Console()
table_combined = Table(title="[bold magenta]Dataset PCA Stats[/bold magenta]")
table_combined.add_column("Dataset", style="cyan", justify="right")
table_combined.add_column("tgt-F1", style="green", justify="right")
table_combined.add_column("F1", style="green", justify="right")
table_combined.add_column("Prec.", style="yellow", justify="right")
table_combined.add_column("Rec.", style="blue", justify="right")
table_combined.add_column("AuPrc", style="blue", justify="right")
table_combined.add_column("AuRoc", style="blue", justify="right")
table_combined.add_column("Threshold", style="blue", justify="right")

# Populate the table with shared features
table_combined.add_row("SMD",
                       f'57.2',
                       f'{f1_smd["f1"]*100:.2f}',
                       f'{f1_smd["precision"]*100:.2f}',
                       f'{f1_smd["recall"]*100:.2f}',
                       f'{f1_smd["auprc"]*100:.2f}',
                       f'{f1_smd["auroc"]*100:.2f}',
                       f'{f1_smd["threshold"]}')
table_combined.add_row("SWAT",
                       f'83.3',
                       f'{f1_swat["f1"]*100:.2f}',
                       f'{f1_swat["precision"]*100:.2f}',
                       f'{f1_swat["recall"]*100:.2f}',
                       f'{f1_swat["auprc"]*100:.2f}',
                       f'{f1_swat["auroc"]*100:.2f}',
                       f'{f1_swat["threshold"]}')
table_combined.add_row(f"WADI-{wadi_dim}",
                       f'50.1',
                       f'{f1_wadi["f1"]*100:.2f}',
                       f'{f1_wadi["precision"]*100:.2f}',
                       f'{f1_wadi["recall"]*100:.2f}',
                       f'{f1_wadi["auprc"]*100:.2f}',
                       f'{f1_wadi["auroc"]*100:.2f}',
                       f'{f1_wadi["threshold"]}')
table_combined.add_row("WADI-112", f'65.5', f'', f'', f'', f'', f'', f'')

# Print the unified table
console.print(table_combined)
