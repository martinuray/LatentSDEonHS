#%%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace

from basic_data_anomaly_detection import calculate_z_normalization_values
from data.ad_provider import ADProvider
from data.aero_provider import AeroDataProvider
from notebooks.utils.analyze import (get_modules, plot_stat,
                                     reconstruct_display_data,
                                     get_anomaly_performance)
from utils.anomaly_detection import calculate_anomaly_threshold, \
    get_number_over_threshold
from utils.scoring_functions import Evaluator

experiment_id_str = "AD_SWaT_250808-15:59:44"
checkpoint_epochs = [3, 90, 2100]

log_file = f"logs/{experiment_id_str}.json"
#% load logfile
with open(log_file,'r') as f:
    logs = json.load(f)

# print experiment settings
from pprint import pprint
pprint(logs['args'])

# print final values
from pprint import pprint
pprint(logs['final'])

# plot loss curves
#plot_stat(logs, stat='loss')
#plt.tight_layout()
#plt.show()

#% load args and data
args = Namespace(**logs['args'])
#args.device = 'cuda:0'

#%
#provider = BasicDataProvider(data_dir='data_dir', data_kind=data_kind, num_features=num_data_features, sample_tp=1.)
provider = ADProvider(data_dir='data_dir', dataset=args.dataset, window_length=args.data_window_length, window_overlap=args.data_window_overlap)
#provider = AeroDataProvider(data_dir="data_dir/aero")

dl_trn = provider.get_train_loader(batch_size=1, shuffle=False)
#dl_val = provider.get_val_loader(batch_size=1)
dl_test = provider.get_test_loader(batch_size=1, shuffle=False)
batch = next(iter(dl_trn))


#%% load models and checkpoint
desired_t = torch.linspace(0, 1.0, provider.num_timepoints, device=args.device).float()

#% load model
modules = [get_modules(args, provider, desired_t) for _ in checkpoint_epochs]

#%% load_model
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

for idx, cpe in enumerate(checkpoint_epochs):
    checkpoint = f"checkpoints/checkpoint_{experiment_id_str}_{cpe}.h5"
    checkpoint = torch.load(checkpoint)
    modules[idx].load_state_dict(checkpoint['modules'])

normalizing_stats = None
if args.normalize_score:
    normalizing_stats = calculate_z_normalization_values(args, dl_trn, modules, desired_t, args.device)


#%% reconstruct training data
from notebooks.utils.analyze import reconstruct_display_data

fig, axs = plt.subplots(nrows=26, ncols=len(checkpoint_epochs), figsize=(24, 36), sharex=True, sharey='row')
for idx, cpe in enumerate(checkpoint_epochs):
    reconstruct_display_data(dl_trn, args,
                             modules_=modules,
                             desired_t_=desired_t,
                             axs=axs,
                             checkpoint_epochs_=checkpoint_epochs,
                             normalizing_stats=normalizing_stats,
                             label=f"train",
                             disturb_value_offset=0.0,
                             dst='out/')
plt.show()

import sys; sys.exit(1)
#%% reconstruct test data
reconstruct_display_data(dl_test, args, modules, desired_t, normalizing_stats, label="test")#, dst='out/export/recon_test/')

#%% reconstruct test data
reconstruct_display_data(dl_test, args, modules, desired_t, normalizing_stats, label="test", dst='out/export/recon_test/')

#%% calculate anomaly threshold
#anomaly_threshold = calculate_anomaly_threshold(dl_trn, args, modules, desired_t)

#n_over_threshold = get_number_over_threshold(dl_trn, args, modules, desired_t, anomaly_threshold)
#n_over_threshold_threshold = n_over_threshold.mean() + 2 * n_over_threshold.std()
#print(n_over_threshold)

#%%  calculate anomaly metrics
import tqdm
from notebooks.utils.analyze import batch_get_log_prob
#n_over_threshold_threshold = n_over_threshold.mean() + 2 * n_over_threshold.std()

#def get_anomaly_performance_mine(dl_test, args, modules, desired_t):
print("Evaluation...")

true_label, pred = [], []

for _, batch_it in tqdm.tqdm(enumerate(dl_test)):
    lg_prb = batch_get_log_prob(batch_it, args, modules, desired_t)

    #pred_tgt = np.zeros(batch_it["aux_tgt"].squeeze().shape)
    # TODO: mean - first guess
    pred_tgt_score = lg_prb.squeeze().mean(axis=1)  # -> anomaly score
    #if pred_tgt.ndim == 1:
    #    #anomaly_selector = anomaly_selector.any(axis=1)
    #    anomaly_selector = (anomaly_selector * 1).sum(axis=1) >  n_over_threshold_threshold

    #pred_tgt_score[pred_tgt_score > clip_value] = clip_value
    #pred_tgt[anomaly_selector] = 1
    pred_tgt_score = pred_tgt_score.detach().cpu()
    true_tgt = batch_it["aux_tgt"].flatten().detach().cpu()

    true_label.append(true_tgt)
    pred.append(pred_tgt_score)

all_predicted_np = np.concatenate(pred)
all_true_labels_np = np.concatenate(true_label)

all_predicted_torch = torch.cat(pred, dtype=torch.float64)
all_true_labels_torch = torch.cat(true_label)

#%%
from utils.scoring_functions import Evaluator
eval_np = Evaluator()

from utils.scoring_functions_torch import Evaluator as TorchEvaluator
eval_torch = TorchEvaluator()


def calculate_f1s_np(scores, true, evaluator=None, clip_value_=100):
    # TODO: to speed shit up
    scores = np.round(scores, 2)
    scores[scores > clip_value_] = clip_value_

    f1 = evaluator.best_f1_score(true, scores)
    f1_ts = evaluator.best_ts_f1_score(true, scores)

    print(f"F1:{f1['f1']:.4f}" )
    print(f"F1_ts:{f1_ts['f1']:.4f}" )

    return f1['f1'], f1_ts['f1']


def calculate_f1s_torch(scores, true, evaluator=None, clip_value_=100):
    # TODO: to speed shit up
    scores = torch.round(scores, decimals=2)
    scores[scores > clip_value_] = clip_value_

    f1 = evaluator.best_f1_score(true, scores)
    f1_ts = evaluator.best_ts_f1_score(true, scores)

    #print(f"F1:{f1['f1']:.4f}" )
    #print(f"F1_ts:{f1_ts['f1']:.4f}" )

    return f1[0], f1_ts[0]


#%%
print("Mean")
calculate_f1s_np(all_predicted_np, all_true_labels_np, evaluator=eval_np)
#%%
calculate_f1s_torch(all_predicted_torch, all_true_labels_torch, evaluator=eval_torch)

#%%
all_predicted_median = np.median(all_predicted_np, axis=1)
print("Median")
calculate_f1s(all_predicted_median, all_true_labels_np)

print("Std")
#all_predicted_std = np.std(all_predicted, axis=1)
#calculate_f1s(all_predicted_std, all_true_labels)

#%
print("Var")
all_predicted_std = np.var(all_predicted_np, axis=1)
f1, f1t = calculate_f1s(all_predicted_std, all_true_labels_np)
print(f1, f1t)

#% irrelevant, same as mean, just not normed
print("Sum")
all_predicted_sum = np.sum(all_predicted_np, axis=1)
calculate_f1s(all_predicted_sum, all_true_labels_np, clip_value_=50 * 31)

#%
print("Max")
all_predicted_max = np.max(all_predicted_np, axis=1)
calculate_f1s(all_predicted_max, all_true_labels_np)

print("75% Q")
all_predicted_75q = np.quantile(all_predicted_np, 0.75, axis=1)
calculate_f1s(all_predicted_75q, all_true_labels_np)

print("90% Q")
all_predicted_90q = np.quantile(all_predicted_np, 0.90, axis=1)
calculate_f1s(all_predicted_90q, all_true_labels_np)

#%
print("95% Q")
all_predicted_95q = np.quantile(all_predicted_np, 0.95, axis=1)
calculate_f1s(all_predicted_95q, all_true_labels_np)

#%% search quantile

def calc_scores_quantiles(pred, true, q):
    pred_q = np.quantile(pred, q, axis=1)
    f1, f1ts = calculate_f1s(pred_q, true)
    return f1, f1ts


quantiles_to_search = np.arange(0.25, 0.99, 0.025)
results_exh_quant_search = [
    calc_scores_quantiles(all_predicted_np, all_true_labels_np, q) for q in tqdm.tqdm(quantiles_to_search)
]

#%%
np_results_exh_quant_search = np.array(results_exh_quant_search)
plt.figure()
plt.plot(quantiles_to_search, np_results_exh_quant_search[0, :], label='$f1$')
plt.plot(quantiles_to_search, np_results_exh_quant_search[1, :], label='$f1_{ts}$')
plt.show()

#%% check according to true label
max_ = 10000000

all_predicted_benign = all_predicted_np[:max_][all_true_labels_np[:max_] == 0, :]
all_predicted_anomaly = all_predicted_np[:max_][all_true_labels_np[:max_] == 1, :]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(all_predicted_anomaly.flatten(), bins=100, color='r', alpha=0.5, label='anomaly')
ax.hist(all_predicted_benign.flatten(), bins=100, color='b', alpha=0.25, label='benign')
ax.set_yscale('log')
plt.legend()
plt.savefig('out.png')
