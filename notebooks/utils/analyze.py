import numpy as np
import os
import zipfile
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score
from torch import nn as nn

from core.models import PhysioNetRecogNetwork, GenericMLP, \
    default_SOnPathDistributionEncoder, PathToGaussianDecoder


def get_modules(args_, provider_, desired_t_):
    recog_net_ = PhysioNetRecogNetwork(
        mtan_input_dim=provider_.input_dim,
        mtan_hidden_dim=args_.h_dim,
        use_atanh=args_.use_atanh
    )

    recon_net_ = GenericMLP(
        inp_dim=args_.z_dim,
        out_dim=provider_.input_dim,
        n_hidden=args_.dec_hidden_dim,
        n_layers=args_.n_dec_layers,
    )

    qzx_net_ = default_SOnPathDistributionEncoder(
        h_dim=args_.h_dim,
        z_dim=args_.z_dim,
        n_deg=args_.n_deg,
        learnable_prior=args_.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t_[-1].item(),
    )

    pxz_net_ = PathToGaussianDecoder(
        mu_map=recon_net_,
        sigma_map=None,
        initial_sigma=0.1)  # TODO: is this initial sigma ok so?
    # initial_sigma=np.sqrt(0.05))

    modules_ = nn.ModuleDict(
        {
            "recog_net": recog_net_,
            "recon_net": recon_net_,
            "pxz_net": pxz_net_,
            "qzx_net": qzx_net_,
        }
    )

    modules_ = modules_.to(args_.device)
    return modules_


def reconstruct_batch(batch_, args_, modules_, desired_t_, n_samples_=500):
    parts_ = batch_to_device(batch_, args_.device)
    inp_ = (parts_["evd_obs"], parts_["evd_msk"], parts_["evd_tps"])
    h_ = modules_["recog_net"](inp_)
    qzx_, _ = modules_["qzx_net"](h_, desired_t_)
    zis_ = qzx_.rsample((n_samples_,))
    pxz_ = modules_["pxz_net"](zis_)
    return pxz_


def batch_get_log_prob(batch_, args_, modules_, desired_t_):
    pxz_ = reconstruct_batch(batch_, args_, modules_, desired_t_)
    lg_prb_ = -pxz_.log_prob(batch_["evd_obs"].to(args_.device)).mean(dim=0)
    lg_prb_ = lg_prb_.to(torch.float64)
    return pxz_,  lg_prb_.squeeze()


def batch_to_device(batch_, device_):
    return {key: val.to(device_) for key, val in batch_.items() if 'evd' in key}


def plot_stat(logs_: dict, stat:str, modes:list = ['trn', 'tst']):
    fig, ax = plt.subplots(figsize=(8,3))
    for mode in modes:
        key = f"{mode}_{stat}"
        val = logs_['all'][key]
        ax.plot(val, label = mode)
    ax.set_xlabel('training epochs')
    ax.set_ylabel(stat)
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    return fig, ax



def zip_directory(directory, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                # archive name keeps the directory structure relative to the zipped folder
                arcname = os.path.relpath(file_path, start=directory)
                zipf.write(file_path, arcname)


def get_start_end_anomal_section(y_, last_index):
    to_anomal = np.where((y_[:-1] == 0) & (y_[1:] != 0))[0]
    from_anomal = np.where((y_[:-1] != 0) & (y_[1:] == 0))[0]

    if y_[0] != 0:
        to_anomal = np.insert(to_anomal, 0, 0)

    if from_anomal.shape[0] >  to_anomal.shape[0]:
        to_anomal = np.insert(to_anomal, 0, 0)
    elif from_anomal.shape[0] < to_anomal.shape[0]:
        from_anomal = np.append(from_anomal, last_index - 1)

    return to_anomal, from_anomal


def reconstruct_display_data(dl_, args_, modules_, desired_t_, normalizing_stats=None, label=None,
                             dst=None, disturb_value_offset=0.0):

    if dst is not None:
        os.makedirs(dst, exist_ok=True)

    for idx, batch_i in enumerate(dl_):
        if label == 'test' and idx < 15 and dst is None:
            continue

        # to put some artificial noise on to the data
        evd_obs = batch_i["evd_obs"]
        batch_i['evd_obs'] += torch.ones_like(evd_obs).to(evd_obs.device) * disturb_value_offset

        my_pxz, lg_prb = batch_get_log_prob(batch_i, args_, modules_, desired_t_)

        evd = batch_i['evd_tps'][0, :]
        evd = evd / evd.max()
        anomaly_indicator = batch_i['aux_tgt'][0, :]
        obs = batch_i['evd_obs'][0, :]

        to_anom, from_anom = get_start_end_anomal_section(anomaly_indicator, obs.shape[0])

        num_subplots = min(lg_prb.shape[1]+1, 11)

        fig, axs = plt.subplots(nrows=num_subplots, figsize=(16, 9), sharex=True)
        for ft, ax in enumerate(axs[:-1]):
            for i in range(500):
                pxz_mean = my_pxz.mean[i, :, :, ft].flatten().detach().cpu()
                ax.plot(torch.linspace(0, 1, pxz_mean.shape[0]),
                        pxz_mean, color='tab:blue', alpha=0.01,
                        linewidth=3)

            ax.set_title(f'{lg_prb[:, ft].mean():.2f}')
            ax.set_ylabel('Signal Amplitude')
            ax.plot(evd[anomaly_indicator==0], obs[anomaly_indicator==0, ft], '+', color='black')
            ax.plot(evd[anomaly_indicator==1], obs[anomaly_indicator==1, ft], '+', color='orange')

            for f, t in zip(to_anom, from_anom):
                ax.axvspan(evd[f], evd[t], color='red', alpha=0.15)

            ac = ax.twinx()
            ac.plot(torch.linspace(0, 1, pxz_mean.shape[0]), lg_prb[:, ft].flatten().detach().cpu(), color='tab:orange')
            ac.set_ylabel("Log Prob")
            ac.set_ylim(0 , 150)
            axs[ft].set_xlabel('Time t / s')

        lg_score = lg_prb[:, :].detach().cpu().mean(axis=1)
        if normalizing_stats is not None:
            lg_score = (lg_score - normalizing_stats['mu'].to('cpu')) / normalizing_stats['sigma'].to('cpu')

        axs[-1].plot(torch.linspace(0, 1, pxz_mean.shape[0]), lg_score, color='tab:blue', label='all features')
        #axs[-1].plot(torch.linspace(0, 1, pxz_mean.shape[0]), lg_prb[:, :-1].detach().cpu().mean(axis=1), color='tab:green', label='continous features only')
        #axs[-1].set_ylim(0, 150)
        axs[-1].legend(loc='upper left')
        axs[-1].set_yscale('log')

        for f, t in zip(to_anom, from_anom):
            axs[-1].axvspan(evd[f], evd[t], color='red', alpha=0.15)

        title_str = f'{label}'
        if disturb_value_offset > 0.0:
            title_str += f' (disturbed {disturb_value_offset})'

        plt.suptitle(title_str)

        if dst is None:
            plt.show()
            if idx > 45:
                break
        else:
            plt.savefig(os.path.join(dst, f'{label}_recondstruction{idx}.png'))

        plt.close()
        if label == 'train':
            break

    if dst is not None:
        dst_zip = f'out/reconstructed_{label}.zip'
        if os.path.isfile(dst_zip):
            os.remove(dst_zip)
        zip_directory(dst, dst_zip)


def get_anomaly_performance(dl_test, args, modules, desired_t, anomaly_threshold):  #, n_over_threshold_threshold):
    print("Evaluation...")
    true_label, pred = [], []
    for _, batch_it in tqdm.tqdm(enumerate(dl_test)):
        _, lg_prb = batch_get_log_prob(batch_it, args, modules, desired_t)

        pred_tgt = np.zeros(batch_it["aux_tgt"].squeeze().shape)
        anomaly_selector = lg_prb.squeeze() > anomaly_threshold
        if pred_tgt.ndim == 1:
            #anomaly_selector = anomaly_selector.any(axis=1)
            anomaly_selector = (anomaly_selector * 1).any() #.sum(axis=1) >  n_over_threshold_threshold

        pred_tgt[anomaly_selector] = 1

        true_tgt = batch_it["aux_tgt"].flatten().detach().cpu()
        true_label.append(true_tgt)
        pred.append(pred_tgt)

    predicted = np.concatenate(pred)
    ture_values = np.concatenate(true_label)
    prec = precision_score(ture_values, predicted)
    rec = recall_score(ture_values, predicted)
    f1 = f1_score(ture_values, predicted)

    print(f"F1:{f1:.4f}" )
    print(f"Prec: {prec:.4f}%")
    print(f"Rec: {rec:.4f}%")
