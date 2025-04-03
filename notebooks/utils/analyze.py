import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
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
    return lg_prb_


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


def reconstruct_display_data(dl_, args_, modules_, desired_t_, num_data_features_=10, label=None):
    for idx, batch_i in enumerate(dl_):

        my_pxz = reconstruct_batch(batch_i, args_, modules_, desired_t_)

        fig, axs = plt.subplots(nrows=num_data_features_, figsize=(16, 9))
        for ft in range(num_data_features_):
            for i in range(500):
                pxz_mean = my_pxz.mean[i, :, :, ft].flatten().detach().cpu()
                axs[ft].plot(torch.linspace(0, 1, pxz_mean.shape[0]),
                             pxz_mean, color='tab:blue', alpha=0.01,
                             linewidth=3)
            evd = batch_i['evd_tps'][0, :]
            axs[ft].set_ylabel('Signal Amplitude')
            axs[ft].plot(evd / evd.max(), batch_i['evd_obs'][0, :, ft], '+',
                         color='black')
            #ac = axs[ft].twinx()
            # ac.plot(torch.linspace(0, 1, pxz_mean.shape[0]), lg_prb[:,:, ft].flatten().detach().cpu(), color='tab:red')
            # ac.axhline(y=anomaly_threshold[ft].item())
            # ac.set_ylabel("Log Prob")
            axs[ft].set_xlabel('Time t / s')
        # plt.ylim(-1.5,1.5);
        plt.suptitle(f'{label}')
        plt.show()
        ttext = input("Press Enter to continue, 'q' to exit...")
        if ttext == 'q':
            break


def get_anomaly_performance(dl_test, args, modules, desired_t, anomaly_threshold):
    accs, f1s = [], []
    for _, batch_it in tqdm.tqdm(enumerate(dl_test)):
        parts = batch_to_device(batch_it, args.device)
        pxz = reconstruct_batch(batch_it, args, modules, desired_t)

        lg_prb = -pxz.log_prob(parts["evd_obs"]).mean(dim=0)
        lg_prb = lg_prb.detach().cpu().squeeze().numpy()

        pred_tgt = np.zeros(batch_it["aux_tgt"].squeeze().shape)
        anomaly_selector = lg_prb.squeeze() > anomaly_threshold
        if pred_tgt.ndim == 1:
            anomaly_selector = anomaly_selector.any(axis=1)

        pred_tgt[anomaly_selector] = 1

        true_tgt = batch_it["aux_tgt"].flatten().detach().cpu()

        acc = accuracy_score(pred_tgt.flatten(), true_tgt)
        f1 = f1_score(pred_tgt.flatten(), true_tgt)
        f1s.append(f1)
        accs.append(acc)
    print("F1:", np.mean(f1s))
    print("Acc:", np.mean(accs))
