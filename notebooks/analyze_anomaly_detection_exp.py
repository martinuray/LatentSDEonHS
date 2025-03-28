#%%
# Toy Experiment with Irregular Sine Data
import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from sklearn.metrics import f1_score, accuracy_score

from data.ad_provider import ADProvider
from data.basicdata_provider import BasicDataProvider

sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import os
from argparse import Namespace
from data.irregular_sine_provider import IrregularSineProvider
import matplotlib.pyplot as plt

data_kind, num_data_features = None, 4
dataset = "anomaly_detection"
experiment_id = 60625   # mine
checkpoint_epoch = 4500
# 108826 bis 2190
experiment_id = 82435   # mine
checkpoint_epoch = 2190

#%%
log_file = f"logs/{dataset}_{experiment_id}.json"

# load logfile
import json
with open(log_file,'r') as f:
    logs = json.load(f)


#% print experiment settings
from pprint import pprint
pprint(logs['args'])


#% print final values
from pprint import pprint
pprint(logs['final'])

#% plot loss curves
import matplotlib.pyplot as plt
def plot_stat(logs: dict, stat:str, modes:list = ['trn', 'tst']):
    fig, ax = plt.subplots(figsize=(8,3))
    for mode in modes:
        key = f"{mode}_{stat}"
        val = logs['all'][key]
        ax.plot(val, label = mode)
    ax.set_xlabel('training epochs')
    ax.set_ylabel(stat)
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    return fig, ax


plot_stat(logs, 'loss')
plt.tight_layout()
plt.show()


# In[8]:
args = Namespace(**logs['args'])
args.device = 'cuda:1'

#provider = BasicDataProvider(data_dir='data_dir', data_kind=data_kind, num_features=num_data_features, sample_tp=1.)
provider = ADProvider(data_dir='data_dir', data_kind="SWaT")

dl_trn = provider.get_train_loader(batch_size=1)
#dl_val = provider.get_val_loader(batch_size=1)
dl_test = provider.get_test_loader(batch_size=1)
batch = next(iter(dl_trn))


# In[9]:
from core.models import ToyRecogNet, ToyReconNet, PathToGaussianDecoder, \
    default_SOnPathDistributionEncoder, PhysioNetRecogNetwork, GenericMLP

# TODO: where do the num_timepoints come from?
desired_t = torch.linspace(0, 1.0, provider.num_timepoints, device=args.device).float()

recog_net = PhysioNetRecogNetwork(
    mtan_input_dim=provider.input_dim,
    mtan_hidden_dim=args.h_dim,
    use_atanh=args.use_atanh
)
recon_net = GenericMLP(
    inp_dim=args.z_dim,
    out_dim=provider.input_dim,
    n_hidden=args.dec_hidden_dim,
    n_layers=args.n_dec_layers,
)
qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
            z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
    )
pxz_net = PathToGaussianDecoder(
    mu_map=recon_net,
    sigma_map=None,
    initial_sigma=0.1) # TODO: is this initial sigma ok so?
    #initial_sigma=np.sqrt(0.05))

modules = nn.ModuleDict(
    {
        "recog_net": recog_net,
        "recon_net": recon_net,
        "pxz_net": pxz_net,
        "qzx_net": qzx_net,
    }
)
modules = modules.to(args.device)

#%% load_model
checkpoint = f"checkpoints/checkpoint_{experiment_id}_{checkpoint_epoch}.h5"
checkpoint = torch.load(checkpoint)
modules.load_state_dict(checkpoint['modules'])


def batch_get_log_prob(batch_):
    parts = batch_to_device(batch_, args.device)
    inp = (parts["evd_obs"], parts["evd_msk"], parts["evd_tps"])
    h = modules["recog_net"](inp)
    for a in inp:
        del a
    qzx, _ = modules["qzx_net"](h, desired_t)
    del h
    zis = qzx.rsample((500,))
    del qzx
    pxz = modules["pxz_net"](zis)
    del zis
    lg_prb = -pxz.log_prob(parts["evd_obs"]).mean(dim=0)
    return lg_prb


def batch_to_device(batch_, device_):
    return {key: val.to(device_) for key, val in batch_.items()}


#%% identify anomaly threshold
log_probs_test = []
for idx, batch_ in enumerate(dl_trn):
    log_probs_test.append(batch_get_log_prob(batch_).detach().cpu().squeeze().numpy())

log_probs_test = np.vstack(log_probs_test)
test_mean = log_probs_test.mean(axis=0)
test_std = log_probs_test.std(axis=0)

anomaly_threshold = test_mean + 1.5 * test_std

#%%  generate reconstructions

accs, f1s = [], []
for idx, batch in enumerate(dl_test):
    parts = batch_to_device(batch, args.device)
    inp = (parts["evd_obs"], parts["evd_msk"], parts["evd_tps"])
    for a in inp:
        del a
    h = modules["recog_net"](inp)
    qzx, _ = modules["qzx_net"](h, desired_t)
    del h
    zis = qzx.rsample((250,))
    del qzx
    pxz = modules["pxz_net"](zis)
    del zis
    lg_prb = -pxz.log_prob(parts["evd_obs"]).mean(dim=0)
    lg_prb = lg_prb.detach().cpu().squeeze().numpy()
    anomaly_selector = lg_prb.squeeze() > anomaly_threshold
    pred_tgt = np.zeros(batch["aux_tgt"].squeeze().shape)
    if pred_tgt.ndim == 1:
        pass
        #anomaly_selector = anomaly_selector.any(axis=1)
    pred_tgt[anomaly_selector] = 1

    acc = accuracy_score(pred_tgt.flatten(), batch["aux_tgt"].flatten().detach().cpu())
    f1 = f1_score(pred_tgt.flatten(), batch["aux_tgt"].flatten().detach().cpu())
    f1s.append(f1)
    accs.append(acc)

print("F1:", np.mean(f1s))
print("Acc:", np.mean(accs))

#%%
fig, axs = plt.subplots(nrows=num_data_features, figsize=(16,9))
for ft in range(num_data_features):
    for i in range(500):
        pxz_mean = pxz.mean[i, :, :, ft].flatten().detach().cpu()
        axs[ft].plot(torch.linspace(0,1,pxz_mean.shape[0]),
                     pxz_mean,color='tab:blue', alpha=0.01, linewidth=3)
        evd = batch['evd_tps'].flatten()
    axs[ft].set_ylabel('Signal Amplitude')
    axs[ft].plot(evd/evd.max(), batch['evd_obs'][:,:,ft].flatten(),
             '+', color='black')
    ac = axs[ft].twinx()
    ac.plot(torch.linspace(0, 1, pxz_mean.shape[0]), lg_prb[:,:, ft].flatten().detach().cpu(), color='tab:red')
    ac.axhline(y=anomaly_threshold[ft].item())
    ac.set_ylabel("Log Prob")
    axs[ft].set_xlabel('Time t / s')
#plt.ylim(-1.5,1.5);
plt.suptitle(f'F1-Score: {f1}')
plt.show()

#%%
t = qzx.t.cpu().detach()
latents = qzx.sample((500,)).cpu().detach()

fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].plot(t, latents[:,0,:,i].permute(1,0), color='tab:blue', alpha=0.1)
plt.show()

#%%
import imageio
import os
sphere_dir = './out/sphere/'
os.makedirs(sphere_dir, exist_ok=True)

images = []

endat = 0 #101
for current_at in range(endat):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(current_at)
    for idx in range(len(latents)):
        ax.plot(latents[idx,0,:current_at,0],latents[idx,0,:current_at,1],latents[idx,0,:current_at,2], c='tab:blue',alpha=.1)
        ax.scatter(latents[idx,0,0,0],latents[idx,0,0,1],latents[idx,0,0,2], c='tab:red', alpha=.1)
        ax.scatter(latents[idx,0,current_at,0],latents[idx,0,current_at,1],latents[idx,0,current_at,2], c='tab:red', alpha=.1)

    # sphere = np.random.randn(50000,3)
    # sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    # ax.scatter(sphere[:,0], sphere[:,1],sphere[:,2], s=.01)

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", alpha=.2)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_aspect("equal")
    ax.axis('off')
    file_name = os.path.join(sphere_dir, f'sphere_{current_at:04d}.png')
    images.append(file_name)

    fig.savefig(file_name, dpi=150)
    plt.close()

with imageio.get_writer(os.path.join(sphere_dir, f'sphere.gif'), mode='I') as writer:
    for fn in images:
        image = imageio.imread(fn)
        writer.append_data(image)
