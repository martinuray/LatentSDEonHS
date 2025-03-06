# Toy Experiment with Irregular Sine Data
import sys

from data.basicdata_provider import BasicDataProvider

sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from data.irregular_sine_provider import IrregularSineProvider
import matplotlib.pyplot as plt


#%%
dataset = "irregular_sine_interpolation"
experiment_id = 82171

#%%
dataset = "basic_data_interpolation_num-ft_1"
experiment_id = 34330   # mine

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
def plot_stat(logs: dict, stat:str, modes:list = ['trn']):
    fig, ax = plt.subplots(figsize=(8,3))
    for mode in modes:
        key = f"{mode}_{stat}"
        val = logs['all'][key]
        ax.plot(val, label = mode)
    ax.set_xlabel('training epochs')
    ax.set_ylabel(stat)
    ax.grid()
    return fig, ax


plot_stat(logs, 'loss')
plt.show()


# In[8]:
args = Namespace(**logs['args'])

if 'sine' in dataset:
    provider = IrregularSineProvider() #data_dir='data_dir')
elif 'basic' in dataset:
    provider = BasicDataProvider(data_dir='data_dir', num_features=1, sample_tp=1.)
else:
    print("Error. Something else.")

dl_trn = provider.get_train_loader(batch_size=1)
batch = next(iter(dl_trn))


# In[9]:
from core.models import ToyRecogNet, ToyReconNet, PathToGaussianDecoder, default_SOnPathDistributionEncoder

# TODO: where do the num_timepoints come from?
desired_t = torch.linspace(0, 1.0, provider.num_timepoints, device=args.device)

recog_net = ToyRecogNet(args.h_dim)
recon_net = ToyReconNet(args.z_dim)
qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
    )
pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=np.sqrt(0.05))

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
epoch = 4500
checkpoint = f"checkpoints/checkpoint_{experiment_id}_{epoch}.h5"
checkpoint = torch.load(checkpoint)
modules.load_state_dict(checkpoint['modules'])


#%%  generate reconstructions
dl = dl_trn
device = args.device
modules = modules.to(device)
for _, batch in enumerate(dl):
    parts = {key: val.to(device) for key, val in batch.items()}
    inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])
    h = modules["recog_net"](inp)
    qzx, pz = modules["qzx_net"](h, desired_t)
    zis = qzx.rsample((500,))
    pxz = modules["pxz_net"](zis)
    break


#%
plt.figure(figsize=(8,3))
for i in range(500):
    pxz_mean = pxz.mean[i].flatten().detach().cpu()
    plt.plot(torch.linspace(0,1,pxz_mean.shape[0]),
             pxz_mean,color='tab:blue', alpha=0.01, linewidth=3)
plt.plot(batch['evd_tps'].flatten(), batch['evd_obs'].flatten(),'+', color='tab:red')
#plt.ylim(-1.5,1.5);
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

endat = 101
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
