#%%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace

from data.ad_provider import ADProvider
from notebooks.utils.analyze import (get_modules, plot_stat,
                                     reconstruct_display_data,
                                     get_anomaly_performance)
from utils.anomaly_detection import calculate_anomaly_threshold

# %% some old stats
data_kind, num_data_features = None, 4
dataset = "anomaly_detection"
experiment_id = 60625   # mine
checkpoint_epoch = 4500
# 108826 bis 2190
experiment_id = 82435   # mine
checkpoint_epoch = 2190

log_file = f"logs/{dataset}_{experiment_id}.json"

#%% SWAT
num_data_features = 10
experiment_id_str = "anomaly_detection_250401-15:38:22"
checkpoint_epoch = 990
log_file = f"logs/{experiment_id_str}.json"


#%% load logfile
with open(log_file,'r') as f:
    logs = json.load(f)

# print experiment settings
from pprint import pprint
pprint(logs['args'])

# print final values
from pprint import pprint
pprint(logs['final'])

# plot loss curves
plot_stat(logs, 'loss')
plt.tight_layout()
plt.show()

#%% load args and data
args = Namespace(**logs['args'])
#args.device = 'cuda:0'

#provider = BasicDataProvider(data_dir='data_dir', data_kind=data_kind, num_features=num_data_features, sample_tp=1.)
provider = ADProvider(data_dir='data_dir', data_kind="SWaT")

dl_trn = provider.get_train_loader(batch_size=1)
#dl_val = provider.get_val_loader(batch_size=1)
dl_test = provider.get_test_loader(batch_size=1)
batch = next(iter(dl_trn))


#%% load models and checkpoint
desired_t = torch.linspace(0, 1.0, provider.num_timepoints, device=args.device).float()

# load model
modules = get_modules(args, provider, desired_t)

# load_model
checkpoint = f"checkpoints/checkpoint_{experiment_id_str}_{checkpoint_epoch}.h5"
checkpoint = torch.load(checkpoint)
modules.load_state_dict(checkpoint['modules'])

#%% reconstruct training data
reconstruct_display_data(dl_trn, args, modules, desired_t, label="train")

#%% reconstruct test data
reconstruct_display_data(dl_test, args, modules, desired_t, label="test")

#%% calculate anomaly threshold
anomaly_threshold = calculate_anomaly_threshold(dl_trn, args, modules, desired_t)

#%%  calculate anomaly metrics
get_anomaly_performance(dl_test, args, modules, desired_t, anomaly_threshold)

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
