#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.ad_provider import ADProvider

provider = ADProvider(data_dir='data_dir', dataset='SWaT',
                      window_length=100, window_overlap=0.99,
                      n_samples=1000)

trn_ldr = provider.get_train_loader(
    batch_size=8,
    shuffle=True,
    collate_fn=None,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)
#%%

for _, batch in enumerate(trn_ldr):
    for idx in range(batch['inp_obs'].shape[0]):
        fig, axs = plt.subplots(nrows=26, ncols=2, figsize=(26, 12), sharex=True)
        axs = axs.flatten()
        for ax_idx in range(batch['inp_obs'].shape[2]):
            axs[ax_idx].plot(batch['inp_obs'][idx, :, ax_idx])
            axs[ax_idx].set_title(f'feature {ax_idx}')
        plt.show()
        plt.close('all')
    break

#%%

