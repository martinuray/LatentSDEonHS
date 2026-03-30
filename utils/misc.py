"""Miscellaneous helper routines.

Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""

import os
import random
import numpy as np
import json
import csv

import torch
from torch import Tensor
from torch.nn import ModuleDict
from typing import Union

def check_exists(dir: str):
    if not os.path.exists(dir):
        return False
    return True


def set_seed(seed: int = 1234, full_reproducibility: bool = False):
    assert seed > 0, "Seed<=0"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if full_reproducibility:
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)


def count_parameters(model: torch.nn.ModuleList):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vec_to_matrix(vec: Tensor, basis: Tensor) -> Tensor:
    return torch.einsum("...d, dij -> ...ij", vec, basis)


def save_stats(args, stats: dict, file: str) -> None:
    base_keys = {"trn", "tst", "val", "oth"}

    def _is_allowed_key(key: str) -> bool:
        return key in base_keys or key == "per_dataset" or key.startswith("macro_")

    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    done = all(_is_allowed_key(key) for key in stats.keys())
    assert done

    all_stats = dict()
    epochs = 0
    for mode in ["trn", "tst", "val", "oth"]:
        if mode not in stats:
            continue

        epoch_stats = stats[mode]
        if not isinstance(epoch_stats, list) or len(epoch_stats) == 0:
            continue

        epochs = max(epochs, len(epoch_stats))
        for key in epoch_stats[0].keys():
            if mode != "oth":
                all_stats[f"{mode}_{key}"] = [
                    str(epoch_stats[epoch].get(key, "")) for epoch in range(len(epoch_stats))
                ]

    last = {key: str(val[-1]) for key, val in all_stats.items() if len(val) > 0}
    if epochs > 0:
        last["epoch"] = epochs

    macro_stats = {
        key: str(val) for key, val in stats.items() if isinstance(key, str) and key.startswith("macro_")
    }
    last.update(macro_stats)

    out = {"args": args.__dict__, "final": last, "all": all_stats}
    if "per_dataset" in stats:
        out["per_dataset"] = _to_jsonable(stats["per_dataset"])
    if len(macro_stats) > 0:
        out["macro"] = macro_stats

    with open(file, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)


def append_final_metrics_csv(csv_path: str, benchmark: str, run_datetime: str, metrics: dict) -> None:
    """Append one run summary row to a CSV file.

    The CSV always contains at least benchmark and run_datetime plus one column
    per metric key. If new metric keys appear later, the file is rewritten with
    the expanded header while preserving existing rows.
    """
    if csv_path in [None, "", "None"]:
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    row = {
        "benchmark": benchmark,
        "run_datetime": run_datetime,
    }

    for key, value in metrics.items():
        if isinstance(value, np.generic):
            row[key] = value.item()
        elif isinstance(value, np.ndarray):
            row[key] = json.dumps(value.tolist())
        elif isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value)
        else:
            row[key] = value

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []

    merged_fields = list(existing_fields)
    for key in row.keys():
        if key not in merged_fields:
            merged_fields.append(key)

    if merged_fields == existing_fields:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow(row)
        return

    existing_rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for current_row in existing_rows:
            writer.writerow(current_row)


class ProgressMessage(object):
    def __init__(self, mask: dict, separator: str = "|", precision: int = 6) -> None:
        assert set(mask.keys()).issubset({"trn", "tst", "val", "oth"})
        self.mask = mask
        self.separator = separator
        self.precision = precision

    def build_progress_message(self, stats: dict, epoch: int) -> str:
        """Builds up a message for command-line output."""
        assert set(stats.keys()).issubset({"trn", "tst", "val", "oth"})
        msg = f"{epoch:04d} {self.separator}"
        for mode, keys in self.mask.items():
            for key in keys:
                val = stats[mode][epoch - 1][key]
                if mode == "oth":
                    if type(val) == float:
                        msg += f" {key}={val:0.6f} {self.separator}"
                    else:
                        msg += f" {key}={val} {self.separator}"
                else:
                    msg += f" {mode}_{key}={val:0.{self.precision}f} {self.separator}"
        return msg


def build_progress_message(epoch: int, stats: dict, mask: dict, sep: str = "|") -> str:
    """Builds up a message for command-line output.

    Parameters:
    -----------
        epoch: int
            Identifies the epoch
        parts: dict
            Dictionary of value to be output
        mask:
            specifies the used dict keys
        sep:
            separator between values
    """

    msg = f"{epoch:04d} {sep}"
    for mode, keys in mask:
        for key in keys:
            val = stats[mode][key]
            if mode == "oth":
                msg += f" {key}={val:0.6f} {sep}"
            else:
                msg += f" {mode}_{key}={val:0.6f} {sep}"
    return msg


def save_checkpoint(args, epoch: Union[int,str], experiment_id: int, modules: ModuleDict, desired_t: Tensor):
    """Save model checkpoint.

    Arguments:
    ----------
        args: argparse.Namespace
            Arguments returned by argparse
        epoch: int | str
            Epoch number or 'best'
        experiment_id: int
            Unique number identifying the experiment
        modules: ModuleDict
            ModuleDict of the form {
                'recog_net': recognition network instance,
                'recon_net': reconstruction network instance,
                'pxz_net: instance of the network parametrizing p(x|z)
                'qzx_net: instance of the network parametrizing the approximate posterior q(z|x)
                'aux_net': instance of the network mapping z's to auxilliary output}
        desired_t: torch.Tensor
            Tensor holding the desired timepoints

    Returns
    -------
        completed: bool
            Returns true if successfully completed, otherwise False
    """
    assert isinstance(epoch, int) or epoch == "best"
    if not args.enable_checkpointing: return False
    if args.checkpoint_dir is None or args.checkpoint_dir == "None": return False

    assert os.path.exists(args.checkpoint_dir), f"{args.checkpoint_dir} does not exist!"
    checkpoint = {"args": args, "desired_t": desired_t, "modules": modules.state_dict()}
    torch.save(
        checkpoint,
        os.path.join(args.checkpoint_dir, f"checkpoint_{experiment_id}_{epoch}.h5"),
    )
    return True


def scatter_obs_and_msk(obs: Tensor, msk: Tensor, tid: Tensor, num_tps: int, mc_samples: int = 1):
    """Scatters observations with corresponding binary mask to size of
    a sample returned by pzx_net.sample(). See the example for a concrete
    use case and the convention for mask formatting.

    Parameters:
    -----------
        obs: torch.Tensor (of size [batch_len, T, obs_shape])
            Available observations, where obs_shape represents the size of the
            observations per t in {0,...,T-1}, e.g., 1x24x24
        msk: torch.Tensor (of size [batch_len, T, obs_shape])
            Available (binary) mask for the observations in obs.
        tid: torch.Tensor
            Tensor with indices into {0,...,K} where K is the
            number of timepoints available from a call to pzx_net.sample()
            which has shape [mc_samples, batch_len, K, obs_shape]
        num_tps: int
            Integer representing K
        mc_samples: int (default=1)
            Represents the number of samples available from a call to
            pzx_net.sample() which has shape [mc_samples, batch_len, K, obs_shape]

    Returns:
    --------
        scattered_obs: torch.Tensor
        scattered_msk: torch.Tensor

    Example:
    --------
        In the following example, obs holds a batch of size 2
        of obs_shape=(1,4,4) observations at 5 timepoints. The
        convention here is that 5 identifies the number of
        potentially available timepoints per batch element. In
        the example, only timepoints at indinces {0,1,2} for the
        first batch element and only timepoints at indices {0,1} are
        available. Mask entries at non-available timepoints need
        to be zero!

        At all valid timepoints, we have a mask ({0,1}) of the
        same shape as obs, identifying valid entries per observation.
        In the example, although obs might represent 4x4 images with
        one color channel, not all pixel might be valid.

        tid holds the indices of timepoints where the available
        observations per batch element should be distributed to. In
        the example, these are {0,3,6} for the first batch element
        and {4,9} for the second batch element.

        >>> obs = torch.rand(2,5,1,4,4)
        >>> obs[0,3:5] = 0
        >>> obs[1,2:5] = 0
        >>> msk = torch.randint(0,2,(2,5,1,4,4))
        >>> msk[:,3:5] = 0
        >>> msk[:,2:5] = 0
        >>> obs[msk==0] = 0.
        >>> tid = torch.tensor([
                [0,3,6,0,0],
                [4,9,0,0,0]
            ])
        >>> num_tps = 10
        >>> scattered_obs, scattered_msk = scatter_obs_and_msk(obs, msk, tid, num_tps, 2)
    """

    assert (
        obs.shape == msk.shape
    ), f"shape mismatch between obs.shape ({obs.shape}) and msk.shape ({msk.shape})"
    assert (
        obs.shape[0:2] == tid.shape
    ), f"shape mismatch between obs.shape[0:2] ({obs.shape[0:2]}) and tid.shape({tid.shape})"
    assert obs[msk == 0].norm() == 0, f"obs is nonzero at msk=0"
    assert (
        tid.max() < num_tps
    ), f" tid max ({tid.max()}) not less than num_tps ({num_tps})"

    batch_len, _, *per_tp_shape = obs.shape
    target_shape = torch.Size([mc_samples, batch_len, num_tps] + per_tp_shape)

    tid = tid.view(tid.shape + torch.Size([1] * len(per_tp_shape))).expand_as(msk)
    obs, msk, tid = [
        x.unsqueeze(0).expand(torch.Size([mc_samples]) + x.shape)
        for x in [obs, msk, tid]
    ]

    zero_obs = torch.zeros(target_shape, dtype=obs.dtype, device=obs.device)
    zero_msk = torch.zeros(target_shape, dtype=msk.dtype, device=msk.device)

    scattered_obs = zero_obs.scatter_reduce(2, tid, obs, "sum")
    scattered_msk = zero_msk.scatter_reduce(2, tid, msk, "sum")

    return scattered_obs, scattered_msk


def check_logging_and_checkpointing(args) -> None:
    if args.enable_file_logging:
        if args.log_dir is None:
            raise FileNotFoundError(f'No logging directory specified!')    
        if not os.path.exists(args.log_dir):
            raise FileNotFoundError(f'Logging directory {args.log_dir} not found!')
    if args.enable_checkpointing:
        if not os.path.exists(args.checkpoint_dir):
            raise FileNotFoundError(f'Checkpointing directory {args.checkpointing_dir} not found!')   
        if len(args.checkpoint_at) == 0:
            print('No checkpoints given. Only checkpoint of best model will be saved.')
        else:
            assert len(args.checkpoint_at) > 0, 'No checkpoints given'
            assert all(x > 0 for x in args.checkpoint_at), 'Checkpoints need to be non-negative!'
            assert max(args.checkpoint_at) <= args.n_epochs, 'Checkpoints out of bounds!'


