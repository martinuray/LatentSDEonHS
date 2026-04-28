import numpy as np
from torcheval.metrics.functional import binary_f1_score
import tqdm

from notebooks.utils.analyze import batch_get_log_prob


def anomaly_detection_performances(aux_tgt, true_tgt):
    return binary_f1_score(aux_tgt.flatten(), true_tgt.flatten(), threshold=0.5)


def get_all_log_probs(args_, desired_t_, dl_, modules_):
    log_probs_test = [
        batch_get_log_prob(batch_it, args_, modules_, desired_t_)
        for _, batch_it in tqdm.tqdm(enumerate(dl_))
    ]
    log_probs_test = [log_prob for pxz, log_prob in log_probs_test]
    log_probs_test = np.vstack(log_probs_test)
    return log_probs_test


def calculate_anomaly_threshold(dl_, args_, modules_, desired_t_):
    # %% identify anomaly threshold
    log_probs_test = get_all_log_probs(args_, desired_t_, dl_, modules_)

    test_mean = log_probs_test.mean(axis=0)
    test_std = log_probs_test.std(axis=0)

    anomaly_threshold = test_mean + 2.5 * test_std

    return anomaly_threshold


def get_number_over_threshold(dl_, args_, modules_, desired_t_, anomaly_threshold):
    probs_test = get_all_log_probs(args_, desired_t_, dl_, modules_)
    probs_test = probs_test > anomaly_threshold
    return probs_test.sum(axis=1)


MIN_AVAILABLE_RATIO = 0.005


def _sample_feature_masked_counts(n_features, max_masked_per_feature, total_masked):
    if n_features == 0 or total_masked == 0:
        return np.zeros(n_features, dtype=int)

    counts = np.zeros(n_features, dtype=int)
    remaining = total_masked
    capacities = np.full(n_features, max_masked_per_feature, dtype=int)

    while remaining > 0:
        active = capacities > 0
        if not np.any(active):
            raise ValueError("Unable to distribute masked samples across features")

        weights = np.random.gamma(shape=1.0, scale=1.0, size=active.sum())
        proposed = np.random.multinomial(remaining, weights / weights.sum())
        assigned = np.minimum(proposed, capacities[active])

        counts[active] += assigned
        capacities[active] -= assigned
        remaining -= assigned.sum()

    return counts


def _set_circular_false(mask_row, start_idx, run_length):
    x_len = mask_row.shape[0]
    end_idx = start_idx + run_length
    if end_idx <= x_len:
        mask_row[start_idx:end_idx] = False
        return

    split_idx = end_idx % x_len
    mask_row[start_idx:] = False
    mask_row[:split_idx] = False


def create_random_burst_mask(n_features, x_len, masked_ratio=0.3, s=0.1, max_false_length = 90):

    p = 1 - masked_ratio
    out_mask = np.empty((n_features, x_len), dtype=bool)

    true_lengths = np.random.geometric(s*p, size=(n_features,x_len))
    false_lengths = np.random.geometric(s*(1 - p), size=(n_features,x_len))
    false_lengths = np.minimum(false_lengths, max_false_length)
    
    state_changes = np.empty((n_features, 2 * x_len), dtype=int)
    state_changes[:,0::2] = true_lengths
    state_changes[:,1::2] = false_lengths

    for i in range(n_features):
        values = np.arange(len(state_changes[i])) % 2 == 0
        mask = np.repeat(values, state_changes[i])  
        start = np.random.randint(0, len(mask))
        idx = (start + np.arange(x_len)) % len(mask)
        out_mask[i] = mask[idx]
    out_mask[:,0] = True
    out_mask[:,1] = True

    return out_mask
