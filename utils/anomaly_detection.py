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


def create_random_burst_mask(n_features, x_len, masked_ratio=0.3):

    max_bursts = x_len // 20 # just set
    min_bursts = x_len // 50 # just set

    if n_features < 0 or x_len < 0:
        raise ValueError("n_features and x_len must be non-negative")
    if not 0.0 <= masked_ratio <= 1.0:
        raise ValueError("masked_ratio must be between 0 and 1")

    if x_len == 0:
        return np.ones((n_features, 0), dtype=bool)

    if min_bursts < 1 or max_bursts < 1 or min_bursts > max_bursts:
        raise ValueError("Require 1 <= min_bursts <= max_bursts")

    mask = np.ones((n_features, x_len), dtype=bool)
    min_available_count = min(x_len, max(int(np.ceil(MIN_AVAILABLE_RATIO * x_len)), 2))
    max_masked_per_feature = x_len - min_available_count

    total_masked = int(np.rint(masked_ratio * n_features * x_len))
    total_masked = int(np.clip(total_masked, 0, n_features * x_len))
    max_total_masked = n_features * max_masked_per_feature

    if total_masked > max_total_masked:
        raise ValueError(
            f"masked_ratio={masked_ratio} is infeasible: each feature must keep at least "
            f"{min_available_count} / {x_len} samples available"
        )

    if total_masked == 0:
        return mask

    feature_masked_counts = _sample_feature_masked_counts(n_features, max_masked_per_feature, total_masked)

    for feature_idx, masked_count in enumerate(feature_masked_counts):
        if masked_count == 0:
            continue

        available_count = x_len - masked_count
        max_feasible_bursts = min(max_bursts, masked_count, available_count)
        if max_feasible_bursts <= 0:
            raise ValueError(
                f"Cannot place {masked_count} masked samples into length {x_len} with separated bursts"
            )

        burst_count = np.random.randint(min(min_bursts, max_feasible_bursts), max_feasible_bursts + 1)

        burst_lengths = 1 + np.random.multinomial(
            masked_count - burst_count,
            np.full(burst_count, 1.0 / burst_count),
        )

        free_gap_budget = available_count - burst_count

        gap_lengths = 1 + np.random.multinomial(
            free_gap_budget,
            np.full(burst_count, 1.0 / burst_count),
        )

        cursor = np.random.randint(0, x_len)
        for burst_idx in range(burst_count):
            burst_len = burst_lengths[burst_idx]
            _set_circular_false(mask[feature_idx], cursor, burst_len)
            cursor = (cursor + burst_len + gap_lengths[burst_idx]) % x_len

    available_per_feature = mask.sum(axis=1)
    if np.any(available_per_feature < min_available_count):
        raise RuntimeError(
            f"Generated mask violates minimum availability: required {min_available_count}, got {available_per_feature.min()}"
        )

    return mask
