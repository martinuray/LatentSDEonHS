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