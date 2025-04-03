import numpy as np
from torcheval.metrics.functional import binary_f1_score
import tqdm

from notebooks.utils.analyze import batch_get_log_prob


def anomaly_detection_performances(aux_tgt, true_tgt):
    return binary_f1_score(aux_tgt.flatten(), true_tgt.flatten(), threshold=0.5)


def calculate_anomaly_threshold(dl_, args_, modules_, desired_t_):
    # %% identify anomaly threshold
    log_probs_test = [
         batch_get_log_prob(batch_it, args_, modules_, desired_t_).to('cpu').detach().numpy()
         for _, batch_it in tqdm.tqdm(enumerate(dl_))
    ]

    log_probs_test = np.vstack(log_probs_test)

    test_mean = log_probs_test.mean(axis=0)
    test_std = log_probs_test.std(axis=0)

    anomaly_threshold = test_mean + 2.5 * test_std

    return anomaly_threshold
