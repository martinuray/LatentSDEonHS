from torcheval.metrics.functional import binary_f1_score

def anomaly_detection_performances(aux_tgt, true_tgt):
    return binary_f1_score(aux_tgt.flatten(), true_tgt.flatten(), threshold=0.5)
