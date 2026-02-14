import numpy as np

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp + 1e-10)


def recall_score(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + 1e-10)


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)


def matthews_corrcoef(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-10
    return numerator / denominator


def roc_auc_score(y_true, y_prob):

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)

    y_prob = y_prob.astype(float)

    sorted_indices = np.argsort(y_prob)
    y_true = y_true[sorted_indices]

    cum_pos = np.cumsum(y_true)
    cum_neg = np.cumsum(1 - y_true)

    tpr = cum_pos / (np.sum(y_true) + 1e-10)
    fpr = cum_neg / (np.sum(1 - y_true) + 1e-10)

    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc
