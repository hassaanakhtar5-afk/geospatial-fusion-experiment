import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
import config


def compute_overall_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100


def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_cohen_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def compute_per_class_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(np.unique(y_true))
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1, "labels": labels}


def compute_high_risk_auc_roc(y_true, y_prob):
    hr_label = config.HIGH_RISK_CLASS
    y_binary = (y_true == hr_label).astype(int)
    hr_prob = y_prob[:, hr_label]
    return roc_auc_score(y_binary, hr_prob)


def compute_high_risk_auc_pr(y_true, y_prob):
    hr_label = config.HIGH_RISK_CLASS
    y_binary = (y_true == hr_label).astype(int)
    hr_prob = y_prob[:, hr_label]
    return average_precision_score(y_binary, hr_prob)


def compute_pr_curve(y_true, y_prob, class_label=None):
    if class_label is None:
        class_label = config.HIGH_RISK_CLASS
    y_binary = (y_true == class_label).astype(int)
    prob = y_prob[:, class_label]
    precision, recall, thresholds = precision_recall_curve(y_binary, prob)
    return precision, recall, thresholds


def compute_roc_curve(y_true, y_prob, class_label=None):
    if class_label is None:
        class_label = config.HIGH_RISK_CLASS
    y_binary = (y_true == class_label).astype(int)
    prob = y_prob[:, class_label]
    fpr, tpr, thresholds = roc_curve(y_binary, prob)
    return fpr, tpr, thresholds


def compute_confusion_matrix(y_true, y_pred, normalize="true"):
    labels = list(range(config.N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    n00 = np.sum(~correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)
    n11 = np.sum(correct_a & correct_b)
    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue


def full_evaluation(y_true, y_pred, y_prob, model_name=""):
    oa = compute_overall_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    kappa = compute_cohen_kappa(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred)
    hr_auc = compute_high_risk_auc_roc(y_true, y_prob)
    hr_auc_pr = compute_high_risk_auc_pr(y_true, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred)

    hr = config.HIGH_RISK_CLASS
    hr_idx = list(per_class["labels"]).index(hr) if hr in per_class["labels"] else -1
    hr_recall = per_class["recall"][hr_idx] if hr_idx >= 0 else 0.0
    hr_f1 = per_class["f1"][hr_idx] if hr_idx >= 0 else 0.0

    return {
        "model": model_name,
        "oa": oa,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_precision": per_class["precision"],
        "per_class_recall": per_class["recall"],
        "per_class_f1": per_class["f1"],
        "class_labels": per_class["labels"],
        "hr_auc_roc": hr_auc,
        "hr_auc_pr": hr_auc_pr,
        "hr_recall": hr_recall,
        "hr_f1": hr_f1,
        "confusion_matrix": cm,
    }
