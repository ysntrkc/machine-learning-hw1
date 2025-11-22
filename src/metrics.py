from typing import Tuple, Optional
import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Gerçek ve tahmin edilen etiketlere göre bir karışıklık matrisi oluşturur.
    Args:
        y_true (np.ndarray): Gerçek etiketler.
        y_pred (np.ndarray): Tahmin edilen etiketler.
    Returns:
        TP (int): True Positive sayısı.
        TN (int): True Negative sayısı.
        FP (int): False Positive sayısı.
        FN (int): False Negative sayısı.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def accuracy(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    conf_matrix: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Doğruluk oranını hesaplar.
    Args:
        y_true (np.ndarray, optional): Gerçek etiketler.
        y_pred (np.ndarray, optional): Tahmin edilen etiketler.
        conf_matrix (Tuple[int, int, int, int], optional): Önceden hesaplanmış confusion matrix (TP, TN, FP, FN).
    Returns:
        float: Doğruluk oranı.

    Not:
        Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.
    """
    if conf_matrix is not None:
        tp, tn, fp, fn = conf_matrix
    elif y_true is not None and y_pred is not None:
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    else:
        raise ValueError("Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.")

    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def precision(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    conf_matrix: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Kesinlik oranını hesaplar.
    Args:
        y_true (np.ndarray, optional): Gerçek etiketler.
        y_pred (np.ndarray, optional): Tahmin edilen etiketler.
        conf_matrix (Tuple[int, int, int, int], optional): Önceden hesaplanmış confusion matrix (TP, TN, FP, FN).
    Returns:
        float: Kesinlik oranı.

    Not:
        Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.
    """
    if conf_matrix is not None:
        tp, _, fp, _ = conf_matrix
    elif y_true is not None and y_pred is not None:
        tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    else:
        raise ValueError("Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.")

    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return tp / denominator


def recall(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    conf_matrix: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Duyarlılık oranını hesaplar.
    Args:
        y_true (np.ndarray, optional): Gerçek etiketler.
        y_pred (np.ndarray, optional): Tahmin edilen etiketler.
        conf_matrix (Tuple[int, int, int, int], optional): Önceden hesaplanmış confusion matrix (TP, TN, FP, FN).
    Returns:
        float: Duyarlılık oranı.

    Not:
        Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.
    """
    if conf_matrix is not None:
        tp, _, _, fn = conf_matrix
    elif y_true is not None and y_pred is not None:
        tp, _, _, fn = confusion_matrix(y_true, y_pred)
    else:
        raise ValueError("Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.")

    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def f1_score(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    conf_matrix: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    F1 skorunu hesaplar.
    Args:
        y_true (np.ndarray, optional): Gerçek etiketler.
        y_pred (np.ndarray, optional): Tahmin edilen etiketler.
        conf_matrix (Tuple[int, int, int, int], optional): Önceden hesaplanmış confusion matrix (TP, TN, FP, FN).
    Returns:
        float: F1 skoru.

    Not:
        Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.
    """
    if conf_matrix is not None:
        prec = precision(conf_matrix=conf_matrix)
        rec = recall(conf_matrix=conf_matrix)
    elif y_true is not None and y_pred is not None:
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
    else:
        raise ValueError("Ya (y_true, y_pred) ya da conf_matrix sağlanmalıdır.")

    denominator = prec + rec
    if denominator == 0:
        return 0.0
    return 2 * (prec * rec) / denominator
