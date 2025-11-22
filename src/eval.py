from typing import Tuple
import numpy as np

from model import predict_probabilities, cross_entropy_loss
from metrics import accuracy, precision, recall, f1_score, confusion_matrix
from utils import log_test_results, print_confusion_matrix


def load_test_data(
    path_prefix: str = "data/normalized",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Test veri setini yükler.
    Args:
        path_prefix (str): Veri seti dosyalarının bulunduğu dizin yolu.
    Returns:
        tuple[np.ndarray, np.ndarray]: Özellik matrisi ve etiket vektörü.
    """
    test_loaded = np.load(f"../{path_prefix}_test.npz")
    X_test = test_loaded["X"]
    y_test = test_loaded["y"]
    return X_test, y_test


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Özellik matrisine bias terimi ekler.
    Args:
        X (np.ndarray): Orijinal özellik matrisi.
    Returns:
        np.ndarray: Bias terimi eklenmiş özellik matrisi.
    """
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    X_bias = np.hstack((bias_column, X))
    return X_bias


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | Tuple[int, int, int, int]]:
    """
    Modeli verilen veri setinde değerlendirir.
    Args:
        X (np.ndarray): Özellik matrisi.
        y (np.ndarray): Gerçek etiketler.
        weights (np.ndarray): Model ağırlıkları.
    Returns:
        dict[str, float]: Hesaplanan metrikler (doğruluk, kesinlik, duyarlılık, F1 skoru, kayıp).
    """
    y_prob = predict_probabilities(X, weights)
    y_pred = (y_prob >= 0.5).astype(int)

    conf_matrix = confusion_matrix(y, y_pred)

    loss = cross_entropy_loss(y, y_prob)
    acc = accuracy(conf_matrix=conf_matrix)
    prec = precision(conf_matrix=conf_matrix)
    rec = recall(conf_matrix=conf_matrix)
    f1 = f1_score(conf_matrix=conf_matrix)

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
    }


if __name__ == "__main__":
    X_test, y_test = load_test_data("data/normalized")
    X_test = add_bias_term(X_test)

    weights = np.load("../results/model/model_weights_latest.npy")

    metrics = evaluate_model(X_test, y_test, weights)

    print("Test Seti Değerlendirme Sonuçları:")
    print(f"Kayıp (Loss): {metrics['loss']:.4f}")
    print(f"Doğruluk (Accuracy): {metrics['accuracy']:.4f}")
    print(f"Kesinlik (Precision): {metrics['precision']:.4f}")
    print(f"Duyarlılık (Recall): {metrics['recall']:.4f}")
    print(f"F1 Skoru: {metrics['f1_score']:.4f}")

    print_confusion_matrix(metrics["confusion_matrix"])

    log_test_results(metrics)
