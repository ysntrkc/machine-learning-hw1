from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import argparse
import logger

from model import predict_probabilities, cross_entropy_loss
from metrics import accuracy, precision, recall, f1_score, confusion_matrix
from utils import (
    log_test_results,
    print_confusion_matrix,
    load_training_params,
    log,
)


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
) -> Tuple[Dict[str, float], Tuple[int, int, int, int]]:
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
    }, conf_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Değerlendirme")
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="both",
        choices=["both", "console", "file"],
        help="Log modu: 'both', 'console', veya 'file'",
    )
    args = parser.parse_args()

    # Setup logger with mode from command line arguments
    logger.setup_logger(mode=args.log)

    log("\n" + "=" * 70)
    log(
        f"MODEL DEĞERLENDİRME BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n".center(
            70
        )
    )

    # Load and display training parameters
    training_params = load_training_params()
    if training_params:
        log("=" * 70)
        log("EĞİTİM PARAMETRELERİ".center(70))
        log("=" * 70)
        log(f"  Öğrenme Oranı (Learning Rate): {training_params['learning_rate']}")
        log(f"  Maksimum Epoch: {training_params['max_epochs']}")
        log(f"  Gerçekleşen Epoch: {training_params['actual_epochs']}")
        if training_params["early_stopping_enabled"]:
            log("  Early Stopping: Aktif")
            log(f"    - Patience: {training_params['patience']}")
            log(f"    - Min Delta: {training_params['min_delta']}")
            if training_params["early_stopped"]:
                log(
                    f"    - Durum: Tetiklendi (epoch {training_params['actual_epochs']}'de durduruldu)"
                )
            else:
                log("    - Durum: Tetiklenmedi (tüm epochlar tamamlandı)")
        else:
            log("  Early Stopping: Devre Dışı")
        log(f"  Eğitim Zamanı: {training_params['timestamp']}")
        log("=" * 70)
        log("")

    X_test, y_test = load_test_data("data/normalized")
    X_test = add_bias_term(X_test)

    weights = np.load("../results/model/model_weights_latest.npy")

    metrics, conf_matrix = evaluate_model(X_test, y_test, weights)

    log("=" * 70)
    log("TEST SETİ DEĞERLENDİRME SONUÇLARI".center(70))
    log("=" * 70)
    log(f"  Kayıp (Loss): {metrics['loss']:.4f}")
    log(f"  Doğruluk (Accuracy): {metrics['accuracy']:.4f}")
    log(f"  Kesinlik (Precision): {metrics['precision']:.4f}")
    log(f"  Duyarlılık (Recall): {metrics['recall']:.4f}")
    log(f"  F1 Skoru: {metrics['f1_score']:.4f}")
    log("=" * 70)

    print_confusion_matrix(conf_matrix)

    log_test_results(metrics)
