from typing import Tuple

import os
import logger
import numpy as np

from datetime import datetime

from dataset import prepare_and_save_data
from model import (
    caclulate_gradient,
    initialize_weights,
    predict_probabilities,
    cross_entropy_loss,
    update_weights,
)
from utils import (
    plot_loss_curve,
    save_weights,
    parse_training_args,
    print_training_config,
    save_training_params,
    plot_decision_boundary,
    log,
)


def load_training_data(
    path_prefix: str = "../data/normalized",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """
    Belirtilen önekle kaydedilmiş eğitim, doğrulama ve test setlerini yükler.
    Args:
        path_prefix (str): Dosya yolu öneki.
    Returns:
        train_data (tuple[np.ndarray, np.ndarray]): Eğitim verisi (X_train, y_train).
        val_data (tuple[np.ndarray, np.ndarray]): Doğrulama verisi (X_val, y_val).
        raw_train_data (tuple[np.ndarray, np.ndarray]): Ham eğitim verisi.
        raw_val_data (tuple[np.ndarray, np.ndarray]): Ham doğrulama verisi.
    """
    log(f"Loading data splits with prefix: {path_prefix}")
    if not os.path.exists(f"{path_prefix}_train.npz") or not os.path.exists(
        f"{path_prefix}_val.npz"
    ):
        raise FileNotFoundError(
            f"Data files with prefix {path_prefix} not found. "
            "Please ensure the data is prepared and saved."
        )

    train_loaded = np.load(f"{path_prefix}_train.npz")
    X_train = train_loaded["X"]
    y_train = train_loaded["y"]

    val_loaded = np.load(f"{path_prefix}_val.npz")
    X_val = val_loaded["X"]
    y_val = val_loaded["y"]

    # Load raw data for visualization
    raw_train_loaded = np.load(f"{path_prefix.replace('normalized', 'raw')}_train.npz")
    X_train_raw = raw_train_loaded["X"]

    raw_val_loaded = np.load(f"{path_prefix.replace('normalized', 'raw')}_val.npz")
    X_val_raw = raw_val_loaded["X"]

    log(f"Data splits loaded with prefix: {path_prefix}")

    return (
        (X_train, y_train),
        (X_val, y_val),
        X_train_raw,
        X_val_raw,
    )


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Özellik matrisine bias terimi ekler.
    Args:
        X (np.ndarray): Orijinal özellik matrisi.
    Returns:
        X_bias (np.ndarray): Bias terimi eklenmiş özellik matrisi.
    """
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    X_bias = np.hstack((bias_column, X))
    return X_bias


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0001,
    early_stopping: bool = True,
) -> Tuple[np.ndarray, list[float], list[float], int, bool]:
    """
    Lojistik regresyon modelini Stochastic Gradient Descent (SGD) ile eğitir.
    Args:
        X_train (np.ndarray): Eğitim özellik matrisi.
        y_train (np.ndarray): Eğitim etiketleri.
        X_val (np.ndarray): Doğrulama özellik matrisi.
        y_val (np.ndarray): Doğrulama etiketleri.
        learning_rate (float): Öğrenme oranı.
        n_epochs (int): Eğitim için epoch sayısı.
        patience (int): Early stopping yapmadan önceki minimum epoch sayısı.
        min_delta (float): Early stopping için minimum iyileşme.
        early_stopping (bool): Early stopping kullanılsın mı.
    Returns:
        w (np.ndarray): Eğitilmiş ağırlıklar.
        train_losses (list[float]): Eğitim kayıplarının listesi.
        val_losses (list[float]): Doğrulama kayıplarının listesi.
        actual_epochs (int): Gerçekleşen epoch sayısı.
        early_stopped (bool): Early stopping tetiklendi mi.
    """
    w = initialize_weights(n_features=X_train.shape[1])

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    early_stopped = False

    log("=" * 70)
    log(f"Starting training for {n_epochs} epochs with learning rate {learning_rate}")
    log("=" * 70)
    for epoch in range(n_epochs):
        epoch_train_losses: list[float] = []

        for i in range(X_train.shape[0]):
            X_i = X_train[i]
            y_i_true = y_train[i]

            y_i_pred = predict_probabilities(X_i.reshape(1, -1), w)[0]
            loss_i = cross_entropy_loss(np.array([y_i_true]), np.array([y_i_pred]))
            epoch_train_losses.append(loss_i)

            gradient = caclulate_gradient(X_i, y_i_true, y_i_pred)

            w = update_weights(w, gradient, learning_rate)

        avg_train_loss = float(np.mean(epoch_train_losses))
        train_losses.append(avg_train_loss)

        y_val_pred = predict_probabilities(X_val, w)
        val_loss = cross_entropy_loss(y_val, y_val_pred)
        val_losses.append(val_loss)

        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_weights = w.copy()
                epochs_no_improve = 0
                improvement_marker = " *"
            else:
                epochs_no_improve += 1
                improvement_marker = ""

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log(
                    f"Epoch {epoch+1:>{len(str(n_epochs))}d}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}{improvement_marker} - No Improve: {epochs_no_improve}"
                )

            if epochs_no_improve >= patience:
                log(f"\n{'='*70}")
                log(f"Early stopping triggered at epoch {epoch + 1}")
                log(f"Best validation loss: {best_val_loss:.4f}")
                log(f"Restoring best weights from epoch {epoch + 1 - patience}")
                log(f"{'='*70}")
                w = np.array(best_weights)
                early_stopped = True
                actual_epochs = epoch + 1
                return w, train_losses, val_losses, actual_epochs, early_stopped
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log(
                    f"Epoch {epoch+1:>{len(str(n_epochs))}d}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

    actual_epochs = n_epochs
    return w, train_losses, val_losses, actual_epochs, early_stopped


def train(
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0001,
    early_stopping: bool = True,
) -> None:
    """
    Veri setini yükler, işler ve lojistik regresyon modelini eğitir.
    Args:
        learning_rate (float): Öğrenme oranı (varsayılan: 0.01).
        n_epochs (int): Epoch sayısı (varsayılan: 100).
        patience (int): Early stopping patience.
        min_delta (float): Early stopping minimum delta.
        early_stopping (bool): Early stopping kullanılsın mı.
    """

    log("\n" + "=" * 70)
    log(
        f"MODEL EĞİTİMİ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n".center(
            70
        )
    )
    print_training_config(learning_rate, n_epochs, patience, min_delta, early_stopping)

    prepare_and_save_data()

    (
        (X_train, y_train),
        (X_val, y_val),
        X_train_raw,
        X_val_raw,
    ) = load_training_data("../data/normalized")

    X_train = add_bias_term(X_train)
    X_val = add_bias_term(X_val)

    w, train_losses, val_losses, actual_epochs, early_stopped = (
        train_logistic_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            patience=patience,
            min_delta=min_delta,
            early_stopping=early_stopping,
        )
    )

    plot_loss_curve(train_losses, val_losses)
    save_weights(w)
    save_training_params(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        actual_epochs=actual_epochs,
        patience=patience,
        min_delta=min_delta,
        early_stopping_enabled=early_stopping,
        early_stopped=early_stopped,
    )

    # Karar sınırı grafiğini çiz (eğitim ve doğrulama setleri için)
    log("\nKarar sınırı grafikleri oluşturuluyor...")
    plot_decision_boundary(X_train_raw, y_train, w, data="train")
    plot_decision_boundary(X_val_raw, y_val, w, data="val")

    log("\n" + "=" * 70)
    log("Training completed and model saved.")
    if early_stopped:
        log(f"Training stopped early at epoch {actual_epochs}/{n_epochs}")
    else:
        log(f"Training completed all {actual_epochs} epochs")
    log("=" * 70)


if __name__ == "__main__":
    args = parse_training_args()

    logger.setup_logger(mode=args.log)

    train(
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stopping=not args.no_early_stopping,
    )
