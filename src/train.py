from typing import Tuple

import os
import numpy as np

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
    log,
    setup_logger,
)


def load_training_data(
    path_prefix: str = "../data/normalized",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Belirtilen önekle kaydedilmiş eğitim, doğrulama ve test setlerini yükler.
    Args:
        path_prefix (str): Dosya yolu öneki.
    Returns:
        train_data (tuple[np.ndarray, np.ndarray]): Eğitim verisi (X_train, y_train).
        val_data (tuple[np.ndarray, np.ndarray]): Doğrulama verisi (X_val, y_val).
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

    log(f"Data splits loaded with prefix: {path_prefix}")

    return (X_train, y_train), (X_val, y_val)


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
) -> Tuple[np.ndarray, list[float], list[float]]:
    """
    Lojistik regresyon modelini Stochastic Gradient Descent (SGD) ile eğitir.
    Args:
        X_train (np.ndarray): Eğitim özellik matrisi.
        y_train (np.ndarray): Eğitim etiketleri.
        X_val (np.ndarray): Doğrulama özellik matrisi.
        y_val (np.ndarray): Doğrulama etiketleri.
        learning_rate (float): Öğrenme oranı.
        n_epochs (int): Eğitim için epoch sayısı.
    Returns:
        w (np.ndarray): Eğitilmiş ağırlıklar.
        train_losses (list[float]): Eğitim kayıplarının listesi.
        val_losses (list[float]): Doğrulama kayıplarının listesi.
    """
    w = initialize_weights(n_features=X_train.shape[1])

    train_losses: list[float] = []
    val_losses: list[float] = []

    log(f"\n{15*'='} Starting Training {'='*16}")
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(
                f"Epoch {epoch+1:>{len(str(n_epochs))}d}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

    return w, train_losses, val_losses


def train(learning_rate: float = 0.01, n_epochs: int = 100) -> None:
    """
    Veri setini yükler, işler ve lojistik regresyon modelini eğitir.
    Args:
        learning_rate (float): Öğrenme oranı (varsayılan: 0.01).
        n_epochs (int): Epoch sayısı (varsayılan: 100).
    """
    print_training_config(learning_rate, n_epochs)

    prepare_and_save_data()

    (X_train, y_train), (X_val, y_val) = load_training_data("../data/normalized")

    X_train = add_bias_term(X_train)
    X_val = add_bias_term(X_val)

    w, train_losses, val_losses = train_logistic_regression(
        X_train, y_train, X_val, y_val, learning_rate=learning_rate, n_epochs=n_epochs
    )

    plot_loss_curve(train_losses, val_losses)
    save_weights(w)

    log("\n" + "=" * 50)
    log("Training completed and model saved.")
    log("=" * 50)


if __name__ == "__main__":
    args = parse_training_args()

    # Setup logger with mode from command line arguments
    setup_logger(mode=args.log)

    train(learning_rate=args.learning_rate, n_epochs=args.epochs)
