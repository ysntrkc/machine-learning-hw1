import numpy as np
import os
from typing import Tuple

from utils import plot_scatter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hw1Data.txt")


def load_data(path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ham veri setini belirtilen yoldan yükler.
    Args:
        path (str): Veri setinin dosya yolu.
    Returns:
        X (np.ndarray): Özellik matrisi.
        y (np.ndarray): Hedef değişken vektörü.
    """
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :2]
    y = data[:, 2]

    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Özellikleri normalize eder: (X - minimum) / (maximum - minimum).
    Args:
        X (np.ndarray): Özellik matrisi.
    Returns:
        X_norm (np.ndarray): Normalize edilmiş özellik matrisi.
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    denominator = X_max - X_min
    denominator[denominator == 0] = 1  # Bölme hatasını önlemek için

    X_norm = (X - X_min) / denominator

    return X_norm


def split_data(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.6, val_ratio: float = 0.2
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Veriyi eğitim, doğrulama ve test setlerine böler.
    İlk %60 eğitim, sonraki %20 doğrulama, kalan %20 test için ayrılır.
    Args:
        X (np.ndarray): Özellik matrisi.
        y (np.ndarray): Hedef değişken vektörü.
        train_ratio (float): Eğitim seti oranı.
        val_ratio (float): Doğrulama seti oranı.
    """
    total_samples = X.shape[0]

    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = int(total_samples * (train_ratio + val_ratio))

    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]

    X_val = X[train_end_idx:val_end_idx]
    y_val = y[train_end_idx:val_end_idx]

    X_test = X[val_end_idx:]
    y_test = y[val_end_idx:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_splits(
    prefix: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Eğitim, doğrulama ve test setlerini belirtilen önekle kaydeder.
    Args:
        prefix (str): Kaydetme yolu için önek.
        train_data (tuple[np.ndarray, np.ndarray]): Eğitim verisi (X_train, y_train).
        val_data (tuple[np.ndarray, np.ndarray]): Doğrulama verisi (X_val, y_val).
        test_data (tuple[np.ndarray, np.ndarray]): Test verisi (X_test, y_test).
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    print(f"Saving data splits with prefix: {prefix}")

    np.savez_compressed(f"{prefix}_train.npz", X=X_train, y=y_train)
    np.savez_compressed(f"{prefix}_val.npz", X=X_val, y=y_val)
    np.savez_compressed(f"{prefix}_test.npz", X=X_test, y=y_test)

    print(f"Data splits saved with prefix: {prefix}")


def prepare_and_save_data() -> None:
    """
    Veri setini yükler, normalleştirir, böler ve kaydeder.
    """
    X_raw, y = load_data()

    # Ham veriyi parçalarına ayır ve kaydet
    raw_train_data, raw_val_data, raw_test_data = split_data(X_raw, y)
    save_splits("../data/raw", raw_train_data, raw_val_data, raw_test_data)

    plot_scatter(X_raw, y, data="whole")
    plot_scatter(raw_train_data[0], raw_train_data[1], data="train")

    # Özellikleri normalleştir
    X_norm = normalize_features(X_raw)

    # Normalize edilmiş veriyi parçalarına ayır ve kaydet
    norm_train_data, norm_val_data, norm_test_data = split_data(X_norm, y)
    save_splits("../data/normalized", norm_train_data, norm_val_data, norm_test_data)


if __name__ == "__main__":
    prepare_and_save_data()
