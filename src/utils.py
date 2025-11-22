from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime


def ensure_dir_exists(directory: str) -> None:
    """
    Belirtilen dizinin var olduğunu garanti eder, yoksa oluşturur.
    Args:
        directory (str): Dizinin yolu.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_scatter(
    X: np.ndarray,
    y: np.ndarray,
    data: str = "tüm",
    save_path: str = "../results/graphs/",
) -> None:
    """
    Verilen veri setini scatter plot olarak çizer.
    Args:
        X (np.ndarray): Özellik matrisi.
        y (np.ndarray): Hedef değişken vektörü.
        data (str): Veri seti türü (örneğin, "train", "val", "test", "tüm").
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    ensure_dir_exists(save_path)

    if X.shape[1] != 2:
        X_plot = X[:, 1:]  # Yalnızca ilk iki özelliği kullan (bias sütununu at)
    else:
        X_plot = X

    class_0 = y == 0
    class_1 = y == 1

    plt.figure(figsize=(8, 6))  # type: ignore
    plt.scatter(  # type: ignore
        X_plot[class_0, 0],
        X_plot[class_0, 1],
        color="red",
        label="Kalanlar (Class 0)",
        alpha=0.5,
        marker="x",
    )
    plt.scatter(  # type: ignore
        X_plot[class_1, 0],
        X_plot[class_1, 1],
        color="blue",
        label="Geçenler (Class 1)",
        alpha=0.5,
        marker="o",
    )

    plt.title(f"{data.capitalize()} Veri Setinin Scatter Plotu")  # type: ignore
    plt.xlabel("Sınav 1")  # type: ignore
    plt.ylabel("Sınav 2")  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True)  # type: ignore
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{data}_scatter_plot.png"))  # type: ignore
    plt.close()


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "../results/graphs/",
) -> None:
    """
    Eğitim ve doğrulama kayıplarını gösteren bir grafik çizer.
    Args:
        train_losses (list[float]): Eğitim kayıpları.
        val_losses (list[float]): Doğrulama kayıpları.
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    ensure_dir_exists(save_path)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))  # type: ignore
    plt.plot(epochs, train_losses, label="Eğitim Kaybı", color="blue")  # type: ignore
    plt.plot(epochs, val_losses, label="Doğrulama Kaybı", color="orange")  # type: ignore
    plt.title("Eğitim ve Doğrulama Kaybı")  # type: ignore
    plt.xlabel("Epoch")  # type: ignore
    plt.ylabel("Kayıp")  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True)  # type: ignore
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))  # type: ignore
    plt.close()


def save_weights(w: np.ndarray, save_dir: str = "../results/model/") -> None:
    """
    Model ağırlıklarını belirtilen dizine kaydeder.
    Model isimlendirmesini "model_weights_{{timestamp}}.npy" formatında yapar.
    Args:
        w (np.ndarray): Ağırlık vektörü.
        save_dir (str): Ağırlıkların kaydedileceği dizin yolu.
    """
    ensure_dir_exists(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_weights_{timestamp}.npy")
    np.save(save_path, w)
    np.save(os.path.join(save_dir, "model_weights_latest.npy"), w)


def log_test_results(
    results: dict[str, float],
    log_file: str = "../results/evaluation/test_results.txt",
) -> None:
    """
    Test sonuçlarını belirtilen dosyaya kaydeder.
    Args:
        results (dict[str, float]): Değerlendirme metrikleri.
        log_file (str): Sonuçların kaydedileceği dosya yolu.
    """
    ensure_dir_exists(os.path.dirname(log_file))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Test Sonuçları - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for metric, value in results.items():
            if metric != "confusion_matrix":
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        f.write("\n")


def parse_training_args():
    """
    Komut satırı argümanlarını parse eder.
    Returns:
        argparse.Namespace: Parse edilmiş argümanlar (learning_rate, epochs).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Lojistik Regresyon Modeli Eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        help="Öğrenme oranı (learning rate)",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Eğitim için epoch sayısı"
    )

    args = parser.parse_args()
    return args


def print_training_config(learning_rate: float, n_epochs: int) -> None:
    """
    Eğitim konfigürasyonunu ekrana yazdırır.
    Args:
        learning_rate (float): Öğrenme oranı.
        n_epochs (int): Epoch sayısı.
    """
    print("=" * 50)
    print("EĞITIM KONFIGÜRASYONU")
    print("=" * 50)
    print(f"Learning Rate: {learning_rate}")
    print(f"Epoch Sayısı: {n_epochs}")
    print("=" * 50)
    print()


def print_confusion_matrix(conf_matrix: Tuple[int, int, int, int]) -> None:
    """
    Confusion matrix'i ekrana tablo şeklinde yazdırır.
    Args:
        conf_matrix (Tuple[int, int, int, int]): Confusion matrix (TP, TN, FP, FN).
    """
    tp, tn, fp, fn = conf_matrix

    # Toplam değerleri hesapla
    total = tp + tn + fp + fn
    actual_positive = tp + fn
    actual_negative = tn + fp
    predicted_positive = tp + fp
    predicted_negative = tn + fn

    # Tablo genişliğini belirle
    col_width = 20

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX".center(70))
    print("=" * 70)
    print()

    # Başlık satırı
    print(
        f"{'':>{col_width}} | {'Actual Positive':^{col_width}} | {'Actual Negative':^{col_width}}"
    )
    print(
        f"{'':>{col_width}} | {'(Class 1)':^{col_width}} | {'(Class 0)':^{col_width}}"
    )
    print("-" * 70)

    print(
        f"{'Predicted Positive':>{col_width}} | {str(tp):^{col_width}} | {str(fp):^{col_width}}"
    )
    print(
        f"{'(Class 1)':>{col_width}} | {'(True Positive)':^{col_width}} | {'(False Positive)':^{col_width}}"
    )
    print("-" * 70)

    print(
        f"{'Predicted Negative':>{col_width}} | {str(fn):^{col_width}} | {str(tn):^{col_width}}"
    )
    print(
        f"{'(Class 0)':>{col_width}} | {'(False Negative)':^{col_width}} | {'(True Negative)':^{col_width}}"
    )
    print("-" * 70)

    print()
    print("ÖZET BİLGİLER:".center(70))
    print("-" * 70)
    print(f"  Toplam Örnek                : {total}")
    print(f"  Gerçek Pozitif (Actual 1)   : {actual_positive}")
    print(f"  Gerçek Negatif (Actual 0)   : {actual_negative}")
    print(f"  Tahmin Pozitif (Predicted 1): {predicted_positive}")
    print(f"  Tahmin Negatif (Predicted 0): {predicted_negative}")
    print("=" * 70)
    print()
