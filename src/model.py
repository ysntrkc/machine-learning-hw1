import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid aktivasyon fonksiyonu.
    Args:
        z (np.ndarray): Girdi değerleri.
    Returns:
        s (np.ndarray): Sigmoid çıktıları.
    """
    s = 1 / (1 + np.exp(-z))
    return s


def predict_probabilities(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Özellik matrisi ve ağırlıklar kullanılarak olasılık tahminleri yapar.
    Args:
        X (np.ndarray): Özellik matrisi.
        w (np.ndarray): Ağırlıklar.
    Returns:
        p (np.ndarray): Olasılık tahminleri.
    """
    z = np.dot(X, w)
    p = sigmoid(z)
    return p


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    İkili sınıflandırma için çapraz entropi kaybını hesaplar.
    Args:
        y_true (np.ndarray): Gerçek etiketler (0 veya 1).
        y_pred (np.ndarray): Tahmin edilen olasılıklar.
    Returns:
        loss (np.ndarray): Her örnek için çapraz entropi kaybı.
    """
    epsilon = 1e-15  # Log(0) hatasını önlemek için küçük bir değer
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss)


def caclulate_gradient(X_i: np.ndarray, y_i_true: int, y_i_pred: float) -> np.ndarray:
    """
    Tek bir örnek için çapraz entropi kaybının gradyanını hesaplar.
    Args:
        X_i (np.ndarray): Özellik matrisi.
        y_i_true (int): Gerçek etiket.
        y_i_pred (float): Tahmin edilen olasılık.
    Returns:
        gradient (np.ndarray): Gradyan vektörü.
    """
    gradient = (y_i_pred - y_i_true) * X_i
    return gradient


def update_weights(
    w: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    """
    Stochastic Gradient Descent (SGD) ile ağırlık günceller.
    Args:
        w (np.ndarray): Mevcut ağırlıklar.
        gradient (np.ndarray): Gradyan vektörü.
        learning_rate (float): Öğrenme oranı.
    Returns:
        w_updated (np.ndarray): Güncellenmiş ağırlıklar.
    """
    w_updated = w - learning_rate * gradient
    return w_updated


def initialize_weights(n_features: int) -> np.ndarray:
    """
    Ağırlıkları -0.01 ile 0.01 arasında rastgele başlatır.
    Args:
        n_features (int): Özellik sayısı.
    Returns:
        w (np.ndarray): Başlatılmış ağırlık vektörü.
    """
    w = np.random.uniform(-0.01, 0.01, size=n_features)
    return w
