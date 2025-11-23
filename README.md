# Makine Öğrenmesi Ödev 1: Lojistik Regresyon

Bu proje, **NumPy** kullanarak sıfırdan lojistik regresyon algoritmasını uygulayan eksiksiz bir makine öğrenmesi pipeline'ıdır. Proje, veri ön işleme, model eğitimi, değerlendirme ve görselleştirme adımlarını içerir.

## 📋 İçindekiler

[Proje Yapısı](#proje-yapısı)
[Kurulum](#kurulum)
[Kullanım](#kullanım)
[Modüllerin Detaylı Açıklaması](#modüllerin-detaylı-açıklaması)
[Algoritma Detayları](#algoritma-detayları)
[Sonuçlar](#sonuçlar)
[Notlar](#notlar)

## 🗂️ Proje Yapısı

```
makine-ogrenmesi-hw1/
│
├── data/                            # Veri setleri
│   ├── hw1Data.txt                  # Ham veri (101 örnek, 2 özellik, 1 etiket)
│   ├── raw_train.npz                # Ham eğitim verisi (%60)
│   ├── raw_val.npz                  # Ham doğrulama verisi (%20)
│   ├── raw_test.npz                 # Ham test verisi (%20)
│   ├── normalized_train.npz         # Normalize edilmiş eğitim verisi
│   ├── normalized_val.npz           # Normalize edilmiş doğrulama verisi
│   └── normalized_test.npz          # Normalize edilmiş test verisi
│
├── docs/                            # Dokümanlar
│   └── ML2025Hw1.pdf                # Ödev açıklaması ve talimatlar
│
├── results/                         # Sonuçlar ve çıktılar
│   ├── evaluation/                  # Değerlendirme sonuçları
│   │   └── test_results.txt         # Test seti metrik sonuçları
│   ├── graphs/                      # Grafikler
│   │   ├── loss_curve.png           # Eğitim/doğrulama kayıp grafiği
│   │   ├── test_decision_boundary.png # Test verisi karar sınırı grafiği
│   │   ├── train_decision_boundary.png # Eğitim verisi karar sınırı grafiği
│   │   ├── tüm_scatter_plot.png     # Tüm verinin scatter plot grafiği
│   │   ├── train_scatter_plot.png   # Eğitim verisinin scatter plot grafiği
│   │   └── val_decision_boundary.png   # Doğrulama verisi karar sınırı grafiği
│   ├── logs/                        # Eğitim logları
│   │   └── training.log             # Epoch bazlı eğitim logları
│   └── model/                       # Eğitilmiş model ağırlıkları
│       ├── model_weights_*.npy      # Zaman damgalı model dosyaları
│       └── model_weights_latest.npy # En son eğitilmiş model
│
├── src/                             # Kaynak kod
│   ├── dataset.py                   # Veri yükleme ve ön işleme
│   ├── model.py                     # Lojistik regresyon modeli
│   ├── train.py                     # Model eğitimi
│   ├── eval.py                      # Model değerlendirme
│   ├── metrics.py                   # Değerlendirme metrikleri
│   ├── logger.py                    # Birleşik loglama sistemi
│   └── utils.py                     # Yardımcı fonksiyonlar
│
├── requirements.txt                 # Gerekli Python kütüphaneleri
└── README.md                        # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.7+
- NumPy
- Matplotlib

### Adımlar

1. Repoyu klonlayın veya indirin:
```bash
git clone https://github.com/ysntrkc/machine-learning-hw1.git
cd makine-ogrenmesi-hw1
```

2. (Opsiyonel) Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

### 1. Model Eğitimi

Modeli varsayılan parametrelerle eğitmek için:

```bash
cd src
python train.py
```

**Komut Satırı Argümanları:**

```bash
python train.py [-lr LEARNING_RATE] [-e EPOCHS] [-p PATIENCE] [-d MIN_DELTA] [--no_early_stopping] [-l LOG_MODE]
```

- `-lr, --learning_rate`: Öğrenme oranı (varsayılan: 0.01)
- `-e, --epochs`: Maksimum epoch sayısı (varsayılan: 500)
- `-p, --patience`: Early stopping patience - iyileşme olmadan beklenecek epoch sayısı (varsayılan: 5)
- `-d, --min_delta`: Early stopping minimum delta - iyileşme olarak kabul edilecek minimum değişim (varsayılan: 0.001)
- `--no_early_stopping`: Early stopping'i devre dışı bırak
- `-l, --log`: Log modu (varsayılan: both)
  - `both`: Konsol ve dosyaya loglama
  - `console`: Sadece konsola loglama
  - `file`: Sadece dosyaya loglama

**Örnek Kullanım:**

```bash
# Varsayılan parametrelerle eğitim (early stopping aktif)
python train.py

# Özel learning rate ve epoch sayısı
python train.py -lr 0.001 -e 200

# Early stopping parametrelerini özelleştirme
python train.py -p 15 -d 0.0005

# Early stopping'i devre dışı bırakma
python train.py --no_early_stopping

# Sadece konsola loglama
python train.py -l console

# Tüm parametrelerle
python train.py -lr 0.005 -e 150 -p 20 -d 0.0001 -l file
```

Bu komut:
- Veriyi yükler ve normalize eder
- Train/val/test setlerine ayırır (%60/%20/%20)
- Scatter plot grafikleri oluşturur (tüm veri ve eğitim verisi)
- Belirtilen epoch sayısı boyunca SGD ile modeli eğitir
- **Early stopping** ile eğitimi izler:
  - Validation loss izlenir
  - Belirlenen patience süresi boyunca iyileşme olmazsa eğitim durdurulur
  - En iyi validation loss'a sahip model ağırlıkları saklanır
  - Early stopping tetiklendiğinde en iyi ağırlıklar geri yüklenir
- Eğitim ilerlemesini konsola ve/veya dosyaya loglar
- Kayıp grafiğini oluşturur (`results/graphs/loss_curve.png`)
- **Karar sınırı grafiklerini oluşturur** (`train_decision_boundary.png`, `val_decision_boundary.png`)
- Model ağırlıklarını iki versiyonda kaydeder:
  - Timestamp'li versiyon: `model_weights_YYYYMMDD_HHMMSS.npy`
  - Son model: `model_weights_latest.npy`
- Eğitim parametrelerini kaydeder (`results/model/training_params.json`)

### 2. Model Değerlendirme

Eğitilmiş modeli test setinde değerlendirmek için:

```bash
python eval.py
```

**Komut Satırı Argümanları:**

```bash
python eval.py [-l LOG_MODE]
```

- `-l, --log`: Log modu (varsayılan: both)
  - `both`: Konsol ve dosyaya loglama
  - `console`: Sadece konsola loglama
  - `file`: Sadece dosyaya loglama

**Örnek Kullanım:**

```bash
# Varsayılan (konsol ve dosyaya)
python eval.py

# Sadece konsola yazdırma
python eval.py -l console

# Sadece dosyaya kaydetme
python eval.py -l file
```

Bu komut şu metrikleri yazdırır:
- **Eğitim Parametreleri**: Modelin eğitildiği parametreler (learning rate, epochs, early stopping bilgileri)
- **Loss (Kayıp)**: Cross-entropy loss
- **Accuracy (Doğruluk)**: Genel doğru tahmin oranı
- **Precision (Kesinlik)**: Pozitif tahminlerin doğruluk oranı
- **Recall (Duyarlılık)**: Gerçek pozitifleri bulma oranı
- **F1 Score**: Precision ve recall'ın harmonik ortalaması
- **Confusion Matrix**: Detaylı tablo formatında confusion matrix
- **Karar Sınırı Grafiği**: Test verisi üzerinde model karar sınırı (`test_decision_boundary.png`)

### 3. Veri Hazırlama (Opsiyonel)

Sadece veri ön işleme yapmak için:

```bash
python dataset.py
```

## 📚 Modüllerin Detaylı Açıklaması

### 1. `dataset.py` - Veri İşleme Modülü

Bu modül, veri yükleme, normalizasyon ve bölme işlemlerini gerçekleştirir.

#### Fonksiyonlar:

**`load_data(path)`**
- Ham veriyi TXT dosyasından ',' ile ayırır ve numPy dizisi olarak yükler
- İlk iki sütun özellikler (features), üçüncü sütun etiket (label)
- 101 örnek, 2 özellik, ikili sınıflandırma (0/1)

**`normalize_features(X)`**
- Min-Max normalizasyonu uygular: `(X - min) / (max - min)`
- Her özelliği [0, 1] aralığına ölçekler
- Bölme hatasını önlemek için özel kontrol içerir

**`split_data(X, y, train_ratio=0.6, val_ratio=0.2)`**
- Veriyi **sıralı olarak** üç sete böler:
  - Eğitim: İlk %60 (60 örnek)
  - Doğrulama: Sonraki %20 (20 örnek)
  - Test: Son %20 (21 örnek)
- **Not**: Random shuffle yapılmaz, veri sıralı bölünür

**`save_splits(prefix, train_data, val_data, test_data)`**
- Train/val/test setlerini `.npz` formatında sıkıştırılmış olarak kaydeder
- Her dosyada `X` (features) ve `y` (labels) arrays bulunur

**`prepare_and_save_data()`**
- Ana veri hazırlama pipeline'ı
- Hem ham hem de normalize edilmiş versiyonları kaydeder
- Scatter plot grafikleri oluşturur:
  - Tüm verinin görselleştirmesi
  - Eğitim verisinin görselleştirmesi

### 2. `model.py` - Lojistik Regresyon Modeli

Lojistik regresyon algoritmasının implementasyonu.

#### Fonksiyonlar:

**`sigmoid(z)`**
```python
σ(z) = 1 / (1 + e^(-z))
```
- Aktivasyon fonksiyonu
- [-∞, +∞] aralığını [0, 1] olasılık aralığına dönüştürür

**`predict_probabilities(X, w)`**
```python
p = σ(X · w)
```
- Özellik matrisi ve ağırlıklar ile olasılık tahmini yapar
- Matris çarpımı sonrası sigmoid uygular

**`cross_entropy_loss(y_true, y_pred)`**
```python
L = -1/N Σ[y·log(p) + (1-y)·log(1-p)]
```
- İkili sınıflandırma için kayıp fonksiyonu
- `epsilon=1e-15` ile log(0) hatasını önler
- `np.mean` kullanarak batch size'dan bağımsız kayıp hesaplar

**`caclulate_gradient(X_i, y_i_true, y_i_pred)`**
```python
∇L = (p - y) · X
```
- Tek bir örnek için gradyan hesaplar
- SGD için gerekli türev

**`update_weights(w, gradient, learning_rate)`**
```python
w_new = w - η · ∇L
```
- Ağırlıkları gradyan descent ile günceller
- η (eta): öğrenme oranı

**`initialize_weights(n_features)`**
- Ağırlıkları [-0.01, 0.01] aralığında rastgele başlatır
- Küçük değerler ile başlamak eğitim stabilitesini artırır

### 3. `train.py` - Model Eğitimi

Lojistik regresyon modelini Stochastic Gradient Descent (SGD) ile eğitir.

#### Ana Fonksiyon: `load_training_data`

**`load_training_data(path_prefix='../data/normalized')`**
- Normalize edilmiş eğitim ve doğrulama verilerini yükler
- **Hata kontrolü**: Eğer veri dosyaları bulunamazsa `FileNotFoundError` fırlatır
- Kullanıcıya önce veri hazırlamasını söyleyen açıklayıcı hata mesajı

#### Ana Fonksiyon: `train_logistic_regression`

**Parametreler:**
- `learning_rate=0.01`: Öğrenme oranı
- `n_epochs=500`: Epoch sayısı
- `patience=5`: Early stopping patience - iyileşme olmadan beklenecek epoch sayısı
- `min_delta=0.001`: Early stopping için minimum iyileşme
- `early_stopping=True`: Early stopping'i etkinleştirir/devre dışı bırakır

**SGD Algoritması:**
```
Her epoch için:
    Her örnek için (tek tek):
        1. Forward pass: tahmin yap
        2. Loss hesapla
        3. Gradyan hesapla
        4. Ağırlıkları güncelle
    Epoch sonu:
        1. Ortalama train loss hesapla
        2. Tüm val seti ile val loss hesapla
        3. Early stopping kontrolü yap
```

**Özellikler:**
- **Bias Term**: Özellik matrisine otomatik bias sütunu eklenir (1'lerden oluşan)
- **Batch-by-Batch**: Her örnek tek tek işlenir (true SGD)
- **Dual Tracking**: Hem eğitim hem doğrulama kaybı kaydedilir
- **Progress Monitoring**: Her epoch'ta kayıplar yazdırılır

#### `add_bias_term(X)`
```python
X_bias = [1, x1, x2, ..., xn]  # Her satıra 1 eklenir
```
- Bias terimi ekler (w0 için)
- n_features → n_features + 1

### 4. `eval.py` - Model Değerlendirme

Eğitilmiş modeli test verisinde değerlendirir.

#### Ana Fonksiyon: `evaluate_model`

**Değerlendirme Adımları:**
1. Olasılık tahminleri yap
2. Threshold=0.5 ile ikili sınıf tahmini yap
3. Tüm metrikleri hesapla

**Dönen Metrikler:**
- Loss
- Accuracy
- Precision
- Recall
- F1 Score

### 5. `metrics.py` - Performans Metrikleri

Sınıflandırma performans metriklerini hesaplar.

#### Confusion Matrix

```
            Gerçek Değer
             1       0
Tahmin  1    TP      FN
Edilen  0    FP      TN
```

**`confusion_matrix(y_true, y_pred)`**
- True Positive (TP): Doğru pozitif tahminler
- True Negative (TN): Doğru negatif tahminler
- False Positive (FP): Yanlış pozitif tahminler (Type I error)
- False Negative (FN): Yanlış negatif tahminler (Type II error)

#### Metrikler

**`accuracy(y_true=None, y_pred=None, conf_matrix=None)`**
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Genel doğruluk oranı
- Tüm doğru tahminlerin oranı
- **İki kullanım şekli:**
  1. `y_true` ve `y_pred` vererek: Otomatik confusion matrix hesaplar
  2. `conf_matrix` vererek: Önceden hesaplanmış confusion matrix kullanır (daha verimli)

**`precision(y_true=None, y_pred=None, conf_matrix=None)`**
```python
Precision = TP / (TP + FP)
```
- Pozitif tahminlerin ne kadarı doğru
- "Tahmin ettiğim pozitiflerin güvenilirliği"
- **İki kullanım şekli:**
  1. `y_true` ve `y_pred` vererek
  2. `conf_matrix` vererek (daha verimli)

**`recall(y_true=None, y_pred=None, conf_matrix=None)`**
```python
Recall = TP / (TP + FN)
```
- Gerçek pozitiflerin ne kadarını bulduk
- "Tüm pozitifleri bulma yeteneğim"
- **İki kullanım şekli:**
  1. `y_true` ve `y_pred` vererek
  2. `conf_matrix` vererek (daha verimli)

**`f1_score(y_true=None, y_pred=None, conf_matrix=None)`**
```python
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Precision ve Recall'ın harmonik ortalaması
- Dengesiz veri setlerinde daha bilgilendirici
- **İki kullanım şekli:**
  1. `y_true` ve `y_pred` vererek
  2. `conf_matrix` vererek (daha verimli)

**Özel Durumlar:**
- Tüm fonksiyonlar division by zero kontrolü içerir
- Tanımsız durumlarda 0.0 döner
- `conf_matrix` parametresi kullanıldığında daha verimli çalışır (confusion matrix'i tekrar hesaplamaz)

**Kullanım Örneği:**
```python
# Metod 1: y_true ve y_pred ile
acc = accuracy(y_true, y_pred)

# Metod 2: Önceden hesaplanmış confusion matrix ile (daha verimli)
conf_mat = confusion_matrix(y_true, y_pred)
acc = accuracy(conf_matrix=conf_mat)
prec = precision(conf_matrix=conf_mat)
rec = recall(conf_matrix=conf_mat)
f1 = f1_score(conf_matrix=conf_mat)
```

### 6. `utils.py` - Yardımcı Fonksiyonlar

Görselleştirme ve dosya yönetimi fonksiyonları.

#### `ensure_dir_exists(directory)`
- Dizin yoksa oluşturur
- `os.makedirs()` ile recursive oluşturma

#### `plot_scatter(X, y, data='tüm', save_path='../results/graphs/')`
- Veriyi 2D scatter plot olarak çizer
- İki sınıfı farklı renklerle gösterir:
  - **Kalanlar (Class 0)**: Kırmızı 'x' - Sınavdan kalan adaylar
  - **Geçenler (Class 1)**: Mavi 'o' - Sınavdan geçen adaylar
- Eksen etiketleri: "Sınav 1" ve "Sınav 2"
- Bias sütununu otomatik atlar
- Grafik dosya adı: `{data}_scatter_plot.png`
- Varsayılan kayıt yolu: `../results/graphs/`

#### `plot_loss_curve(train_losses, val_losses, save_path='../results/graphs/')`
- Eğitim ve doğrulama kayıplarını epoch'a göre çizer
- Overfitting/underfitting tespiti için kritik
- İki eğriyi aynı grafikte gösterir
- Grafik dosya adı: `loss_curve.png`
- Varsayılan kayıt yolu: `../results/graphs/`

#### `plot_decision_boundary(X_normalized, y, weights, X_raw, data='test', save_path='../results/graphs/')`
- Veri noktalarını ve lojistik regresyon karar sınırını birlikte çizer
- **Orijinal (normalize edilmemiş) değerleri kullanır** - daha anlaşılır görselleştirme
- **Karar Sınırı Hesaplama:**
  - Model normalize edilmiş verilerle eğitilir: `w0 + w1*x1_norm + w2*x2_norm = 0`
  - Karar sınırı orijinal ölçeğe dönüştürülür
  - Bu doğru, sigmoid fonksiyonunun 0.5 değerini aldığı noktaları gösterir
  - Doğrunun üstündeki noktalar Class 1, altındakiler Class 0 olarak tahmin edilir
- **Parametreler:**
  - `X_normalized`: Normalize edilmiş özellik matrisi (bias terimi içerebilir)
  - `y`: Gerçek etiketler
  - `weights`: Model ağırlıkları [w0 (bias), w1, w2]
  - `X_raw`: Ham (normalize edilmemiş) özellik matrisi
  - `data`: Veri seti türü (grafik başlığı için)
- **Görselleştirme:**
  - Kırmızı 'x': Kalanlar (Class 0)
  - Mavi 'o': Geçenler (Class 1)
  - Yeşil çizgi: Karar sınırı (Decision Boundary)
  - Eksen etiketleri: "Sınav 1" ve "Sınav 2" (orijinal değerler)
- Grafik dosya adı: `{data}_decision_boundary.png`
- Yüksek çözünürlük (150 DPI)
- Eğitim sonrası otomatik olarak train, val ve test setleri için oluşturulur

#### `save_weights(w, save_dir='../results/model/')`
- Model ağırlıklarını `.npy` formatında kaydeder
- **İki ayrı dosya olarak kaydeder**:
  1. Timestamp ile isimlendirilen versiyon: `model_weights_YYYYMMDD_HHMMSS.npy`
  2. En son model: `model_weights_latest.npy` (her eğitimde üzerine yazılır)
- Varsayılan kayıt yolu: `../results/model/`
- Timestamp'li versiyon farklı eğitimleri karıştırmadan saklar

#### `parse_training_args()`
- Komut satırı argümanlarını parse eder
- Desteklenen argümanlar:
  - `-lr, --learning_rate`: Öğrenme oranı (float, varsayılan: 0.01)
  - `-e, --epochs`: Maksimum epoch sayısı (int, varsayılan: 100)
  - `-p, --patience`: Early stopping patience (int, varsayılan: 10)
  - `-d, --min_delta`: Early stopping minimum delta (float, varsayılan: 0.0001)
  - `--no_early_stopping`: Early stopping'i devre dışı bırak (flag)
  - `-l, --log`: Log modu (str, varsayılan: "both")
- `argparse.Namespace` objesi döndürür

#### `print_training_config(learning_rate, n_epochs, patience, min_delta, early_stopping_enabled)`
- Eğitim konfigürasyonunu formatlı şekilde ekrana yazdırır
- Gösterilen bilgiler:
  - Learning rate
  - Epoch sayısı
  - Early stopping durumu (aktif/devre dışı)
  - Early stopping parametreleri (patience, min_delta)
- Eğitim başlamadan önce çağrılır

#### `save_training_params(learning_rate, n_epochs, actual_epochs, patience, min_delta, early_stopping_enabled, early_stopped, save_file='../results/model/training_params.json')`
- Eğitim parametrelerini JSON formatında kaydeder
- Kaydedilen bilgiler:
  - `learning_rate`: Öğrenme oranı
  - `max_epochs`: Maksimum epoch sayısı
  - `actual_epochs`: Gerçekleşen epoch sayısı
  - `early_stopping_enabled`: Early stopping kullanıldı mı
  - `early_stopped`: Early stopping tetiklendi mi
  - `patience`: Early stopping patience
  - `min_delta`: Early stopping minimum delta
  - `timestamp`: Eğitim tarihi ve saati
- Evaluation sırasında bu parametreler otomatik olarak gösterilir

#### `load_training_params(load_file='../results/model/training_params.json')`
- Kaydedilmiş eğitim parametrelerini yükler
- JSON dosyasını okur ve dictionary döndürür
- Dosya yoksa `None` döndürür
- `eval.py` tarafından test sonuçlarını gösterirken kullanılır

#### `print_confusion_matrix(conf_matrix)`
- Confusion matrix'i tablo formatında görselleştirir
- TP, TN, FP, FN değerlerini gösterir
- Özet bilgiler:
  - Toplam örnek sayısı
  - Gerçek pozitif/negatif sayıları
  - Tahmin pozitif/negatif sayıları
- Kullanıcı dostu tablo formatı

#### `log_test_results(results, log_file='../results/evaluation/test_results.txt')`
- Test sonuçlarını dosyaya kaydeder
- Timestamp ile birlikte kaydedilir
- Tüm metrikleri (loss, accuracy, precision, recall, f1_score) içerir

### 7. `logger.py` - Birleşik Loglama Sistemi

Proje genelinde birleşik loglama sağlayan modül. Konsola, dosyaya veya her ikisine birden loglama yapabilir.

#### `Logger` Sınıfı

**`__init__(log_file='../results/logs/training.log', mode='both')`**
- Loglama sistemi için ana sınıf
- **Parametreler:**
  - `log_file`: Log dosyasının yolu
  - `mode`: Loglama modu
    - `"both"`: Hem konsol hem dosya
    - `"console"`: Sadece konsol
    - `"file"`: Sadece dosya
- Context manager destekler (`with` statement)

**`log(message, end='\n')`**
- Mesajı seçilen moda göre loglar
- `print()` gibi çalışır ama dosyaya da yazar
- Otomatik flush ile anında yazma

**`close()`**
- Log dosyasını kapatır
- Kaynakları temizler

#### Yardımcı Fonksiyonlar:

**`setup_logger(log_file='../results/logs/training.log', mode='both')`**
- Global logger instance'ı oluşturur ve yapılandırır
- Önceki logger varsa kapatır ve yenisini oluşturur
- Train ve eval modülleri tarafından kullanılır

**`get_logger()`**
- Global logger instance'ını döndürür
- Yoksa otomatik olarak oluşturur

**`log(message, end='\n')`**
- Kolaylık fonksiyonu
- Global logger'ı kullanarak mesaj loglar
- Tüm modüllerde `from logger import log` ile import edilir

**Kullanım Örneği:**
```python
from logger import setup_logger, log

# Logger'ı yapılandır
setup_logger(mode='both')

# Log kullan
log("Training started")
log(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## 🧮 Algoritma Detayları

### Lojistik Regresyon Matematiği

#### 1. Hipotez Fonksiyonu
```
h(x) = σ(w^T · x) = 1 / (1 + e^(-w^T·x))
```

#### 2. Karar Kuralı
```
y_pred = 1  if h(x) ≥ 0.5
y_pred = 0  if h(x) < 0.5
```

#### 3. Kayıp Fonksiyonu (Cross-Entropy)
```
L(w) = -1/m Σ[y^(i) log(h(x^(i))) + (1-y^(i)) log(1-h(x^(i)))]
```

#### 4. Gradyan
```
∂L/∂w = 1/m Σ[(h(x^(i)) - y^(i)) · x^(i)]
```

#### 5. Güncelleme Kuralı (SGD)
```
w := w - η · (h(x^(i)) - y^(i)) · x^(i)
```

### Stochastic Gradient Descent (SGD)

Bu implementasyon **true SGD** kullanır:
- Her örnekte ağırlık güncellenir
- Mini-batch veya batch GD değil

### Early Stopping

**Overfitting'i önlemek** için validation loss bazlı early stopping kullanılır:

#### Parametreler:
- **patience**: İyileşme olmadan beklenecek epoch sayısı (varsayılan: 10)
- **min_delta**: İyileşme olarak kabul edilecek minimum değişim (varsayılan: 0.0001)

#### Algoritma:
```
Her epoch sonunda:
    Eğer (val_loss < best_val_loss - min_delta):
        best_val_loss = val_loss
        best_weights = current_weights
        epochs_no_improve = 0
    Değilse:
        epochs_no_improve += 1
    
    Eğer (epochs_no_improve >= patience):
        Eğitimi durdur
        best_weights'i geri yükle
```

#### Log Çıktısı:
```
Epoch  50/100 - Train Loss: 0.3245 - Val Loss: 0.3412 * - No Improve: 0
Epoch  60/100 - Train Loss: 0.3201 - Val Loss: 0.3445   - No Improve: 10

==================================================
Early stopping triggered at epoch 60
Best validation loss: 0.3412
Restoring best weights from epoch 50
==================================================
```

**Not:** `*` işareti validation loss'ta iyileşme olduğunu gösterir.

### Normalizasyon

**Min-Max Scaling** kullanılır:
```
X_norm = (X - X_min) / (X_max - X_min)
```

**Neden Normalizasyon?**
- Farklı ölçeklerdeki özellikleri eşitler
- Gradyan descent'i hızlandırır
- Sayısal stabiliteyi artırır
- Öğrenme oranı seçimini kolaylaştırır

## 📊 Sonuçlar

### Model Performansı

Model başarılı şekilde eğitilir ve şu metrikler hesaplanır:

- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin güvenilirliği
- **Recall**: Tüm pozitifleri yakalama oranı
- **F1 Score**: Precision ve recall dengesi

### Çıktı Dosyaları

1. **Scatter Plots** (`results/graphs/`)
   - `tüm_scatter_plot.png`: Tüm veri setinin görselleştirmesi
   - `train_scatter_plot.png`: Eğitim verisinin görselleştirmesi
   - Her sınıf farklı renk ve işaretle gösterilir
   - Eksenler: Sınav 1 ve Sınav 2 skorları

2. **Decision Boundary Plots** (`results/graphs/`)
   - `train_decision_boundary.png`: Eğitim verisi üzerinde karar sınırı
   - `val_decision_boundary.png`: Doğrulama verisi üzerinde karar sınırı
   - `test_decision_boundary.png`: Test verisi üzerinde karar sınırı
   - Yeşil çizgi: Lojistik regresyon karar sınırı (decision boundary)
   - Kırmızı 'x': Kalanlar (Class 0)
   - Mavi 'o': Geçenler (Class 1)
   - **Orijinal (normalize edilmemiş) Sınav 1 ve Sınav 2 skorları kullanılır**
   - Modelin sınıfları nasıl ayırdığını görsel olarak gösterir
   - Karar sınırı normalize edilmiş modelden hesaplanır ve orijinal ölçeğe dönüştürülür

3. **Loss Curve** (`results/graphs/loss_curve.png`)
   - Eğitim ve doğrulama kayıplarının epoch'a göre değişimi
   - Overfitting kontrolü için kullanılır
   - Mavi: Eğitim kaybı, Turuncu: Doğrulama kaybı

4. **Model Weights** (`results/model/`)
   - `model_weights_YYYYMMDD_HHMMSS.npy`: Timestamp'li versiyon
   - `model_weights_latest.npy`: En son eğitilmiş model
   - Her ikisi de `numpy.load()` ile yüklenebilir
   - Timestamp'li versiyon her çalıştırmada yeni dosya oluşturur
   - Latest versiyon her eğitimde güncellenir

5. **Training Parameters** (`results/model/training_params.json`)
   - Eğitim parametrelerini JSON formatında saklar
   - İçerik:
     - `learning_rate`: Öğrenme oranı
     - `max_epochs`: Maksimum epoch sayısı
     - `actual_epochs`: Gerçekleşen epoch sayısı
     - `early_stopping_enabled`: Early stopping kullanıldı mı
     - `early_stopped`: Early stopping tetiklendi mi
     - `patience`: Early stopping patience değeri
     - `min_delta`: Early stopping minimum delta değeri
     - `timestamp`: Eğitim zamanı
   - Test sonuçları yazdırılırken otomatik olarak gösterilir
   - Latest versiyon her eğitimde güncellenir

## 📝 Notlar

### Veri Seti Özellikleri
- **Toplam örnek**: 101
- **Özellik sayısı**: 2 (Sınav 1 ve Sınav 2 skorları)
- **Sınıf sayısı**: 2 (binary classification)
  - **Class 0**: Kalanlar (sınavdan geçemeyen adaylar)
  - **Class 1**: Geçenler (sınavdan geçen adaylar)
- **Format**: CSV (virgülle ayrılmış)
- **Split**: 60-20-20 (train-val-test)
- **Dosya yolları**: Göreli yollar kullanılır (`../data/`, `../results/`)

### Hiperparametreler
- **Learning Rate**: 0.01 (özelleştirilebilir: `-lr` flag)
- **Max Epochs**: 500 (özelleştirilebilir: `-e` flag)
- **Early Stopping**: Aktif (devre dışı bırakılabilir: `--no_early_stopping` flag)
  - **Patience**: 5 (özelleştirilebilir: `-p` flag)
  - **Min Delta**: 0.001 (özelleştirilebilir: `-d` flag)
- **Weight Initialization**: Uniform(-0.01, 0.01)
- **Threshold**: 0.5 (classification)
- **Epsilon**: 1e-15 (numerical stability)
