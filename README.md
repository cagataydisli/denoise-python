# denoise-python

**Speech Denoising with Full- and Sub-band LSTM Network**

---

## 🚀 Projenin Amacı

Bu proje, konuşma sinyallerindeki gürültüyü kaldırmak için **FullSubNet** (Full- and Sub-band LSTM) tabanlı bir derin öğrenme modelini PyTorch ile uygulamaktadır.
Matlab'da geliştirilmiş prototipin Python’a çevrimi; veri hazırlama (STFT tabanlı mask üretimi), model mimarisi, eğitim ve çıkarım (inference) adımlarını içerir. Ayrıca **veri artırma** (data augmentation) olarak:

* **Rastgele SNR** (Signal-to-Noise Ratio)
* **Mixture of Noises** (farklı gürültü kaynaklarını karıştırma)

yaklaşımları, modelin genelleme performansını artırmak için kullanılmıştır.

---

## 📁 Proje Dizini (Klasör Yapısı)

```
denoise-python/
├─ .gitignore
├─ README.md
├─ train.py                 # Eğitim (training) scripti
├─ inference.py             # Bütün uzunlukta ses dosyaları için çıkarım (inference) scripti
├─ models/
│   └─ denoise_model.py     # FullSubNet mimarisi ve yardımcı katmanlar
├─ utils/
│   ├─ ci_mask.py           # cIRM (Complex Ideal Ratio Mask) hesaplama fonksiyonu
│   ├─ speech_features.py   # DNSDataset: STFT + mask üretimi + veri artırma
│   ├─ generate_synthetic_data.py  # Sentetik temiz/gaussian gürültü verisi üretimi örneği
│   ├─ split_audio.py       # Uzun WAV dosyalarını eşit parçalara (chunks) bölme
│   └─ resample_and_pad.py  # Ses dosyalarını 8 kHz’e düşürme ve pad/truncate işlemleri
├─ data/
│   ├─ clean_fullband/      # Temiz konuşma sinyalleri (WAV)
│   ├─ noise_fullband/      # Gürültü sinyalleri (WAV)
│   └─ test/                # Eğitim dışında kullanılan test WAV dosyaları
├─ checkpoints/             # Küçük çaplı eğitim için kaydedilen modeller
├─ checkpoints_big/         # Büyük veri setiyle eğitimde kaydedilen modeller
└─ output/                  # İnference sonucu oluşturulan denoised WAV dosyaları
```

> **Not:**
>
> * `data/clean_fullband/` altına 8 kHz’e yeniden örneklenmiş (resample) temiz konuşma dosyalarınız olmalıdır.
> * `data/noise_fullband/` altına 8 kHz’e düşürülmüş gürültü WAV’leri yerleştirilir.
> * `data/test/` klasöründe, çıkarım (inference) aşaması için gürültülü veya uzun test WAV’ler bulunur.

---

## 📦 Bağımlılıklar (Dependencies)

Aşağıdaki paketler ve araçlar önceden kurulmuş olmalıdır:

* Python ≥ 3.8
* PyTorch ≥ 1.9 (CUDA destekli: `torch.cuda.is_available() == True`)
* torchaudio
* numpy
* soundfile (torchaudio arka ucu olarak)
* Git (proje yönetimi için)

```bash
# Örnek conda kurulumu:
conda create -n denoise-env python=3.9
conda activate denoise-env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy soundfile
```

---

## 📊 Model Mimarisi: FullSubNet

```
FullSubNet:
  ├─ Input: [B, 1, F=257, T=time_steps]
  ├─ Full-Band Bölümü:
  │   ├─ PadLayer: T → T+2
  │   ├─ ‣ LSTM1 (num_hidden_fb)
  │   ├─ ‣ LSTM2 (num_hidden_fb)
  │   ├─ ‣ FullyConnected(Freq → F)
  │   └─ ‣ ReLU + Unsqueeze(2) → [B, F=257, 1, T+2]
  │
  ├─ Sub-Band Bölümü:
  │   ├─ UnfoldLayer: Sliding window (±15 frekans komşusu) → [B, F=257, 31, T]
  │   ├─ Pad (time eksenine 1er örnek ekle) → T+2
  │   ├─ reshape→[B*F, T+2, 31]
  │   ├─ LSTM3 (num_hidden_sb)
  │   ├─ LSTM4 (num_hidden_sb)
  │   ├─ FC2 (→ 2 real/imag mask boyutu)
  │   └─ RelabelLayer & FinalLayer → [B, 2, 257, T]
  │
  └─ Çıktı: [B, 2, 257, T] (real+imag cIRM mask)
```

* **F** = 257 (1-257 arası tek taraflı FFT frekans sayısı)
* **T** = `time_steps` (örneğin 92 frame)
* **num\_hidden\_fb** = 768 (Full-band LSTM gizli boyutu)
* **num\_hidden\_sb** = 512 (Sub-band LSTM gizli boyutu)
* **--time\_steps** parametresi, STFT çıktı zaman çerçevesi sayısıdır. İnference ve eğitimde bu aynı değerle çalışmalıdır.

Detaylı katmanlar ve ileri-besleme (**forward**) akışı, `models/denoise_model.py` dosyasında Python sınıfı olarak tanımlanmıştır.

---

## 🔄 Veri Hazırlama ve Data Augmentation

### 1. Temiz / Gürültü Dosyalarının Hazırlanması

* `data/clean_fullband/` klasörüne konuşma WAV dosyalarınızı koyun. Tüm dosyalar 8 kHz olmalı (eğer 48 kHz ise `utils/resample_and_pad.py` ile 8 kHz’e düşürebilirsiniz).
* `data/noise_fullband/` klasöründe çeşitli gürültü WAV dosyaları bulunsun (örneğin: trafik, vantilatör, yağmur, kalabalık vb.). Yine 8 kHz’e resample edildiklerinden emin olun.

### 2. Chunk’lama (Split Audio)

Eğer elinizde uzun tek bir WAV varsa (örneğin 1 saatlik kayıt), otomatik olarak 3 saniyelik (24 000 örnek) parçalara bölmek için:

```bash
python utils/split_audio.py --input_dir data/clean_fullband --output_dir data/clean_fullband
python utils/split_audio.py --input_dir data/noise_fullband --output_dir data/noise_fullband
```

* Bu script, dosyaları `*_chunk01.wav`, `*_chunk02.wav` … şeklinde 3 saniyelik segmentlere böler.
* Böylece hem clean hem de noise dosyalarınız doğrudan 3 s/24 000 örnek uzunluğunda olur.

### 3. Sentetik Veri Üretme (Opsiyonel)

Sabit frekanslı (sinüs) işaretler veya Gaussian gürültü gibi basit örnekleri `utils/generate_synthetic_data.py` scriptiyle oluşturabilirsiniz:

```bash
python utils/generate_synthetic_data.py
```

* Çıktıda `data/clean_fullband/sine_200Hz.wav`, `sine_400Hz.wav`, … gibi sinüs sinyalleri
* `data/noise_fullband/noise_1.wav`, … gibi beyaz gürültü dosyaları elde edilir.
* Bu basit veriler, modelin ilk prototip eğitimleri veya test amaçlı kullanılabilir.

### 4. DNSDataset İçinde Veri Artırma (Data Augmentation)

`speech_features.py` dosyasındaki `DNSDataset` sınıfında iki önemli augmentation adımı bulunur:

1. **Mixture of Noises**

   * Her `__getitem__` çağrısında rastgele iki farklı noise dosyası seçilir (`random.sample`)
   * Bu ikisinin eşit ağırlıklı ortalaması alınır:

     ```python
     combined_noise_np = (noise_seg1 + noise_seg2) / 2.0
     ```

2. **Rastgele SNR**

   * Seçilebilecek dB değerleri: `[-5, 0, 5, 10]` (örnek)
   * Rastgele bir `snr_db` seçilir, lineer ölçeğe çevrilir:

     ```python
     snr_lin = 10 ** (snr_db / 10)
     ```
   * “clean” ve “combined\_noise” güçleri hesaplanır ve `noise_scaled_np = combined_noise_np * sqrt(clean_power / (noise_power * snr_lin))` formülüyle ölçeklenir.
   * Son olarak `noisy_np = clean_np + noise_scaled_np` oluşturulur.

Böylece her eğitim adımı (batch) için farklı SNR seviyesinde, farklı gürültü karışımları kullanılarak zengin bir veri kümesi sunulur.

---

## 🏗️ Dataset Yapısı

```
data/
├─ clean_fullband/
│   ├─ 001_chunk01.wav
│   ├─ 001_chunk02.wav
│   ├─ 002_chunk01.wav
│   └─ … (3 saniyelik temiz parçalara bölünmüş WAV dosyaları)
│
├─ noise_fullband/
│   ├─ 004_chunk01.wav
│   ├─ 004_chunk02.wav
│   ├─ 005_chunk01.wav
│   └─ … (3 saniyelik gürültü parçalara bölünmüş WAV dosyaları)
│
└─ test/
    ├─ test_003.wav          # 3 sn’lik test dosyası
    └─ test_long_003.wav     # >3 sn: inference için örnek uzun WAV
```

* `clean_fullband/` ve `noise_fullband/` içindeki her dosya 8000 Hz, 16 bit, tek kanallı (mono), **3 saniye** (24 000 örnek) uzunluğundadır.
* Eğitim: her `__getitem__` çağrısında `i` indeksli clean vs. rastgele seçilen 2 noise birleştirilir.
* Test: `data/test/` altında hem kısa (3 saniye) hem de uzun (örneğin 10 saniye) dosyalar bulunur. Uzun dosya, “arbitrary length inference” (isteğe bağlı uzunlukta inference) koduyla parçalara bölünerek işlenir.

---

## 📈 Model Eğitimi (Training)

### Küçük Ölçekli Eğitim

```bash
python train.py \
  --data_dir data \
  --batch_size 8 \
  --epochs 15 \
  --lr 5e-4 \
  --time_steps 92 \
  --checkpoint_dir checkpoints
```

* `--data_dir data`
* `--batch_size 8`
* `--epochs 15`
* `--lr 5e-4`
* `--time_steps 92` (3 saniye için \~92 STFT frame)
* `--checkpoint_dir checkpoints`

### Büyük Ölçekli Eğitim (Data Augmentation + Uzun Epoch)

```bash
python train.py \
  --data_dir data \
  --batch_size 16 \
  --epochs 100 \
  --lr 5e-4 \
  --time_steps 92 \
  --checkpoint_dir checkpoints_big \
  --log_interval 10 \
  --patience 10
```

* `--batch_size 16` (GPU belleğine bağlı olarak ayarlanabilir)
* `--epochs 100`
* `--log_interval 10` (Her 10 adımda bir validation raporu)
* `--patience 10` (Validation loss 10 epoch iyileşmediyse erken durdurma)
* En düşük validation loss için `checkpoints_big/best_epoch_{i}.pth` dosyası kaydedilir.

---

## 🎯 Çıkarım (Inference)

### Kısa (3 s) Test Dosyası

```bash
python inference.py \
  --model_path checkpoints_big/best_epoch_11.pth \
  --input_wav data/test/test_003.wav \
  --output_wav output/denoised_test_003.wav \
  --sample_rate 8000 \
  --time_steps 92
```

* `--model_path`: Eğitilmiş model ağırlığı (`.pth`).
* `--input_wav`: Gürültülü test WAV (tam 3 saniye).
* `--output_wav`: Denoise edilmiş WAV’ın kaydedileceği yol.
* `--sample_rate 8000`, `--time_steps 92` (eğitimle aynı STFT frame sayısı).

### Uzun (Arbitrary-Length) Test Dosyası

```bash
python inference.py \
  --model_path checkpoints_big/best_epoch_11.pth \
  --input_wav data/test/test_long_003.wav \
  --output_wav output/denoised_test_long_003.wav \
  --sample_rate 8000 \
  --time_steps 92
```

* Uzun sinyali, modelin kabul ettiği `time_steps` (92 frame) boyutunda parçalara bölerek her parça için çıkarım yapar.
* Ardından “overlap-add” yöntemiyle parçaları birleştirip tek bir denoised çıkış üretir.

---

## 🔧 Parametre Açıklamaları

| Argüman            | Açıklama                                                                   |
| ------------------ | -------------------------------------------------------------------------- |
| `--data_dir`       | `clean_fullband/` ve `noise_fullband/` altındaki verilerin kök klasörü.    |
| `--batch_size`     | GPU’daki bellek durumuna göre ayarlanabilir.                               |
| `--epochs`         | Maksimum eğitim döngüsü sayısı.                                            |
| `--lr`             | Başlangıç öğrenme oranı (learning rate).                                   |
| `--time_steps`     | STFT’ın zaman ekseni çerçeve sayısı (3 s için ≈ 92).                       |
| `--checkpoint_dir` | Eğitim sırasında kaydedilecek model ağırlıklarının dizini.                 |
| `--log_interval`   | Kaç iteration’da bir validation raporu verileceği (training script’te).    |
| `--patience`       | Validation loss belirli epoch sayısı boyunca iyileşmediyse erken durdurma. |

---

## 📜 Lisans ve Katkıda Bulunma

* Bu proje, eğitim ve araştırma amaçlı **MIT Lisansı** altında yayınlanmıştır.
* Katkıda bulunmak, hata bildirmek veya iyileştirme önerileri sunmak için GitHub Issues sayfasını kullanabilirsiniz.
