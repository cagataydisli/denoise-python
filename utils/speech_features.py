# utils/speech_features.py

import os
import random
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from .ci_mask import compute_complex_ideal_ratio_mask

def _check_activity(y: np.ndarray, threshold: float = 0.01) -> float:
    """
    MATLAB’daki check_activity fonksiyonunun eşdeğeri:
      percent_active = sum(abs(y) > threshold) / length(y)
    """
    return np.sum(np.abs(y) > threshold) / len(y)

class DNSDataset(Dataset):
    """
    DNS (Deep Noise Suppression) tarzı veri seti.
    Her örnek için:
      - 48 kHz’lik .wav dosyasını 8 kHz’e indirger (resample)
      - Rastgele bir gürültü dosyası seçer, 8 kHz’e düşürür.
      - Her ikisini de 3 saniyeye (3*8000=24000 örnek) eşitler.
      - SNR’a göre gürültüyü ölçekler, clean + noise → noisyAudio.
      - STFT alıp, noisy magnitude (predictor) ve cIRM (target) üretir.
    """
    def __init__(self,
                 clean_dir: str,
                 noise_dir: str,
                 sample_rate: int = 8000,
                 expected_length: int = 3,   # saniye cinsinden
                 window_length: int = 512,
                 overlap: int = 512 - 256,
                 n_fft: int = 512):
        super(DNSDataset, self).__init__()
        self.clean_files = []
        self.noise_files = []
        self.sample_rate = sample_rate
        self.expected_samples = expected_length * sample_rate
        self.window_length = window_length
        self.overlap = overlap
        self.n_fft = n_fft

        # Tüm .wav dosyalarını clean ve noise klasörlerinden topla
        for root, _, files in os.walk(clean_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    self.clean_files.append(os.path.join(root, f))
        for root, _, files in os.walk(noise_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    self.noise_files.append(os.path.join(root, f))

        assert len(self.clean_files) > 0, f"Clean klasörü boş: {clean_dir}"
        assert len(self.noise_files) > 0, f"Noise klasörü boş: {noise_dir}"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # 1) Temiz dosyayı oku (48 kHz olabilir), 8 kHz'e indirgeme
        clean_path = self.clean_files[idx]
        clean_wav, sr_c = torchaudio.load(clean_path)  # [1, num_samples] veya [2, num_samples]
        # Mono: birleştir
        if clean_wav.shape[0] > 1:
            clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)
        # 48k -> 8k indirgeme (factor=6)
        if sr_c != self.sample_rate:
            clean_wav = torchaudio.functional.resample(clean_wav, sr_c, self.sample_rate)

        # -------------------------------
        # 2) İki farklı gürültü (noise) dosyası seç, yükle ve karıştır
        # -------------------------------
        # random.sample ile birbirinden farklı 2 index seçiyoruz
        noise_idx1, noise_idx2 = random.sample(range(len(self.noise_files)), 2)
        noise_path1 = self.noise_files[noise_idx1]
        noise_path2 = self.noise_files[noise_idx2]

        # Gürültüleri yükle
        noise_wav1, sr_n1 = torchaudio.load(noise_path1)
        noise_wav2, sr_n2 = torchaudio.load(noise_path2)

        # Stereo → Mono indirgeme
        if noise_wav1.dim() > 1:
            noise_wav1 = torch.mean(noise_wav1, dim=0, keepdim=True)
        if noise_wav2.dim() > 1:
            noise_wav2 = torch.mean(noise_wav2, dim=0, keepdim=True)
        # 8 kHz’e indir (32 kHz, 48 kHz vb. olabilir)
        if sr_n1 != self.sample_rate:
            noise_wav1 = torchaudio.functional.resample(
                noise_wav1, orig_freq=sr_n1, new_freq=self.sample_rate
            )
        if sr_n2 != self.sample_rate:
            noise_wav2 = torchaudio.functional.resample(
                noise_wav2, orig_freq=sr_n2, new_freq=self.sample_rate
            )

        # Beklenen uzunluğa (expected_length × sample_rate) göre pad/truncate
        expected_len = self.expected_samples  # örn: 3 s × 8000 = 24000
        # Noise1
        noise_seg1 = noise_wav1.squeeze().cpu().numpy()
        if noise_seg1.shape[0] >= expected_len:
            noise_seg1 = noise_seg1[:expected_len]
        else:
            pad_amt = expected_len - noise_seg1.shape[0]
            noise_seg1 = np.concatenate([noise_seg1, np.zeros(pad_amt)])
        # Noise2
        noise_seg2 = noise_wav2.squeeze().cpu().numpy()
        if noise_seg2.shape[0] >= expected_len:
            noise_seg2 = noise_seg2[:expected_len]
        else:
            pad_amt = expected_len - noise_seg2.shape[0]
            noise_seg2 = np.concatenate([noise_seg2, np.zeros(pad_amt)])

        # İki gürültüyü eşit ağırlıkta karıştır
        combined_noise_np = (noise_seg1 + noise_seg2) / 2.0  # NumPy array, [expected_len]



        # -------------------------------
        # 4) Temiz sinyalin uzunluğunu eşitle (pad/truncate)
        # -------------------------------
        clean_np = clean_wav.squeeze().cpu().numpy()  # [N]
        if clean_np.shape[0] >= expected_len:
            clean_np = clean_np[:expected_len]
        else:
            pad_amt = expected_len - clean_np.shape[0]
            clean_np = np.concatenate([clean_np, np.zeros(pad_amt)])

        # 5) Rastgele SNR seçme ve ölçekleme
        snr_choices = [-5, 0, 5, 10]
        snr_db = random.choice(snr_choices)
        snr_lin = 10 ** (snr_db / 10)

        # Burada 'noise_seg' yerine 'combined_noise_np' kullanıyoruz:
        noise_power = np.sum(combined_noise_np ** 2) + 1e-8
        clean_power = np.sum(clean_np ** 2) + 1e-8

        scale = np.sqrt(clean_power / (noise_power * snr_lin))
        noise_scaled_np = combined_noise_np * scale
        noisy_np = clean_np + noise_scaled_np

        # 6) STFT hesaplama (PyTorch/Torchaudio)
        #     - Temiz sinyal STFT
        #     - Gürültülü sinyal STFT
        #     - Magnitude al, cIRM hesapla
        # Torchaudio’nun stft çıktısı: complex tensor (örn. [1, freq_bins, time_frames])
        clean_tensor = torch.from_numpy(clean_np).unsqueeze(0)   # [1, samples]
        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0)   # [1, samples]

        # Window ve overlap
        window = torch.hamming_window(self.window_length, periodic=True)
        clean_stft = torch.stft(clean_tensor,
                                n_fft=self.n_fft,
                                hop_length=self.window_length - self.overlap,
                                win_length=self.window_length,
                                window=window,
                                return_complex=True,
                                center=False)        # [1, freq_bins, time_frames], complex
        noisy_stft = torch.stft(noisy_tensor,
                                n_fft=self.n_fft,
                                hop_length=self.window_length - self.overlap,
                                win_length=self.window_length,
                                window=window,
                                return_complex=True,
                                center=False)

        # 7) Magnitude (predictor) ve cIRM (target) üretme
        #    - noisy_mag: |noisy_stft|
        #    - clean_real, clean_imag, noisy_real, noisy_imag: ayrı ayrı alıp cIRM hesapla
        noisy_mag = torch.abs(noisy_stft).squeeze(0)   # [freq_bins, time_frames]
        clean_real = clean_stft.real.squeeze(0)        # [freq_bins, time_frames]
        clean_imag = clean_stft.imag.squeeze(0)
        noisy_real = noisy_stft.real.squeeze(0)
        noisy_imag = noisy_stft.imag.squeeze(0)

        # cIRM hesaplama (utils/ci_mask.py’dan)
        cirm = compute_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)
        # cirm → [freq_bins, time_frames, 2]

        # 8) PyTorch’a döndür: predictor ve target
        #    - Predictor: noisy_mag  (isteğe bağlı log-normalize edilebilir; MATLAB’da abs(…) → direkt besliyor olabilir)
        #    - Target: cIRM
        # Model’e girdi olarak genellikle [1, freq_bins, time_frames] biçiminde eklenir
        predictor = noisy_mag.unsqueeze(0)            # [1, freq_bins, time_frames]
        target = cirm.permute(2, 0, 1)                # [2, freq_bins, time_frames] (real / imag mask)

        return predictor, target
