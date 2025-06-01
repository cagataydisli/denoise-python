# utils/ci_mask.py

import torch

def compute_complex_ideal_ratio_mask(noisy_real: torch.Tensor,
                                     noisy_imag: torch.Tensor,
                                     clean_real: torch.Tensor,
                                     clean_imag: torch.Tensor,
                                     eps: float = 1e-8) -> torch.Tensor:
    """
    MATLAB’daki buld_complex_ideal_ratio_mask fonksiyonunun PyTorch’daki karşılığı.
    Girişler:
      - noisy_real, noisy_imag: [freq_bins, time_frames] boyutlu tensor’lar (STFT real ve imag bileşenleri)
      - clean_real, clean_imag: [freq_bins, time_frames] boyutlu tensor’lar
    Başlangıçta:
      complex_noisy = noisy_real + j*noisy_imag
      complex_clean = clean_real + j*clean_imag
    cIRM (complex ideal ratio mask) şöyle hesaplanır:
      M = (complex_clean / complex_noisy)
    Çıkış: M’in real ve imag bileşenlerinin ayrı ayrı veya birleşik olarak dönülmesi
    Çoğunlukla:
      M_real = Re{complex_clean * conj(complex_noisy)} / (|noisy|^2 + eps)
      M_imag = Im{complex_clean * conj(complex_noisy)} / (|noisy|^2 + eps)
    Ancak MATLAB’daki implementasyona birebir uyacak şekilde uyarlamalısın.
    """
    # Örnek basit cIRM hesaplama (bunu MATLAB koduna göre gerektiğinde değiştir)
    # complex_noisy = noisy_real + 1j*noisy_imag
    # complex_clean = clean_real + 1j*clean_imag
    # M = complex_clean / (complex_noisy + eps)

    # Aşağıda daha yaygın kullanılan formül:
    #       R = (Re(noisy)*Re(clean) + Im(noisy)*Im(clean)) / (|noisy|^2 + eps)
    #       I = (Re(noisy)*Im(clean) - Im(noisy)*Re(clean)) / (|noisy|^2 + eps)
    denom = noisy_real.pow(2) + noisy_imag.pow(2) + eps
    M_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denom
    M_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denom

    # Sonuçta [freq_bins, time_frames, 2] biçiminde bir tensor döndürebiliriz
    # Veya her biri ayrı tensor döndürüp sonra birleştime işlemi yapabiliriz. Burada
    # sıklıkla network, “[RealMask; ImagMask]” biçiminde girişe alır.
    return torch.stack([M_real, M_imag], dim=-1)  # [freq_bins, time_frames, 2]
