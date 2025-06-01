# utils/resample_and_pad.py

import os
import glob
import torch
import torchaudio

def resample_and_pad(input_dir, output_dir, target_sr=8000, duration_s=3):
    """
    - input_dir: Kaynak wav/mp3/m4a dosyalarının bulunduğu klasör
    - output_dir: Çıktı wav'ların kaydedileceği klasör
    - target_sr: Hedef örnekleme hızı (örneğin 8000)
    - duration_s: Sabitlenen uzunluk (saniye cinsinden, burada 3 saniye)
    """
    os.makedirs(output_dir, exist_ok=True)
    expected_samples = target_sr * duration_s  # 3 * 8000 = 24000 örnek

    # .wav, .mp3, .m4a uzantılı bütün dosyaları al
    patterns = ["*.wav", "*.mp3", "*.m4a"]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(input_dir, p)))

    if len(all_files) == 0:
        print(f"[!] {input_dir} içinde *.wav, *.mp3 veya *.m4a bulunamadı.")
        return

    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            waveform, sr = torchaudio.load(file_path)  # [channel, num_samples]
        except Exception as e:
            print(f"[!] Dosya yüklenirken hata: {filename} → {e}")
            continue

        # 1) Stereo ise mono'ya düşür
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, num_samples]

        # 2) Farklı sr ise 8 kHz'e yeniden örnekle
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        # 3) Uzunluğu tam 3 saniyeye çek (24000 örnek)
        num_samples = waveform.shape[1]
        if num_samples > expected_samples:
            # 3 saniyeden uzun → sadece ilk 24000 örneği al
            waveform = waveform[:, :expected_samples]
        elif num_samples < expected_samples:
            # 3 saniyeden kısa → sonuna sıfır pad ekle
            pad_size = expected_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # 4) Kaydet (çeşitli kötü uzantılar olsa da torchaudio.save .wav extension kullanır)
        out_filename = os.path.splitext(filename)[0] + ".wav"
        out_path = os.path.join(output_dir, out_filename)
        torchaudio.save(out_path, waveform, target_sr)
        print(f"✔ İşlendi: {filename} → {out_path}  (shape={waveform.shape}, sr={target_sr})")


if __name__ == "__main__":
    # 1) Clean konuşma dosyaları
    resample_and_pad(
        input_dir="data/clean_fullband",
        output_dir="data/clean_fullband_8k_3s",
        target_sr=8000,
        duration_s=3
    )
    # 2) Gürültü dosyaları
    resample_and_pad(
        input_dir="data/noise_fullband",
        output_dir="data/noise_fullband_8k_3s",
        target_sr=8000,
        duration_s=3
    )
