# utils/split_audio.py

import os
import glob
import torch
import torchaudio

def split_directory(input_dir, output_dir, chunk_s=3, target_sr=8000):
    """
    input_dir: ham wav dosyalarının bulunduğu dizin (ör. data/clean_fullband)
    output_dir: parçalanmış 3 saniyelik dosyaların kaydedileceği dizin
    chunk_s: her bir parça uzunluğu (saniye)
    target_sr: 8 kHz, aynı olmalı
    """
    os.makedirs(output_dir, exist_ok=True)
    expected_samples = chunk_s * target_sr

    patterns = ["*.wav"]
    wav_files = []
    for p in patterns:
        wav_files.extend(glob.glob(os.path.join(input_dir, p)))

    if len(wav_files) == 0:
        print(f"[!] {input_dir} içinde WAV dosyası bulunamadı.")
        return

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        name, _ = os.path.splitext(filename)

        # 1) Dosyayı yükle
        waveform, sr = torchaudio.load(wav_path)  # [kanal, örnek]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        # 2) Mono’ya indir (gerekirse)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        total_samples = waveform.shape[1]
        num_full_chunks = total_samples // expected_samples

        # 3) Art arda tam parçaları yaz
        for i in range(num_full_chunks):
            start = i * expected_samples
            end = (i + 1) * expected_samples
            chunk = waveform[:, start:end]  # [1, expected_samples]
            out_name = f"{name}_chunk{i+1:02d}.wav"
            out_path = os.path.join(output_dir, out_name)
            torchaudio.save(out_path, chunk, target_sr)
            print(f"✔ Oluşturuldu: {out_path}  (saniye: {i*chunk_s}-{(i+1)*chunk_s})")

        # 4) Son parça (eksik kalmışsa)
        rem = total_samples - num_full_chunks * expected_samples
        if rem > 0:
            last_chunk = waveform[:, num_full_chunks * expected_samples : ]
            # Eksik kalan kısmı 0 ile pad’le
            pad_size = expected_samples - rem
            last_chunk = torch.nn.functional.pad(last_chunk, (0, pad_size))
            out_name = f"{name}_chunk{num_full_chunks+1:02d}.wav"
            out_path = os.path.join(output_dir, out_name)
            torchaudio.save(out_path, last_chunk, target_sr)
            print(f"✔ Oluşturuldu: {out_path}  (saniye: {num_full_chunks*chunk_s}-{num_full_chunks*chunk_s + rem/target_sr:.2f} + pad)")

if __name__ == "__main__":
    # 1) Clean konuşma dosyalarını ayır
    split_directory(
        input_dir="data/clean_fullband",
        output_dir="data/clean_fullband_chunks",
        chunk_s=3,
        target_sr=8000
    )
    # 2) Gürültü dosyalarını ayır
    split_directory(
        input_dir="data/noise_fullband",
        output_dir="data/noise_fullband_chunks",
        chunk_s=3,
        target_sr=8000
    )
