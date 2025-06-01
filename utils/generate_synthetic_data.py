# denoise-python/utils/generate_synthetic_data.py

import os
import torch
import numpy as np
import soundfile as sf    # <-- soundfile'i import ettik

def generate_sine_wave(frequency, duration, sr=8000, amplitude=0.5):
    t = torch.arange(0, int(duration * sr)) / sr
    wave = amplitude * torch.sin(2 * np.pi * frequency * t)
    return wave.unsqueeze(0)  # [1, num_samples]

def generate_white_noise(duration, sr=8000):
    num_samples = int(duration * sr)
    noise = torch.randn(1, num_samples) * 0.1
    return noise

if __name__ == "__main__":
    # Script'in bulunduğu klasöre göre proje kök dizinini bulun:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    clean_dir = os.path.join(project_root, "data", "clean_fullband")
    noise_dir = os.path.join(project_root, "data", "noise_fullband")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)

    sr = 8000
    duration = 3  # saniye

    freqs = [200, 400, 800]
    for f in freqs:
        sine = generate_sine_wave(f, duration, sr=sr)  # [1, samples]
        # PyTorch tensor'ı NumPy dizisine çevir (float32)
        sine_np = sine.squeeze(0).cpu().numpy().astype(np.float32)  # [samples]
        out_path = os.path.join(clean_dir, f"sine_{f}Hz.wav")
        # soundfile ile kaydet: sf.write(yol, data, sr)
        sf.write(out_path, sine_np, sr)
    
    for i in range(1, 4):
        noise = generate_white_noise(duration, sr=sr)  # [1, samples]
        noise_np = noise.squeeze(0).cpu().numpy().astype(np.float32)
        out_path = os.path.join(noise_dir, f"noise_{i}.wav")
        sf.write(out_path, noise_np, sr)

    print("Sentetik veri üretildi:")
    print(f" - {clean_dir} içinde: sine_{freqs[0]}Hz.wav, sine_{freqs[1]}Hz.wav, sine_{freqs[2]}Hz.wav")
    print(f" - {noise_dir} içinde: noise_1.wav, noise_2.wav, noise_3.wav")
