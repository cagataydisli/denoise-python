"""
Batch Inference Script
----------------------
Bu betik, eğitilmiş **FullSubNet** modelini kullanarak
`data/test` (veya belirtilen başka bir klasör) altındaki tüm WAV dosyalarını
denoise eder ve sonuçları `output` klasörüne `denoised_<orjinal>` adıyla kaydeder.

Örnek kullanım:
```
python batch_inference.py \
  --model_path checkpoints_big/best_epoch_11.pth \
  --in_dir data/test \
  --out_dir output \
  --sample_rate 8000 \
  --time_steps 92
```
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torchaudio
from models.denoise_model import FullSubNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Inference device: {device}")

def load_model(ckpt_path: str, time_steps: int):
    model = FullSubNet(
        num_features=257,
        time_steps=time_steps,
        num_hidden_fb=768,
        num_hidden_sb=512,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denoise_one(model: FullSubNet, wav_path: Path, out_path: Path, sr_target: int, time_steps: int):
    wav, sr = torchaudio.load(str(wav_path))
    # Mono
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
        sr = sr_target

    wav = wav.to(device)

    # STFT params (match training)
    n_fft = 512
    hop = 256
    win = 512
    window = torch.hann_window(win, device=device)

    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                      return_complex=True, center=True, pad_mode="reflect")
    mag = torch.abs(stft)  # [1, 257, T]
    B, F, T = mag.shape

    # Pad to multiple of time_steps
    chunks = (T + time_steps - 1) // time_steps
    pad_T = chunks * time_steps - T
    if pad_T:
        mag = torch.nn.functional.pad(mag, (0, pad_T))

    real_mask = torch.zeros_like(mag)
    imag_mask = torch.zeros_like(mag)

    with torch.no_grad():
        for i in range(chunks):
            s, e = i * time_steps, (i + 1) * time_steps
            m_chunk = mag[:, :, s:e].unsqueeze(1)  # [1,1,257,Tc]
            out = model(m_chunk)
            real_mask[:, :, s:e] = out[:, 0]
            imag_mask[:, :, s:e] = out[:, 1]

    if pad_T:
        real_mask = real_mask[:, :, :T]
        imag_mask = imag_mask[:, :, :T]

    real_noisy, imag_noisy = stft.real, stft.imag
    real_den = real_noisy * real_mask - imag_noisy * imag_mask
    imag_den = real_noisy * imag_mask + imag_noisy * real_mask
    den_complex = torch.complex(real_den, imag_den)

    denoised = torch.istft(den_complex, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=window, center=True, length=wav.shape[-1])
    denoised = denoised.cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), denoised, sr)
    print(f"[✓] {wav_path.name} → {out_path.name}")


def main():
    ap = argparse.ArgumentParser("Batch FullSubNet inference")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_rate", type=int, default=8000)
    ap.add_argument("--time_steps", type=int, default=92)
    args = ap.parse_args()

    model = load_model(args.model_path, args.time_steps)

    wav_files = sorted(Path(args.in_dir).glob("*.wav"))
    if not wav_files:
        print("[!] in_dir içinde .wav yok"); return

    for fp in wav_files:
        out_fp = Path(args.out_dir) / f"denoised_{fp.name}"
        denoise_one(model, fp, out_fp, args.sample_rate, args.time_steps)

    print("\n[Done] Batch inference completed.")


if __name__ == "__main__":
    main()
