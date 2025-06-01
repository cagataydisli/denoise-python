"""
Mix Clean + Noise Script (v2)
---------------------------
Bu betik, `data/clean_fullband` altındaki temiz WAV dosyaları ile
`data/noise_fullband` altındaki gürültü WAV dosyalarını rastgele eşleştirir;
istenen adette (ör. 100) gürültülü dosya üretir ve `data/test` klasörüne
`noisy_<index>_<cleanname>.wav` formatında kaydeder.

Python 3.8/3.9 uyumlu.

Örnek:
```
python mix_clean_noise.py \
  --clean_dir data/clean_fullband \
  --noise_dir data/noise_fullband \
  --out_dir   data/test \
  --snr 0 \
  --num 100
```
* `--snr` opsiyonel (dB).
* `--num` üretilmek istenen karışık dosya sayısı.
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torchaudio

TARGET_SR = 8000  # Hz

def _load_mono(path: str, target_sr: int = TARGET_SR):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr


def mix_one(clean_path: str, noise_path: str, out_path: Path, snr_db: Optional[float]):
    clean, sr = _load_mono(clean_path)
    noise, _  = _load_mono(noise_path, sr)

    if noise.numel() < clean.numel():
        rep = int(np.ceil(clean.numel() / noise.numel()))
        noise = noise.repeat(rep)[: clean.numel()]
    else:
        noise = noise[: clean.numel()]

    if snr_db is not None:
        cp = clean.pow(2).mean()
        npow = noise.pow(2).mean() + 1e-8
        scale = (cp / (npow * 10 ** (snr_db / 10))) ** 0.5
        noise = noise * scale

    mixed = clean + noise
    mixed = mixed / mixed.abs().max().clamp(min=1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path.as_posix(), mixed.unsqueeze(0), sr)
    print(f"[+] {out_path.name}  <-  {Path(clean_path).name} + {Path(noise_path).name}")


def main():
    ap = argparse.ArgumentParser("Random Clean+Noise Mixer")
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--num", type=int, default=100, help="Üretilecek dosya sayısı")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    clean_files = list(Path(args.clean_dir).glob("*.wav"))
    noise_files = list(Path(args.noise_dir).glob("*.wav"))
    if not clean_files or not noise_files:
        raise RuntimeError("Temiz veya gürültü klasörlerinde WAV bulunamadı.")

    for i in range(args.num):
        clean_fp = random.choice(clean_files)
        noise_fp = random.choice(noise_files)
        out_name = f"noisy_{i:03d}_{clean_fp.stem}.wav"
        out_fp = Path(args.out_dir) / out_name
        mix_one(clean_fp.as_posix(), noise_fp.as_posix(), out_fp, args.snr)

    print(f"\nTamamlandı: {args.num} dosya ⇒ {args.out_dir}")


if __name__ == "__main__":
    main()
