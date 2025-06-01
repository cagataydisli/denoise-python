"""
Batch evaluation metrics (PESQ + STOI)
-------------------------------------
* 8 kHz → PESQ narrow‑band (nb), 16 kHz → wide‑band (wb)
* Kısa klipler (< 0.25 s) pad edilir, ancak "no utterances" veya benzeri
  hatalar ortaya çıkarsa metrik **NaN** olarak işaretlenir.
* STOI, varsayılan olarak ≥ 0.5 s gereksinimini karşılamayan dosyalar
  için NaN döner.

Komut örneği:
```
python evaluate_metrics.py --clean_dir data/test --denoised_dir output --output_csv metrics_results.csv
```
"""

import os
import glob
import argparse
import csv
from typing import List, Tuple, Optional

import numpy as np
import torchaudio
from pesq import pesq   # pip install pesq
from pystoi import stoi # pip install pystoi

# --------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------

def _to_contiguous_1d(x: np.ndarray) -> np.ndarray:
    """Ensure C‑contiguous float32 1‑D array (PESQ/STOI gereksinimi)."""
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def _pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] < length:
        return np.pad(x, (0, length - x.shape[0]))
    return x

# --------------------------------------------------
# Metrik hesaplama (hata‑toleranslı)
# --------------------------------------------------

def compute_metrics(
    clean_wav: np.ndarray,
    denoised_wav: np.ndarray,
    sr: int,
    stoi_min_sec: float = 0.5,
    pesq_min_sec: float = 0.25,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (PESQ, STOI). Hata veya kısıt ihlali durumunda NaN."""
    mode = "nb" if sr == 8000 else "wb"

    clean = _to_contiguous_1d(clean_wav)
    den   = _to_contiguous_1d(denoised_wav)

    # Uzunlukları eşitle
    L = min(len(clean), len(den))
    clean, den = clean[:L], den[:L]

    # --- PESQ ---
    pesq_req = int(pesq_min_sec * sr)
    clean_p  = _pad_to_length(clean, pesq_req)
    den_p    = _pad_to_length(den,   pesq_req)
    try:
        pesq_val = pesq(sr, clean_p, den_p, mode)
    except Exception as e:
        print(f"  [PESQ skip] {e}")
        pesq_val = np.nan

    # --- STOI ---
    stoi_req = int(stoi_min_sec * sr)
    if len(clean) < stoi_req:
        stoi_val = np.nan
    else:
        try:
            stoi_val = stoi(clean, den, sr, extended=False)
        except Exception as e:
            print(f"  [STOI skip] {e}")
            stoi_val = np.nan

    return pesq_val, stoi_val

# --------------------------------------------------
# Dosya eşleştirme
# --------------------------------------------------

def collect_pairs(clean_dir: str, den_dir: str) -> List[Tuple[str, str, str]]:
    pairs = []
    for clean_fp in glob.glob(os.path.join(clean_dir, "*.wav")):
        name = os.path.basename(clean_fp)
        den_fp = os.path.join(den_dir, f"denoised_{name}")
        if os.path.exists(den_fp):
            pairs.append((name, clean_fp, den_fp))
        else:
            print(f"[Skip] {den_fp} yok")
    return pairs

# --------------------------------------------------
# Ana akış
# --------------------------------------------------

def main(args):
    pairs = collect_pairs(args.clean_dir, args.denoised_dir)
    if not pairs:
        print("[Hata] Eşleşen dosya yok"); return

    results = []
    for name, c_fp, d_fp in pairs:
        c_t, sr_c = torchaudio.load(c_fp)
        d_t, sr_d = torchaudio.load(d_fp)
        if sr_c != sr_d:
            print(f"[SR Mismatch] {name}, atlandı."); continue

        c_np = c_t.squeeze().cpu().numpy()
        d_np = d_t.squeeze().cpu().numpy()

        pesq_v, stoi_v = compute_metrics(c_np, d_np, sr_c)
        print(f"{name:<30} PESQ {pesq_v if not np.isnan(pesq_v) else 'nan':>5} | STOI {stoi_v if not np.isnan(stoi_v) else 'nan'}")
        results.append((name, pesq_v, stoi_v))

    if not results:
        print("Hiç ölçüm yok"); return

    pesq_vals = [r[1] for r in results if not np.isnan(r[1])]
    stoi_vals = [r[2] for r in results if not np.isnan(r[2])]

    print("\n=== Ortalama ===")
    if pesq_vals:
        print(f"Average PESQ: {np.mean(pesq_vals):.3f}")
    else:
        print("Average PESQ: n/a")

    if stoi_vals:
        print(f"Average STOI: {np.mean(stoi_vals):.3f}")
    else:
        print("Average STOI: n/a (geçerli klip yok)")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["filename", "pesq", "stoi"])
            for fn, p, s in results:
                wr.writerow([fn, "nan" if np.isnan(p) else f"{p:.3f}", "nan" if np.isnan(s) else f"{s:.3f}"])
        print(f"CSV kaydedildi → {args.output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("PESQ + STOI batch evaluator")
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--denoised_dir", required=True)
    ap.add_argument("--output_csv", default=None)
    main(ap.parse_args())
