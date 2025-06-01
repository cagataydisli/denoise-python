import os
import glob
import argparse
import csv
from typing import Tuple, List

import numpy as np
import torchaudio
from pesq import pesq               # pip install pesq
from pystoi import stoi             # pip install pystoi

# -------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------

def _to_contiguous_1d(x: np.ndarray) -> np.ndarray:
    """Ensure a C‑contiguous, float32, 1‑D array (PESQ/STOI requirement)."""
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def _pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Pad 1‑D array with zeros up to *length* samples."""
    if x.shape[0] < length:
        return np.pad(x, (0, length - x.shape[0]))
    return x

# -------------------------------------------------------------
#  Core metric function
# -------------------------------------------------------------

def compute_metrics(
    clean_wav: np.ndarray,
    denoised_wav: np.ndarray,
    sr: int,
    stoi_min_sec: float = 0.5,
    pesq_min_sec: float = 0.25,
) -> Tuple[float, float]:
    """Return (PESQ, STOI).  If signal < *stoi_min_sec*, STOI=nan."""
    mode = "nb" if sr == 8000 else "wb"   # PESQ narrow / wide band selection

    clean = _to_contiguous_1d(clean_wav)
    den   = _to_contiguous_1d(denoised_wav)

    # --- unify length ---
    L = min(len(clean), len(den))
    clean, den = clean[:L], den[:L]

    # --- PESQ: pad up to pesq_min_sec if necessary ---
    pesq_req = int(pesq_min_sec * sr)
    clean_p = _pad_to_length(clean, pesq_req)
    den_p   = _pad_to_length(den,   pesq_req)
    pesq_val = pesq(sr, clean_p, den_p, mode)

    # --- STOI: require >= stoi_min_sec; else nan ---
    stoi_req = int(stoi_min_sec * sr)
    if len(clean) < stoi_req:
        stoi_val = np.nan
    else:
        stoi_val = stoi(clean, den, sr, extended=False)

    return pesq_val, stoi_val

# -------------------------------------------------------------
#  Main routine
# -------------------------------------------------------------

def collect_pairs(clean_dir: str, den_dir: str) -> List[Tuple[str, str, str]]:
    """Find matching (clean, denoised) wav pairs."""
    pairs = []
    for clean_fp in glob.glob(os.path.join(clean_dir, "*.wav")):
        name = os.path.basename(clean_fp)
        den_fp = os.path.join(den_dir, f"denoised_{name}")
        if os.path.exists(den_fp):
            pairs.append((name, clean_fp, den_fp))
        else:
            print(f"[Skip] {den_fp} bulunamadı.")
    return pairs


def main(args):
    pairs = collect_pairs(args.clean_dir, args.denoised_dir)
    if not pairs:
        print("[Hata] Değerlendirme için eşleşen dosya bulunamadı.")
        return

    results = []
    for fname, clean_fp, den_fp in pairs:
        clean_t, sr_c = torchaudio.load(clean_fp)
        den_t,   sr_d = torchaudio.load(den_fp)
        if sr_c != sr_d:
            print(f"[Uyarı] SR uyuşmazlığı {fname}, atlandı.")
            continue

        c_np = clean_t.squeeze().cpu().numpy()
        d_np = den_t.squeeze().cpu().numpy()

        pesq_v, stoi_v = compute_metrics(c_np, d_np, sr_c)
        print(f"{fname:<20} PESQ {pesq_v:5.3f} | STOI {stoi_v if not np.isnan(stoi_v) else 'nan'}")
        results.append((fname, pesq_v, stoi_v))

    if not results:
        print("Hiç ölçüm yapılmadı.")
        return

    pesq_mean = np.mean([r[1] for r in results])
    stoi_vals = [r[2] for r in results if not np.isnan(r[2])]
    stoi_mean = np.nan if not stoi_vals else np.mean(stoi_vals)

    print("\n=== Ortalama ===")
    print(f"Average PESQ: {pesq_mean:.3f}")
    if not np.isnan(stoi_mean):
        print(f"Average STOI: {stoi_mean:.3f}")
    else:
        print("Average STOI: n/a (çok kısa klipler)")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["filename", "pesq", "stoi"])
            for row in results:
                wr.writerow([row[0], f"{row[1]:.3f}", "nan" if np.isnan(row[2]) else f"{row[2]:.3f}"])
        print(f"CSV kaydedildi → {args.output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("PESQ + STOI değerlendirme scripti")
    ap.add_argument("--clean_dir", required=True, help="Temiz referans WAV klasörü")
    ap.add_argument("--denoised_dir", required=True, help="Model çıktıları klasörü")
    ap.add_argument("--output_csv", default=None, help="Opsiyonel: sonuçları CSV'ye yaz")
    main(ap.parse_args())
