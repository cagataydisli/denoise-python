import os
import argparse
import torch
import torchaudio
from models.denoise_model import FullSubNet


def arbitrary_length_inference(args):
    # 1) Cihaz seçimi: Eğer GPU (CUDA) varsa onu, yoksa CPU’yu kullan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Inference cihazı: {device}")

    # 2) Modeli oluşturun ve GPU/CPU'ya taşıyın
    model = FullSubNet(
        num_features=257,
        time_steps=args.time_steps,
        num_hidden_fb=768,
        num_hidden_sb=512
    ).to(device)

    # 3) Checkpoint’i yükleyin
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4) Girdi WAV dosyasını yükleyin
    waveform, sr = torchaudio.load(args.input_wav)  # [kanal, örnek]
    if waveform.dim() > 1:
        # Stereo ise mono’ya indir
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # 5) Örnekleme hızını (sample_rate) ayarlayın
    if sr != args.sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=args.sample_rate)
        sr = args.sample_rate

    # 6) STFT parametreleri (eğitimde kullanılanla aynı)
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length).to(device)

    # 7) Model tek seferde tüm sinyali işleyemediğinden parçalama yapacağız
    #    Önce tüm sinyali GPU’ya taşıyın
    waveform = waveform.to(device)  # [1, N_samples]

    # 8) Tüm sinyal için STFT alın (kompleks çıktısı)
    stft_complex = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect"
    )  # [1, 257, T_frames]

    mag = torch.abs(stft_complex)  # [1, 257, T_frames]
    # phase bilgisi aslında son adımda iSTFT için kullanılacak,
    # burada doğrudan model maskesi uygulayacağımız için fazı tutmamıza gerek yok.
    # Ancak ileride ince ayar yaparken phase’a ihtiyaç duyarsanız şöyle alabilirsiniz:
    # phase = torch.angle(stft_complex)

    B, F, T = mag.shape
    time_steps = args.time_steps
    # Kaç parça (chunk) olacak?
    num_chunks = (T + time_steps - 1) // time_steps

    # Eğer T, tam katı değilse pad yapacağız
    pad_frames = num_chunks * time_steps - T
    if pad_frames > 0:
        mag = torch.nn.functional.pad(mag, (0, pad_frames), "constant", 0)  # [1, 257, T_padded]

    # Bütün maskeleri saklayacağımız tensorlar
    real_mask_full = torch.zeros((B, F, num_chunks * time_steps), device=device)
    imag_mask_full = torch.zeros((B, F, num_chunks * time_steps), device=device)

    # 9) Parça parça modelden geçirme
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * time_steps
            end = start + time_steps
            mag_chunk = mag[:, :, start:end]  # [1, 257, time_steps]
            predictor = mag_chunk.unsqueeze(1).to(device)  # [1, 1, 257, time_steps]
            mask_out = model(predictor)  # [1, 2, 257, time_steps]
            real_mask_chunk = mask_out[:, 0, :, :]  # [1, 257, time_steps]
            imag_mask_chunk = mask_out[:, 1, :, :]  # [1, 257, time_steps]
            real_mask_full[:, :, start:end] = real_mask_chunk
            imag_mask_full[:, :, start:end] = imag_mask_chunk

    # 10) Orijinal frame sayısına geri kes (pad varsa)
    if pad_frames > 0:
        real_mask_full = real_mask_full[:, :, :T]
        imag_mask_full = imag_mask_full[:, :, :T]

    # 11) Denoised STFT’ı oluşturun
    real_noisy = stft_complex.real  # [1, 257, T]
    imag_noisy = stft_complex.imag  # [1, 257, T]
    real_denoised = real_noisy * real_mask_full - imag_noisy * imag_mask_full
    imag_denoised = real_noisy * imag_mask_full + imag_noisy * real_mask_full
    denoised_complex = torch.complex(real_denoised, imag_denoised)  # [1, 257, T]

    # 12) iSTFT → Zaman domenine geri dönün
    denoised_waveform = torch.istft(
        denoised_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=waveform.shape[-1]  # Orijinal örnek sayısı
    )  # [1, N_samples]

    # 13) Sonucu CPU’ya alın ve kaydedin
    denoised_waveform = denoised_waveform.cpu()
    os.makedirs(os.path.dirname(args.output_wav), exist_ok=True)
    torchaudio.save(args.output_wav, denoised_waveform, args.sample_rate)
    print(f"Denoised wav kaydedildi → {args.output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Örn: checkpoints_big/fullsubnet_epoch15.pth")
    parser.add_argument("--input_wav", type=str, required=True,
                        help="Denoise edilecek gürültülü WAV dosyasının yolu")
    parser.add_argument("--output_wav", type=str, required=True,
                        help="Denoised{" "}çıktısını kaydedeceğiniz WAV dosyasının yolu")
    parser.add_argument("--sample_rate", type=int, default=8000,
                        help="Örnekleme hızı (model 8 kHz ile eğitildiği için 8000 girilmeli)")
    parser.add_argument("--time_steps", type=int, default=92,
                        help="Frame sayısı (3 s x 8000Hz, FFT=512, hop=256 → ~92)")
    args = parser.parse_args()

    arbitrary_length_inference(args)
