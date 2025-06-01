# models/denoise_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 1. ÖZEL KATMANLARIN PYTORCH EŞDEĞERLERİ
################################################################################

class PadLayer(nn.Module):
    """
    MATLAB’daki padLayer:
      - Giriş: X [batch, features, time] (dlarray biçiminde STB)
      - predict fonksiyonu:
          formated = extractdata(X)
          formated = padarray(formated, [0,2,0,0],0,'post');
          maxVal = max(formated,[], 'all') + 1e-5;
          formated = formated / maxVal;
          formated = dlarray(formated,'STB');
      - Yani: (1) zaman ekseni sonuna 2 sıfır ekle (padding), 
              (2) tüm tensor’u max+eps ile normalize et,
              (3) çıktı olarak aynı formatı geri ver.
    """
    def __init__(self, eps: float = 1e-5):
        super(PadLayer, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features, time]
        # (1) Zaman ekseni sonunda 2 sıfır ekle (pad right by 2)
        #    PyTorch’ta: F.pad ile (left_pad, right_pad) şeklinde verilir.
        #    Burada sadece “post” olarak 2 ekliyoruz:
        x_padded = F.pad(x, (0, 2), mode='constant', value=0.0)  # [B, F, time+2]

        # (2) Normalize et:    maxVal = max(x_padded) + eps
        max_val = x_padded.max()
        normed = x_padded / (max_val + self.eps)

        # (3) Çıktı: [batch, features, time+2]
        return normed


class UnfoldLayer(nn.Module):
    """
    MATLAB’daki unfoldLayer:
      - Giriş: X [freq, channels=1, frames, time] muhtemelen (dlarray STB? biçiminden önce)
      - Kodda: 
          output = reshape(X, size(X,1), 1, size(X,2), size(X,3));
          output = permute(output, [3,2,1,4]);   # → [batch, 1, freq, frames]
          sub_band_unit_size = num_neighbors*2 + 1  (num_neighbors=15 → 31)
          output = padarray(output, [0,0,num_neighbors,0], 'symmetric','both');
          output = im2col_cus(output);
          output = reshape(output, [batch, 1, sub_band_unit_size, frames, freq]);
          output = permute(output, [1,5,2,3,4]);
          output = reshape(..., [batch, freq, sub_band_unit_size, frames]);
          output = dlarray(output,'BSCT');
      - Yani temel mantık: “freq-axis üzerinde ±15 komşu genliğini al, her bir frekansı 
        31’lik bir pencere içine yerleştir, sonra da [batch, freq_bins, 31, time_frames] boyutuna getir.”
    """
    def __init__(self, num_neighbors: int = 21):
        super(UnfoldLayer, self).__init__()
        self.num_neighbors = num_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, freq_bins, time_frames]
        Adımlar:
          1) [B, F, T] → [B, 1, F, T]
          2) Frekans eksenini pad’le: ± num_neighbors, ‘symmetric’
          3) PyTorch’un unfold ile “sliding window” mantığını kullanarak 
             her frekans üzerindeki pencereyi çıkart
          4) Çıktıyı [B, freq_bins, sub_band_unit_size, time_frames]
        """
        B, F, T = x.shape
        # 1) [B, 1, F, T]
        out = x.unsqueeze(1)  # [B, 1, F, T]

        # 2) Pad frekans ekseninde ± num_neighbors (symmetric): 
        pad_amt = self.num_neighbors
        # PyTorch’ta 2-boyutlu pad: (left_time, right_time, top_freq, bottom_freq)
        # Zaman eksenini padlemiyoruz, frekans eksenini padliyoruz: 
        # “top_freq”=pad_amt, “bottom_freq”=pad_amt 
        out = torch.nn.functional.pad(out, (0, 0, pad_amt, pad_amt), mode='reflect')
        # → [B, 1, F + 2*num_neighbors, T]

        # 3) Unfold: 
        #    PyTorch: out.unfold(dimension, size, step)
        #    Kongre: 
        #      - dimension=2 (freq-axis) 
        #      - size = 2*num_neighbors + 1 
        #      - step = 1
        subband_size = 2 * self.num_neighbors + 1
        # out: [B, 1, F + 2*p, T]
        # Önce freq eksenini “sliding window” mantığıyla ayır:
        # .unfold(2, subband_size, 1) → boyut: [B, 1, (F + 2*p - subband_size + 1), T, subband_size]
        # Not: (F + 2*p - subband_size + 1) = F
        out_unfold = out.unfold(2, subband_size, 1)  # [B, 1, F, T, subband_size]

        # 4) Sırası gerekir: MATLAB’da sonra permute edip [B, F, subband_size, T]
        #en son olarak [batch, freq_bins, subband_unit_size, time_frames]
        out_unfold = out_unfold.squeeze(1)            # [B, F, T, subband_size]
        out_unfold = out_unfold.permute(0, 1, 3, 2)    # [B, F, subband_size, T]

        return out_unfold  # [batch, freq_bins, 31, time_frames]


class NormLayer(nn.Module):
    """
    MATLAB’daki normLayer(isSpecial, Name):
      - Eğer isSpecial=false:
          normed = stripdims(X);
          normed = permute(normed, [1,3,2]);   # [freq, time, batch] → [freq, batch, time] vb. 
          means = mean(normed, [1,2]);
          newNormed = normed / (means + 1e-5);
          normed = dlarray(newNormed,'STB');
        → Python’da x: [B, freq, time], 
                    → [freq, B, time], ortalamaya göre böl → → PyTorch’ta yeniden [B, freq, time].
      - Eğer isSpecial=true:
          normed = permute(X, [1,2,4,3]);
          means = mean(normed, [1,2,3]);
          newNormed = normed / (means + eps);
          normed = permute(newNormed, [4,1,2,3]);
          normed = reshape(normed, [size(normed,1)*size(normed,2), size(normed,3), size(normed,4)]);
          normed = dlarray(normed,'TCU');
    """
    def __init__(self, is_special: bool = False, eps: float = 1e-5):
        super(NormLayer, self).__init__()
        self.is_special = is_special
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, freq, time]
        """
        B, F, T = x.shape

        if not self.is_special:
            # 1) [B, F, T] → [F, B, T]
            y = x.permute(1, 0, 2).contiguous()
            # 2) Mean over dim(1,2) yani (batch ve time) → [F]
            mean_vals = y.mean(dim=(1, 2), keepdim=True)  # [F,1,1]
            # 3) Norm: y / (means + eps)
            y = y / (mean_vals + self.eps)
            # 4) Geri [F, B, T] → [B, F, T]
            y = y.permute(1, 0, 2).contiguous()
            return y  # [B, F, T]
        else:
            # MATLAB’daki isSpecial branch (sub-band tarafı için)
            # x: [batch, freq, time] 
            # 1) [B, F, T] → [B, F, 1, T]  (çünkü “channels=1” muhabbeti)
            y = x.unsqueeze(2)  # [B, F, 1, T]
            # 2) Permute: [B, F, 1, T] → [B, 1, T, F] (MATLAB’da [1,2,4,3])
            y = y.permute(0, 1, 3, 2).contiguous()  # [B, F, T, 1] → dikkat sıralama
            #    Aslında [B, C, T, U] bekleniyor: burada C=1, U=F
            #    Doğru permute: [B, F, 1, T] → [B, 1, T, F]
            y = y.permute(0, 2, 3, 1).contiguous()  # [B, 1, T, F]
            # 3) Ortalama: mean(y, dim=(1,2,3)) → tek bir skaler her batch için. 
            #    MATLAB’da means = mean(normed,[1,2,3])  → [1,1,1,F]? 
            mean_vals = y.mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
            # 4) Norm: y / (mean_vals + eps)
            y = y / (mean_vals + self.eps)  # [B,1,T,F]
            # 5) Geri permute & reshape:
            #    MATLAB: permute → [4,1,2,3] → [F, B, 1, T]? 
            #    sonra reshape → [F*B, 1, T] 
            # Python’da:
            y = y.permute(3, 0, 2, 1).contiguous()  # [F, B, T, 1]
            y = y.view(F * B, 1, T)                 # [F*B, 1, T]
            return y  # [F*B, 1, T]


class RelabelLayer(nn.Module):
    """
    MATLAB’daki relabelLayer(timeSize, newLabel):
      - Giriş: X [batch, channels, timeSize] (örneğin [32, 1285, 94], format 'TCU' veya 'CBT')
      - predict:
          Z = reshape(X, size(X,1), [], timeSize);
          Z = dlarray(Z, newLabel);
        → Özeti: X: [B*C, U, timeSize] → [B, C*U, timeSize], label’i 'CBT' vb. yap.
    Python karşılığı:
      - x: [B*C, U, timeSize]
      - B*C = x.shape[0], U = x.shape[1], timeSize = x.shape[2]
      - “？” Bu satır aslında: “B = original_batch, C = original_channels, U = some unit”
        → B*C tek tek bölünüp → [B, C*U, timeSize].
    Burada, “timeSize” zaten biliniyor. “newLabel” PyTorch’ta etiket değil; 
    biz output’ta [B, C*U, timeSize] tensor’u üretip döndüreceğiz.
    """
    def __init__(self, time_size: int, new_label: str = "CBT"):
        super(RelabelLayer, self).__init__()
        self.time_size = time_size
        self.new_label = new_label  # PyTorch’ta önemli değil, sadece bilgi.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*C, U, time_size]
        B*C = x.shape[0], U = x.shape[1], time_size = x.shape[2]
        Orijinal B ve C değerlerini bilmediğimiz için, genellikle
        B = x.shape[0] // C? Ama burada, B ve C’yi önceki 
        layer’lardan takip edip el ile vermek zorundayız.
        Ancak MATLAB’daki bağlamda:
         - x: [freq*batch, 1, time] (örnek)
         - timeSize: time (MATLAB kodunda “time+2” → 96 vb.)
         - reshape → [batch, freq, timeSize]
        Yani:
         x: [freq*batch, U, timeSize] 
         → → [batch, freq*U, timeSize] 
        Python’da:
        """
        B_times_C, U, T = x.shape
        T_expected = self.time_size
        assert T == T_expected, f"Zaman boyutu beklenenden farklı: {T} vs {T_expected}"

        # batch_sayisi * channel_sayisi = B_times_C
        # Orijinal bölünmüş batch ve channel bilgisi, 
        # önceki layer’da “B” ve “C” bilinebilir olmalı. 
        # Dolayısıyla, bu metodu kullanmadan önce 
        # x’i [B, C, U, T] boyutuna yeniden getirmeli 
        # veya B ve C’yi dışarıdan aktarmalıyız.

        # Aşağıda örnek bir yaklaşım: 
        # Eğer “C”=2 ise (örneğin real/imag mask), 
        # B = B_times_C // 2, C=2
        # Bu bilginin network’ün içinde doğru tutulması gerekir.
        # Burada c=2 kabul edelim (MATLAB’da “fullyConnectedLayer(2)” → 2 out).
        C = 2
        B = B_times_C // C

        # 1) [B*C, U, T] → [B, C, U, T]
        y = x.view(B, C, U, T).contiguous()
        # 2) [B, C, U, T] → [B, C*U, T]
        y = y.view(B, C * U, T).contiguous()  # [B, (C*U), T]

        return y  # [batch, channel*unit, timeSize]


class FinalLayer(nn.Module):
    """
    MATLAB’daki finalLayer:
      - Giriş: X [2, 1285, 94] format 'CBT'
      - predict:
          formated = reshape(X, size(X,1), [], 257, timeSize);
          formated = formated(:,:,:,3:end);
          formated = dlarray(formated,"CBSS");
        Yani:
          1) X shape: [2, (some), time] 
            →  reshape ile [2, ?, 257, timeSize] 
            →  timeSize’den 3’ten sonrasını al (3:end) 
            →  Çıktı format: [2, ?, 257, timeSize-2]
        Python’da adımlar:
          - x: [2, (batch*?), timeSize]
          - batch*? = x.shape[1] // 257  (çünkü 257 freq bin)
          - timeSize: x.shape[2] 
        """
    def __init__(self, time_size: int, n_freq: int = 257):
        super(FinalLayer, self).__init__()
        self.time_size = time_size
        self.n_freq = n_freq  # 257

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [2, batch_times_, timeSize]
        Burada “batch_times_ = batch * n_freq” 
        Adımlar:
         1) [2, batch_times_, timeSize] → [2, batch, n_freq, timeSize]
         2) formated = formated[:, :, :, 2:]  # PyTorch’da index 2 karşılık MATLAB’daki 3
         3) → [2, batch, n_freq, timeSize-2]
         4) Çıktı → [batch, 2, n_freq, timeSize-2] veya [batch, 2, timeSize-2, n_freq], 
             MATLAB’da “CBSS”: C=2, B=?, S=257, S=time-2. 
             PyTorch’da genelde [B, C, F, T] biçimidir.
        """
        C, B_times, T = x.shape
        assert C == 2, "FinalLayer girişinin ilk boyutu 2 olmalı (real/imag mask boyutu)."
        assert T == self.time_size, f"Zaman boyutu uyumsuz: {T} vs {self.time_size}"

        # 1) batch sayısını bul:
        batch = B_times // self.n_freq
        # 2) [2, batch*n_freq, timeSize] → [2, batch, n_freq, timeSize]
        y = x.view(2, batch, self.n_freq, T).contiguous()
        # 3) Zaman ekseninde 2’den sonrası: (MATLAB: “3:end” demek → PyTorch: index 2 itibariyle)
        y = y[:, :, :, 2:].contiguous()  # [2, batch, 257, timeSize-2]

        # 4) PyTorch’ta sıklıkla [batch, channels, freq, time] desiriz:
        y = y.permute(1, 0, 2, 3).contiguous()  # [batch, 2, 257, timeSize-2]

        return y  # [B, 2, 257, T-2]


################################################################################
# 2. FullSubNet BENZERİ BİR MİMARİ ŞABLONU
################################################################################

class FullSubNet(nn.Module):
    def __init__(self,
                 num_features: int = 257,
                 time_steps: int = 96,
                 num_hidden_fb: int = 512,
                 num_hidden_sb: int = 384):
        super(FullSubNet, self).__init__()
        self.num_features = num_features
        self.time_steps = time_steps
        self.num_hidden_fb = num_hidden_fb
        self.num_hidden_sb = num_hidden_sb

        # Full-band branch:
        self.pad = PadLayer()
        self.norm1 = NormLayer(is_special=False)
        self.lstm1 = nn.LSTM(input_size=self.num_features,
                             hidden_size=self.num_hidden_fb,
                             batch_first=True,
                             bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=self.num_hidden_fb,
                             hidden_size=self.num_hidden_fb,
                             batch_first=True,
                             bidirectional=False)
        self.fc1 = nn.Linear(self.num_hidden_fb, self.num_features)
        self.relu = nn.ReLU(inplace=True)

        # Sub-band branch:
        self.unfold = UnfoldLayer(num_neighbors=15)
        subband_size = 2 * 15 + 1  # 31
        self.lstm3 = nn.LSTM(input_size=subband_size,
                             hidden_size=self.num_hidden_sb,
                             batch_first=True,
                             bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=self.num_hidden_sb,
                             hidden_size=self.num_hidden_sb,
                             batch_first=True,
                             bidirectional=False)
        self.fc2 = nn.Linear(self.num_hidden_sb, 2)

        # Final layer:
        self.final = FinalLayer(time_size=self.time_steps + 2,
                                n_freq=self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, num_features, time_steps]  yani [B,1,257, T=92]
        """
        B = x.shape[0]
        # 1) [B, 1, F, T] → [B, F, T]
        x = x.squeeze(1)  # [B, 257, 92]
        # (print(f"x shape after squeeze: {x.shape}")  # debug)

        # 2) Pad: [B, F, T] → [B, F, T+2]
        x_padded = self.pad(x)  # [B, 257, 94]
        # (print(f"x_padded shape: {x_padded.shape}")

        # 3) Unfold (Sub-band features): [B, F, T] → [B, F, 31, T]
        subband_feats = self.unfold(x)  # [B, 257, 31, 92]
        # (print(f"subband_feats shape: {subband_feats.shape}")

        # ------------------------------
        # Full-Band Branch (LHS)
        # ------------------------------
        fb = self.norm1(x_padded)               # [B, 257, 94]
        fb_seq = fb.permute(0, 2, 1).contiguous()  # [B, 94, 257]
        fb_out1, _ = self.lstm1(fb_seq)         # [B, 94, 512]
        fb_out2, _ = self.lstm2(fb_out1)        # [B, 94, 512]
        fb_fc = self.fc1(fb_out2)               # [B, 94, 257]
        fb_act = self.relu(fb_fc)               # [B, 94, 257]
        fb_act = fb_act.permute(0, 2, 1).unsqueeze(2).contiguous()  # [B, 257, 1, 94]

        # ------------------------------
        # Sub-Band Branch (RHS) – Düzeltilmiş
        # ------------------------------
        # 4) subband_feats: [B, 257, 31, 92]
        rhs = subband_feats
        #    Zaman eksenini pad’le: (1,1) => [B, 257, 31, 94]
        rhs = torch.nn.functional.pad(rhs, (1, 1), mode='constant', value=0)  # [B, 257, 31, 94]
        # (print(f"rhs shape after pad: {rhs.shape}")

        # 5) LSTM3 için sequence formatına getir:
        #    rhs: [B, 257, 31, 94]
        #    Permute ile [B, 257, 94, 31]
        rhs_perm = rhs.permute(0, 1, 3, 2).contiguous()  # [B, 257, 94, 31]
        #    View ile [B*257, 94, 31]
        B_fuse, F_fuse, T_fuse, C_fuse = rhs_perm.shape  # B_fuse=B, F_fuse=257, T_fuse=94, C_fuse=31
        seq = rhs_perm.view(B_fuse * F_fuse, T_fuse, C_fuse)  # [B*257, 94, 31]
        # (print(f"seq shape before LSTM3: {seq.shape}")

        # 6) Sub-band LSTM katmanı
        sb_out1, _ = self.lstm3(seq)  # [B*257, 94, 384]
        sb_out2, _ = self.lstm4(sb_out1)  # [B*257, 94, 384]

        # 7) FC2 → [B*257, 94, 2]
        sb_fc = self.fc2(sb_out2)  # [B*257, 94, 2]
        #    Permute ile [2, B*257, 94]
        sb_fc = sb_fc.permute(2, 0, 1).contiguous()  # [2, B*257, 94]

        # 8) B*257= batch_size*frequency = 2*257 = 514; time=T_fuse=94
        #    [2, 514, 94] → FinalLayer’a verilecek
        #    FinalLayer bu input’u [2, 514, 94] olarak alacak
        sb_out = sb_fc  # [2, 514, 94]

        # ------------------------------
        # Final Katman
        # ------------------------------
        # FinalLayer, input olarak [B, C*F, T+2] bekliyor.
        # Burada: C=2, F=257, T+2=94 → [2, 514, 94]
        final_out = self.final(sb_out)  # [B, 2, 257, 92]
        return final_out