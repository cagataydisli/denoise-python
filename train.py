import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils.speech_features import DNSDataset
from models.denoise_model import FullSubNet

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==> Eğitimde kullanılacak cihaz: {device}")

    # 1) DNSDataset ve train/val split
    dataset = DNSDataset(
        clean_dir=os.path.join(args.data_dir, "clean_fullband"),
        noise_dir=os.path.join(args.data_dir, "noise_fullband"),
        sample_rate=8000,
        expected_length=3,
        window_length=512,
        overlap=256,
        n_fft=512
    )
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    # 2) Model, criterion, optimizer
    model = FullSubNet(
        num_features=257,
        time_steps=args.time_steps,
        num_hidden_fb=768,
        num_hidden_sb=512
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 3) Scheduler: val loss platoda kalırsa lr*=0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=1
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (predictor, target) in enumerate(train_loader, 1):
            predictor = predictor.to(device)
            target    = target.to(device)

            optimizer.zero_grad()
            output = model(predictor)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                print(f"Epoch [{epoch}/{args.epochs}] "
                      f"Step [{batch_idx}/{len(train_loader)}] "
                      f"Train Loss: {avg_loss:.6f}")
                running_loss = 0.0

        # 4) Validation aşaması
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for predictor, target in val_loader:
                predictor = predictor.to(device)
                target    = target.to(device)
                output = model(predictor)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{args.epochs}] Validation Loss: {val_loss:.6f}")

        # 5) Scheduler’ı güncelle
        scheduler.step(val_loss)

        # 6) En iyi modeli kaydet / Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_epoch_{epoch}.pth"))
            print(f"==> Yeni en iyi model kaydedildi: epoch {epoch}, val_loss {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"==> Validation Loss {args.patience} epoch boyunca iyileşmedi. Training kesiliyor.")
                break

    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--time_steps', type=int, default=92)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_big')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3,
                        help="validation loss kaç epoch artarsa training durmalı")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
