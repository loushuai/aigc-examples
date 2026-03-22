"""VAE training + sampling on 64x64 cartoon faces."""
import torch
import torchvision
from pathlib import Path
from PIL import Image

from vae.model import VAE

IMG_SIZE = 64
DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "data" / "cartoonset100k" / "cartoonset100k"
SAVE_DIR = Path(__file__).resolve().parent
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
LATENT_DIM = 128
KL_WEIGHT = 1.0  # beta in beta-VAE; 1.0 = standard VAE
SAVE_EVERY = 10


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    return Image.open(path).convert("RGB")


def get_data_loader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.CenterCrop(IMG_SIZE),
        torchvision.transforms.ToTensor(),          # [0, 1]
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1]
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=str(DATA_ROOT),
        loader=load_image,
        is_valid_file=lambda x: x.endswith(".png"),
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2,
    )


def train(model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_recon, total_kl, count = 0.0, 0.0, 0

        for bidx, (x, _) in enumerate(loader):
            x = x.to(device)
            x_recon, mu, log_var = model(x)
            recon_loss, kl_loss = VAE.loss(x, x_recon, mu, log_var)
            loss = recon_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            count += 1

            if bidx % 100 == 0:
                print(f"  batch {bidx}/{len(loader)}  recon={recon_loss.item():.4f}  kl={kl_loss.item():.4f}")

        scheduler.step()
        avg_recon = total_recon / count
        avg_kl = total_kl / count
        print(f"Epoch {epoch}/{NUM_EPOCHS}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  total={avg_recon + KL_WEIGHT * avg_kl:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), SAVE_DIR / f"model_epoch_{epoch}.pth")
            save_samples(model, device, SAVE_DIR / f"samples_epoch_{epoch}.png")
            save_reconstructions(model, device, loader, SAVE_DIR / f"recon_epoch_{epoch}.png")


@torch.no_grad()
def save_samples(model, device, filename, n=16):
    model.eval()
    samples = model.sample(n, device)
    samples = (samples * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    grid = torchvision.utils.make_grid(samples, nrow=4)
    torchvision.transforms.functional.to_pil_image(grid).save(filename)
    print(f"Saved {n} samples to {filename}")


@torch.no_grad()
def save_reconstructions(model, device, loader, filename, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    x_recon, _, _ = model(x)
    # Interleave original and reconstruction
    combined = torch.stack([x, x_recon], dim=1).flatten(0, 1)
    combined = (combined * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(combined, nrow=n)
    torchvision.transforms.functional.to_pil_image(grid).save(filename)
    print(f"Saved reconstructions to {filename}")


def main():
    device = get_device()
    print(f"Device: {device}")

    loader = get_data_loader()
    print(f"Dataset: {len(loader.dataset)} images, {len(loader)} batches/epoch")

    model = VAE(in_channels=3, latent_dim=LATENT_DIM).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, loader, device)

    torch.save(model.state_dict(), SAVE_DIR / "model_final.pth")
    save_samples(model, device, SAVE_DIR / "samples_final.png", n=16)


if __name__ == "__main__":
    main()
