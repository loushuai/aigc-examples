"""Minimal Glow training + sampling on cartoon faces."""
import math
import torch
import torchvision
from pathlib import Path
from PIL import Image

from normalizing_flows.model import GlowModel

IMG_SIZE = 64
DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "data" / "cartoonset100k" / "cartoonset100k"
SAVE_DIR = Path(__file__).resolve().parent
NUM_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-4
MAX_BATCHES = None  # use all batches per epoch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_crop(path):
    """Load RGBA image, crop to non-transparent bounding box, convert to RGB."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bbox = img.getchannel("A").getbbox()
        if bbox is not None:
            img = img.crop(bbox)
    return img.convert("RGB")


def get_data_loader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.CenterCrop(IMG_SIZE),
        torchvision.transforms.ToTensor(),  # [0, 1]
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=str(DATA_ROOT),
        loader=load_and_crop,
        is_valid_file=lambda x: x.endswith(".png"),
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2,
    )


def train(model, loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_epochs = 5
    lr_lambda = lambda epoch: (
        epoch / warmup_epochs if epoch < warmup_epochs
        else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    n_pixels = 3 * IMG_SIZE * IMG_SIZE
    ln2 = torch.log(torch.tensor(2.0)).item()

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        count = 0
        for bidx, (x, _) in enumerate(loader):
            if MAX_BATCHES is not None and bidx >= MAX_BATCHES:
                break
            x = x.to(device)
            # Add uniform noise for dequantization, then rescale to [-0.5, 0.5]
            x = (x * 255 + torch.rand_like(x)) / 256.0 - 0.5

            log_prob = model.log_prob(x)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1
            if bidx % 50 == 0:
                bpd = loss.item() / (n_pixels * ln2)
                total_batches = min(len(loader), MAX_BATCHES) if MAX_BATCHES is not None else len(loader)
                print(f"  batch {bidx}/{total_batches}  loss={loss.item():.2f}  bpd={bpd:.3f}")

        scheduler.step()
        avg = total_loss / max(count, 1)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  avg_loss={avg:.2f}  bpd={avg / (n_pixels * ln2):.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), SAVE_DIR / f"model_epoch_{epoch}.pth")
            generate_samples(model, device, filename=SAVE_DIR / f"samples_epoch_{epoch}.png")


@torch.no_grad()
def generate_samples(model, device, n=16, temperature=0.9, filename=None):
    model.eval()
    samples = model.sample(n, device, temperature=temperature)
    # Rescale from [-0.5, 0.5] back to [0, 1]
    samples = (samples + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    img = torchvision.transforms.functional.to_pil_image(grid)
    out = filename or (SAVE_DIR / "samples.png")
    img.save(out)
    print(f"Saved {n} samples to {out}")
    model.train()


def main():
    device = get_device()
    print(f"Device: {device}")

    loader = get_data_loader()
    print(f"Dataset: {len(loader.dataset)} images, {len(loader)} batches/epoch")

    model = GlowModel(in_channels=3, num_levels=3, num_steps=32, hidden_channels=512).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, loader, device)

    # Final generation
    torch.save(model.state_dict(), SAVE_DIR / "model_final.pth")
    generate_samples(model, device, n=16, filename=SAVE_DIR / "samples_final.png")


if __name__ == "__main__":
    main()
