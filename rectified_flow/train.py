import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_device
from torch.utils.data import DataLoader

import losses
import sde_lib
from dataset.dataset import get_data_loader
from models.model_configs import instantiate_model


def train(model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 2e-4,
    device: torch.device = torch.device("cpu"),) -> None:
    model.train(True)

    sde = sde_lib.RectifiedFlow()
    loss_fn = losses.get_rectified_flow_loss_fn(sde, train=True, reduce_mean=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95)
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=num_epochs,
        start_factor=1.0,
        end_factor=1e-4,
    )

    max_loop = 400
    data_len = min(len(data_loader), max_loop)
    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(data_loader):
            optimizer.zero_grad()
            x = x.to(device)

            x = x * 2.0 - 1.0

            loss = loss_fn(model, x)
            loss.backward()

            optimizer.step()
            if bidx % 20 == 0:
                print(f"Batch: {bidx} / {data_len}, Loss: {loss.item()}")

            total_loss += loss.item()

            if bidx >= max_loop:
                break

        print(f"Epoch: {i+1} Loss: {total_loss/(max_loop if max_loop < len(data_loader) else len(data_loader))}")
        lr_scheduler.step()

        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{i+1}.pth")


def main():
    device = get_device()
    data_loader = get_data_loader()
    model = instantiate_model("score_matching")
    model.to(device)

    train(model, data_loader, device=device)


if __name__ == "__main__":
    main()
