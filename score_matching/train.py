import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import get_data_loader

from models.model_configs import instantiate_model
from models.cond_refinenet_dilated import CondRefineNetDilated
from models.refinenet_dilated_baseline import RefineNetDilated
from utils.utils import get_device
from utils.arg_parser import get_args_parser
from dsm import anneal_dsm_score_estimation, dsm_score_estimation


def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 2e-4,
    device: torch.device = torch.device("cpu"),) -> None:
    model.train(True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95)
    )
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=num_epochs,
        start_factor=1.0,
        end_factor=1e-4,
    )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    sigma_begin = 30
    sigma_end = 0.01
    T = 100
    sigma = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               T))).float().to(device)

    max_loop = 400
    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(data_loader):
            optimizer.zero_grad()
            x = x.to(device)

            x = x * 2.0 - 1.0
            sample = x / 256. * 255. + 2.0 * torch.rand_like(x) / 256.
            # sample = x
            t = torch.randint(0, len(sigma), (sample.shape[0],), device=device)
            loss = anneal_dsm_score_estimation(model, sample, t, sigma, 2.0)
            # loss = dsm_score_estimation(model, sample, 0.01)
            loss.backward()
            optimizer.step()

            if bidx % 20 == 0:
                print(f"Batch: {bidx} / {len(data_loader)}, Loss: {loss.item()}")
            total_loss += loss.item()

            if bidx >= max_loop:
                break

        print(f"Epoch: {i+1} Loss: {total_loss/(max_loop if max_loop < len(data_loader) else len(data_loader))}")
        lr_scheduler.step()

        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{i+1}.pth")


def main(args):
    device = get_device()
    data_loader = get_data_loader()
    model = instantiate_model("ddpm")
    # model = RefineNetDilated()
    model.to(device)

    train(model, data_loader, device=device)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    main(args)
