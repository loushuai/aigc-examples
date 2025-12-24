import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import get_data_loader

from models.model_configs import instantiate_model
from utils.utils import get_device
from utils.arg_parser import get_args_parser
from grad_scaler import NativeScalerWithGradNormCount
from flow_matching.path.affine import CondOTProbPath


def train(model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),) -> None:
    model.train(True)

    path = CondOTProbPath()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95)
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=num_epochs,
        start_factor=1.0,
        end_factor=1e-8 / lr,
    )
    loss_scaler = NativeScalerWithGradNormCount()

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(data_loader):
            x = x.to(device)

            # Scaling to [-1, 1] from [0, 1]
            samples = x * 2.0 - 1.0
            noise = torch.randn_like(samples).to(device)
            t = torch.rand(samples.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                loss = torch.pow(model(x_t, t, extra={}) - u_t, 2).mean()
            
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=True,
            )

            if bidx % 20 == 0:
                print(f"Batch: {bidx} / {len(data_loader)}, Loss: {loss.item()}")

        print(f"Epoch: {i+1} Loss: {total_loss/len(data_loader)}")
        lr_scheduler.step()

    torch.save(model.state_dict(), f"model_epoch.pth")

def main(args):
    device = get_device()
    data_loader = get_data_loader()
    model = instantiate_model("flow_matching")
    model.to(device)

    train(model, data_loader, device=device)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)