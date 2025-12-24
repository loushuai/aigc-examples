import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model_configs import instantiate_model
from models.schedulers import DDPM_Scheduler
from dataset.dataset import get_data_loader
from utils.utils import get_device
from utils.arg_parser import get_args_parser
import torch.optim as optim


def train(
    model: torch.nn.Module,
    scheduler: DDPM_Scheduler,
    data_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),) -> None:
    model.train(True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss(reduction='mean')

    batch_size = data_loader.batch_size if data_loader.batch_size else 1
    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(data_loader):
            x = x.to(device)
            t_step = torch.randint(0, scheduler.num_time_steps, (batch_size,))
            t = t_step / scheduler.num_time_steps
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t_step].view(batch_size,1,1,1).to(device)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if bidx % 20 == 0:
                print(f"Batch: {bidx} / {len(data_loader)}, Loss: {loss.item()}")

        print(f"Epoch: {i+1} Loss: {total_loss/len(data_loader)}")
        lr_scheduler.step()

    torch.save(model.state_dict(), f"model_epoch.pth")


def main(args):
    device = get_device()
    data_loader = get_data_loader()
    model = instantiate_model("ddpm")
    model.to(device)
    scheduler = DDPM_Scheduler(num_time_steps=1000)
    train(model=model, scheduler=scheduler, data_loader=data_loader, device=device)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
