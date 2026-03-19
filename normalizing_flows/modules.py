import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg as scipy_linalg


class ActNorm(nn.Module):
    """Activation Normalization layer from Glow.

    Performs an affine transformation with learnable per-channel scale and bias,
    initialized so that the output has zero mean and unit variance given the
    first mini-batch (data-dependent initialization).

    Uses logscale_factor (default 3) for numerical stability, matching the
    TF reference: logs are stored as logs/factor and used as logs*factor.
    """

    def __init__(self, num_channels, logscale_factor=3.0):
        super().__init__()
        self.logscale_factor = logscale_factor
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def initialize(self, x):
        with torch.no_grad():
            # x: (B, C, H, W) — center first, then compute variance
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            x_centered = x - mean
            var = (x_centered ** 2).mean(dim=[0, 2, 3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(
                torch.log(1.0 / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
            )
            self.initialized.fill_(1.)

    def forward(self, x):
        if self.initialized.item() == 0:
            self.initialize(x)
        h, w = x.shape[2], x.shape[3]
        logs = self.log_scale * self.logscale_factor
        log_det = h * w * logs.sum()
        return (x + self.bias) * torch.exp(logs), log_det

    def inverse(self, z):
        logs = self.log_scale * self.logscale_factor
        return z * torch.exp(-logs) - self.bias


class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution from Glow.

    Uses LU decomposition for efficient computation of the log-determinant.
    """

    def __init__(self, num_channels):
        super().__init__()
        # Initialize with a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0]
        p, l, u = scipy_linalg.lu(w_init)
        p = torch.tensor(p, dtype=torch.float32)
        l = torch.tensor(l, dtype=torch.float32)
        u = torch.tensor(u, dtype=torch.float32)
        # Separate into P (fixed), L (lower triangular), U (upper triangular)
        s = torch.diag(u)
        log_s = torch.log(torch.abs(s))
        sign_s = torch.sign(s)
        u_mask = torch.triu(torch.ones_like(u), diagonal=1)

        self.register_buffer('p', p)
        self.register_buffer('sign_s', sign_s)
        self.register_buffer('l_mask', torch.tril(torch.ones_like(l), diagonal=-1))
        self.register_buffer('u_mask', u_mask)
        self.register_buffer('eye', torch.eye(num_channels))

        self.l = nn.Parameter(l)
        self.u = nn.Parameter(u)
        self.log_s = nn.Parameter(log_s)

    def _get_weight(self):
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        w = self.p @ l @ u
        return w

    def forward(self, x):
        w = self._get_weight()
        h, w_dim = x.shape[2], x.shape[3]
        log_det = h * w_dim * self.log_s.sum()
        # 1x1 conv is equivalent to a matrix multiply on the channel dimension
        weight = w.unsqueeze(2).unsqueeze(3)
        z = F.conv2d(x, weight)
        return z, log_det

    def inverse(self, z):
        w = self._get_weight()
        w_inv = torch.inverse(w)
        weight = w_inv.unsqueeze(2).unsqueeze(3)
        return F.conv2d(z, weight)


class CouplingNetwork(nn.Module):
    """Convolutional network used inside affine coupling layers.

    Matches TF reference: internal convs use actnorm, final conv is zero-init
    with a learnable log-scale output multiplier.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=256, logscale_factor=3.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.actnorm1 = ActNorm(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.actnorm2 = ActNorm(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.conv_out.weight.data.zero_()
        self.conv_out.bias.data.zero_()
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        h, _ = self.actnorm1(F.relu(self.conv1(x)))
        h, _ = self.actnorm2(F.relu(self.conv2(h)))
        h = self.conv_out(h)
        h = h * torch.exp(self.logs * self.logscale_factor)
        return h


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer matching TF Glow reference.

    Splits channels into two halves. First half is unchanged, second half
    is transformed via shift and scale predicted from the first half.

    Forward: y1 = x1, y2 = (x2 + t(x1)) * s(x1)   where s = sigmoid(h + 2)
    Inverse: x1 = y1, x2 = y2 / s(y1) - t(y1)
    Log-det: sum(log(s(x1)))
    """

    def __init__(self, num_channels, hidden_channels=256):
        super().__init__()
        half_channels = num_channels // 2
        out_channels = num_channels - half_channels
        self.half_channels = half_channels
        # Output has 2x channels: even indices for shift, odd for scale (pre-sigmoid)
        self.net = CouplingNetwork(half_channels, out_channels * 2, hidden_channels)

    def forward(self, x):
        x1, x2 = x[:, :self.half_channels], x[:, self.half_channels:]
        h = self.net(x1)
        shift = h[:, 0::2]
        log_scale_pre = h[:, 1::2]
        scale = torch.sigmoid(log_scale_pre + 2.0)
        y2 = (x2 + shift) * scale
        log_det = torch.log(scale).sum(dim=[1, 2, 3])
        return torch.cat([x1, y2], dim=1), log_det

    def inverse(self, z):
        z1, z2 = z[:, :self.half_channels], z[:, self.half_channels:]
        h = self.net(z1)
        shift = h[:, 0::2]
        log_scale_pre = h[:, 1::2]
        scale = torch.sigmoid(log_scale_pre + 2.0)
        x2 = z2 / scale - shift
        return torch.cat([z1, x2], dim=1)


class SplitPrior(nn.Module):
    """Learned prior for the split operation, matching TF split2d_prior.

    Predicts mean and log-std for the factored-out channels from the
    kept channels, using a zero-initialized conv with learnable log-scale.
    """

    def __init__(self, num_channels, logscale_factor=3.0):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels * 2, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.logs = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1))
        self.logscale_factor = logscale_factor

    def forward(self, z1):
        h = self.conv(z1) * torch.exp(self.logs * self.logscale_factor)
        mean, log_std = h.chunk(2, dim=1)
        return mean, log_std

    def log_prob(self, z1, z2):
        mean, log_std = self.forward(z1)
        return -0.5 * (np.log(2 * np.pi) + 2 * log_std + (z2 - mean) ** 2 / torch.exp(2 * log_std)).sum(dim=[1, 2, 3])

    def sample(self, z1, temperature=1.0):
        mean, log_std = self.forward(z1)
        eps = torch.randn_like(mean) * temperature
        return mean + torch.exp(log_std) * eps


def squeeze(x):
    """Squeeze operation: trade spatial dimensions for channel dimensions.
    (B, C, H, W) -> (B, 4C, H/2, W/2)
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 4, h // 2, w // 2)
    return x


def unsqueeze(x):
    """Unsqueeze operation: trade channel dimensions for spatial dimensions.
    (B, 4C, H/2, W/2) -> (B, C, H, W)
    """
    b, c, h, w = x.shape
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)
    return x
