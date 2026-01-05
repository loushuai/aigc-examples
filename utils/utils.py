import torch
from torch import Tensor
from typing import Optional


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def gradient(
    output: torch.Tensor,
    x: torch.Tensor,
    grad_outputs: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))
