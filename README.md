# AIGC Examples

Educational implementations of modern generative AI algorithms for image synthesis.

## Setup

```bash
conda env create -f environment.yml
conda activate aigc-examples
```

## Algorithms

| Algorithm | Description | Train | Inference | Details |
|-----------|-------------|-------|-----------|---------|
| **DDPM** | Classic diffusion model with fixed noise schedule and iterative denoising | [train.py](ddpm/train.py) | [inference.ipynb](ddpm/inference.ipynb) | - |
| **Score Matching** | Score-based modeling with Langevin dynamics and multiple noise levels | [train.py](score_matching/train.py) | [inference.ipynb](score_matching/inference.ipynb) | [README](score_matching/README.md) |
| **Flow Matching** | Neural ODE with conditional flow matching and Gaussian probability paths | [train.py](flow_matching/train.py) | [inference.ipynb](flow_matching/inference.ipynb) | [README](flow_matching/README.md) |
| **Rectified Flow** | Optimal transport with straight-line trajectories for efficient sampling | [train.py](rectified_flow/train.py) | [inference.ipynb](rectified_flow/inference.ipynb) | [README](rectified_flow/README.md) |

## Dependencies

- Python 3.12
- PyTorch (CUDA)
- torchvision
- numpy, tqdm, matplotlib
- einops, torchdiffeq
