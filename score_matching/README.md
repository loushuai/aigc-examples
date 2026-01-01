# Score Matching

## Langevine Dynamics
The core of score matching is Langevine dynamics:
```math
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon} \mathbf{z}_{t}
```
where $\mathbf{z}\_t \sim \mathcal{N}(0, I)$. The gradient of the logarithm of a distribution is defined as the score.

Given a fixed step size $\epsilon>0$ and an initial value $\tilde{\mathbf{x}}\_0\sim\pi(\mathbf{x})$ with $\pi$ being an prior distribution, we can use this process to produce samples from a probability density $p(x)$ using only the score function $\nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1})$.\
You can find an intuitve introduction of Langevine dynamics in <a name="reference-1"> [1] </a>.

## Denoising Score Matching
In order to obtain samples from $p_{data}(x)$, we can train a score network such that such that $\mathbf{s}_{\theta}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ and then approximately obtain samples with Langevin dynamics using $\mathbf{s}_{\theta}(\mathbf{x})$. This is the key idea of
our framework of score-based generative modeling.<a name="reference-2"> [2] </a>

we can find the best model of the score function by optimizing the following regression problem <a name="reference-3"> [3] </a>
$$\ell(\theta)=\frac{1}{2} \int \Big\| \mathbf{s}_{\theta}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) \Big\|^2 \, p_{\text{data}}(\mathbf{x}) \, d\mathbf{x}$$


## Reference
[[1]](#reference-1) [A Simplified Overview of Langevin Dynamics](https://friedmanroy.github.io/blog/2022/Langevin/)

[[2]](#reference-2) [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

[[3]](#reference-3) [Estimation of Non-Normalized Statistical Models
by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
