# Score Matching

## Langevine Dynamics
The core of score matching is Langevine dynamics:
```math
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon} \mathbf{z}_{t}
```
where $\mathbf{z}\_t \sim \mathcal{N}(0, I)$. The gradient of the logarithm of a distribution is defined as the score.

Given a fixed step size $\epsilon>0$ and an initial value $\tilde{\mathbf{x}}\_0\sim\pi(\mathbf{x})$ with $\pi$ being an prior distribution, we can use this process to produce samples from a probability density $p(x)$ using only the score function $\nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}\_{t-1})$.\
You can find an intuitve introduction of Langevine dynamics in <a name="reference-1"> [1] </a>.

## Denoising Score Matching
In order to obtain samples from $p_{data}(x)$


## Reference
[[1]](#reference-1) [A Simplified Overview of Langevin Dynamics](https://friedmanroy.github.io/blog/2022/Langevin/)
