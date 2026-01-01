# Flow Matching

The goal of generative modeling is sampling from a data distribution $`p_\text{data}`$. Flow matching is a type of methodologies that achive the goal via the transformation of samples from a simple distribution $`p_\text{init}`$, such as the Gaussian $`\mathcal{N}(0, I_d)`$, to samples from the target distribution $`p_\text{data}`$.

## Flow Models
A flow model is described by the ODE
```math
X_0 \sim p_{\text{init}} 
```
```math
\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\theta}(X_t)
```
where the vector field $`u_t^{\theta}`$ is a neural network with parameters $`\theta`$. Our goal is to make the endpoint $`X_1`$ of the trajectory have distribution $`p_{\text{data}}`$, i.e.
```math
X_1 \sim p_{\text{data}} \;\;\Longleftrightarrow\;\; \psi_1^{\theta}(X_0) \sim p_{\text{data}}
```

