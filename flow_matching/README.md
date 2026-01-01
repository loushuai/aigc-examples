# Flow Matching

The goal of generative modeling is sampling from a data distribution $`p_\text{data}`$. Flow matching is a type of methodologies that achive the goal via the transformation of samples from a simple distribution $`p_\text{init}`$, such as the Gaussian $`\mathcal{N}(0, I_d)`$, to samples from the target distribution $`p_\text{data}`$. The transformation velocity fields can be estimated by neural network model and we could predicted samples by simulating ordinary differential equations.

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
where $`\psi_t^{\theta}`$ describes the flow induced by $`u_t^{\theta}`$.

## Training Target
An intuitive way of obtaining $`u_t^{\theta} \approx u_t^{\text{target}}`$ is to use a mean-squared error
```math
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, \, x \sim p_t} \big[ \| u_t^{\theta}(x) - u_t^{\text{target}}(x) \|^2 \big]
```
where $`u_t^{\text{target}}(x)`$ is the training target that we would like to approximate. $`\text{Unif}=\text{Unif}_{[0,1]}`$ is the uniform distribution on the interval [0,1], and by $`\mathbb{E}`$ the expected value of a random variable. $`p_t(x)`$ is the marginal probability path which we obtain by first sampling a data point $`z \sim p_\text{data}`$ from the data distribution and then sampling from $`p_t(\,\cdot \mid z)`$.
```math
z \sim p_{\text{data}}, \quad x \sim p_t(\cdot \mid z) \;\;\Rightarrow\;\; x \sim p_t
```
```math
p_t(x) = \int p_t(x \mid z) \, p_{\text{data}}(z) \, \mathrm{d}z
```
Theorem 10 in [[1]][#reference-1] indicates that $`u_t^{\text{target}}(x)`$ can be computed from the conditional velocity field $`u_t^{\text{target}}(x \mid z)`$ by
```math
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z) \, \frac{p_t(x \mid z) \, p_{\text{data}}(z)}{p_t(x)} \, \mathrm{d}z
```
Unfortunaetly, we cannot compute it efficiently because the above integral is intractable. We can compute the conditional flow matching loss instead:
```math
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, \, z \sim p_{\text{data}}, \, x \sim p_t(\cdot \mid z)} \big[ \| u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z) \|^2 \big]
```
It can be approved ([[1]][#reference-1] Theorem 18)
```math
\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C
```
where C is independent of $`\theta`$. Therefore, their gradients coincide:
```math
\nabla_{\theta} \mathcal{L}_{\text{FM}}(\theta) = \nabla_{\theta} \mathcal{L}_{\text{CFM}}(\theta)
```
In particular, for the minimizer $`\theta^*`$ of $`\mathcal{L}_{\text{CFM}}(\theta)`$, it will hold that $`u_t^{\theta^*} = u_t^{\text{target}}`$.

## Gaussian Conditional Probability Paths
If we define Gaussian probability paths $`p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z; \beta_t^2 I_d)`$, then we may sample from the conditional path via
```math
\epsilon \sim \mathcal{N}(0, I_d) \implies x_t = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot \mid z)
```
It can be derived that
```math
u_t^{\text{target}}(x \mid z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
```
where $`\dot{\alpha}_t = \partial_t \alpha_t`$ and $`\dot{\beta}_t = \partial_t \beta_t`$ denote respective time derivatives of $`\alpha_t`$ and $`\beta_t`$. Plugging in this formula, the conditional flow matching loss reads
```math
\begin{aligned}
\mathcal{L}_{\mathrm{CFM}}(\theta)
&= \mathbb{E}_{t \sim \mathrm{Unif},\; z \sim p_{\mathrm{data}},\; x \sim \mathcal{N}(\alpha_t z,\, \beta_t^2 I_d)}
\bigl[ \,\|\, u_t^{\theta}(x) - \Bigl( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \Bigr) z - \frac{\dot{\beta}_t}{\beta_t} x \,\|^2 \bigr] \\
&= \mathbb{E}_{t \sim \mathrm{Unif},\; z \sim p_{\mathrm{data}},\; \epsilon \sim \mathcal{N}(0, I_d)}
\bigl[ \,\|\, u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - \bigl(\dot{\alpha}_t z + \dot{\beta}_t \epsilon\bigr) \,\|^2 \bigr]
\end{aligned}
```
Notice how easy it is to calculate $`\mathcal{L}_{\mathrm{CFM}}`$: We sample a data point $`z`$ from training dataset, sample some noise $`\epsilon`$ and then we take a mean squared error. To make it more concrete, let's take a special case of $`\alpha_t=t`$ and $`\beta_t=1-t`$. The corresponding probability
```math
p_t(x \mid z) = \mathcal{N}(t z, (1-t)^2 I_d)
```
which is sometimes referred to as the (Gaussian) CondOT probability path. Then we have $`\dot{\alpha}_t = 1`$, $`\quad \dot{\beta}_t = -1`$, so that
```math
\mathcal{L}_{\mathrm{cfm}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\; z \sim p_{\mathrm{data}},\; \epsilon \sim \mathcal{N}(0, I_d)}
\bigl[ \,\|\, u_t^{\theta}(t z + (1 - t)\epsilon) - (z - \epsilon) \,\|^2 \bigr]
```


#### Algorithm 1: Flow Matching Training Procedure  
*(here for Gaussian CondOT path \( p_t(x|z) = \mathcal{N}(tz, (1 - t)^2) \))*

---

**Require:** A dataset of samples \( z \sim p_{\text{data}} \), neural network \( u_t^\theta \)

1. **for** each mini-batch of data **do**
2. &nbsp;&nbsp;&nbsp;&nbsp;Sample a data example \( z \) from the dataset.
3. &nbsp;&nbsp;&nbsp;&nbsp;Sample a random time \( t \sim \text{Unif}_{[0,1]} \).
4. &nbsp;&nbsp;&nbsp;&nbsp;Sample noise \( \epsilon \sim \mathcal{N}(0, I_d) \).
5. &nbsp;&nbsp;&nbsp;&nbsp;Set \( x = tz + (1 - t)\epsilon \).
6. &nbsp;&nbsp;&nbsp;&nbsp;Compute loss:

```math
\mathcal{L}(\theta) = \|u_t^\theta(x) - (z - \epsilon)\|^2
\quad \text{(General case: } x \sim p_t(\cdot|z), \; \|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\text{)}
```

## Inference
After training, we can generate samples from the learned flow model by solving the ODE from $`t=0`$ to $`t=1`$ using an ODE solver such as the Euler method or the Runge-Kutta method. The procedure is summarized in the following algorithm:


#### Algorithm 2: Sampling from a Flow Model with Euler method

---

**Require:** Neural network vector field \( u_t^\theta \), number of steps \( n \)

1. Set \( t = 0 \)
2. Set step size \( h = \frac{1}{n} \)
3. Draw a sample \( X_0 \sim p_{\text{init}} \)
4. **for** \( i = 1, \dots, n \) **do**
5. &nbsp;&nbsp;&nbsp;&nbsp;\( X_{t+h} = X_t + h u_t^\theta(X_t) \)
6. &nbsp;&nbsp;&nbsp;&nbsp;Update \( t \leftarrow t + h \)
7. **end for**
8. **return** \( X_1 \)

---


## Reference
[1] [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070)

[#reference-1]: https://arxiv.org/abs/2506.02070

