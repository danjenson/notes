---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 17: Sequential Importance Sampling & Non-linear Time Series (2022-12-01)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

### Example 1

- Our goal is to generate $k$ weighted samples from
  distribution $\mathscr{D}(\vec{Z}_n\mid\vec{Y}_n)$.

  $$
  \begin{aligned}
  x_i
  &\sim \operatorname{Normal}\left(\mu,\Sigma\right) \\
  x_i
  &=(z_i,y_i)\quad y_i\text{ is observed, and }z_i\text{ is unobserved} \\
  n
  &=269\rightarrow 88\text{ complete, }40\text{ missing 6th component} \\
  \vec{X}_i
  &=(x_1,\ldots,x_i)\quad\text{partial dataset containing up to index }i \\
  \vec{Z}_i
  &=(z_1,\ldots, z_i) \\
  \vec{Y}_i
  &=(y_1,\ldots,y_i) \\
  \end{aligned}
  $$

- Denote $(\vec{Z}_n^{(1)},w^{(1)}),\ldots,(\vec{Z}_n^{(k)},w^{(k)})$.
- Then $$\mathscr{D}(\mu,\Sigma\mid \vec{Y}_n)=\sum_{k=1}^K
  \alpha_k\mathscr{D}(\mu\Sigma\mid \vec{X}_n^{(k)})$$ where $$\alpha_k\propto \frac{w^{(k)}}{\sum_{k=1}^K w_n^{(k)}}$$.

## **Use Sequential Importance Sampling (SIS)**

$\mathscr{D}(\vec{Z}_n\mid\vec{Y}_n)=\pi_n(\vec{Z}_n)$

- Set $\pi_i(\vec{Z}_i)=\mathscr{D}(\vec{Z}_i\mid\vec{Y}_i)$ - $\pi_1(\vec{Z}_1)=\mathscr{D}(z_1\mid y_1)$ - $\pi_2(\vec{Z}_2)=\mathscr{D}(z_1,z_2\mid y_1,y_2)$ - $\pi_3(\vec{Z}_3)=\mathscr{D}(z_1,z_2,z_3\mid y_1,y_2,y_3)$ - $\ldots$
- Suppose we know how to sample from $$\pi_{i-1}(\vec{Z}_{i-1})$$, i.e. we have
  $$(\vec{Z}_{i-1}, w_{i-1})$$ from $$\pi_{i-1}(\vec{Z}_{i-1})$$, i.e.
  $\vec{Z}_{i-1}$ is drawn from trial density, $$q_{i-1}(\vec{Z}_{i-1})$$ and
  $$w_{i-1}\propto\frac{\pi_{i-1}(\vec{Z}_{i-1})}{q_{i-1}(\vec{Z}\_{i-1})}$$
- We extend it to $(\vec{Z}_i,w_i)$ by one of two methods:
  1. Method A (preferable if feasible because it uses more data/information)
  - Draw $$\vec{Z}_i$$ from $$\mathscr{D}(\vec{Z}_i\mid
         \vec{Z}_{i-1},\vec{Y}_{i-1}, y_i)$$
  - Update weight: $$w_i=w_{i-1}p(\vec{Y}_i\mid
    \vec{Z}_{i-1},\vec{Y}_{i-1})=w_{i-1}p(y_i\mid\vec{X}_{i-1})$$
  2. Method B
  - Draw $$\vec{Z}_i$$ from $$\mathscr{D}(z_i\mid \vec{X}_{i-1})$$
  - Set $w_i=w_{i-1}p(y_i\mid\vec{X}_{i-1}, z_i)$
- **Proof of Correctness**:

  1. Method A

  - $w_i$ should be
    $$\propto\frac{\pi_i(\vec{Z}_i)}{q_{i-1}(\vec{Z}_{i-1})p(z_i\mid \vec{X}_{i-1},y_i)}$$
  - Now,

    $$
    \begin{aligned}
    \pi_i(\vec{Z}_i)
    &=p(\vec{Z}_i\mid\vec{Y}_i) \\
    &\propto p(\vec{Z}_i\mid \vec{Y}_i) \\
    &=p(\vec{X}_i) \\
    &=p(\vec{X}_{i-1})p(z_i,y_i\mid \vec{X}_{i-1}) \\
    &\propto \underbrace{p(\vec{Z}_{i-1}\mid\vec{Y}_{i-1})}_{\pi_{i-1}(\vec{Z}_{i-1})}\left[p(z_i\mid\vec{X}_{i-1},y_i)p(y_i\mid\vec{X}_{i-1})\right] \\
    \end{aligned}
    $$

  - Hence, $$w_i\propto\frac{\pi_{i-1}(\vec{Z}_{i-1})p(z_i\mid\vec{X}_{i-1},y_i)p(y_i\mid\vec{X}_{i-1})}{\pi_{i-1}(\vec{Z}_{i-1})p(z_i\mid\vec{X}_{i-1},y_i)}\propto w_{i-1}p(y_i\mid\vec{X}_{i-1})$$
  - The proof is similar for method B.

### Example 1 continued...

- Applying Method A
- To do this, we need some facts about the multivariate normal
- $$\mathscr{D}(x_i\mid \vec{X}_{i-1})$$ is multivariate t-distribution
  - Chapter 3 in Bayesian Data Analysis
- **Definition**: A random variable $x\in \mathbb{R}^d$ is
  $t_d(\mu,\Sigma_{d\times d},\nu)$ distribution if it has density
  $$\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\left(\nu\pi\right)^{\frac{d}{2}}\sqrt{\det(\Sigma)}}\left(1+\frac{1}{\nu}(\vec{x}-\vec{\mu})^\intercal\Sigma^{-1}(\vec{x}-\vec{\mu})\right)^{-(\nu+p)/2}$$
- Then if $x=\begin{bmatrix}Y\\ Z\end{bmatrix}$ and $\vec{Y}\in \mathbb{R}^{d_1}$ and
  $\vec{Z}\in \mathbb{R}^{d_2}$ where $d_1+d_2=d$ then we have $Y\sim
  t_{d_1}(\mu_1,\Sigma_{1,1},\nu)$ with $$\vec{\mu}=\begin{bmatrix}\mu_1 \\\mu_2\end{bmatrix},\Sigma=\begin{bmatrix}\Sigma_{1,1} & \Sigma_{1,2} \\ \Sigma_{2,1} & \Sigma_{2,2}\end{bmatrix}$$
- $\vec{Z}\mid\vec{Y}=y\sim t_{d_2}(\mu_{2\mid 1},c\Sigma_{2\mid1},\nu+d_1)$
  where $\mu_{2\mid1}=\mu_2+\Sigma_{2,1}\Sigma_{1,1}^{-1}(y-\mu_1),\Sigma_{2\mid
  1}=\Sigma_{2,2}-\Sigma_{2,1}\Sigma_{1,1}^{-1}\Sigma_{1,2}$ and
  $c=\frac{\nu+(y-\mu_1)^\intercal\Sigma_{1,1}^{-1}(y-\mu_1)}{\nu+d_1}$
- $x=\mu +\frac{1}{\sqrt{V}}U$ where $V\sim \operatorname{Gamma}\left(\frac{\nu}{2},\frac{\nu}{2}\right)$ and $U\sim \operatorname{Normal}\left(0,\Sigma\right)$
- Using these facts about the multivariate t-distribution, you can implemented
  method A.
- The of the imputation is very important $\rightarrow$ process data points with least
  missingness first.

## Importance Resampling

- The SIS algorithm builds K segments in parallel
- $(\vec{Z}_1^{(1)},w_1^{(1)})\to(\vec{Z}_2^{(1)},w_2^{(1)})\to(\vec{Z}_3^{(1)},w_3^{(1)})\cdots(\vec{Z}_i^{(1)},w_i^{(1)})\cdots$
- $(\vec{Z}_1^{(2)},w_1^{(2)})\to(\vec{Z}_2^{(2)},w_2^{(2)})\to(\vec{Z}_3^{(2)},w_3^{(2)})\cdots(\vec{Z}_i^{(2)},w_i^{(2)})\cdots$
- $\ldots$
- $(\vec{Z}_1^{(K)},w_1^{(K)})\to(\vec{Z}_2^{(K)},w_2^{(K)})\to(\vec{Z}_3^{(K)},w_3^{(K)})\cdots(\vec{Z}_i^{(K)},w_i^{(K)})\cdots$
- This is a weighted sample of from $\pi_i(\vec{Z}_i)$
  - To know if this is a good weighted sample, compute
    $$k_{\text{eff}}=\frac{k}{1+(cv)^2}$$ where $cv=$ coefficient of variation of the
    weights. This will fail when some points have very large weights and other
    have much lower weights.
  - If $k_\text{eff}$ is small, then do importance resampling.
  - Resample $$\{\vec{Z}_{i-1}^{(k)},k=1,\ldots, K\}$$ using the Importance
    Sampling weights to get a new set of samples, now equally weighted.

### Example 2 (Time Series)

- Consider a non-linear system described by the following equations:

$$
\begin{aligned}
z_{t+1}
&=f_t(z_t,u_t) \\
y_t
&=h_t(z_t)+v_t \\
u_t
&\sim \phi(\cdot)\;\text{i.i.d.} \\
v_t
&\sim \psi(\cdot)\;\text{i.i.d.}
\end{aligned}
$$

- Imagine the $z_t$s are unobserved latent states and $y_t$s are observed data
  generated from $z_t$.
- Based on $\vec{y}_t$, how can we infer $\vec{z}_t$s?
- Paper by Gordon, Salman, and Smith (1993) proposed the particle filter to do
  this inference.
- Let $\vec{Z}_t=(z_1,\ldots,z_t)$ be a weighted sample from
  $p(\vec{Z}_t\mid\vec{Y}_t)$.
  - Apply method B to impute the next state.
    - Draw $$z_{t+1}\mid\vec{Z}_t,\vec{Y}_t$$ (draw $$u_t\sim\phi(\cdot)$$ and set
      $$z_{t+1}=f_t(z_t,u_t)$$
    - Then $$w_{t+1}=w_tp(y_{t+1}\mid \vec{Z}_t,\vec{Y}_t,z_{t+1})=w_t\phi(y_{t+1}-h_t(z_{t+1}))$$
    - Do importance re-sampling every step (this is a special case of SIS).
    - You can do this online.
