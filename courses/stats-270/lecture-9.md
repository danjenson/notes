---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 9: Hierarchical & Empirical Bayes"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Example 1:

- $j=1,\ldots, J$
- $Y_{ij}\sim \operatorname{Normal}\left(\theta_j,\sigma^2\right)$ iid
- Let $$\bar{y}_{\cdot j}=\frac{1}{n_j}\sum_{i=1}^{n_j}y_{ij}$$
  $\sigma^2_j=\theta^2/n_j$
- $$
  \mathcal{L}(\theta; x)=\prod_{j=1}^J
  \frac{1}{\sqrt{2\pi\sigma_j^2}}\exp\left(-\frac{1}{2\sigma^2_j}(\bar{y_{\cdot j}-\theta_j})^2\right)
  $$
- If $n_j >> J$, then we treat $\sigma^2,\theta_1,\ldots,\theta_J$ as
  independent parameters, and can use Jeffrey's prior.
- This leads to using $$\bar{y}_{\cdot j}$$ to estimate $\theta_j$ and
  $$\sum_{i=1}^J \left(\sum_j^{n_j}(y_{ij}-\bar{y}_{ij})^2\right)$$ for inference
  of $\sigma^2$ with degrees of freedom (df) = $\sum_{i=1}^J (n_j-1)$.
- This returns the same estimate as MLE, UMVUE (?)
- But if $J$ is large, then $$\mathbb{E}_\theta\lVert \hat{\theta}^{\text{MLE}}\rVert^2 = \sum_{i=1}^J(\theta_j^2 + \sigma^2_j) >
  lVert \theta\rVert^2$$
  - Always biased and it can be substantial if $J>>n$
- For simplicity, assume $\sigma^2$ is known.
- Stein (1955) showed that with respect to the squared error loss,
  $\hat{\theta}^{\text{MLE}}$ is inadmissible if $J>>3$.
- Later, Janes & Stein (1961) gave a simple estimate that dominates $\hat{\theta}^{\text{MLE}}$.
- $\hat{\theta}_{JS}=(1-\hat{B})\hat{\theta}^{\text{MLE}}$ with
  $\hat{B}=(J-2)\frac{\sigma^2}{n}\cdot\frac{1}{S}=\frac{J-2}{J}\cdot\frac{\sigma^2/n}{S/J}$.
  - $S=\sum_{i=1}^J\bar{y}_{\cdot j}^2$
  - $\frac{S}{J}=\frac{\sum_{j=1}^J\theta_j^2}{J}+\frac{\sum_{j=1}^J\sigma^2/n}{J}=$signal + noise.
  - $\frac{\sigma^2/n}{S/J}$ is the noise fraction. If this is large, $\hat{B}$
    is large.
  - **shrinkage** $(1-\hat{B})$ is large if noise/signal ratio is large.
- When $J$ is large, Jeffrey's prior actually puts in some strong information.
  This is hard to avoid in general if $\Omega$ is high dimensional.
- We need to put some structure on the parameters, i.e. they are no longer
  independent. In many applications, it is reasonable to assume that $\theta_j$s are
  drawn from a distribution, the same distribution.
- This means that $\theta_1,\ldots,\theta_j$ are iid from some distribution.

## Baseball Example (Efron & Morris, 1975, JASA)

- 18 major league players (1970)
- Player, first 45 at bat, remainder of season
  - Clemente, 0.400, 0.346
  - TODO
- Let $x_i$ be the batting average in the first 45 games, then $nx_i\sim
  \operatorname{Binomial}\left(n,p_i\right)$ (n=45). You are interested in
  estimating $p_i$.
- Let $y=\sqrt{n}\arcsin(2x-1)=f(x)$; $\theta=f(p)$.
- Then, $y_i=f(x_i)\sim \operatorname{Normal}\left(\theta_i,1\right)$.
  - Designed to transform binomial into normal approximation.
  - This approximation is very good unless $p_i$ is close to 0 or 1.
- Reasonable to assume that the $\theta$s are drawn from some population.

## Example 1 cont...

- $\bar{y}_{\cdot j}\mid\theta_j\sim
  \operatorname{Normal}\left(\theta_j,\sigma^2/n_j\right)$ where
  $\sigma^2/n_j=\sigma^2_j$.
- Assume $\theta_j \sim \operatorname{Normal}\left(\mu,\tau^2\right)$ where
  $\mu$ and $\tau^2$ are hyper-parameters.
- You can set priors on these and they are called **hierarchical priors**.
- How do you do inference?
- Joint posterior $$p(\mu,\tau,\theta \mid \mathbf{\bar{y}})\propto
  p(\mu,\tau)p(\theta\mid\mu,\tau)p(\mathbf{\bar{y}}\mid\theta)$$. Then,
  assuming $\mu$ is uniformly distributed:
- $$p(\mu,\tau,\theta\mid \mathbf{\bar{y}})\propto p(\tau)\prod_{j=1}^J \operatorname{Normal}\left(\theta_j\mid \mu,\tau^2\right)\prod_{j=1}^J \operatorname{Normal}\left(\bar{y}_{\cdot j}\mid \theta_j,\sigma_j^2\right)$$

i) Conditional distribution of $\theta\mid\mu,\tau^2,\mathbf{\bar{y}}\sim \operatorname{Normal}\left(\hat{\theta}_j,v_j^2\right)$

$$
\begin{aligned}
\bar{y}_{\cdot j}
&\sim\operatorname{Normal}\left(\theta_j,\sigma_j^2\right) \\
\theta_j
&\sim \operatorname{Normal}\left(\mu,\tau^2\right) \\
\hat{\theta}_j
&=\frac{\frac{1}{\sigma_j^2}\bar{y}_{\cdot j}+\frac{1}{\tau^2}\mu}{\frac{1}{\sigma_j^2}+\frac{1}{\tau^2}} \\
&=\frac{\tau^2\bar{y}_{\cdot j}+\sigma_j^2\mu}{\tau^2+\sigma_j^2} \\
\frac{1}{v_j}
&=\frac{1}P\sigma_j^2+\frac{1}{\tau^2} \\
\end{aligned}
$$

- To infer hyper-parameters, consider $$p(\mu,\tau\mid
  \mathbf{\bar{y}})=p(\tau\mid \mathbf{\bar{y}})\cdot p(\mu\mid
  \tau,\mathbf{\bar{y}})$$.
- From $$\bar{y}_{\cdot j}\sim \operatorname{Normal}\left(\theta_j,\sigma_j^2\right)=\theta_j+\varepsilon_j$ when $\varepsilon_j\sim \operatorname{Normal}\left(0,\sigma_j^2\right)$$
- So, $\bar{y}_{\cdot j}\sim \operatorname{Normal}\left(\mu,\sigma_j^2+\tau^2\right)$ but $\theta_j\sim \operatorname{Normal}\left(\mu,\tau^2\right)$, i.e. $\theta$s are gone.

ii) $\mu\mid\tau,\mathbf{\bar{y}}\sim \operatorname{Normal}\left(\hat{mu},\nu(\tau)\right)$

$$
\begin{aligned}
\hat{mu}(\tau)
&=\left(\sum_{j=1}^J \frac{1}{\sigma_j^2+\tau^2}\bar{y}_{\cdot j}\right)/\left(\sum_{j=1}^J \frac{1}{\sigma_j^2+\tau^2}\right) \\
\frac{1}{\nu(\tau)}
&= \sum_{j=1}^J\frac{1}{\sigma_j^2+\tau^2} \\
\end{aligned}
$$

iii) The following is a function of $\tau$ alone.
{% sidenote 'no-mu' "why isn't this conditional on mu?" %}

$$
\begin{aligned}

p(\tau\mid \mathbf{\bar{y}})
&=\frac{p(\mu,\tau\mid \mathbf{\bar{y}})}{p(\mu\mid\tau,\mathbf{\bar{y}})} \text{true for any $\mu$.} \\
&\propto \frac{p(\tau)\prod_{j=1}^J \operatorname{Normal}\left(\bar{y}_{\cdot j}\mid\mu,\sigma_j^2+\tau^2\right)}{\operatorname{Normal}\left(\mu\mid \hat{\mu}, \nu(\tau)\right)}
\end{aligned}
$$

- E & M point out that $\hat{\theta}_{JS}$ is related to Hierarchical Bayes.
- Assume $n_j=n$, so $\theta_j\mid \mu,\tau,\mathbf{\bar{y}}\sim \operatorname{Normal}\left(\hat{\theta}\_j,\nu_j\right)$
- $$\hat{\theta_j}=\frac{\tau^2 \bar{y}_{\cdot j}+\left(\frac{\sigma^2}{n}\right)\mu}{\tau^2+\sigma^2/n}=(1-B)\bar{y}_{\cdot j}+B\mu$$, $$B=\frac{\theta^2/n}{\theta^2/n+\tau^2}$$
- $\frac{1}{\nu_j}=\frac{n}{\sigma^2}+\frac{1}{\tau^2}$.
- Let $\mu=0$, then $\bar{y}_{\cdot j}\sim \operatorname{Normal}\left(0,\sigma^2/n +\tau^2\right)$
- So $S=\sum_{j=1}^J\bar{y}_{\cdot j}^2\sim \left(\sigma^2/n+\tau^2\right)\cdot\chi_J^2$.
- Using the $\chi_J^2$ distribution, we can derive an unbiased estimate for $B$.
- $\hat{B}=(J-2)\frac{\sigma^2/n}{S}$, then
  $\hat{\theta_j}=(1-\hat{B})\bar{y}_{\cdot j}$, which is the same as
  $\hat{\theta}^{\text{JS}}$!
- This is called Empirical Bayes: combines Hierarchical Bayes with empirical
  estimates of hyper-parameters.
- James is Willard D. James: Professor of Mathematics and CS at Cal State Long
  Beach (1967-1989).
