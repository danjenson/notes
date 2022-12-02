---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 16: Importance Sampling (2022-11-29)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Importance Sampling

- Target density: $p(x)$
- Proposal/trial density: $q(x)$
- Assume both $p, q$ can be evaluated

### Simple Importance Sampling

- Draw $x_i\sim q$ for $i=1,\ldots,n$
- Compute $w_i=w(x_i)$ then $$
w(x)=
\begin{cases}
&=p(x)/q(x) &\text{if } q(x) > 0 \\
&=0 &\text{otherwise}
\end{cases}
$$
- The set $(x_i, w_i)$ for $i=1,\ldots,n$ is a weighted sample.
- If we want $$\alpha=\mathbb{E}_{p}\left[h(x)\right]$$, then we can use the
  weighted sample $$\hat{\alpha}=\frac{1}{n}\sum_{i=1}^n h(x_i)w_i$$.
- **Theorem**:
  - If:
    1. $\\{x: p(x) > 0\\}\subset\\{x: q(x) > 0\\}$
    2. Variance of $w(x)$ is finite, i.e. $\sigma_w^2$
    3. $\left\|h(x)\right\| < M < \infty$
  - Then:
    - $\mathbb{E}\left[\hat{\alpha}\right]=\alpha$
    - $\operatorname{var}\left[\hat{\alpha}\right]\le M^2(\sigma_w^2+1)/n$
- **Proof**:

$$
\begin{aligned}
\hat{\alpha}
&=\frac{1}{n} \sum_{i=1}^n h(x_i)w(x_i) \\
\mathbb{E}\left[\hat{\alpha}\right]
&= \mathbb{E}_{q}\left[h(x)w(x)\right] \\
&= \int_{x:q(x) > 0} h(x)w(x)q(x)\dd x \\
&= \int_{x:q(x) > 0} h(x)\frac{p(x)}{q(x)}q(x)\dd x \\
&= \int_{x:p(x)>0,q(x) > 0} h(x)p(x)\dd x \\
&= \int_{x:p(x)>0} h(x)p(x)\dd x \\
&= \mathbb{E}_{p}\left[h(x)\right] \\
&= \alpha \\
\operatorname{var}\left[\hat{\alpha}\right]
&= \frac{\operatorname{var}\left[h(x)w(x)\right]}{n} \\
&\le \frac{\mathbb{E}\left[(h(x)w(x))^2\right]}{n} \\
&\le \frac{M^2 \mathbb{E}\left[(w(x))^2\right]}{n} \\
&= \frac{M^2(\sigma^2_w+1)}{n} \\
\mathbb{E}\left[(w(x))^2\right]
&= \operatorname{var}\left[w(x)\right]+\left(\mathbb{E}\left[w(x)\right]\right)^2 \\
&= \sigma_w^2 + 1 \\
\mathbb{E}\left[w(x)\right]
&=\int_q \frac{p(x)}{q(x)}q(x)\dd x \\
&=\int_{q>0}p(x)\dd x \\
&=\int_{p>0}p(x)\dd x \\
&=1
\end{aligned}
$$

{% marginfigure 'p-inside-q' 'courses/stats-270/figures/lecture-16/p-inside-q.png' 'p inside q.' %}

- The renormalized case:

$$
\begin{aligned}
s_i
&= s(x_i)=h(x_i)w(x_i) \\
w_i
&= w(x_i) \\
r(s,w)
&=\frac{s}{w} \\
\hat{\alpha}
&=\frac{\sum_{i=1}^n h(x_i)w(x_i)}{\sum_{i=1}^n w(x_i)} \\
&=\frac{\frac{1}{n}\sum_{i=1}^n s_i}{\frac{1}{n}\sum_{i=1}^n w_i} \\
&=\frac{\bar{s}}{\bar{w}}=r(\bar{s},\bar{w}) \\
\end{aligned}
$$

- Objective is to expand $r(\bar{s},\bar{w})$ around $(\mathbb{E}\left[\bar{s}\right],\mathbb{E}\left[\bar{w}\right])=(\mu_s,\mu_w)=(\alpha,1)$

$$
\begin{aligned}
\hat{\alpha}
&=r(\mu_s,\mu_w)+\pdv{r}{s}(\mu_s,\mu_w)(\bar{s}-\mu_s)+\frac{1}{2}\pdv[2]{r}{s}(\mu_s,\mu_r)(\bar{s}-\mu_s)^2 \\
&=\alpha+\frac{1}{\mu_w}(\bar{s}-\mu_s)-\frac{\mu_s}{\mu_w}(\bar{w}-\mu_w)+\frac{1}{2}\pdv[2]{r}{w}(\mu_s,\mu_w)(\bar{w}-\mu_w)^2 \\
&=\alpha+(\bar{s}-\alpha)-\alpha(\bar{w}-1)+\alpha(\bar{w}-1)^2-(\bar{s}-\alpha)(\bar{w}-1)\cdots+O_p\left(n^{-\frac{3}{2}}\right)\\
\mathbb{E}\left[\hat{\alpha}\right]
&=\alpha+\alpha\cdot\frac{\sigma_w^2}{n}-\frac{1}{n}\cdot\rho\cdot\sigma_w\cdot\sigma_s \\
\operatorname{var}\left[\hat{\alpha}\right]
&\approx
\operatorname{var}\left[\bar{s}-\alpha\bar{w}\right]=\operatorname{var}\left[\operatorname{mean}(s-\alpha w)\right]=\frac{\operatorname{var}\left[s-\alpha w\right]}{n} \\
&= \frac{\operatorname{var}\left[s-\alpha w\right]}{n} \\
&= \frac{1}{n}\mathbb{E}\left[((h(x)-\alpha)w(x))^2\right] \\
&\le \frac{4M^2 \mathbb{E}\left[w^2\right]}{n} \\
&= \frac{4M^2(\sigma_w^2+1)}{n} \\
&= \frac{4M^2}{n_\operatorname{eff}} \\
n_\operatorname{eff}
&=\frac{n}{1+\sigma_w^2} \\
\end{aligned}
$$

- Compare this with the case when $x_i\sim p(\cdot)$, then
  $\operatorname{var}\left[\frac{1}{n}\sum_{i=1}^n h(x_i)\right]\le
  \frac{\operatorname{var}\left[h(x)\right]}{n}\le\frac{M^2}{n}$

- **Lemma**: $\sigma_w^2$ is the coefficient of variation of
  $u(x)=\frac{f(x)}{g(x)}$
- **Proof**:

$$
\begin{aligned}
u(x)
&= c w(x) \\
c
&= \frac{Z_p}{Z_q} \\
\mathbb{E}\left[u\right]
&= c\mathbb{E}\left[w\right]=c \\
(cv)^2
&=\frac{\operatorname{var}\left[u\right]}{\mathbb{E}^2\left[u\right]} \\
&=\frac{\operatorname{var}\left[c u\right]}{\mathbb{E}^2\left[c u\right]} \\
&=\frac{\operatorname{var}\left[w\right]}{\mathbb{E}^2\left[w\right]} \\
&=\frac{\sigma_w^2}{1} \\
&=\sigma_w^2
\end{aligned}
$$

- In practice, compute $u_i$ for $i=1,\ldots,n$ and find its coefficient of
  variation (sd / mean).
- Remarks:
  - Importance sampling is useful when $x$ is of low dimension and you can guess
    when $p(x)$ is large.
  - But in high dimensions, this is not feasible unless you have convexity and
    good bounds
  - Try an example when you have time:
    - $p\sim\frac{1}{2}\operatorname{Normal}\left(0,I_d\right)+\operatorname{Normal}\left(\mu,I_d\right)$
    - Proposal $q\sim t_3(0, k^2 I_d)$
    - See if you can control $w(x)^2$ by varying $k$.
    - You will see that it is not possible to make $w^2$ smaller than
      $O(\|\mu^d\|)$ (curse of dimensionality); so, in practice importance
      sampling is used sequentially

## Examples

- Target density is $p(x)=\frac{f(x)}{Z_p}$
- Trial density is $q(x)=\frac{g(x)}{Z_q}$
- We can evaluate $f$ and $g$, but we do not know the normalizing constants
  $Z_p$ and $Z_q$.
- Then we use importance sampling with normalization:
  1. Draw $x_i$ for $i=1,\ldots,n$ iid with $q(\cdot)$
  2. Compute $u_i=\frac{f(x_i)}{g(x_i)}=c\cdot w_i$ where $c=\frac{Z_p}{Z_q}$
  3. Use $\hat{\alpha}=\frac{\sum_{i=1}^n h(x_i)w(x_i)}{\sum_{i=1}^n
     w(x_i)}=\frac{\sum_{i=1}^n h(x_i)u_i}{\sum_{i=1}^n u_i}$

### Example 1

- $p(Y_i=1)=1-p(Y_i=0)=\frac{e^{\theta x_i}}{1+e^{\theta x_i}}$ for
  $i=1,\ldots,n=100$
- $\pi(\theta)\sim \operatorname{Normal}\left(0,100\right)$
- Target density $$p(\theta\mid \{(x_i,y_i)\}_{i=1}^{100})=cf(\theta)$$
- Likelihood: $$f(\theta)=\exp\left(-\frac{\theta^2}{100}+\theta\cdot \sum_{i=1}^n X_iY_i-\sum_{i=1}^n \log(1+e^{\theta x_i})\right)$$
- What kind of trial (proposal) distribution do you want to use?
  - Use the prior? If $q$ is the prior density, then sampling is very
    inefficient.
    {% marginfigure 'bad-sample-prior' 'courses/stats-270/figures/lecture-16/bad-sample-prior.png' 'Sampling from prior is inefficient.' %}
  - Better way:
    - Note that log(f) is a convex, so find $\hat{\theta}_\operatorname{MLE}$
      and $\sigma^2=-\frac{1}{\pdv[2]{\theta}\mathcal{L}_n(\theta)}$
    - Then use $t_\nu(\hat{\theta}_\operatorname{MLE};\sigma)$ when $\nu$ is small,
      e.g. 5
