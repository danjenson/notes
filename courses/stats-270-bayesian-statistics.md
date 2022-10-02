---
title: "STATS 270: Bayesian Statistics"
toc: true
---

# Typed Notes

## Terminology

### Terms

- **Bias**: The difference between the expected value of an estimator and the
  true value of the parameter. There can be mean or median bias.
  Mean-unbiasedness is not preserved under non-linear transformations, while
  median-unbiasedness is.
- **Conditional likelihood**: A likelihood of data given a sufficient statistic
  for the nuisance parameters so that the likelihood does not depend on them.
- **Consistency**: Consistent estimators converge to the true value of the
  parameter, but may be biased or unbiased.
- **Expected value**: A generalization of the weighted average (weighted by
  probability). Informally, it is the arithmetic mean of a large number of
  independently sampled outcomes of a random variable.
- **Inference**: The process of drawing reliable conclusions from
  data subject to random variation. In particular, based on data, inference
  draws a conclusion about a parameter $\theta$. Note that in ML/Deep Learning,
  "inference" is used slightly differently and often refers to passing input
  through a trained model to get predictions.
- **Likelihood function**: Any function
  $\mathcal{L}(\theta;\mathbf{y})=c(\mathbf{y})p(\theta;\mathbf{y})$ that is
  proportional to $p(\theta;\mathbf{y})$ for any function $c(\mathbf{y})>0$ that
  is independent of the parameter $\theta$ but may depend on $\mathbf{y}$.
  Typically, this is written as $\mathcal{L}(\theta\mid\mathbf{y})$, which
  emphasizes that it is a function of $\theta$ conditional on observed data.
  This is also written as $P(\mathbf{y}\mid\theta)$ or the probability of the
  data given a value for $\theta$. Further, note that this is invariant under
  linear transformations, i.e. if $\mathbf{z}=g(\mathbf{y})$, mapping $\mathbb{R}^n\to
  \mathbb{R}^n$, then
  $\mathcal{L}(\theta;\mathbf{z})=p\left(\theta;g^{-1}(\mathbf{z})\right)\left|\pdv{\mathbf{y}}{\mathbf{z}}\right|$
  where the last term is the absolute value of the determinant of the Jacobian.
- **Marginal likelihood**: A likelihood based only on part of the information in
  the data to remove the nuisance parameters.
- **Maximum likelihood estimate (MLE)**: A maximizer of the likelihood function.
- **Nuisance paramter**: A parameter that the function depends on but that is
  not a parameter of interest.
- **Posterior**: Likelihood $\times$ Prior / Evidence, i.e.
  $\frac{\mathcal{L}(\theta\mid\mathbf{y})\times P(\theta)}{\int
  P(\mathbf{y}\mid\theta)P(\theta)\dd\theta}$.
- **Statistic**: a measurable function of $\mathbf{Y}$, i.e. $S=S(\mathbf{Y})$.
- **Sufficient Statistic**: A statistic such that no other statistic calculated
  form the same sample can provide any more information about the parameter of
  interest. If a density can be factorized as
  $f_{\mathbf{X}}(\mathbf{x})=h(\mathbf{x})g(\theta,T(\mathbf{x}))$, then
  $T(\mathbf{x})$ is a sufficient statistic. In particular, from this
  factorization, you can see that the MLE estimate, $\arg\max_\theta
  h(\mathbf{x})g(\theta,T(\mathbf{x}))$ depends only on
  $g(\theta,T(\mathbf{x}))$. Another way of understanding this is that the
  conditional distribution of $\mathbf{X}$ given $T(\mathbf{X})$ remains the
  same over $\mathcal{F}=\\{\Pr_\theta(\mathbf{x}): \theta\in\Omega\\}$, i.e.
  $\Pr(\mathbf{X}\mid T(\mathbf{X}),\theta)=\Pr(\mathbf{X}\mid T(\mathbf{X}))$.

### Notation

- $p(\theta;\mathbf{y})$: $\theta$ given $\mathbf{y}$, not necessarily
  conditional on in an event sense. This is often used in optimization.
- $p(\mathbf{y}\mid\theta)$: $\mathbf{y}$ conditional on the state of affairs or
  event that $\theta$ has happened. This usage is restricted to statistics.
- $\Pr_\theta(\mathbf{y})$: Joint probability of $\mathbf{y}$ under parameters
  $\theta$, equivalent to $\Pr(\mathbf{y}\mid\theta)$.
- $\mathcal{L}(\theta;\mathbf{y})$: The likelihood function of $\theta$ given $\mathbf{y}$.
- $\mathbb{E}\_{\theta}\left[T_n\right]=\mathbb{E}\_{y\mid\theta}\left[T_n\right]$:
  The expected value of statistic $T_n=T_n(\mathbf{X})$ over $\mathbf{x}$
  conditional on $\theta$, i.e. $\int_{-\infty}^\infty
T_n(\mathbf{x})p(\mathbf{x}\mid\theta)\dd x$.

## Chapter 1: Statistical Inference

### Likelihood and Inference

#### Overview

- Given $\mathbf{Y}=(Y_1,\ldots,Y_n)$ is observed, i.e.
  $\mathbf{y}=(y_1,\ldots,y_n)$, our goal is to make an inferential statement
  about $\theta$, the parameters of the data generating function.
- Example 1.1: Sample statistic $S=S(\mathbf{Y})=\sum_i Y_i$ with Bernoulli
  data:
  - Given a sample of $\mathbf{Y}$ where each $Y_i\sim
    \operatorname{Ber}(\theta)$, then the probability of the joint distribution
    is $$\Pr_\theta(\mathbf{y}) \equiv \Pr(\mathbf{y} \mid
    \theta)=\prod_{i=1}^n
    \theta^{y_i}(1-\theta)^{1-y_i}=\theta^s(1-\theta)^{n-s}$$.
  - $\mathbf{Y}$ Can also be viewed as following $\operatorname{Bin}(n;\theta)$,
    namely $\Pr_\theta(S=s)=\binom{n}{s}\theta^s(1-\theta)^{n-s}$.
  - Now, $\Pr_\theta(\mathbf{Y}=\mathbf{y}\mid
  S=s)=\frac{\Pr_\theta(\mathbf{Y}=\mathbf{y},S=s)}{\Pr(S=s)}=\frac{\theta^s(1-\theta)^{n-s}}{\binom{n}{s}\theta^s(1-\theta)^{n-s}}=\frac{1}{\binom{n}{s}}$,
    which doesn't depend on $\theta$. Thus, $S$ is a sufficient statistic for
    $\theta$.
  - Next, let $N$ be the number of trials required to observe $S=s$ successes,
    i.e. negative binomial:
    $\Pr_\theta(N=s\mid\theta)=\binom{n-1}{s-1}\theta^s(1-\theta)^{n-s}$.
  - Thus, you can see that the following functions share the same likelihood
    function, $\mathcal{L}(\theta;\mathbf{y})=\theta^{\sum_{i=1}^n
    y_i}(1-\theta)^{n-\sum_{i=1}^n y_i}=\theta^s(1-\theta)^{n-s}$:
    - $\Pr(\mathbf{y}\mid\theta)=\prod_{i=1}^n\theta^{y_i}(1-\theta)^{1-y_i}=\theta^s(1-\theta)^{n-s}\to
    c\Pr(\mathbf{y}\mid\theta)=\mathcal{L}(\theta;\mathbf{y})\quad \text{i.e. } c=1$
    - $\Pr(S=s\mid\theta)=\binom{n}{s}\theta^s(1-\theta)^s\to
    c'\Pr(S=s\mid\theta)=\mathcal{L}(\theta;\mathbf{y})\quad\text{i.e.
    }c'=\frac{1}{\binom{n}{s}}$
    - $\Pr(N\mid\theta)=\binom{n-1}{s-1}\theta^s(1-\theta)^{n-s}\to
    c'' \mathcal{L}(\theta;\mathbf{y})\quad\text{i.e.
    }c''=\frac{1}{\binom{n-1}{s-1}}$
- Example 1.2 Linear regression:

  - Given: $$Y=\beta_0+\sum_{j=1}^p \beta_j x_j+\varepsilon, \quad \varepsilon \sim N\left(0, \sigma^2\right)$$
  - You can see from the following that the function depends on $\mathbf{Y}$
    through
    $RSS=(\mathbf{y}-\mathbf{X}\hat{\beta})^\intercal(\mathbf{y}-\mathbf{X}\hat{\beta})$
    (residual sum square) and $\hat{\beta}=(\mathbf{X}^\intercal
    \mathbf{X})^{-1}\mathbf{X}^\intercal \mathbf{Y}$, which means $Y\mid
    (RSS,\hat{\beta})$ is independent of $\theta$, and, thus $(RSS,\hat{\beta})$
    is a sufficient statistic for $\theta$:

    $$
    \begin{aligned} p(\theta ; \mathbf{y}) &=\left(2 \pi \sigma^2\right)^{-n / 2}
    \exp \left(-\frac{1}{2 \sigma^2}(\mathbf{y}-\mathbf{X}
    \beta)^T(\mathbf{y}-\mathbf{X} \beta)\right) \\ &\left.=\left(2 \pi
    \sigma^2\right)^{-n / 2} \exp \left(-\frac{1}{2
    \sigma^2}(\mathbf{y}-\mathbf{X} \widehat{\beta})^T(\mathbf{y}-\mathbf{X}
    \widehat{\beta})+(\widehat{\beta}-\beta)^T\left(\mathbf{X}^T
    \mathbf{X}\right)(\widehat{\beta}-\beta)\right)\right) \\ &=\left(2 \pi
    \sigma^2\right)^{-n / 2} \exp \left(-\frac{1}{2 \sigma^2}\left(R S
    S+(\widehat{\beta}-\beta)^T\left(\mathbf{X}^T
    \mathbf{X}\right)(\widehat{\beta}-\beta)\right)\right), \end{aligned}
    $$

  - [Proof](https://stats.stackexchange.com/a/419934/358356) of independence of $\hat{\beta}$ and $RSS$.

- In many situations, a likelihood is a function of multiple parameters, but
  only a small number of parameters are of interest, the remainder being
  **nuisance parameters**. To eliminate these, there are several alternatives,
  including the marginal, conditional, and profile likelihoods.
- In general, constructing the conditional and marginal likelihoods may be
  non-trivial. The following is a useful method:
  - Given $\mathbf{Y}=(V,W)$ can be partitioned into two parts with the
    likelihood function of $\mathbf{Y}$ being $p((\theta,\phi);\mathbf{y})$ where
    $\theta$ is the parameter of interest and $\phi$ is the nuisance parameter.
  - Consider the marginal likelihood of $V$ and the conditional likelihood of
    $W$ given $V$; we may choose $V$ so that there is no information about
    $\phi$ on $W$. {% sidenote 'm-vw' 'How would you do this?' %}
- Example 1.3 Separating Bernoulli and Poisson Likelihoods:
  - Given $\mathbf{Y}=(Y_1,\ldots,Y_N)$ where
    $Y_i\sim\operatorname{Bernoulli}(\theta)$ and
    $N\sim\operatorname{Poisson}(\phi)$, you can see that the likelihood
    functions decompose (below) and that $N$ is a sufficient statistic for
    $\phi$; thus, given $n$, the conditional likelihood $\mathcal{L}_{S\mid
    N}(\theta; s,n)$ becomes a function of $\theta$ alone.

$$
\begin{aligned}
\mathcal{L}((\theta, \phi) ; \mathbf{y}) & \propto p(\theta ; \mathbf{y} \mid
N=n) p(\phi ; n) \\
&=\prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i} \phi^n \exp (-n
\phi) / n ! \\ & \propto \theta^s(1-\theta)^{n-s} \phi^n \exp (-n \phi) \\
&=\mathcal{L}_{S \mid N}(\theta ; s , n) \mathcal{L}_N(\phi ; n)
\end{aligned}
$$

- Example 1.4 Marginal Likelihood with Normal Distribution:
  {% sidenote 'm-indep', 'Go through this example more slowly' %}

  - Let $Y_ji\sim \operatorname{Normal}(\mu_i,\sigma^2)$ for
    $j=1,2,\;i=1,\dots,n$.
  - The parameter of interest is $\sigma^2$ and the nuisance parameters are
    $\mu_i$ for $i=1,\ldots,n$; hence $\theta=\sigma^2$ and
    $\phi=(\mu_i,\ldots,\mu_n)$
  - $\bar{y}\_i=(y_{1i}+y_{2i})/2$.
  - The likelihood function is:
    $$
    L((\theta, \phi) ; \mathbf{y})=\left(2 \pi \sigma^2\right)^{-n} \exp
    \left(-\frac{1}{2 \sigma^2}\left(2
    \sum_{i=1}^n\left(\bar{y}_i-\mu_i\right)^2+\sum_{i=1}^n \sum_{j=1}^2\left(y_{
    ji}-\bar{y}_i\right)^2\right)\right)
    $$
  - When $V=\sum_{i=1}6n\sum_{j=1}^2(Y_{ji}-\bar{Y}_i)^2$ and
    $W=(\bar{Y}_1,\ldots,\bar{Y}_n)$, $V$ carries no information about the
    nuisance parameters $(\mu_1,\ldots,\mu_n)$.
  - Further, the following orthogonal transformations
    $V_i=(Y_{1i}-Y_{2i})/\sqrt{2}$ and $W_i=(Y_{1i}+Y_{2i})/\sqrt{2}$ imply
    $V_i\sim \operatorname{Normal}(0,\sigma^2)$ and $W_i\sim
  \operatorname{Normal}(\mu_i,\sigma^2)$. Also note that $V_i \perp W_i$, and
    thus $V=\sum_{i=1}^n V_i^2$ and $W$ are independent.
  - Using the preceding, you can see that the marginal likelihoods of $V$ and
    $W$ are $\mathcal{L}\_V(\sigma^2;v)$, depending only on $\sigma^2$, and
    $\mathcal{L}\_W(\mu_1,\ldots,\mu_n,\sigma^2;w)$, or the conditional likelihood
    of $W$ given $V$ because $V$ and $W$ are independent. This means that the
    MLE based on $\mathcal{L}\_V(\sigma^2;v)$ yields a consistent estimate of
    $\hat{\sigma}^2=n^{-1}\sum_{i=1}^n\sum_{j=1}^2(Y_{ji}-\bar{Y}\_i)^2$ of
    $\sigma^2$. This is in contrast to the inconsistent MLE based on the full
    likelihood
    $\hat{\sigma}^2=(2n)^{-1}\sum_{i=1}^n\sum_{j=1}^2(Y_{ji}-\bar{Y}_i)^2$.
    Curiously, the inconsistency is "magically" resolved when the marginal
    likelihood of $V$ is used.

$$
\begin{aligned}
\mathcal{L}_V\left(\sigma^2 ; v\right) &=\exp \left(-\frac{1}{2
\sigma^2}\left(\sum_{i=1}^n \sum_{j=1}^2\left(y_{i
j}-\bar{y}_i\right)^2\right)\right), \\ \mathcal{L}_W\left(\mu_1, \cdots, \mu_n, \sigma^2
; w\right) &=\exp \left(-\frac{1}{2 \sigma^2}\left(2
\sum_{i=1}^n\left(\bar{y}_i-\mu_i\right)^2\right)\right) \\ \mathcal{L}((\theta, \phi) ;
\mathbf{y}) &\propto \mathcal{L}_V\left(\sigma^2 ; v\right)
\mathcal{L}_W\left(\mu_1, \cdots, \mu_n,
\sigma^2 ; w\right)
\end{aligned}
$$

- There are several inferential techniques, including MLE, Bayesian posterior,
  and the decision-theoretical approach.

#### Minimum Mean Squares

- This technique minimizes the mean squared error (MSE)
  $r(T_n,\theta)=\mathbb{E}\_\theta\left[T_n-\theta\right]$ where
  $\mathbb{E}\_\theta$ is the expectation under $\Pr_\theta$ and
  $T_n=T_n(\mathbf{Y})$.
  - For any given $\theta$, you may be able to obtain $T_n$ by solving the
    minimization, i.e. $T_n$ is a vector of weights $\boldsymbol\theta$.
  - $T_n=T_n(\theta)$ may depend on an underlying class $T_n$ and more generally
    $\theta$. {% sidenote 's-cls-tn' 'What does this mean?' %} Fortunately,
    however, $\frac{1}{n}\sum_{i=1}^n Y_i$ is the "optimal estimator" for all
    $\theta$ in the sense of minimizing the MSE within the class of unbiased
    estimates.
- When you cannot minimize $r(T_n,\theta)$ independently of $\theta$, one
  partial solution is to exclude at least the $T_n$s worse than another.

#### Maximum Likelihood

- The maximum likelihood estimator (MLE) seeks $T_n$ to maximize
  $\mathcal{L}\_n(\theta)$ with respect to $\theta$. Mathematically,
  $T_n=\hat{\theta}=\arg\max_\theta \mathcal{L}_n(\theta;\mathbf{y})$.
- **Lemma 1.1: Optimality**: In Example 1.1, $S\sim
  \operatorname{Binomial}(n,\theta)$ with
  $\mathbb{E}\_{y\mid\theta}\left[\frac{S}{n}\right]=\theta$. For any $\theta$
  and an unbiased estimator $T_n=\hat{\theta}$ with $\mathbb{E}_{y\mid
  \theta}\left[T_n\right]=\theta$, $\frac{S}{n}$ is optimal in that
  $r(T_n,\theta)=\mathbb{E}\_{y\mid\theta}\left[T_n-\theta\right]\ge
  \mathbb{E}\_{y\mid\theta}\left[\frac{S}{n}-\theta\right]^2=r\left(\frac{S}{n},\theta\right)$.
  The following is a proof:

$$
\begin{aligned}
\mathbb{E}_{y\mid\theta}\left[T_n-\theta\right]
&=\mathbb{E}_{y\mid\theta}\left[\left(T_n-\frac{S}{n}\right)+\left(\frac{S}{n}-\theta\right)\right]^2
\\
&=\mathbb{E}_{y\mid\theta}\left[T_n-\frac{S}{n}\right]^2+\mathbb{E}_{y\mid\theta}\left[\frac{S}{n}-\theta\right]^2+2
\mathbb{E}_{y\mid\theta}\left[\left(T_n-\frac{S}{n}\right)\left(\frac{S}{n}-\theta\right)\right]
\end{aligned}
$$

- Now let $W_n=T_n-\frac{S}{n}$, which makes the last term above become $2
  \mathbb{E}\_theta\left[W_n S\right]$, i.e. plug $S=n(T_n-W_n)$ in to
  $\frac{S}{n}-\theta$ yields TODO

# Lectures

## Lecture 1

## Lecture 2
