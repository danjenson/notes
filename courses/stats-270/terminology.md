---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Terminology"
toc: true
---

# Terminology

## Terms

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

## Notation

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
- $\mathcal{D}(Y)$: Distribution of $Y$.
