---
title: "STATS 270: Bayesian Statistics"
toc: true
---

# Typed Notes

## Terminology

### Terms

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
- **Posterior**: Likelihood $\times$ Prior / Evidence, i.e.
  $\frac{\mathcal{L}(\theta\mid\mathbf{y})\times P(\theta)}{\int
  P(\mathbf{y}\mid\theta)P(\theta)\dd\theta}$.
- **Statistic**: a measurable function of $\mathbf{Y}$, i.e. $S=S(\mathbf{Y})$.
- **Sufficient Statistic**: A statistic such that no other statistic calculated
  form the same sample can provide any more information about the parameter of
  interest. Mathematically, $T(\mathbf{Y})$ is a sufficient statistic if
  $P(\mathbf{Y}\mid T(\mathbf{Y}),\theta)=P(\mathbf{Y}\mid T(\mathbf{Y}))$.
  Another way of expressing this is that if $P(\mathbf{Y}\mid T(\mathbf{Y}))$
  remains the same over the set
  $\mathscr{F}=\\{\Pr_\theta(\mathbf{y}):\theta\in\Omega\\}$, then it is
  sufficient. One can think of it as $P_{T(\mathbf{Y})}(\mathbf{y})$.

### Notation

- $p(\theta;\mathbf{y})$: $\theta$ given $\mathbf{y}$, not necessarily
  conditional on in an event sense. This is often used in optimization.
- $p(\mathbf{y}\mid\theta)$: $\mathbf{y}$ conditional on the state of affairs or
  event that $\theta$ has happened. This usage is restricted to statistics.
- $\Pr_\theta(\mathbf{y})$: Joint probability of $\mathbf{y}$ under parameters
  $\theta$, equivalent to $\Pr(\mathbf{y}\mid\theta)$.
- $\mathcal{L}(\theta;\mathbf{y})$: The likelihood function of $\theta$ given $\mathbf{y}$.

## Chapter 1: Statistical Inference

### Likelihood and Inference

- Given $\mathbf{Y}=(Y_1,\ldots,Y_n)$ is observed, i.e.
  $\mathbf{y}=(y_1,\ldots,y_n)$, our goal is to make an inferential statement
  about $\theta$, the parameters of the data generating function.

- Example 1.1: Sample statistic $S=S(\mathbf{Y})=\sum_i Y_i$ with Bernoulli data

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

# Lectures

## Lecture 1

## Lecture 2
