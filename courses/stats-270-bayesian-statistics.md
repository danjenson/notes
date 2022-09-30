---
title: "STATS 270: Bayesian Statistics"
toc: true
---

# Typed Notes

## Terminology

### Terms

- **Statistical inference**: The process of drawing reliable conclusions from
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
  This is also written as $P(\mathbf{Y}\mid\theta)$ or the probability of the
  data given a value for $\theta$.
- **Posterior**: Likelihood$\times$Prior$\div$Evidence, i.e.
  $\mathcal{L}(\theta\mid\mathbf{y})\times P(\theta)\div \int
  P(\mathbf{y}\mid\theta)P(\theta)\dv{theta}$

### Notation

- $p(\theta;\mathbf{y})$: $\theta$ given $\mathbf{y}$, not necessarily
  conditional on in an event sense. This is often used in optimization.
- $p(\mathbf{y}\mid\theta)$: $\mathbf{y}$ conditional on the state of affairs or
  event that $\theta$ has happened. This usage is restricted to statistics.

## Chapter 1: Statistical Inference

### Likelihood and Inference

- Given $\mathbf{Y}=(Y_1,\ldots,Y_n)$ is observed, i.e.
  $\mathbf{y}=(y_1,\ldots,y_n)$, our goal is to make an inferential statement
  about $\theta$, the parameters of the data generating function.

# Lectures

## Lecture 1

## Lecture 2
