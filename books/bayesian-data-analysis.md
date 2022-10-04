---
title: Bayesian Data Analysis
title_url: https://www.amazon.com/Bayesian-Analysis-Chapman-Statistical-2013-11-01/dp/B01JNVQ2QC/
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\sd}{\op{sd}}
$$

# Terminology

## Terms

- **Coefficient of variation**: $\op{sd}(\theta)/\mathbb{E}\left[\theta\right]$.
- **Estimand**: Something estimated from data.
- **Exchangeability**: The concept that permutations of the data doesn't affect their uncertainty.
- **Explanatory variables**: Non-random variables or covariates.
- **Geometric mean**: $\exp\left(\mathbb{E}\left[\log(\theta)\right]\right)$.
- **Geometric standard deviation**: $\exp(\op{sd}(\log(\theta)))$.
- **Likelihood function**: $\mathcal{L}(\theta\mid y)$ or $p(y\mid\theta)$.
- **Posterior density**: $p(\theta\mid y)=\frac{p(\theta)p(y\mid\theta)}{\int
  p(\theta)p(y\mid\theta)\dd\theta}$.
- **Posterior odds**: The prior ratio multiplied by the likelihood ratio, i.e. $\frac{p\left(\theta_1 \mid y\right)}{p\left(\theta_2 \mid y\right)}=\frac{p\left(\theta_1\right) p\left(y \mid \theta_1\right) / p(y)}{p\left(\theta_2\right) p\left(y \mid \theta_2\right) / p(y)}=\frac{p\left(\theta_1\right)}{p\left(\theta_2\right)} \frac{p\left(y \mid \theta_1\right)}{p\left(y \mid \theta_2\right)}$.
- **Posterior predictive distribution**: $p(\tilde{y}\mid y)$. It is posterior
  because it is conditional on observing $y$ and predictive because it is
  observable.
- **Prior predictive distribution**: Also known as the marginal distribution of
  $y$: $p(y)=\int p(y,\theta)\dd\theta=\int p(\theta)p(y\mid\theta)\dd\theta$.
  It is prior because it is not conditional on a previous observation of the
  process and predictive because it is the distribution of a quantity that is
  observable.
- **Sampling distribution**: $p(y\mid\theta)$. Also known as the **data distribution**.
- **Unit**: A record or single object measured, i.e. a person. Each unit may be associated with many observables.

## Notation

- $\theta$: Parameters.
- $y$: Observed data.
- $\tilde{y}$: Unknown, but potentially observable, quantities.
- $p(\cdot\mid\cdot)$: A conditional probability distribution.
- $p(\cdot)$: A marginal probability distribution.
- $\theta\sim \operatorname{Normal}\left(\mu,\sigma^2\right)$: Equivalent to
  $p(\theta)=p(\theta\mid\mu,\sigma^2)=\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$
- $\operatorname{Normal}\left(\mu,\sigma^2\right)$: Typically used for random
  variables.
- $\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$: Typically used for density functions.

# Chapter 1: Probability and inference

## 1.1 The three steps of Bayesian data analysis

1. Setting up a full probability model informed by knowledge, the problem, and
   the data collection process.
2. Conditioning on observed data and calculating the posterior distribution.
3. Evaluating the fit and implications of the posterior.

## 1.2 General notation for statistical inference

- Two kinds of estimands:
  1. Potential observable quantities, i.e. future outcomes or outcomes under
     treatments not received.
  2. Parameters governing the data generating process.
- Often, data is assumed to be exchangeable, i.e. the order doesn't matter: $y =
  (y_1,y_2,\ldots,y_n)\equiv(y_2,y_n,\ldots,y_1)$.

## Bayesian inference

- Prior predictive distribution: $p(y)$.
- Posterior predictive distribution: $p(\tilde{y}\mid y)$. Assuming the
  conditional indpendence of $y$ and $\tilde{y}$ given $\theta$:

$$
\begin{aligned}
p(\tilde{y}\mid y)
&=\int p(\tilde{y},\theta\mid y)\dd\theta
\\ &=\int p(\tilde{y}\mid\theta, y)p(\theta\mid y)\dd\theta
\\ &=\int p(\tilde{y}\mid\theta)p(\theta\mid y)\dd\theta
\end{aligned}
$$

- You can rarely be sure that the model you have selected is correct.

## Discrete examples: genetics and spell checking

- Spelling example is excellent, p.9-11.

## Probability as a measure of uncertainty

- Two common notions:
  - Symmetry or exchangeability: assuming equally likely possibilities, it is
    the number of favorable outcomes over total number of possibilities.
  - Frequency: With a large number of repeated trials, one would expect this
    event to happen in proportion to the number of favorable outcomes over the
    total number of outcomes.
- The Frequentist perspective embeds probability questions in a long sequence of
  identical events, which runs into difficultly for rare events.
- Probability is a reasonable way of quantifying uncertainty for the following
  reasons:
  1. By analogy: physical randomness induces uncertainty, so it seems reasonable
     to describe uncertainty in the language of random events.
  2. Axiomatic or normative approach: related to decision theory, this approach
     places all statistical inference in the context of decision-making with
     gains and losses. Then reasonable axioms (ordering, transitivity, etc)
     imply that uncertainty _must_ be represented in terms of probability.
  3. Coherence of bets. _Define_ the probability $p$ attached by you to an event
     $E$ as the fraction $p\in[0,1]$ at which you would bet \\$p for a return of
     \\$1 if $E$ occurs. Namely, if $E$ occurs, you get \\$$(1-p)$ and if $\neg E$
     occurs, you lose \\$p.
- **Whenever there is replication, in the sense of many exchangeable units
  observed, there is scope for estimating features of a probability distribution
  from data and thus making the analysis more objective.**

## Example: probabilities from football point spreads

- p. 13
