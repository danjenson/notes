---
title: Bayesian Data Analysis
title_url: "."
subtitle: "Chapter 1: Probability and inference"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\sd}{\op{sd}}
\newcommand{\var}{\op{var}}
\newcommand{\logit}{\op{logit}}
\newcommand{\J}{\op{J}}
$$

# 1.1 The three steps of Bayesian data analysis

1. Setting up a full probability model informed by knowledge, the problem, and
   the data collection process.
2. Conditioning on observed data and calculating the posterior distribution.
3. Evaluating the fit and implications of the posterior.

# 1.2 General notation for statistical inference

- Two kinds of estimands:
  1. Potential observable quantities, i.e. future outcomes or outcomes under
     treatments not received.
  2. Parameters governing the data generating process.
- Often, data is assumed to be exchangeable, i.e. the order doesn't matter: $y =
  (y_1,y_2,\ldots,y_n)\equiv(y_2,y_n,\ldots,y_1)$.

# 1.3 Bayesian inference

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

# 1.4 Discrete examples: genetics and spell checking

- Spelling example is excellent, p.9-11.

# 1.5 Probability as a measure of uncertainty

- Two common notions:
  - Symmetry or exchangeability: assuming equally likely possibilities, it is
    the number of favorable outcomes over total number of possibilities.
  - Frequency: With a large number of repeated trials, one would expect this
    event to happen in proportion to the number of favorable outcomes over the
    total number of outcomes.
- The Frequentist perspective embeds probability questions in a long sequence
  of identical events, which runs into difficultly for rare events.
- Probability is a reasonable way of quantifying uncertainty for the following
  reasons:
  1. By analogy: physical randomness induces uncertainty, so it seems
     reasonable to describe uncertainty in the language of random events.
  2. Axiomatic or normative approach: related to decision theory, this approach
     places all statistical inference in the context of decision-making with
     gains and losses. Then reasonable axioms (ordering, transitivity, etc)
     imply that uncertainty _must_ be represented in terms of probability.
  3. Coherence of bets. _Define_ the probability $p$ attached by you to an
     event $E$ as the fraction $p\in[0,1]$ at which you would bet \\$p for a
     return of \\$1 if $E$ occurs. Namely, if $E$ occurs, you get \\$$(1-p)$
     and if $\neg E$ occurs, you lose \\$p.
- **Whenever there is replication, in the sense of many exchangeable units
  observed, there is scope for estimating features of a probability
  distribution from data and thus making the analysis more objective.**

# 1.6 Example: probabilities from football point spreads

- p. 13

# 1.7 Example: calibration for record linkage

- p.16
- The distribution can be thought of as a mixture of two distributions:
  matching and non-matching distributions:
  $p(y)=\Pr(\text{match})p(y\mid\text{match})+\Pr(\text{non-match})p(y\mid\text{non-match})$.

# 1.8 Some useful results from probability theory

- When $H$ refers to the set of hypotheses or assumptions used to define the
  model, $p(\theta,y\mid H)=p(\theta\mid H)p(y\mid\theta,H)$.
- In general, we prefer to model complexity with hierarchical structure using
  additional variables rather than with complicated marginal distributions,
  even when the additional variables are unobserved or even unobservable.
- Iterated expectation first averages over the target random variable
  conditional on the second and then over the conditional variable, averaging
  the conditional averages:

$$
\begin{aligned}
\mathbb{E}\left[u\right]
&=\mathbb{E}\left[\mathbb{E}\left[u\mid v\right]\right]
\\ &= \int\int u\cdot p(u,v)\dd u\dd v
\\ &=\int p(v)\int u\cdot p(u\mid v)\dd u\dd v
\\ &= \int \mathbb{E}\left[u\mid v\right]p(v)\dd v
\end{aligned}
$$

- Equivalently:

$$
\begin{aligned}
\mathbb{E}\left[u\right]
&=\mathbb{E}\left[\mathbb{E}\left[u\mid v\right]\right]
\\ &= \int \left[\int u\cdot f_{U\mid V}(u\mid v)\dd u\right] f_V(v)\dd v
\\ &= \int \left[\int u\cdot \frac{f_{U,V}(u,v)}{f_V(v)}\dd u\right]f_V(v)\dd v
\end{aligned}
$$

- Law of total variance: $\var(u)=\mathbb{E}\left[\var(u\mid v)\right] +
  \var(\mathbb{E}\left[u\mid v\right])$ (also holds for vectors/matrices).

  $$
  \begin{aligned}
  \mathrm{\mathbb{E}}(\operatorname{var}(u \mid v))+\operatorname{var}(\mathrm{\mathbb{E}}(u \mid v)) &=\mathrm{\mathbb{E}}\left(\mathrm{\mathbb{E}}\left(u^2 \mid v\right)-(\mathrm{\mathbb{E}}(u \mid v))^2\right)+\mathrm{\mathbb{E}}\left((\mathrm{\mathbb{E}}(u \mid v))^2\right)-(\mathrm{\mathbb{E}}(\mathrm{\mathbb{E}}(u \mid v)))^2 \\
  &=\mathrm{\mathbb{E}}\left(u^2\right)-\mathrm{\mathbb{E}}\left((\mathrm{\mathbb{E}}(u \mid v))^2\right)+\mathrm{\mathbb{E}}\left((\mathrm{\mathbb{E}}(u \mid v))^2\right)-(\mathrm{\mathbb{E}}(u))^2 \\
  &=\mathrm{\mathbb{E}}\left(u^2\right)-(\mathrm{\mathbb{E}}(u))^2 \\
  &=\operatorname{var}(u)
  \end{aligned}
  $$

- Transformation of variables:
  - Let $p_u(u)$ be the density of vector $u$.
  - Let $v=f(u)$ where $f:\mathbb{R}^n\to \mathbb{R}^n$.
  - If $p_u$ is a discrete distribution and $f$ is a one-to-one function, then $p_v(v)=p_u(f^{-1(v)})$.
  - If $f$ is a many-to-one function, then a sum appears on the right hand side
    with one term corresponding to each of the branches of the inverse function.
  - If $p_u$ is a continuous distribution and $f$ is a one-to-one
    transformation, then the joint density of the transformed vector is
    $p_v(v)=|\det\J|p_u(f^{-1}(v))$ where the $(i,j)$the entry in $\J$ is
    $\pdv{u_i}{v_j}$.
- In one dimension, the logarithm is often used to transform the parameter space
  from $(0,\infty)$ to $(-\infty,\infty)$.
- When working with parameters defined on the open unit interval, $(0,1)$, we
  often use the logistic transformation
  $\logit(u)=\log\left(\frac{u}{1-u}\right)$ whose inverse is
  $\logit^{-1}(v)=\frac{e^v}{1+e^v}$.
- Another common choice is the probit transformation, $\Phi^{-1}(u)$ where
  $\Phi$ is the standard normal cumulative distribution function, to transform
  from $(0,1)$ to $(-\infty,\infty)$.

# 1.9 Computation and software

- General approach is to fit many models, gradually increasing the complexity.
- The cumulative density function:

  $$
  \begin{aligned}
  F\left(v_*\right) &=\operatorname{Pr}\left(v \leq v_*\right) \\
  &= \begin{cases}\sum_{v \leq v_*} p(v) & \text { if } p \text { is discrete } \\
  \int_{-\infty}^{v_*} p(v) d v & \text { if } p \text { is continuous. }\end{cases}
  \end{aligned}
  $$

- Simple example of sampling an exponential:
  - Solve $U=F(v)=1-e^{-\lambda v}$ for $v$, which yields
    $-\frac{\log(1-U)}{\lambda}$, but since $1-U$ has the same distribution as
    $U$, you can say $v=-\frac{\log(U)}{\lambda}$. Then, you can simulate random
    uniforms to generate exponentials.
- Chart of sample indexing on p.24.

# 1.10 Bayesian inference in applied statistics

- Benefits of a Bayesian approach:
  - Flexibility in combining multiple levels of uncertainty and sources of information.
  - Most intervals are interpreted naturally in a Bayesian sense.
  - If the Bayesian answers vary dramatically over a range of scientifically
    reasonable assumptions athat are unassailable by the data, then the resultant
    range of possible conclusions must be entertained as legitimate.
- Other important themes:
  - A willingness to use many parameters.
  - Hierarchical modeling, which is essential for partial pooling of estimates
    and compromising scientifically between alternative sources of information.
  - Model checking.
  - An emphasis on inference in the form of distributions or at least intervals
    rather than point estimates.
  - The use of simulation as the primary method of computation.
  - The importance of including as much background information as possible.
  - The importance of designing studies that have the property that inferences
    for estimands of interest will be robust to model assumptions.
