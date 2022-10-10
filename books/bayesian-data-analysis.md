---
title: Bayesian Data Analysis
title_url: https://www.amazon.com/Bayesian-Analysis-Chapman-Statistical-2013-11-01/dp/B01JNVQ2QC/
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\sd}{\op{sd}}
\newcommand{\var}{\op{var}}
\newcommand{\logit}{\op{logit}}
\newcommand{\J}{\op{J}}
$$

# Terminology

## Terms

- **Coefficient of variation**: $\op{sd}(\theta)/\mathbb{E}\left[\theta\right]$.
- **Conjugacy**: If $\mathcal{F}$ is a class of sampling distributions
  $p(y\mid\theta)$, and $\mathcal{P}$ is a class of prior distributions for
  $\theta$, then the class $\mathcal{P}$ is conjugate for $\mathcal{F}$ if
  $p(\theta\mid y)\in \mathcal{P}\quad\forall p(\cdot\mid\theta)\in \mathcal{F}\quad\forall\;
  p(\cdot)\in \mathcal{P}$. This is trivial if you take $\mathcal{P}$ to be the
  class of all distributions. Ergo, we are often interested in **natural
  conjugate prior families** which share the same functional form as the
  likelihood.
- **Estimand**: Something estimated from data.
- **Exchangeability**: The concept that permutations of the data doesn't affect their uncertainty.
- **Explanatory variables**: Non-random variables or covariates.
- **Geometric mean**: $\exp\left(\mathbb{E}\left[\log(\theta)\right]\right)$.
- **Geometric standard deviation**: $\exp(\op{sd}(\log(\theta)))$.
- **Hyperparameters**: The parameters of a distribution, i.e. $\alpha$ and
  $\beta$ in $\operatorname{Beta}\left(\alpha,\beta\right)$.
- **Likelihood function**: $\mathcal{L}(\theta\mid y)$ or $p(y\mid\theta)$.
- **Posterior density**: $p(\theta\mid y)=\frac{p(\theta)p(y\mid\theta)}{\int p(\theta)p(y\mid\theta)\dd\theta}$.
- **Posterior odds**: The prior ratio multiplied by the likelihood ratio, i.e. $\frac{p\left(\theta_1 \mid y\right)}{p\left(\theta_2 \mid y\right)}=\frac{p\left(\theta_1\right) p\left(y \mid \theta_1\right) / p(y)}{p\left(\theta_2\right) p\left(y \mid \theta_2\right) / p(y)}=\frac{p\left(\theta_1\right)}{p\left(\theta_2\right)} \frac{p\left(y \mid \theta_1\right)}{p\left(y \mid \theta_2\right)}$.
- **Posterior predictive distribution**: $p(\tilde{y}\mid y)$. It is posterior because it is conditional on observing $y$ and predictive because it is observable.
- **Precision**: The inverse of the variance, $\frac{1}{\sigma^2}$.
- **Prior predictive distribution**: Also known as the marginal distribution of $y$: $p(y)=\int p(y,\theta)\dd\theta=\int p(\theta)p(y\mid\theta)\dd\theta$. It is prior because it is not conditional on a previous observation of the process and predictive because it is the distribution of a quantity that is observable.
- **Sampling distribution**: $p(y\mid\theta)$. Also known as the **data distribution**.
- **Unit**: A record or single object measured, i.e. a person. Each unit may be associated with many observables.

## Notation

- $\mathbb{E}\left[u\mid v\right]$: The conditional expectation of $u$ with $v$ held fixed, i.e. it is a function of $v$.
- $\mathbb{E}\left[u\right]=\int up(u)\dd u$. This is the expectation of $u$ averaging over any conditioning variables, e.g. $v$, as well as $u$.
- $\operatorname{Normal}\left(\mu,\sigma^2\right)$: Typically used for random variables.
- $\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$: Typically used for density functions.
- $\theta$: Parameters.
- $\theta\sim \operatorname{Normal}\left(\mu,\sigma^2\right)$: Equivalent to $p(\theta)=p(\theta\mid\mu,\sigma^2)=\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$
- $\tilde{y}$: Unknown, but potentially observable, quantities.
- $\var(u)=\int(u-\mathbb{E}\left[u\right])^2p(u)\dd u$. While for vectors, the covariance matrix is $\var(u)=\int(u-\mathbb{E}\left[u\right])(u-\mathbb{E}\left[u\right])^\intercal p(u)\dd u$
- $p(\cdot)$: A marginal probability distribution.
- $p(\cdot\mid\cdot)$: A conditional probability distribution.
- $y$: Observed data.
- $(\theta^s,\tilde{y}^s)$: $s$ indexes the simulation draws $s=1,\ldots,S$.

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

## 1.3 Bayesian inference

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

## 1.4 Discrete examples: genetics and spell checking

- Spelling example is excellent, p.9-11.

## 1.5 Probability as a measure of uncertainty

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

## 1.6 Example: probabilities from football point spreads

- p. 13

## 1.7 Example: calibration for record linkage

- p.16
- The distribution can be thought of as a mixture of two distributions:
  matching and non-matching distributions:
  $p(y)=\Pr(\text{match})p(y\mid\text{match})+\Pr(\text{non-match})p(y\mid\text{non-match})$.

## Some useful results from probability theory

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

## 1.9 Computation and software

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

## 1.10 Bayesian inference in applied statistics

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

# Chapter 2: Single-parameter models

## 2.1 Estimating a probability from binomial data

- For the binomial, $p(y\mid\theta)=\operatorname{Binomial}\left(y\mid
  n,\theta\right)=\binom{n}{y}\theta^y(1-\theta)^{n-y}$, $n$ is suppressed on
  the left hand side because it is regarded as part of the experimental design
  that is considered fixed.
- The posterior of the binomial is $\theta\mid y\sim
  \operatorname{Beta}\left(y+1,n-y+1\right)$.
- Jacob Bernoulli identified the "weak law of large numbers", namely if $y\sim
  \operatorname{Binomial}\left(n,\theta\right)$ then
  $\Pr\left(\left|\frac{y}{n}-\theta\right|>\epsilon\mid\theta\right)\to 0$ as
  $n\to\infty$.
- Pierre Simon Laplace and Reverend Thomas Bayes inverted the probability
  statement to $\Pr(\theta\in(\theta_1,\theta_2)\mid y)$.
- The posterior predictive distribution can be represented as follows:

$$
\begin{aligned}
\operatorname{Pr}(\tilde{y}=1 \mid y)
&=\int_0^1 \operatorname{Pr}(\tilde{y}=1 \mid \theta, y) p(\theta \mid y) \dd \theta \\
\\ &=\int_0^1 \theta \operatorname{Beta}\left(y+1,n-y+1\right) \dd \theta
\\ &=\mathbb{E}[\theta \mid y]
\\ &=\frac{y+1}{n+2}
\end{aligned}
$$

- This result, based on the uniform prior distribution, is known as "Laplace's
  law of succession". When $y=0$, this law predicts $\frac{1}{n+2}$, and when
  $y=n$, this law predicts $\frac{n+1}{n+2}$.

## 2.2 Posterior as compromise between data and prior information

- The prior mean is the expectation over posterior means, i.e.
  $\mathbb{E}\left[\theta\right]=\mathbb{E}\left[\mathbb{E}\left[\theta\mid
  y\right]\right]$. In other words, **the prior mean of $\theta$ is the average
  of all possible posterior means over the distribution of possible data**,
  distributed as $p(y)$.
- In general, during Bayesian inference which progresses form $p(\theta)$ to
  $p(\theta\mid y)$, we expect that the posterior will be less variable than the
  prior because we have more information. This can be seen from the
  decomposition of variance, which separates into two terms:
  1. The mean of the posterior variances.
  2. The variance of the posterior means.

$$
\begin{aligned}
\var(u)&=\mathbb{E}\left[\var(u\mid v)\right] + \var(\mathbb{E}\left[u\mid v\right])
\end{aligned}
$$

- This suggests that on average, the posterior variance is lower than the prior
  variance because does not incorporate the variance of the posterior means over
  the distribution of possible data.
- The greater the variation in posterior means, the more potential for reducing
  uncertainty regarding our estimate of $\theta$.
- The posterior is always a compromise between the data and the prior.

## 2.3 Summarizing posterior inference

- Commonly used summaries of location are the mean, median, and mode(s) of the
  distribution.
- Variation is commonly summarized by the standard deviation, interquartile
  range, and other quantiles.
- The mode or most likely posterior value is often easier to compute than the
  mean or median.
- In addition to point summaries, it is important to report posterior
  uncertainty. These usually take the form of quantiles of the posterior
  distribution of the estimands of interest.
- The **central posterior interval** can differ significantly from the **highest
  posterior density** region, as shown by the following graphic:

{% fullwidth 'books/figures/bayesian-data-analysis/cpi-vs-hpd.png' '' %}

## 2.4 Informative prior distributions

- Two justifications for priors:
  - _Population_ interpretation: the prior represents a population of possible
    parameter values from which $\theta$ has been drawn.
  - _State of knowledge_ interpretation: we must express our knowledge and
    uncertainty about $\theta$ as if its value could be thought of as a random
    realization from a prior distribution.
- Typically, the prior should include all plausible values of $\theta$, and even
  if the prior is not centered around the true value, the data will far
  outweigh _any_ reasonable prior.
- Probability distributions that belong to an **exponential family** have
  natural conjugate prior distributions.
- The class $\mathcal{F}$ is an exponential family if all its members have the
  form:
  - $\theta,y_i,\phi(\theta),u(y_i)\in \mathbb{R}^n$
  - $\phi(\theta)$ is the **natural parameter**

$$
p(y_i\mid\theta)=f(y_i)g(\theta)e^{\phi(\theta)^\intercal u(y_i)}
$$

- The likelihood corresponding to i.i.d. $y_i$ is

$$
p(y\theta)=\left(\prod_{i=1}^n
f(y_i)\right)g(\theta)^n\exp\left(\phi(\theta)^\intercal\sum_{i=1}^n u(y_i)\right)
$$

- And for all $n$ and $y$, this has a fixed form as a function of $\theta$ where
  $t(y)=\sum_{i=1}^n u(y_i)$:

$$
p(y\mid\theta)\propto g(\theta)^ne^{\phi(\theta)^\intercal t(y)}
$$

- Here $t(y)$ is said to be a **sufficient statistic** for $\theta$ because the
  likelihood for $\theta$ depends on the data $y$ only through the value of
  $t(y)$.
- If the prior density is $p(\theta)\propto g(\theta)^\eta
  \exp(\phi(\theta)^\intercal \nu)$, then the posterior is $p(\theta\mid
  y)\propto g(\theta)^{\eta+n}\exp(\phi(\theta)^\intercal(\nu+t(y)))$.
- In general, the exponential families are the only classes of distributions
  that have natural conjugate prior distributions, since, apart from certain
  irregular cases, the only distributions having a fixed number of sufficient
  statistics for all $n$ are of the exponential type.
- When applying the normal to a proportion, it is useful to logit transform it,
  $\log\left(\frac{\theta}{1-\theta}\right)$ so that the unit interval becomes
  the real line.

## 2.5 Normal distribution with known variance

- The central limit theorem helps to justify using the normal likelihood in many
  problems as an approximation to a less analytically convenient actual
  likelihood.
- The normal density is
  $\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2\sigma^2}(y-\mu)^2\right)$
- Considered as a function of $\mu$, the likelihood is an exponential of a
  quadratic from in $\mu$, so conjugate priors look like

$$
p(\mu)=\exp\left(A\mu^2+B\mu+C\right)
$$

- We parameterize this family as $\mu\sim \operatorname{Normal}\left(\mu_0,\sigma_0^2\right)$:

$$
p(\mu)\propto\exp\left(-\frac{1}{2\sigma_0^2}(\mu-\mu_0)^2\right)
$$

- The posterior is then:

$$
p(\theta\mid y)\propto\exp\left(-\frac{1}{2}\left(\frac{(y-\mu)^2}{\sigma^2}+\frac{(\mu-\mu_0)^2}{\sigma_0^2}\right)\right)
$$

- And, combining terms, $\theta\mid y\sim \operatorname{Normal}\left(\mu_1,\sigma_1^2\right)$

$$
\begin{aligned}
p(\theta\mid y)&\propto\exp\left(-\frac{1}{2\sigma_1^2}(\mu-\mu_1)^2\right) \\
\mu_1&=\frac{\frac{1}{\sigma_0^2}\mu_0+\frac{1}{\sigma^2}y}{\frac{1}{\sigma_0^2}+\frac{1}{\sigma^2}} \\
\frac{1}{\sigma_1^2}&=\frac{1}{\sigma_0^2}+\frac{1}{\sigma^2}
\end{aligned}
$$

- The posterior mean can be expressed as a weighted average of the prior mean
  and the observed value, $y$, with weights proportional to the precisions.
- Alternatively, we can express $\mu_1$ as the prior mean adjusted toward the
  observed $y$:

$$
\begin{aligned}
\mu_1&=\mu_0+(y-\mu_0)\frac{\sigma_0^2}{\sigma^2+\sigma_0^2} \\
\end{aligned}
$$

- Or, as the data shrunk toward the prior mean:

$$
\mu_1=y-(y-\mu_0)\frac{\sigma^2}{\sigma^2+\sigma_0^2}
$$

- Or, even as:

$$
\mu_1=\mu_0 \left(\frac{\sigma^2}{\sigma^2+\sigma_0^2}\right)+y
\left(\frac{\sigma_0^2}{\sigma^2+\sigma_0^2}\right)
$$

- This makes it clear that

$$
\begin{aligned}
\mu_1&=\mu_0\text{ if }y=\mu_0\text{ or }\sigma_0^2=0 \\
\mu_1&=y\text{ if }y=\mu_0\text{ or }\sigma_0^2=0 \\
\end{aligned}
$$

- The posterior predictive distribution can be calculated using integration:

$$
\begin{aligned}
p(\tilde{y}\mid y)&=\int p(\tilde{y}\mid\mu)p(\mu\mid y)\dd\mu \\
&\propto\int\exp\left(-\frac{1}{2\sigma^2}(\tilde{y}-\mu)^2\right)\exp\left(-\frac{1}{2\sigma_1^2}(\mu-\mu_1)^2\right)\dd\mu
\end{aligned}
$$

- You can determine the mean and variance of the posterior predictive
  distribution using the knowledge from the posterior distribution that
  $\mathbb{E}\left[\tilde{y}\mid\mu\right]=\mu$ and
  $\operatorname{var}\left[\tilde{y}\mid\theta\right]=\sigma^2$:

$$
\begin{aligned}
\mathbb{E}\left[\tilde{y}\mid y\right]
&=\mathbb{E}\left[\mathbb{E}\left[\tilde{y}\mid\mu,y\right]\mid
y\right]=\mathbb{E}\left[\mu\mid y\right]=\mu_1 \\
\operatorname{var}\left[\tilde{y}\mid y\right]
&= \mathbb{E}\left[\operatorname{var}\left[\tilde{y}\mid \mu,y\mid y\right]\right]+ \operatorname{var}\left[\mathbb{E}\left[\tilde{y},\mu,y\right]\mid y\right] \\
&= \mathbb{E}\left[\sigma^2\mid y\right]+ \operatorname{var}\left[\mu\mid y\right] \\
&= \sigma^2+\sigma_1^2
\end{aligned}
$$

- Thus, the posterior predictive for $\tilde{y}$ has a mean equal to the
  posterior mean of $\mu$ and variance equal to the predictive variance
  $\sigma^2$ and the variance $\sigma_1^2$ due to the uncertainty in $\mu_1$.
- With multiple $y_i$, this becomes:

$$
\begin{aligned}
p(\mu \mid y) & \propto p(\mu) p(y \mid \mu) \\
&=p(\theta) \prod_{i=1}^n p\left(y_i \mid \theta\right) \\
& \propto \exp \left(-\frac{1}{2 \sigma_0^2}\left(\mu-\mu_0\right)^2\right) \prod_{i=1}^n \exp \left(-\frac{1}{2 \sigma^2}\left(y_i-\mu\right)^2\right) \\
& \propto \exp \left(-\frac{1}{2}\left(\frac{1}{\sigma_0^2}\left(\mu-\mu_0\right)^2+\frac{1}{\sigma^2} \sum_{i=1}^n\left(y_i-\mu\right)^2\right)\right)
\end{aligned}
$$

- Algebraic simplification of this shows that the posterior depends on $y$ only
  through the sample mean, $\bar{y}=\frac{1}{n}\sum_{i=1}^n y_i$; namely,
  $\bar{y}$ is a **sufficient statistic** for the model. In fact, since
  $\bar{y}\mid\mu\sigma^2\sim \operatorname{Normal}\left(\mu,\sigma^2\right)$,
  the results derived for the single normal observation apply immediately
  (treating $\bar{y}$) as a single observation:

$$
\begin{aligned}
p(\mu\mid y_1,\ldots,y_n)=p(\mu\mid\bar{y})&=\operatorname{Normal}\left(\mu\mid\mu_n,\sigma_n^2\right) \\
\mu_n&=\frac{\frac{1}{\sigma_0^2}\mu_0+\frac{n}{\sigma^2}\bar{y}}{\frac{1}{\sigma_0^2}+\frac{1}{\sigma^2}} \\
\frac{1}{\sigma_n^2}&=\frac{1}{\sigma_0^2}+\frac{n}{\sigma^2}
\end{aligned}
$$

- As $n\to\infty$, $p(\u\mid y)\approx \operatorname{Normal}\left(\mu\mid \bar{y},\sigma^2/n\right)$.

## 2.6 Other standard single-parameter models

- The normal distribution with known mean but unknown variance provides an
  introductory example of the estimation of a scale parameter.
