---
title: Bayesian Data Analysis
title_url: "."
subtitle: "Terminology"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\sd}{\op{sd}}
\newcommand{\var}{\op{var}}
\newcommand{\logit}{\op{logit}}
\newcommand{\J}{\op{J}}
$$

# Terms

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
- **Noninformative prior**: A prior which is often flat or diffuse and has
  little effect on the posterior.
- **Pivotal quantity**: If the density of $y$ is such that $p(y-\theta\mid\theta)$ is free of $\theta$ and $y$, then $y-\theta$ is a pivotal quantity. Furthermore $\theta$ is called a **location** parameter. Similarly, if $p\left(\frac{y}{\theta}\mid\theta\right)$ is a function free of $\theta$ and $y$, then $\theta$ is called a **scale** parameter.
- **Posterior density**: $p(\theta\mid y)=\frac{p(\theta)p(y\mid\theta)}{\int p(\theta)p(y\mid\theta)\dd\theta}$.
- **Posterior odds**: The prior ratio multiplied by the likelihood ratio, i.e. $\frac{p\left(\theta_1 \mid y\right)}{p\left(\theta_2 \mid y\right)}=\frac{p\left(\theta_1\right) p\left(y \mid \theta_1\right) / p(y)}{p\left(\theta_2\right) p\left(y \mid \theta_2\right) / p(y)}=\frac{p\left(\theta_1\right)}{p\left(\theta_2\right)} \frac{p\left(y \mid \theta_1\right)}{p\left(y \mid \theta_2\right)}$.
- **Posterior predictive distribution**: $p(\tilde{y}\mid y)$. It is posterior because it is conditional on observing $y$ and predictive because it is observable.
- **Precision**: The inverse of the variance, $\frac{1}{\sigma^2}$.
- **Prior predictive distribution**: Also known as the marginal distribution of $y$: $p(y)=\int p(y,\theta)\dd\theta=\int p(\theta)p(y\mid\theta)\dd\theta$. It is prior because it is not conditional on a previous observation of the process and predictive because it is the distribution of a quantity that is observable.
- **Proper prior density**: A prior density that does not depend on the data and integrates to 1. Improper prior densities can still lead to proper posterior densities.
- **Sampling distribution**: $p(y\mid\theta)$. Also known as the **data distribution**.
- **Unit**: A record or single object measured, i.e. a person. Each unit may be associated with many observables.
- **Weakly informative priors**: Priors that contain enough information to "regularize" the posterior distribution.

# Notation

- $\mathbb{E}\left[u\mid v\right]$: The conditional expectation of $u$ with $v$ held fixed, i.e. it is a function of $v$.
- $\mathbb{E}\left[u\right]=\int up(u)\dd u$. This is the expectation of $u$ averaging over any conditioning variables, e.g. $v$, as well as $u$. - $\operatorname{Normal}\left(\mu,\sigma^2\right)$: Typically used for random variables.
- $\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$: Typically used for density functions.
- $\theta$: Parameters.
- $\theta\sim \operatorname{Normal}\left(\mu,\sigma^2\right)$: Equivalent to $p(\theta)=p(\theta\mid\mu,\sigma^2)=\operatorname{Normal}\left(\theta\mid\mu,\sigma^2\right)$
- $\tilde{y}$: Unknown, but potentially observable, quantities.
- $\var(u)=\int(u-\mathbb{E}\left[u\right])^2p(u)\dd u$. While for vectors, the covariance matrix is $\var(u)=\int(u-\mathbb{E}\left[u\right])(u-\mathbb{E}\left[u\right])^\intercal p(u)\dd u$
- $p(\cdot)$: A marginal probability distribution.
- $p(\cdot\mid\cdot)$: A conditional probability distribution.
- $y$: Observed data.
- $(\theta^s,\tilde{y}^s)$: $s$ indexes the simulation draws $s=1,\ldots,S$.
