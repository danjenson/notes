---
title: Bayesian Data Analysis
title_url: "."
subtitle: "Chapter 2: Single-parameter models"
toc: true
---

# 2.1 Estimating a probability from binomial data

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

# 2.2 Posterior as compromise between data and prior information

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

# 2.3 Summarizing posterior inference

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

{% fullwidth 'books/bayesian-data-analysis/figures/chapter-2/cpi-vs-hpd.png' '' %}

# 2.4 Informative prior distributions

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

# 2.5 Normal distribution with known variance

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

- As $n\to\infty$, $p(\mu\mid y)\approx \operatorname{Normal}\left(\mu\mid \bar{y},\sigma^2/n\right)$.

# 2.6 Other standard single-parameter models

- The normal distribution with known mean but unknown variance provides an
  introductory example of the estimation of a scale parameter.
- For $p(y\mid\mu,\sigma^2)=\operatorname{Normal}\left(y\mid\mu,\sigma^2\right)$
  with $\mu$ known and $\sigma^2$ unknown, the likelihood for a vector $y$ of
  $n$ independent i.i.d. observations is:

$$
\begin{aligned}
p\left(y \mid \sigma^2\right) & \propto \sigma^{-n} \exp \left(-\frac{1}{2
\sigma^2} \sum_{i=1}^n\left(y_i-\mu\right)^2\right) \\
&=\left(\sigma^2\right)^{-n / 2} \exp \left(-\frac{n}{2 \sigma^2} v\right) .
\end{aligned}
$$

- The **sufficient statistic** is $v=\frac{1}{n}\sum_{i=1}^n(y_i-\mu)^2$, and
  the corresponding conjugate prior density is the inverse-gamma
  $p(\sigma^2)\propto(\sigma^2)^{(-\alpha+1)}e^{-\beta/\sigma^2}$.
- A convenient parameterization is as a scaled inverse-$\chi^2$ distribution with
  $\sigma_0^2$ and $\nu_0$ degrees of freedom; that is, the prior distribution
  of $\sigma^2$ is taken to be $\sigma_0^2\nu_0/X$ where $X$ is a
  $\chi_{\nu_0}^2$ random variable. Using a convenient but non-standard
  notation: $\sigma^2\sim\operatorname{Inv}-\chi^2(\nu_0,\sigma_0^2)$. The
  resulting posterior is:

$$
\begin{aligned}
p\left(\sigma^2 \mid y\right) \propto & p\left(\sigma^2\right) p\left(y \mid \sigma^2\right) \\
\propto &\left(\frac{\sigma_0^2}{\sigma^2}\right)^{\nu_0 / 2+1} \exp \left(-\frac{\nu_0 \sigma_0^2}{2 \sigma^2}\right) \cdot\left(\sigma^2\right)^{-n / 2} \exp \left(-\frac{n}{2} \frac{v}{\sigma^2}\right) \\
\propto &\left(\sigma^2\right)^{-\left(\left(n+\nu_0\right) / 2+1\right)} \exp \left(-\frac{1}{2 \sigma^2}\left(\nu_0 \sigma_0^2+n v\right)\right) \\
& \sigma^2 \mid y \sim \operatorname{Inv-} \chi^2\left(\nu_0+n, \frac{\nu_0 \sigma_0^2+n v}{\nu_0+n}\right)
\end{aligned}
$$

- The Poisson density is $p(y\mid\theta)=\frac{\theta^y e^{-\theta}}{y!}$ for
  $y=0,1,2,\ldots$ and the likelihood is

$$
p(y\mid\theta)=\prod_{i=1}^n\frac{1}{y_i!}\theta^{y_i}e^{-\theta}\propto\theta^{t(y)}e^{-n\theta}
$$

- Thus, $t(y)-\sum_{i=1}^ny_i$ is the **sufficient statistic**.
- As an exponential, this can be written as:

$$
p(y\mid\theta)\propto e^{-n\theta}e^{t(y)\log\theta}
$$

- This suggests that the natural parameter $\phi(\theta)=\log\theta$ and the
  natural conjugate prior distribution is

$$
p(\theta)\propto (e^{-\theta})^\eta e^{\nu\log\theta}
$$

- The likelihood is of the form $\theta^ae^{-b\theta}$, so the conjugate prior
  density must be of the form $p(\theta)\propto\theta^Ae^{-B\theta}$. A more
  conventional parameterization would be the gamma:

$$
p(\theta)\propto e^{-\beta\theta}\theta^{\alpha-1}
$$

- Thus, the posterior density is $\theta\mid y\sim
  \operatorname{Gamma}\left(\alpha+n\bar{y},\beta+n\right)$

- With conjugate families, the known form of the prior and posterior densities
  can be used to find the marginal distribution, $p(y)$, using the formula

$$
p(y)=\frac{p(y\mid\theta)p(\theta)}{p(\theta\mid
y)}=\frac{p(y,\theta)p(y)}{p(y,\theta)}
$$

- For the Poisson with a single observation $y$, the prior predictive
  distribution is $y\sim\operatorname{Negative-binomial}(\alpha,\beta)$:

$$
\begin{aligned}
p(y) &=\frac{\operatorname{Poisson}(y \mid \theta) \operatorname{Gamma}(\theta \mid \alpha, \beta)}{\operatorname{Gamma}(\theta \mid \alpha+y, 1+\beta)} \\
&=\frac{\Gamma(\alpha+y) \beta^\alpha}{\Gamma(\alpha) y !(1+\beta)^{\alpha+y}} \\
p(y)&=\binom{\alpha+y-1}{y}\left(\frac{\beta}{\beta+1}\right)^\alpha\left(\frac{1}{\beta+1}\right)^y
\end{aligned}
$$

- This illustrates that the negative binomial distribution is a _mixture_ of
  Poisson distributions with rates, $\theta$, that follow the gamma
  distribution:

$$
\operatorname{Negative-binomial}(y\mid\alpha,\beta)=\int
\operatorname{Poisson}\left(y\mid\theta\right)\operatorname{Gamma}\left(\theta\mid\alpha,\beta\right)\dd\theta
$$

- The Poisson can be extended to the form $y_i\sim
  \operatorname{Poisson}\left(x_i\lambda\right)$ where $x_i$ are the values of
  an explanatory variable called the _exposure_. $\theta$ is often called the
  _rate_. Ignoring non-$\theta$, the likelihood then becomes:

$$
p(y\mid\theta)\propto \theta^{\left(\sum_{i=1}^n
y_i\right)}e^{-\left(\sum_{i=1}^n x_i\right)\theta}
$$

- So, the gamma distribution for $\theta$ is conjugate, so with the prior
  $\theta\sim \operatorname{Gamma}\left(\alpha,\beta\right)$, the posterior
  becomes

  $$
  \theta\mid y\sim \operatorname{gamma}\left(\alpha+\sum_{i=1}^ny_i,\beta
  \sum_{i=1}^nx_i\right)
  $$

- the exponential distribution is often used to model waiting times.
  - the density is $p(y\mid\theta)=\theta\exp(-y\theta)$ for $y>0$.
  - the exponential is a special case of the gamma distribution with parameters
    $(\alpha,\beta)=(1,\theta)$.
  - the exponential is **memoryless**: $\Pr(y>t+s\mid
  y>s,\theta)=\Pr(y>t\mid\theta)$.
  - The conjugate prior for $\theta$ is $\operatorname{Gamma}\left(\alpha,\beta\right)$.
  - The posterior is $\operatorname{Gamma}\left(\alpha+1,\beta+y\right)$
- The sampling distribution of $n$ independent exponential observations is
  $p(y\mid\theta)=\theta^n\exp(-n\bar{y}\theta)$ for $\bar{y}\ge 0$
  - When viewed as the likelihood of $\theta$ for fixed $y$, this is
    proportional to $\operatorname{Gamma}\left(n+1,n\bar{y}\right)$; this can be
    viewed as $\alpha-1$ exponential observations with total waiting time $\beta$.

# 2.7 Example: informative prior distribution for cancer rates

- Introduces hierarchical modeling

# 2.8 Noninformative prior distributions

- When prior distributions have no population basis, they can be difficult to
  construct, so there is a desire for "reference prior distributions", which are
  described as vague, flat, diffuse, or _noninformative_.
- Jeffreys' invariance principle: by considering one-to-one transformations of
  the parameter $\phi=h(\theta)$. By transformation of variables, the prior
  density $p(\theta)$ is equivalent to the following:

$$
p(\phi)=p(\theta)\left|\dv{\theta}{\phi}\right|=p(\theta)\left|h'(\theta)\right|^{-1}
$$

- The transformed model should determine the same distribution:
  $p(y,\phi)=p(\phi)p(y\mid\phi)$. This leads to the noninformative prior
  density $p(\theta)\propto[J(\theta)]^{1/2}$ where $J(\theta)$ is the **Fisher
  information** for $\theta$:

$$
J(\theta)=\mathrm{E}\left(\left(\frac{d \log p(y \mid \theta)}{d \theta}\right)^2 \big\lvert \theta\right)=-\mathrm{E}\left(\frac{d^2 \log p(y \mid \theta)}{d \theta^2} \big\lvert \theta\right)
$$

- To see that Jeffreys' prior model is invariant to parameterization, evaluate
  $J(\phi)$ at $\theta=$ $h^{-1}(\phi)$ :

$$
\begin{aligned}
J(\phi) &=-\mathrm{E}\left(\frac{d^2 \log p(y \mid \phi)}{d \phi^2}\right) \\
&=-\mathrm{E}\left(\frac{d^2 \log p\left(y \mid \theta=h^{-1}(\phi)\right)}{d \theta^2}\left|\frac{d \theta}{d \phi}\right|^2\right) \\
&=J(\theta)\left|\frac{d \theta}{d \phi}\right|^2
\end{aligned}
$$

- When the number of parameters in a problem is large, its useful to use
  hierarchical models over noninformative priors.
- Pivotal quantities, location and scale parameters on p.54.
- Problems with noninformative priors:
  - Searching for a prior is misguided if the likelihood is dominant. Blindingly
    applying a reference prior is possibly inappropriate.
  - Noninformative priors may be flat or uniform in one parameterization but not
    in another.
  - Difficulties arise when averaging over a set of competing models that have
    improper prior distributions.
- Noninformative priors are still useful when it does not seem to be worth the
  effort to codify real prior knowledge.

# 2.9 Weakly informative prior distributions

- Weakly informative means that it is proper but weaker than any actual prior
  knowledge that is available.
- Rather than trying to model complete ignorance, we prefer to use weakly
  informative priors that include a small amount of real-world information --
  enough to make sure the posterior makes sense.
- Guidance for weakly informative priors:
  - Start with some version of a noninformative prior and then add enough
    information so that inferences are constrained to be reasonable.
  - Start with a strong, highly informative prior and broaden it to account for
    uncertainty in one's prior beliefs and in the applicability of any
    historically based prior distribution to new data.
- Prior distributions should not pull inferences in any predetermined
  direction. If anything, a prior that leans _against_ a hypothesis might be
  advisable.
