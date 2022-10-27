---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 10: Hierarchical Bayes & The Federalist Papers (2022-10-27)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Federalist Papers

- David Wallace of UChicago and Frederick Mosteller applied hierarchical Bayes
  to the Federalist papers.
- 77 newspaper essays
- 63 have known authorship
- 12 have disputed authorship (Hamilton vs. Madison)
- Techniques:
  - Using words as discriminators
- Let $x$ be the count of a word in paper length $w$, modeled as a
  $\operatorname{Poisson}\left(w\mu\right),\mu=mu_H\text{ or }\mu=mu_M$
  - $\log(p\mid\mu)=x\cdot\log(w\mu)-w\mu$
  - If we have $n$ independent words, then the log-likelihood ratio is:
    $\sum_{i=1}^n \left[x_i\log(\mu_{iH}/\mu_{iM})-w(\mu_{iH}-\mu_{iM})\right]$
  - If this is positive, it favors Hamilton.
  - This should provide good discriminative power if there are enough words.
- What are the difficulties?
  - Selection of words.
    1. Low frequency words create noise.
    2. Some introduce bias, i.e. content is related to the paper/context.
  - Independence of words.
- Selection:
  - Identified 70 high frequency, non-contextual words.
  - Added discriminatory words from screening.
  - Added discriminatory words from 70,000 words of texts (50% from each author.)
- Second difficulty: unknown parameters. Solution: use hierarchical Bayes.
- $$P(X=4\mid H)=\int\int P(X=4\mid H,\mu_h)p(\mu_H,\mu_M)\dd\mu_H\dd\mu_M$$
- You use the joint distribution because the means are likely not independent.
- Similar calculation for $P(x=4\mid M)$.
- At the end you want the ratio $P(X=4\mid H)/P(X=4\mid M)$
- Posterior given known papers $X_H$ and $X_M$: $p(\mu_H,\mu_M\mid X_H,\X_M)$.
  Note that the rates might be dependent in this joint distribution.
- Let $\sigma=\mu_H+\mu_M$, $\tau=\frac{\mu_H}{\mu_H+\mu_M}$. The parameter of
  interest is $\tau$.
- "For authors writing on the same topics at the same period, we suppose that the
  prior distribution for $\tau$ for any word should be nearly symmetric and
  unimodal. The spread may depend on $\sigma$."
- "We like to have prior distributions based on data, even feebly."
- To do this, M & W plotted the rates of 90 unselected words in known papers by
  Hamilton and Madison.
- The data suggest the priors

$$
\begin{aligned}
\sigma
&\sim \operatorname{Uniform}\left(\cdot,\cdot\right) \\
\tau\mid\sigma
&\sim \operatorname{Beta}\left(\gamma,\gamma\right) \\
\gamma
&=\beta_1+\beta_2\sigma \\
p(\sigma,\tau)
&\propto \tau^{\beta_1+\beta_2\sigma}(1-\tau)^{\beta_1+\beta_2\sigma} \\
\mathbb{E}\left[\tau\mid\sigma\right]
&=\frac{1}{2} \\
\operatorname{var}\left[\tau\mid\sigma\right]
&=\frac{1/4}{2\gamma+1}
\end{aligned}
$$

{% marginfigure 'beta-parameterization' 'courses/stats-270/figures/lecture-10/beta-parameterization.png' 'Beta Parameterization.' %}

- Assuming $\beta_1,\beta_2$ is known, although it was tested for 4 pairs of
  values: (10,0), (15,0), (5,5), and (5,1).
- The posterior given known papers is available in closed form:

$$
\begin{aligned}
p(\mu_H,\mu_M\mid X_H,X_M)
&= Cp(\mu_H,\mu_M\mid\beta_1,\beta_2)p(X_H\mid \mu_H)p(X_M\mid\mu_M)
\end{aligned}
$$

- The odds of Hamilton vs Madison for each disputed paper is evaluated by
  integrating out $\mu_H$ and $\mu_M$ in the $P(X=4\mid H)$ expressions using
  the above calculated prior.
- They calculate the log-odds on known papers, and the results are quite strong
  and correct, even for external papers.
- The Poisson ultimately becomes inadequate, so they switch to the negative
  binomial.
  {% marginfigure 'log-odds' 'courses/stats-270/figures/lecture-10/log-odds.png' 'Log-odds under Beta parameterizations.' %}
- Using the negative binomial, they find that all 12 disputed papers were by
  Madison.
- The log-odds strongly favor Madison.
- The publication of the results had a huge splash.
- Main contributions:
  - Pioneering use of Hierarchical Bayes analysis in the context of a real
    problem.
  - Discussion of the shrinkage effect, towards $\tau=1/2$.
  - Laplace approximation in Bayesian computation: $$\int\int p(x\mid\theta,
  H)p(\theta\mid X_H,X_M)\dd\theta$$. This is a 4-dimensional integral under the
    Negative Binomial.
  - "The negative binomial introduces many complications that strongly affected
    our allocation of efforts, but few new ideas."
  - In a parallel analysis using the Frequentist approach, they made pioneering
    use of "machine learning" concepts such as training data, calibration, marker
    (feature) selection, etc.
- The paper is called _Deciding Authorship_ by Wallace and Mosteller.
