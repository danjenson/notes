---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 2: Multinomial, Beta, and Dirichlet"
---

- Dirichlet is conjugate prior to multinomial and categorical distributions.
- **Lemma 1**: If $\theta\sim \operatorname{Dirichlet}\left(\vec{\alpha}\right)$
  then $\theta\sim \operatorname{Beta}\left(\alpha_1,\sum_{i\ne 1}\alpha_i\right)$
- **Lemma 2**: If $\theta\sim\operatorname{Beta}(\alpha,\beta)$ then
  $\mathbb{E}\left[\theta\right]=\frac{\alpha}{\alpha+\beta}$ and
  $$\operatorname{var}\left[\frac{\mathbb{E}\left[\theta\right](1-\mathbb{E}\left[\theta\right])}{\alpha+\beta+1}\right]$$
- Posterior expectation and variance are combination of prior and data. Example
  with normal.
