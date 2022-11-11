---
title: "STATS 270: Bayesian Statistics"
title_url: "."
subtitle: "Lecture 13: Data Augmentation and Latent Variables (2022-11-10)"
toc: true
---

$$
\newcommand{\op}{\operatorname}
\newcommand{\var}[1]{\op{var}\left[#1\right]}
\newcommand{\sd}[1]{\op{sd}\left[#1\right]}
\newcommand{\cov}[2]{\op{cov}\left[#1, #2\right]}
$$

## Data Augmentation and Latent Variables

- This lecture was a collection of examples that were provided as typed pdf.
- The key idea was when you are trying to augment your dataset, introduce a
  latent variable that makes it easier to calculate your parameter $\theta$.
  Then iterate (1) sampling that latent variable conditional on your data and the
  parameters and (2) sampling the parameters based on the data and latent
  variable.
- One nice property to look for is data independence given your
  parameters.
- There is an extended example about protein binding sites in the pdf.
