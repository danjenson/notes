---
title: Bayesian Data Analysis
title_url: "."
subtitle: "Chapter 3: Introduction to Multiparameter models"
toc: true
---

- Although a problem can include several parameters of interest, conclusions
  will often be drawn about one or only a few parameters.
- In this case, the ultimate aim of Bayesian analysis is the marginal posterior
  distribution of particular parameters of interest.
- The method is clear:
  1. Obtain the joint distribution of all unknowns.
  2. Integrate out the nuisance parameters.
- Or, equivalently, using simulation:
  1. Draw samples from the joint posterior and ignore the values of the other
     unknowns.

# 3.1 Averaging over 'nuisance parameters'

- A parameter used in the joint distribution but that is not one of interest is
  called a **nuisance parameter**.
- Often the joint posterior density can be factored to yield:

$$
p\left(\theta_1 \mid y\right)=\int p\left(\theta_1 \mid \theta_2, y\right)
p\left(\theta_2 \mid y\right) d \theta_2
$$

# 3.2 Normal data with a noninformative prior distribution

# 3.3 Normal data with a conjugate prior distribution

# 3.4 Multinomial model for categorical data

# 3.5 Multivariate normal model with known variance

# 3.6 Multivariate normal with unknown mean and variance

# 3.7 Example: analysis of bioassay experiment

# 3.8 Summary of elementary modeling and computation
