---
title: Quadratic Approximation of the Posterior -- Deep Dive
date: 2020-12-04


# Example:
# https://github.com/wowchemy/starter-academic/tree/master/exampleSite/content/post/writing-technical-content

---


In this article, we go over how to find:
* the maximum a posteriori (MAP) of a Gaussian model with priors
* the marginal distribution of each parameter via Quadratic Approximation

This is called a Quadratic Approximation "deep dive" because we're going to go 
through steps in as much detail as possible: Taylor Expansion, 
the Covariance Matrix $\Sigma$, it's inverse $\Sigma^{-1}$, and it's relationship
to the Hessian \$mathcal{H}$.  

(aka Laplace Approximation) to
