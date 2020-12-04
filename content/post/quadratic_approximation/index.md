---
title: Quadratic Approximation of the Posterior -- Deep Dive
date: 2020-12-04
math: true
---

In this article, we go over how to find:
* the maximum a posteriori (MAP) of a Gaussian model with priors
* the marginal distribution of each parameter via Quadratic Approximation

This is called a Quadratic Approximation "deep dive" because we're going to go 
through steps in as much detail as possible: Taylor Expansion, 
the Covariance Matrix $\sigma$, it's inverse $\sigma^{-1}$, and it's relationship
to the Hessian $H$.  

(aka Laplace Approximation) to
