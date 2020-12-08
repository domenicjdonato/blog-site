---
title: Quadratic Approximation of the Posterior — Deep Dive
date: 2020-12-04


# Example:
# https://github.com/wowchemy/starter-academic/tree/master/exampleSite/content/post/writing-technical-content

---

I'm calling this a "deep dive" because we're going to go through many steps as we solve this problem. Many of these steps are intermediary and usually glossed over or stated as fact in other explanations of this topic. This post is most suited to those who are doing interview preparation or who want a detailed understand of the topic.

Topics covered:
* Multivariate Gaussian Distribution $N(\mu, \Sigma)$
* Gradient Decent to find the *maximum a posteriori* (MAP)
* Quadratic Approximation (aka Laplace Approximation)
* Taylor Expansion
* Covariance Matrix $\Sigma$ and it's inverse $\Sigma^{-1}$
* The Hessian $\mathcal{H}$ and it's relationship to $\Sigma^{-1}$
* Empirical (tabular) estimation of second partial derivatives
* Errors associated with the Gaussian assumption

[Companion Colab Notebook](https://colab.research.google.com/drive/1REwGPMOk_elQcalsqhKzqQ3WYizB_T37?usp=sharing)

{{% toc %}}

## Preamble - modeling thought process

When it comes to modeling, it is easy to get so involved in the process of optimizing that sometimes the bigger picture is lost. For this reason, let's go through the modeling thought process step by step. 

* There is some phenomena $p \in \mathcal{P}$ 
* Our goal $g \in \mathcal{G}$ is to understand, predict, or describe $p$
* To accomplish our goal empirically, we need
  * data $d \in \mathcal{D}$
  * a model $m \in \mathcal{M}$
  * parameters $\theta \in \Theta$, please note that when $m$ is non-parametric $\theta = d$

We'd like to select each of these components to maximize the probability of us achieving our goal. To put it formally we are looking for

$$
\underset{d, m, \theta}{arg\max}\ P(\mathcal{D} = d, \mathcal{M} = m, \Theta = \theta | p, g)
$$

Although it's true that $p$ may not exist and $g$ may be the wrong goal, at some point we make the decision to proceed with our modeling process. To make the notation less cluttered we implicitly condition on these two elements. This means we are left with the joint distribution

$$
\tag{1}
P(\mathcal{D}, \mathcal{M}, \Theta)
$$

which will be important to remember as we proceed.

## Problem setup

Richard McElreath does a great job of explaining the topics of Probabilistic Modeling and Bayesian Inference. The problem we are going to cover is in Chapter 4 of Statistical Rethinking, which is to describe the distribution of !Kung adult heights. In particular, we're going to manually do what is accomplished from this line of code in the text

```python
result = maximum_a_posteriori(model, data=df)
``` 

It turns out there is a lot happening behind the scene here.

### Data

The data provided by Richard is partial census data from where the !Kung people live, which is near the Kalahari Desert in the southern part of Africa. The data contains a mix of adults and children. Filtering it to those with an age of at least 18 leaves us with 352 data points. Inspecting the histogram shows that the data is roughly Gaussian.

!['Histogram of adult height data.'](images/data_histogram.png)

**Modeling decision:** $\mathcal{D} = d$ here we've decided to use the data provided to us.

### Model

The first model that Richard has us use is a Gaussian with priors on both of it's parameters.

$$
\tag{2}
\begin{aligned}
h_i &\sim \mathcal{M}(\theta) = N(\mu, \sigma) \\\\
\mu &\sim N(178, 20) \\\\
\sigma &\sim U(0, 50)
\end{aligned}
$$

From the data and what we know about the world, a mean height prior of `178` is a bit high. However, the flexibility provided by the standard deviation of `20` means that the model can still fit our data. It's also useful for illustrating how data can overcome incorrect priors so long as they are not too strong. We double check that our model is capable of describing the data by plotting samples from its prior distribution. 

!['Histogram of height data and model prior.'](images/data_and_prior_histogram.png)

## Posterior

TODO: Derive the posterior from the joint distribution in eq (1).

Just as our data has a distribution, so do the parameters of our model. There are infinitely many values our parameters could take on but only a small subset of these make sense given our data and priors. This distribution of parameters is called the _posterior_.     We'd like to find the parameters that maximize the probability of both the data and our priors. We use Bayes' rule to isolate the probability of our parameters, which is called the _posterior_.

$$
\tag{2}
\overbrace{P(\theta | \mathcal{D})}^\text{Posterior} = \frac{\overbrace{P(\mathcal{D} | \theta)}^\text{Likelihood} \cdot \overbrace{P(\theta)}^\text{Prior}}{\underbrace{P(\mathcal{D})}_\text{Space being considered}}
$$

One thing to note is that $P(\mathcal{D})$ is a _normalizing constant_. It is not something we can usually know and is used to ensure that the sum of all our probabilities in the space being considered sum to `1`. Sometimes this is referred to as the probability of the data, but it's important to remember that $P(\mathcal{D})$ is really the probability of the data given our choice of model and the priors we placed on the model's parameters. A more explicit way of writing formula (2) is

$$
\tag{3}
P(\theta | \mathcal{D}, \mathcal{M}) = \frac{P(\mathcal{D} | \theta, \mathcal{M}) \cdot P(\theta | \mathcal{M}) \cdot P(\mathcal{M})}{P(\mathcal{D},\mathcal{M})}
$$

where $\mathcal{M}$ stands for our modelling assumptions. This formulation helps to remind us that our conclusions and confidences are conditioned on the probability that we made correct modeling assumptions. It's safe to say that in most cases $P(\mathcal{M}) < 1$. 


## Find MAP estimate of parameters



Since $P(\mathcal{D})$ is fixed by the time we get this far, more on this later, we can drop it from the function  start optimizing
TODO: Expand equation (2) to our concrete example/model.

Now that we know we're trying to optimize the posterior, let's learn a way to do this. There are a few optimization techniques that can be used for this type of problem and I'm going to use the one that's most familiar to me. I come from a deep learning background and we use gradient decent often so this is how I'm going to find the maximum of our posterior distribution. Before we get into this though, we're going to make the function we optimizer easier to deal with.

TODO: Explain the Log transformation.

TODO: Explain gradient decent. This is full batch gradient decent.


{{% callout note %}}
The mode of a distribution is its *maximum a posteriori* which translates to *maximum of the posterior*.
{{% /callout %}}


## Quadratic approximation of the posterior

Now that we have found the mode of the posterior distribution, we can use quadratic approximation to estimate the full posterior distribution. Remember that our posterior distribution is a multivariate Gaussian

$$
\begin{aligned}
\mathbf{\theta} &\sim N(\hat{\mathbf{\theta}}, \Sigma) \\\\
&\sim \frac{1}{(2\pi)^{\frac{d}{2}} |\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(\mathbf{\theta} - \hat{\mathbf{\theta}})^\intercal \Sigma^{-1}(\mathbf{\theta} - \hat{\mathbf{\theta}})}
\end{aligned}
$$

and in our case $\mathbf{\theta} := \\{\mu, \sigma\\}$ since these are the two, $d = 2$, parameters of our height model. Using gradient decent, we've found that $\hat{\mu} = 154.60$ and $\hat{\sigma} = 7.73$ which means that $\hat{\mathbf{\theta}} = \\{154.60, 7.73\\}$. What's still unknown to us is the variance of the posterior, $\Sigma$. Let's figure this out now.

### Approximating the covariance matrix

First, let's take a look at the empirical posterior joint distribution. This was 

!['Histogram of adult height data.'](images/posterior_distribution.png)

