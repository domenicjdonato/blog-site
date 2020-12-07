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

## Preamble

Richard McElreath does a great job of explaining the topics of Probabilistic Modeling and Bayesian Inference. This post is based on the problem he uses in Chapter 4 of Statistical Rethinking. In particular, this post is about what happens when a single line of code is called 

```python
result = maximum_a_posteriori(model, data=df)
``` 

It turns out there is a lot happening behind the scene here and we're going to cover it.

## Problem setup

We want to use a Gaussian to describe the distribution of !Kung adult heights. The data provided by Richard is partial census data from where they live, which is near the Kalahari Desert in the southern part of Africa. 

### Data

The data contains a mix of adults and children. Filtering it to those with an age of at least 18 leaves us with 352 data points. Inspecting the histogram shows that the data is roughly Gaussian.

!['Histogram of adult height data.'](images/data_histogram.png)

### Model

The first model that Richard has us use is a Gaussian with priors on both of it's parameters.

$$
\tag{1}
\begin{aligned}
h_i &\sim N(\mu, \sigma) \\\\
\mu &\sim N(178, 20) \\\\
\sigma &\sim U(0, 50)
\end{aligned}
$$

From the data and what we know about the world, a mean height prior of `178` is a bit high. However, the flexibility provided by the standard deviation of `20` means that the model can still fit our data. It's also useful for illustrating how data can overcome incorrect priors so long as they are not too strong. We double check that our model is capable of describing the data by plotting samples from its prior distribution. 

!['Histogram of height data and model prior.'](images/data_and_prior_histogram.png)

### Posterior

We'd like to find the parameters that maximize the probability of both the data and our priors. We use Bayes' rule to isolate the probability of our parameters, which is called the _posterior_.

$$
\tag{2}
\overbrace{P(\theta | \mathcal{D})}^\text{Posterior} = \frac{\overbrace{P(\mathcal{D} | \theta)}^\text{Likelihood} \cdot \overbrace{P(\theta)}^\text{Prior}}{\underbrace{P(\mathcal{D})}_\text{Space being considered}}
$$

TODO: Expand equation (2) to our concrete example/model.

#### Aside

One thing to note is that $P(\mathcal{D})$ is a _normalizing constant_. It is not something we can usually know and is used to ensure that the sum of all our probabilities in the space being considered sum to `1`. Sometimes this is referred to as the probability of the data, but it's important to remember that $P(\mathcal{D})$ is really the probability of the data given our choice of model and the priors we placed on the model's parameters. A more explicit way of writing formula (2) is

$$
\tag{3}
P(\theta | \mathcal{D}, \mathcal{M}) = \frac{P(\mathcal{D} | \theta, \mathcal{M}) \cdot P(\theta | \mathcal{M}) \cdot P(\mathcal{M})}{P(\mathcal{D},\mathcal{M})}
$$

where $\mathcal{M}$ stands for our modelling assumptions. This formulation aids in reminding us that our conclusions and confidences are conditioned on the probability that we made correct modeling assumptions. I think it's safe to say that in most cases $P(\mathcal{M}) \neq 1$. 


## Find MAP estimate of parameters

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

