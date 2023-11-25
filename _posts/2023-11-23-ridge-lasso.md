---
title: Ridge and Lasso
date: 2023-11-23 18:00:00 +0530
categories: [Data Science]
tags: [machine learning]     # TAG names should always be lowercase
toc : true
math: true
mermaid: true
---

## Introduction

We know the equation for linear regression:

$$
y = X\beta + \epsilon
$$

Where y is the regressor variable, X is the data, $$ \beta $$ is the parameter and $$ \epsilon $$ is the error term.

**Assumption:**

$$
\epsilon \sim N(0, \sigma^2)
$$

$$
y_i \sim N(\beta^T x_i, \sigma^2)
$$

One of the issues in $$ \beta $$ estimated from Ordinary Least Square(OLS) is that , $$\hat{\beta_{ols}}$$ have high variance. It means. small change in X leads to big change in $$\beta$$ . That is not favourable in machine learning.

So, we use the method of regularization. That comes in two forms i.e Lasso(L1 norm) and Ridge(L2 norm). So here the optimization problem is slightly different.

$$
Lasso:   \hat{\beta_{L1}} = argmin_\beta[||y-X\beta^2|| + \lambda\sum|\beta_j|] 
$$

$$
Ridge:   \hat{\beta_{L2}} = argmin_\beta[||y-X\beta^2|| + \lambda\sum\beta_j^2]
$$

In the above two equations, the first term is the error in linear regression and second term is the regularization term. There are two distinct goals for the optimization:
1. Minimize the error term by fitting this model on some setting of $$\beta$$.
2. The absolute values of $$\beta$$ to be small. This is coming from the regularization term. Because the $$\beta$$ from OLS have high variance means a large set of values. So keeping the regularization term small to keep the $$\beta$$ small.

But where the regularization terms come from ?

**Bayesian Approach**:

We will look into posterior probability first and will try to maximize it. The posterior probability we will be looking at is 

$$
{P(\beta|y)}
$$

First lets understand what it means. $$\beta$$ is an unknown parameter that we are trying to solve for. y is some known response variable. so $$P(\beta \mid y)$$ is asking the question about given the known parameter vector y, what is the probability of getting some settings of $$\beta$$. So the natural thing to do is maximize $$\beta$$.

$$
\beta_{MAP} = argmax_\beta P(\beta|y)
$$

which means, that this setting of $$\beta$$ is most likely given the data we actually observe. $$\beta_{MAP}$$ maximizes the posterior. Now let's solve it.

$$
\begin{equation}
  \begin{aligned}
\beta_{MAP} & = argmax_\beta P(\beta|y) \\
            & = argmax_\beta \frac{P(y|\beta)P(\beta)}{P(y)}  \text{     using bayes theroem}\\
            & = argmax_\beta P(y|\beta)P(\beta)  \text{       the denominator doesn't have beta, so we can ignore them}\\
  \end{aligned}
\end{equation}
$$

Here $$P(y \mid \beta)$$ is the likelihood and $$P(\beta)$$ is the prior. $$P(\beta)$$ is prior, it means it's asking the question, before observing the data, what is the probability of this setting of $$\beta$$ unconditional on anything.

Now, we will take log on both the sides, because maximizing/minimizing something is similar to maximizing/minimizing it's log value.

$$
log(\beta_{MAP}) = argmax_\beta[log(P(y|\beta)) + log(P(\beta))]  ... (i)
$$

Now we will calculate the values of both the terms separately and combine them later. The first term is very easy to calculate. We know from OLS assumption stated at the begining that $$y_i \sim N(\beta^Tx_i , \sigma^2)$$ . So, as y is a vector of n observations $$P(y\mid \beta)$$ will be multiplication of n probability density functions each of which is a normal pdf.

$$
\begin{equation}
 \begin{aligned}
P(y \mid \beta) & = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(y_i - \beta^Tx_i)^2}{2\sigma^2}} \\
log(P(y \mid \beta)) & = \sum_{i=1}^{n}[log\frac{1}{\sigma\sqrt{2\pi}} - \frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2]\\
log(P(y \mid \beta)) & \sim \sum_{j=1}^{n}[ - \frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2]  ...(ii)
 \end{aligned}
\end{equation}
$$

The first part doesn't have $$\beta$$ term. So, we care about the second term only.

Now comes the second term of equation (i) which is the prior. Let's assume the prior distribution of $$\beta$$ is normal with mean 0 and variance \tau^2. So that means, $$\beta_j \sim N(0,\tau^2) \forall j = 1,2...,p

X is a nXp matrix, so there are p $$\beta$$

$$
\begin{equation}
 \begin{aligned}
P(\beta) & = \prod_{j=1}^{p} \frac{1}{\tau\sqrt{2\pi}} e^{-\frac{\beta_j^2}{2\tau^2}} \\
log(P(\beta)) & = \sum_{j=1}^{p}[log\frac{1}{\tau\sqrt{2\pi}} - \frac{\beta_j^2}{2\tau^2}] \\
log(P(\beta)) & \sim \sum_{j=1}^{p}[ - \frac{\beta_j^2}{2\tau^2}]  ...(iii)
 \end{aligned}
\end{equation}
$$

The first part doesn't have $$\beta$$ term. So, we care about the second term only.

So, each of the $$\beta_j$$ has the pdf of normal distribution.

There is a lot of mass around 0. This is where the regularization come in. We are trying to achieve the vakues of $$\beta$$ close to zero in regularization. So, the prior distribution is trying to do the same thing.

So, again going back to our main equation (i) and putting the values from (ii) and (iii), we get:

$$
\begin{equation}
 \begin{aligned}
argmax_\beta [log(P(y \mid \beta)) + log(P(\beta))] & = argmax_\beta [\sum[ - \frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2 + \sum[ - \frac{\beta_j^2}{2\tau^2}]]\\
                 									& = - argmax_\beta [\sum[\frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2 + \sum[\frac{\beta_j^2}{2\tau^2}]]\\
                 									& = - argmax_beta \frac{\sigma^2}{2}[\sum(y_i-\beta^Tx_i)^2 + \frac{\sigma^2}{\tau^2}\sum\beta_j^2]\\
                 									& = argmin[||y-X\beta||^2 + \lambda \sum\beta_j^2]
 \end{aligned}
\end{equation}
$$

Here $$\lambda = \frac{\sigma^2}{\tau^2}$$. The above equation is nothing but the Ridge Regularization. Isn't is fascinating.

Similarly, if we pick Laplacian prior $$\beta_j \sim \frac{1}{2b} e^{- \mid\beta_j\mid/b} $$, the we get the Lasso Regularization. (You can do the calculation of your own). 

![lr_likelihood_1](/assets/img/laplace.jpeg)

Now $$\lambda$$ controls the amount of regularization. More the value of $$\lambda$$ is, more the $$\beta$$s will be driven towards zero. How ?

Let explain: $$\lambda = \frac{\sigma^2}{\tau^2}$$. If $$\lambda$$ is big, that means $$\tau$$ is small. Small value of $$\tau$$ makes prior distribution more narrow or peaked like the following.

![lr_likelihood_1](/assets/img/normal_2.jpeg)

That means we have more belief that $$\beta$$s are close to zero.




































