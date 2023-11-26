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

Now comes the second term of equation (i) which is the prior. Let's assume the prior distribution of $$\beta$$ is normal with mean 0 and variance $$\tau^2$$. So that means, $$\beta_j \sim N(0,\tau^2) \forall j = 1,2...,p$$

X is a nxp matrix, so there are p $$\beta$$

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

![lr_likelihood_1](/assets/img/normal_1.jpeg)

There is a lot of mass around 0. This is where the regularization come in. We are trying to achieve the values of $$\beta$$ close to zero in regularization. So, the prior distribution is trying to do the same thing.

So, again going back to our main equation (i) and putting the values from (ii) and (iii), we get:

$$
\begin{equation}
 \begin{aligned}
argmax_\beta [log(P(y \mid \beta)) + log(P(\beta))] & = argmax_\beta [\sum[ - \frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2 + \sum[ - \frac{\beta_j^2}{2\tau^2}]]\\
                 									& = - argmax_\beta [\sum[\frac{1}{2\sigma^2}(y_i-\beta^Tx_i)^2 + \sum[\frac{\beta_j^2}{2\tau^2}]]\\
                 									& = - argmax_\beta \frac{1}{2\sigma^2}[\sum(y_i-\beta^Tx_i)^2 + \frac{\sigma^2}{\tau^2}\sum\beta_j^2]\\
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

## Why Does Ridge Regression Improve Over Least Squares?

Ridge regression’s advantage over least squares is rooted in thebias-variance trade-off. As $$\lambda$$ increases,the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. In general, in situations where the relationship between the response and the predictors is close to linear, the least squares estimates will have low bias but may have high variance. This means that a small change in the training data can cause a large change in the least squares coefficient estimates. In particular,when the number of variables p is almost as large as the number of observations n, the least squares estimates will be extremely variable. And if p>n, then the least squares estimates do not even have a unique solution,whereas ridge regression can still perform well by trading off a small increase in bias for a large decrease in variance. Hence,ridge regression works best in situations where the least squares estimates have high variance.

## Disadvantages of Ridge Regression

Ridge regression does have one obvious disadvantage. Ridge regression will include all p predictors in the final model. The penalty $$\lambda\sum\beta_j^2$$ will shrink all of the coefficients towards zero,but it will not set any of them exactly to zero(unless $$\lambda = \infty).This may not be a problem for prediction accuracy, but it can create a challenge in model interpretation in settings in which the number of variables p is quite large. However, ridge regression will always generate a model involving all predictors. Increasing the value of $$\lambda$$ will tend to reduce the magnitudes of the coefficients, but will not result in exclusion of any of the variables. The lasso is a relatively recent alternative to ridge regression that overcomes this disadvantages.

## Lasso Regression

The lasso shrinks the coefficient estimates towards zero. However,in the case of the lasso,the l1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $$\lambda$$ is sufficiently large. Hence,much like best subset selection,the lasso performs variable selection. As a result,models generated from the lasso are generally much easier to interpret than those produced by ridge regression. We say that the lasso yields sparse models—that is, sparse models that involve only a subset of the variables. As in ridge regression, selecting a good value of $$\lambda$$ for the lasso is critical.

## Geometric Interpretetion of The Variable Selection Property of the Lasso

We can write the Ridge and Lasso in the following form:

![lr_likelihood_1](/assets/img/ridge_lasso.jpeg)

Whenp=2, then (6.8) indicates that the lasso coefficient estimates have the smallest RSS out of all points that lie within the diamond define dby |β1|+|β2|≤s. Similarly, the ridge regression estimates have the smallest RSS out of all points that lie within the circle defined by $$\beta_1^2 + \beta_2^2 <s$$. We can think of(6.8) as follows. When we perform the lasso we are trying to find the set of coefficient estimates that lead to the smallest RSS,subject to the constraint that there is a budget s for how large $$\sum{j=1}{p}|\beta_j|$$ can be. When s is extremely large,then this budget is not very restrictive, and so the coefficient estimates can be large. In fact, if s is large enough that the least squares solution falls within the budget, then(6.8) will simply yield the least square ssolution. In contrast, if s is small,then $$\sum{j=1}{p}|\beta_j|$$ must be small in order to avoid violating the budget. Similarly,(6.9)indicates that when we performr idge regression,we seek a set of coefficient estimates such that the RSS is as small as possible, subject to the requirement that $$\sum{j=1}{p}\beta_j^2$$ not exceed the budgets.

Why is it that the lasso, unlike ridge regression, results in coefficient estimates that are exactly equal to zero? The formulations(6.8)and(6.9) can be used to shed light on the issue. the following figure illustrates the situation. The least squares solution is marked as $$\cap{\beta}$$, while the blue diamond and

![lr_likelihood_1](/assets/img/ridge_lasso_2.jpeg)

circle represent the lasso and ridge regression constraints in(6.8)and(6.9), respectively. If s is sufficiently large, then the constraint regions will contain $$cap{\beta}$$, and so the ridge regression and lasso estimates will be the same as the least squares estimates. (Such a large value of s corresponds to $$\lambda=0$$ )However, in the above figure the least squares estimates lie outside of the diamond and the circle,and so the least squares estimates are not the same as the lasso and ridge regression estimates. The ellipses that are centered around $$\cap{\beta}$$ represent regions of constant RSS. In other words, all of the points on a given ellipse share a common value of the RSS. As the ellipse s expand away from the least squares coefficient estimates, the RSS increases. Equations (6.8)and(6.9) indicate that the lasso and ridge regression coefficient estimates are given bythe first point at which an ellipse contacts the constraint region. Since ridge regression has a circular constraint with no sharp points, this intersection will not generally occur on an axis,and so the ridge regression coefficient estimates will be exclusively non-zero. However, the lasso constraint has corners at each of the axes, and so the ellipse will often intersect the constraint region at an axis.When this occurs,one of the coefficients will equal zero. In higher dimensions,many of the coefficient estimates may equal zero simultaneously. In the above figure,the intersection occurs at $$\beta_1 = 0$$,and so the resulting model will only include $$\beta_2$$. In the above figure, we considered the simple case of p=2. Whenp=3, then the constraint region for ridge regression becomes a sphere,and the constraint region for the lasso becomes a polyhedron.



































