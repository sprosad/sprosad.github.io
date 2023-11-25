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

One of the issues in $$ \beta $$ estimated from Ordinary Least Square(OLS) is that , $$\hat{\beta_ols} have high variance.





