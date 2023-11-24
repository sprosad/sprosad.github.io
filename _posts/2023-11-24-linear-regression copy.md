---
title: Linear Regression
date: 2023-11-23 17:00:00 +0800
categories: [Data Science]
tags: [machine learning]     # TAG names should always be lowercase
toc : true
math: true
mermaid: true
---

## Introduction

Linear regression is called "linear" because it models the relationship between a dependent variable and one or more independent variables using a linear equation. This linear equation is typically of the form:


The term "linear" in linear regression refers to the linearity of the model in its coefficients. This means that each predictor (independent variable) is multiplied by a coefficient and summed up to predict the output. The relationship between each predictor and the dependent variable is linear with respect to the coefficients, even if the predictors themselves are transformed or interact with each other in non-linear ways.

It's important to note that linearity in linear regression refers to the linearity of the model with respect to the coefficients, and not necessarily to the shape of the relationship between the independent and dependent variables, which can sometimes be non-linear.

Linear refers to the relationship between the parameters that you are estimating (e.g., 𝛽) and the outcome (e.g., 𝑦𝑖). Hence, $$ 𝑦=𝑒^𝑥 𝛽+𝜖 $$ is linear, but $$ 𝑦=𝑒^𝛽 𝑥+𝜖 $$
 is not. A linear model means that your estimate of your parameter vector can be written 𝛽̂ =\sum_{i}𝑤_i 𝑦_i
, where the {𝑤_i} are weights determined by your estimation procedure. Linear models can be solved algebraically in closed form, while many non-linear models need to be solved by numerical maximization using a computer.
