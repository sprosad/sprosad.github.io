---
title: Logistic Regression
date: 2023-11-23 17:00:00 +0800
categories: [Data Science]
tags: [machine learning]     # TAG names should always be lowercase
toc : true
math: true
mermaid: true
---

## Logistic Regression: A Simple Explanation
Logistic regression is a statistical method used for analyzing datasets in which there are one or more independent variables that determine an outcome. The outcome is dichotomous, which means it only has two possible outcomes (e.g., "yes" or "no", "success" or "failure").

Imagine you have a bunch of oranges and apples, and you want to build a machine that can tell them apart. You notice that, generally, oranges are rounder and apples are a bit heavier. In logistic regression, you'd use these features (roundness and weight) to predict whether a fruit is an orange or an apple. The method looks at the data you have (known fruits), learns the patterns (how roundness and weight are related to being an apple or an orange), and then uses this knowledge to predict the type of new fruits it sees.

## The Nitty-Gritty Details
**Binary Outcome**: Logistic regression is used when the dependent variable (the one you want to predict) is categorical and binary, like predicting if a tumor is malignant (1) or benign (0).

**Odds Ratio and Logit Function**: The core of logistic regression is the logit function, which is the natural logarithm of the odds ratio. This function links the probability of the binary outcome to the independent variables.

**Sigmoid Curve**: Logistic regression models the probability that the dependent variable belongs to a particular category. This is done through a sigmoid (S-shaped) curve which ensures that the output probability stays between 0 and 1.

**Maximum Likelihood Estimation (MLE)**: Unlike linear regression which uses least squares, logistic regression uses MLE to find the best fitting model. This method estimates the parameters which maximize the likelihood of observing the sample data.

**Coefficient Interpretation**: The coefficients in logistic regression are in terms of log odds. Unlike linear regression where the coefficients represent the mean change in the dependent variable for a one unit change in the independent variable, in logistic regression, they represent the change in the log odds of the dependent variable for a one unit change in the independent variable.

**Multicollinearity and Feature Selection**: It's important to check for multicollinearity in logistic regression. Highly correlated independent variables can distort the results and interpretations of the coefficients.

**Model Evaluation**: The performance of a logistic regression model is not assessed using R-squared. Instead, metrics like AUC-ROC curve, confusion matrix, accuracy, precision, recall, and F1 score are used.

**Assumptions**: Logistic regression assumes that there is no high intercorrelation (multicollinearity) among the independent variables. It also assumes linearity of independent variables and log odds.

In summary, logistic regression is a predictive analysis used for classification problems. It's based on the concept of probability and is particularly useful when the dependent variable is dichotomous. The interpretation of results and model assessment, however, differ significantly from linear regression.
