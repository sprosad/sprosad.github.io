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

## Geometric Interpretetion

Fitting a logistic regression model geometrically can be understood through the concept of a sigmoid function and how it relates to the data in a multidimensional space.

1. **The Sigmoid Function**: The core of logistic regression is the sigmoid function, also known as the logistic function. This function takes any real-valued number and maps it between 0 and 1. Mathematically, it's represented as $$ \sigma(z) = 1/(1+ e^{-z}) $$ where z is a linear combination of independent variables (e.g. $$ z = \beta_0 + \beta_1 x_1 +\beta_2 x_2 +.... \beta_n x_n) $$ where $$ \beta_i $$ are the coefficients and $$ x_i $$ are variables. 

2. **Geometric Representation**:

	- **Linear Boundary in Feature Space**: In the feature space, logistic regression determines a linear decision boundary. This boundary is the set of points where the model is uncertain (predicts a 50% chance for either class). On one side of this boundary, the model predicts the probability greater than 50% for one class, and on the other side, it predicts more than 50% for the other class.
	Transforming Through Sigmoid: The linear combination of variables and coefficients (the z value) is fed into the sigmoid function. This function compresses the output to a range between 0 and 1, representing the probability.
	Sigmoid Curve in Probability Space: If you plot the sigmoid function, it looks like an "S" shape. The bottom part of the "S" approaches zero, and the top part approaches one. The middle part, where the curve is steepest, represents the area around the decision boundary.

3. **Fitting the Model**:
	- **Maximum Likelihood Estimation (MLE)**: Geometrically, fitting a logistic regression model involves finding the line (in the case of two variables, a plane in higher dimensions) that best separates the classes in the feature space. This is done using MLE, which adjusts the coefficients $$ \beta_i ** to maximize the probability of observing the sample data.
	- **Probability Contours**: In a multidimensional feature space, this boundary can be visualized as a contour where each point on the contour represents a particular probability (like 0.5). Points inside one region of the contour have probabilities leaning towards one class and points on the other side lean towards the other class.

4. **Interpreting the Fit**:
	- The distance of a point from the decision boundary in the feature space can be interpreted as the confidence of the prediction. The further away from the boundary, the higher the confidence.
	- The slope of the decision boundary determines how quickly probabilities change as you move in the feature space.

In summary, geometrically, logistic regression fits a model by finding a linear decision boundary in the feature space. This boundary corresponds to a transition in probabilities, as modeled by the sigmoid function, from one class to another. The fitting process involves adjusting this boundary to best separate the classes based on the likelihood of observing the given data.
























