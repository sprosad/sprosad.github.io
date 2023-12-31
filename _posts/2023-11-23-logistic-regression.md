---
title: Logistic Regression
date: 2023-11-23 17:00:00 +0530
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

## Some Mathematics

**Derivative of sigmoid function**

![sigmoid_derivative](/assets/img/sigmoid_derivative.jpeg)

Lets assume, $$ h_\theta (x) $$ is the probability of prediction class 1.
$$ y_i $$ follows Bernoulli's distribution.

Where $$ h_\theta (x) = 1/(1 + e^{-\theta^T x})$$

$$ y_i = 1 $$ with probability $$ h_\theta (x) $$ 

$$ P(y=1|x;\theta)=h_\theta (x) $$ 

$$ y_i = 0 $$ with probability $$ (1 - h_\theta (x)) $$ 

$$ P(y=0|x;\theta) =  1 - h_\theta (x) $$

Note that this can be written more compactly as

$$
P(y|x;\theta) = (h_\theta (x))^y * (1 - h_\theta (x))^{1-y}
$$

Assuming that the n training examples were generated independently, we can then write down the likelihood of the parameters as

![lr_likelihood_1](/assets/img/lr_likelihood_1.jpeg)

It will be easier to maximize the log-likelihood:

![lr_likelihood_2](/assets/img/lr_likelihood_2.jpeg)


How do we maximize the likelihood? Similar to our derivation in the case of linear regression, we can use gradient ascent.
Let’s start by working with just one training example (x, y), and take derivatives to derive the stochastic gradient ascent rule

![lr_likelihood_3](/assets/img/lr_likelihood_3.jpeg)

Above, we used the fact that $$ g\prime(z) = g(z)(1-g(z)) $$
This therefore gives us the stochastic gradient ascent rule

![lr_likelihood_4](/assets/img/lr_likelihood_4.jpeg)

## Why not Least Square for optimization ?

**Binary Outcomes and Non-linear Relationships**: Logistic regression is used for binary outcomes (e.g., 0 or 1, success or failure). The relationship between the independent variables and the probability of the outcome is not linear but sigmoidal. The least squares method, on the other hand, assumes a linear relationship and is designed to minimize the distance between observed values and the values predicted by a linear equation. This linear approach doesn't fit well with the non-linear, S-shaped curve of the logistic function.

**Heteroscedasticity**: In logistic regression, the variance of the error terms is not constant. The variability in the outcome differs at different values of the independent variables. This phenomenon, known as heteroscedasticity, violates one of the key assumptions of the least squares method, which assumes homoscedasticity (constant variance of errors). Least squares, therefore, would not provide the best, most reliable estimates for logistic regression.

**Probability and Odds**: Logistic regression models the probability that an outcome will occur, specifically in terms of odds. The output is not a direct value but a logit - the natural log of odds that the outcome will occur. Least squares is not equipped to handle this transformation and direct relationship with odds and probabilities.

**Outliers and Extreme Values**: The least squares method is sensitive to outliers. In logistic regression, using least squares can lead to significant distortions if there are extreme values or outliers in the data, as it tries to minimize squared errors, which gives more weight to larger errors (caused by outliers).

**Maximum Likelihood Estimation (MLE)**: Instead, logistic regression uses MLE, which is more appropriate for models with binary outcomes. MLE seeks to find the parameter values that make the observed data most probable. This method is better suited for the probabilistic nature of logistic regression and effectively handles the non-linear relationship between independent variables and the outcome.

**Model Interpretation**: Logistic regression coefficients represent the change in the log odds of the dependent variable for a one-unit change in an independent variable. This interpretation is more meaningful for binary outcomes and is not aligned with the interpretations derived from least squares, which is more about changes in the actual values of the dependent variable.

In essence, the least squares method is not suitable for logistic regression due to its underlying assumptions of linearity, constant error variance, and sensitivity to outliers, which do not align with the characteristics of logistic regression models.

## Why not MSE as cost function ?

Using Mean Squared Error (MSE) as the cost function in logistic regression is not advisable due to several reasons related to the nature of logistic regression and the implications of using MSE in this context:

**Non-Convexity**: The primary issue with using MSE in logistic regression is that it leads to a non-convex cost function. Logistic regression, when combined with a non-linear function like the sigmoid function, and then applied with MSE, results in a cost function that has multiple local minima. This is problematic because gradient descent, the optimization algorithm commonly used to find the minimum of the cost function, might get stuck in these local minima instead of finding the global minimum.

**Mismatched Error Profile**: Logistic regression models probabilities and its output is bounded between 0 and 1. MSE, by squaring the errors, can disproportionately penalize small deviations in predictions when the true outcomes are near the boundaries (0 or 1). This is not ideal for a probability model where such deviations are common and not necessarily indicative of a poor model.

**Poor Interpretation with Binary Outcomes**: The binary nature of outcomes in logistic regression (0 or 1) makes MSE less intuitive as a measure of model performance. MSE is more suited to continuous data where the idea of "distance" between the predicted and actual values is more meaningful.

**Incorrect Assumptions About the Data Distribution**: MSE implicitly assumes that the errors (residuals) are normally distributed. However, in logistic regression, the residuals do not follow a normal distribution due to the binary outcomes. This mismatch can lead to inefficiencies and biases in parameter estimation.

**Impact on Gradient Descent**: The use of MSE with logistic regression can impact the gradient descent optimization process. The gradients may be very small when the predicted probability is close to 0 or 1, which can slow down the convergence of the algorithm or lead to convergence at suboptimal points.

**Alternative**: Cross-Entropy Loss: Instead of MSE, logistic regression typically uses a cross-entropy loss (or log loss) function. This function is convex for logistic regression, ensuring a single global minimum, and it directly aligns with the probabilistic interpretation of the model, providing a more natural measure of the model's performance.

If you were to use MSE as the cost function in logistic regression, the issues mentioned above could result in poor model performance, slow and unreliable convergence during training, and ultimately, less accurate predictions. The model could be less effective in distinguishing between the binary classes, and the optimization process might not yield the most effective parameters for the logistic regression model.
































