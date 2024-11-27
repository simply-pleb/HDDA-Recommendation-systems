---
title: HDDA Project - Recommendation System
author: Ahmadsho Akdodshoev, Ayhem Bouabid
---

# Introduction

    Let’s denote $X$ the $m$-by-$n$ dimensional preference matrix ($m$ users, $n$ films) and $r$ the number of ‘feature’ users.

$$W_{u,i} = \begin{cases}1 & \text{if}\  (u,i)\in \Omega \\ 0 & \text{otherwise} \end{cases}$$

Let us define the following sets

- The set $A$ is defined as a set of all users in the system and the set $B$ is defined as a set of all items in the system
- The set $\Upsilon$ is defined such that for any user $u \in \Upsilon$  there exists such item $i\in B$ that $(u,i)\in\Omega$, and
- The set $\Iota$ is defined such that for any item $i \in \Iota$ there exists such user $u\in A$ that $(u,i)\in\Omega$

Thus, we have defined set $\Upsilon$ of all users that have rated some item, and set $\Iota$ of all items that have a rating.

# Models

## Baseline model

$$\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}}{\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left[\left(X - UV^T\right)^2 + \lambda\left(\beta_u + b_i + ||U_u||^2 + ||V_i||^2\right)\right]_{u,i}}$$

### Update rules

By defining $e_{u,i}$ as $$e_{u,i} = \left(X - UV^T\right)_{u,i}$$

The updates can be expressed as:

- $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
- $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $\alpha$ is the step size and $\lambda$ is regularization parameter.

### Stochastic Gradient Descent 

1. Initialize $V$ and $U$
1. For $k$ to $K$:
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

### Block Coordinate Descent 

1. Initialize $V$ and $U$
1. For $k$ to $K$:
    1.  For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

### Prediction

Prediction $\hat y$ for a $(u, i)$ pair is made as follows

$$\hat y = \begin{cases} UV^T & \text{if}\  (u,i)\in \Omega \\ \mu & \text{otherwise} \end{cases}$$

where $\mu$ is the average prediction as is equal to $$\mu = \frac{1}{\sum_{u=1}^m\sum_{i=1}^n W_{ij}}\sum_{u=1}^m\sum_{i=1}^n{W_{ij}X_{ij}}$$

<!-- ### Decent direction of $U$

In order to derive the decent direction of $U$ we have to fix $V$.

The prediction of $U$ for some $(u, i)\in \Omega$ will be a function $f_u(\theta) = V_i^T\theta$,
where $\theta\in \mathbb{R}^k$ is the latent representation of user $u$ and $V_i\in \mathbb{R}^k$ is the latent representation of item $i$.

We have to minimize the following objective $$\min_{\theta\in \mathbb{R}^k} ||f_u(\theta) - X_{u,i}||_2^2,$$
where $X_{u,i}\in \mathbb R$ is the target rating.

Let us denote the objective as $J(\theta)$ as reexpress it as

$$J(\theta) = \left(\sum_{j=1}^{k}V_{i_j}\theta_j - X_{u,i}\right)^2$$

Let us find the gradient of objective $J$

$$\frac{\partial J}{\partial \theta_k}(\theta) = 2V_{i_j}\left(\sum_{j=1}^{k}V_{i_j}\theta_j - X_{u,i}\right)$$

$$\nabla_\theta J(\theta) = 2\left(V_{i}^T\theta - X_{u,i}\right)V_{i}$$

Thus, the descent direction is $$p = -2\left(V_{i}^T\theta - X_{u,i}\right)V_{i}$$

or

$$p = 2\left(X_{u,i} - V_{i}^T\theta\right)V_{i}$$

Let us define $e_{u,i}$ as $$e_{u,i} = X_{u,i} - V_i^T\theta$$

Hence, the decent direction can be expressed as $p = 2e_{u,i}\cdot V_i$ or simply $$p = e_{u,i}\cdot V_i$$ -->
<!-- $$\begin{aligned}\nabla_\theta J &= (f_u(\theta) - X_{u,i})\nabla_\theta f_u(\theta)
\\ &= (V_i\theta - X_{u,i})V_i
\end{aligned}$$ -->

<!-- To find the local minimum of the objective we need to find such $\theta^*$ that $\nabla_\theta J(\theta^*) = 0$

$$2V_{i}^T\left(V_{i}\theta - X_{u,i}\right) = 0$$

$$2V_{i}^T V_{i}\theta - V_{i}^TX_{u,i} = 0$$ -->

<!-- ### Decent direction of $V$

The descent direction of $V$ can be found itendically by fixing $U$. Which will result in $$p = e_{u,i}\cdot U_u$$

where $(u,i)\in \Omega$ and $e_{u,i}$ is defined as $$e_{u,i} = X_{u,i} - U_u^T\theta$$
where $\theta\in \mathbb{R}^k$ is the latent representation of item $i$ and $U_u\in \mathbb{R}^k$ is the latent representation of user $u$. -->



## Improved model

The main downside of the baseline model is its inability to make prediction for users or items that do not have any ratings. The baseline model is capable to predict a rating for a $(u, i)$ pair only if such user $u \in \Upsilon$ and $i \in \Iota$.

The improved model explicitly models the bias of every user $u\in\Upsilon$ and item $i\in\Iota$. Thus, it is able to make predictions even if $(u, i)\not \in \Omega$.

$$\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}}{\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left[\left(X - \left(\mu + \beta_u + b_i +  UV^T\right)\right)^2 + \lambda\left(\beta_u + b_i + ||U_u||^2 + ||V_i||^2\right)\right]_{u,i}}$$

where $\mu$ is the average rating, $\beta$ is the bias of each user, and $b$ is the bias of each item.

### Update rules

By defining $e_{u,i}$ as $$e_{u,i} = \left(X - \left(\mu + \beta_u + b_i +  UV^T\right)\right)_{u,i}$$

The updates can be expressed as:

- $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i \right)$
- $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
- $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
- $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $\alpha$ is the step size and $\lambda$ is regularization parameter.

### Stochastic Gradient Descent 

1. Initialize $V$, $U$, $\beta$ and $b$
1. For $k$ to $K$:
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i\right)$
            - $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

### Block Coordinate Descent 

1. Initialize $V$, $U$, $\beta$ and $b$
1. For $k$ to $K$:
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i\right)$
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
    1. For every pair $(u, i) \in \Omega$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

### Prediction

Prediction $\hat y$ for a $(u, i)$ pair is made as follows

$$\hat y = \begin{cases} \mu + \beta_u + b_i + UV^T & \text{if}\  (u,i)\in \Omega \\ \mu + \beta_u & \text{if}\  u\in \Upsilon \text{ and } i\not \in \Iota \\ \mu + b_i & \text{if}\  u\not\in \Upsilon \text{ and } i \in \Iota \\ \mu & \text{otherwise} \end{cases}$$

where $\mu$ is the average prediction as is equal to $$\mu = \frac{1}{\sum_{u=1}^m\sum_{i=1}^n W_{ij}}\sum_{u=1}^m\sum_{i=1}^n{W_{ij}X_{ij}}$$

<!-- ### Decent direction of $U$ and $V$

It can be observed that the descent direction of $U$ and $V$ in the improved model are identical to the decent direction of $U$ and $V$ in the baseline model.

### Decent direction of $\beta$

To find the decent direction of $\beta$, parameters $U$, $V$ and $b$ have to be fixed.

### Decent direction of $b$ -->


# Results

# Reference

TODO: add reference

---

<!-- # Algorithms

## Stochasic Gradient Descent

By having the update rules defined for both models, the SGD algorithm can be applied on both models.

Let us first define the algorithm for the baseline model with $K$ as the number of iterations:

1. Initialize $V^{(0)}$ and $U^{(0)}$
1. For $k$ to $K$:
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$


The SGD algorithm for the improved model will have three modifications:

1. Initialize $V^{(0)}$, $U^{(0)}$, $\beta^{(0)}$ and $b^{(0)}$
1. For $k$ to $K$:
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = \mu + \beta_{u} + b_i + (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i\right)$
            - $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

Where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

## Block Coordinate Descent

In block coordinate descent all parameters except one are fixed. Let us name the parameter that was not fixed as update parameter and the rest as fixed parameters.

Unlike SGD that updates every parameter on each iteration, BCD only updates the update parameter per iteration.

Thus, the BCD algorithm for the baseline model can be defined as follows:

1. Initialize $V$ and $U$
1. For $k$ to $K$:
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

The BCD for the improved model is defined as:

1. Initialize $V$, $U$, $\beta$ and $b$
1. For $k$ to $K$:
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i\right)$
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (UV^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply update:
            - $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
    1. For every pair $(u, i)$ such that $W_{u,i}=1$:
        1. $\text{target} = X_{u,i}$
        1. $\text{prediction} = (U^{}V^T)_{u,i}$
        1. $e_{u,i} = \text{target} - \text{prediction}$
        1. Apply updates:
            - $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

Where $K$ is the number of iterations, $\alpha$ is the step size and $\lambda$ is the regularization parameter.

# Results -->
