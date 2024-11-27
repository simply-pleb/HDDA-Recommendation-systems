---
title: HDDA Project
author: Ahmadsho Akdodshoev, Ayhem Bouabid
---

# Introduction

...
$m$ users $n$ items

$$W_{u,i} = \begin{cases}1 & \text{if} (u,i)\in \Omega \\ 0 & \text{otherwise} \end{cases}$$

Let us define

- set $A$ of all users in the system and set $B$ of all items in the system
- set $\Upsilon$ such that for any user $u \in \Upsilon$  there exists such $i\in B$ that $(u,i)\in\Omega$, and
- set $\Iota$ such that for any user $i \in \Iota$  there exists such $u\in A$ that $(u,i)\in\Omega$

Thus, we have defined set $\Upsilon$ of all users that have rated some item, and set $\Iota$ of all items that have a rating.

# Models

## Baseline model

$$\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}}{\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left(X - UV^T\right)_{u,i}^2}$$

### Decent direction of $U$

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

Hence, the decent direction can be expressed as $p = 2e_{u,i}\cdot V_i$ or simply $$p = e_{u,i}\cdot V_i$$
<!-- $$\begin{aligned}\nabla_\theta J &= (f_u(\theta) - X_{u,i})\nabla_\theta f_u(\theta)
\\ &= (V_i\theta - X_{u,i})V_i
\end{aligned}$$ -->

<!-- To find the local minimum of the objective we need to find such $\theta^*$ that $\nabla_\theta J(\theta^*) = 0$

$$2V_{i}^T\left(V_{i}\theta - X_{u,i}\right) = 0$$

$$2V_{i}^T V_{i}\theta - V_{i}^TX_{u,i} = 0$$ -->

### Decent direction of $V$

The descent direction of $V$ can be found itendically by fixing $U$. Which will result in $$p = e_{u,i}\cdot U_u$$

where $(u,i)\in \Omega$ and $e_{u,i}$ is defined as $$e_{u,i} = X_{u,i} - U_u^T\theta$$
where $\theta\in \mathbb{R}^k$ is the latent representation of item $i$ and $U_u\in \mathbb{R}^k$ is the latent representation of user $u$.

### Update rules

By defining $e_{u,i}$ as $$e_{u,i} = \left(X - UV^T\right)_{u,i}$$

The updates can be expressed as:

- $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
- $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $\alpha$ is the step size and $\lambda$ is regularization parameter.

## Improved model

The main downside of the baseline model is its inability to make prediction for users or items that do not have any ratings. The baseline model is capable to predict a rating for a $(u, i)$ pair only if such user such user $u \in \Upsilon$ and $i \in \Iota$.

$$\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}}{\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left(X - \left(\mu + \beta_u + b_i +  UV^T\right)\right)_{u,i}^2}$$

where $\mu$ is the average rating, $\beta$ is the bias of each user, and $b$ is the bias of each item.

### Decent direction of $U$ and $V$

It can be observed that the descent direction of $U$ and $V$ in the improved model are identical to the decent direction of $U$ and $V$ in the baseline model.

### Decent direction of $\beta$

To find the decent direction of $\beta$, parameters $U$, $V$ and $b$ have to be fixed.

### Decent direction of $b$

### Update rules

By defining $e_{u,i}$ as $$e_{u,i} = \left(X - \left(\mu + \beta_u + b_i +  UV^T\right)\right)_{u,i}$$

The updates can be expressed as:

- $b_i \gets b_i + \alpha \left(e_{u,i} - \lambda b_i \right)$
- $\beta_u \gets \beta_u + \alpha \left(e_{u,i} - \lambda \beta_u\right)$
- $V_i \gets V_i + \alpha \left(e_{u,i}\cdot U_u - \lambda V_i\right)$
- $U_u \gets U_u + \alpha \left(e_{u,i}\cdot V_i - \lambda U_u\right)$

where $\alpha$ is the step size and $\lambda$ is regularization parameter.

# Algorithms

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

<!-- Notice that the update of $U_u$ at iteration $(k+1)$ uses $V_i$ but not $V_i$ -->

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

# Results
