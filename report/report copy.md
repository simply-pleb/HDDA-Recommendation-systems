---
title: HDDA Project
authors: Ahmadsho Akdodshoev, Ayhem Bouabid
---

# Introduction
In this project, we attempt to develop a simple Recommender System. Given training data consisting of $m$ users and $i$ items. We would like to build a mathematical model that would enable us to predict the ratings of: 

1. new users on given items
2. new items by already existing users


# Notation
The matrix $W$ is an indicator matrix indicating whether an item $i$ was rated by a given user $u$. More formally: $W \in \mathbb{R}^{m, n}$ wwhere

$$
W_{u,i} = \begin{cases}1 & \text{if} (u,i)\in \Omega \\ 0 & \text{otherwise} \end{cases}$$

The ratings matrix is denoted by $X$. The known ratings can be mathematically expressed in matrix form as: 

$$ 
W \circ X
$$

where $\circ$ is the element-wise matrix product.

Further more we define: 

* $S_u$ as the set of users in the training data
* $S_i$ as the set of items in the training data.
* $\Omega$ as the set of pairs $(u, i)$ where user $u$ rated the item $i$

We assume that each item $i \S_i$ has been rated by at least one user, and similarly each user $u \in S_u$ rated at least one givem item.



# Baseline model

The initial step is to to build numerical / mathematical representations for both users and items denoted by $U \in \mathbb{R}^{m, k}$ and $V \in \mathbb{R}^{m, k}$ respectively. The $i$ -th row in $U$ represents the $i$ -th user and similarity, the $j$ -th row in $V$ represents the $j$ -th item.  

Modeling the rating of a the $i$ -th user $u$ to the $i$ -th item as the dot product between the representations, then one to `learn` good representations from the data is to minimize the following mathematical objective:


$$ 

\begin{equation}

\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}} L = {\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left(X - UV^T\right)_{u,i}^2}

\end{equation}
$$

For the $k$ -th user and the $j$ -th item, we define $ $ 

$$ 
\begin{equation}
f(U_k, V_j) = (X_{j, k} - U_k ^ T \cdot V_j) ^  2
\end{equation}
$$

It is easy to see that 

$$ 
\begin{equation}
L = \sum _ {(i, j) \in \Omega} f(U_k, V_j)
\end{equation}
$$



## Minimizing the optimizaton objetive

We consider 2 algorithms to solve the optimization problem $(1)$:

* Stochastic Gradient Desent (SGD)
* Block Coordinate Descent (BCD)

### SGD
Given $K$ as the number of iterations and $\alpha$ as the step size.

1. Initialize $V$ and $U$

2. For $k$ to $K$:
    1. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item: 

        * apply both updates simultaneously: 
            - $V_k \gets V_k - \alpha \cdot \nabla f(V_k, V_k)$    while fixing $U_j$
            - $U_j \gets U_j - \alpha \cdot \nabla f(V_k, U_j)$    while fixing $U_k$



### BCD 

1. Initialize $V$ and $U$
2. For $k$ to $K$:
    1. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item: 
    
        1. update items representations: 
            - $V_k \gets V_k - \alpha \cdot \nabla f(V_k, V_k)$ 
            while fixing $U_j$


    2. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item:

        1. update users representations: 
            - $U_j \gets U_j - \alpha \cdot \nabla f(V_k, U_j)$ while fixing $U_k$        


## Prediction

Prediction $\hat y$ for a $(u, i)$ pair is made as follows

$$

\hat y = \begin{cases} UV^T & \text{if}\  (u,i)\in \Omega \\ \mu & \text{otherwise} \end{cases}

$$

where $\mu$ is the average prediction computed from the training data 

$$
\mu = \frac{1}{\sum_{u=1}^m\sum_{i=1}^n W_{ij}}\sum_{u=1}^m\sum_{i=1}^n{W_{ij}X_{ij}}
$$


## Mathematical Derivations

To do later


# Improved Model

The main downside of the baseline model is its inability to make prediction for users or items that do not have any ratings. The baseline model is capable to predict a rating for a $(u, i)$ pair only if such user such user $u \in \S_u$ and $i \in \S_i$.

$$

\min_{U\in\mathbb{R}^{m\times r}, V\in\mathbb{R}^{n\times r}} L = {\sum_{u=1}^m\sum_{i=1}^n W_{u,i}\left(X - \left(\mu + \beta_u + b_i +  UV^T\right)\right)_{u,i}^2}

$$

where $\mu$ is the average rating, $\beta$ is the bias of each user, and $b$ is the bias of each item.

## Minimizing the mathematical Objective

We used the same optimizers as as for the baseline model: 

### SGD

1. Initialize $V$ and $U$

2. For $k$ to $K$:
    1. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item: 

        * apply both updates simultaneously: 
            - $V_k \gets V_k - \alpha \cdot \nabla f(V_k, V_k)$
            - $U_j \gets U_j - \alpha \cdot \nabla f(V_k, U_j)$
            - $b_k \gets b_k - \alpha \cdot \nabla f(V_k, V_k, b_k, \beta_j)$
            - $\beta_j \gets \beta_j - \alpha \cdot \nabla f(V_k, V_k, b_k, \beta_j)$
    
            - Each update for a given parameter is carried out while fixing the others.


### BCD

1. Initialize $V$ and $U$
2. For $k$ to $K$:
    1. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item: 
    
        1. update $V$: 
            - $V_k \gets V_k - \alpha \cdot \nabla f(V_k, V_k)$ 


    2. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item:

        1. update $U$: 
            - $U_j \gets U_j - \alpha \cdot \nabla f(V_k, U_j)$ 

    3. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item:

        1. udpate $b$: 
            - $b_k \gets b_k - \alpha \cdot \nabla f(V_k, V_k, b_k, \beta_j)$
        
    4. iterate through all pairs rated (user, item) pairs, For the $j$ -th user and $k$ -th item:

        1. update $\beta$: 
            - $\beta_j \gets \beta_j - \alpha \cdot \nabla f(V_k, V_k, b_k, \beta_j)$


# Results
