# HDDA-Recommendation-systems

This repository contains a JAX implementation and comparison of two basic recommendation models:

- SVD model $$X_{u,i} = U_u V_i^T$$
- SVD model with user and item biases $$X_{u,i} = \mu + b_u + b_i + U_u V_i^T$$

where $X_{u,i}$ is the rating of user $u$ for item $i$, $\mu$ is the global average rating, $b_u$ is the user bias, $b_i$ is the item bias, $U_u$ is the user latent factors and $V_i$ is the item latent factors.

Two optimization were implemented using JAX:

- Stochastic gradient descent
- Block coordinate descent (alternating least squares)

At the moment, optimization algorithms are implemented per each model. They can be found in corresponding `jax_solver.py` files.

## Data

The data is based on the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/).

## Usage

<!-- For now, to run a model of your choice, execute the following command:

```bash
python src/main.py
```

and edit the `src/main.py` file to choose the model you want to run.

You can also choose the optimizer that you want to use:

- SGD
- BCD (ALS) -->

use

```
pip install -e src/
```

## Results

The optimization problem and the derivations are available in the `report/` folder.

This work was a part of a kaggle competition for the High Dimensional Data Analysis course in Innopolis University. The RMSE on the validation set was $0.8758$ for the SVD model with user and item biases, 10 latent factors and BCD optimizer.

## References

- [funk-svd github repo](https://github.com/gbolmier/funk-svd)
- [Wiki page on Matrix factorization :)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
