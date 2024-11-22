import jax


def shuffle(X, key):
    """
    Shuffles the array X using JAX.

    Parameters
    ----------
    X : jax.numpy.ndarray
        The array to shuffle.
    key : jax.random.PRNGKey
        The random key for reproducibility.

    Returns
    -------
    shuffled_X : jax.numpy.ndarray
        The shuffled array.
    """
    shuffled_indices = jax.random.permutation(key, X.shape[0])
    shuffled_X = X[shuffled_indices]
    return shuffled_X