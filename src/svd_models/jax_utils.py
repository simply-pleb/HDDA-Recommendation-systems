import jax
from jax import numpy as jnp

from functools import partial


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


@partial(jax.jit, static_argnames=("batch_size"))
def create_batches(X, batch_size):
    # Compute the number of batches
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Round up
    
    # Pad X to make it divisible by batch_size
    pad_size = num_batches * batch_size - num_samples
    X_padded = jnp.pad(X, ((0, pad_size), (0, 0)))  # Pad along the first axis
    
    # Split into batches
    batches = X_padded.reshape(num_batches, batch_size, -1)
    return batches
