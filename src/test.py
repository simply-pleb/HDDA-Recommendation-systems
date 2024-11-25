import jax
from jax import numpy as jnp

@jax.jit
def mse(x, y):
    return jnp.square(x - y)

# Compute the gradient of MSE with respect to `x`
mse_grad_x = jax.grad(mse, argnums=0)

# Example input arrays
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([1.5, 3, 3.5])

# Compute the gradient
grad_x = jax.vmap(mse_grad_x, in_axes=[0, 0])(x, y)

print("Gradient of MSE with respect to x:", grad_x)