from functools import partial
import jax
from jax import numpy as jnp

# Define the Mean Squared Error (MSE)
@jax.jit
def mse(target, pred):
    return jnp.mean(jnp.square(target - pred))

# Define the prediction function
@jax.jit
def pred_func(params, x):
    # Example: params is a tuple (w, b)
    w, b = params
    return w * x + b

# Example parameters and input
params = (2.0, 0.5)  # Example weights and bias
x = jnp.array([1.0, 2.0, 3.0])
target = jnp.array([3.0, 5.0, 7.0])

# Function to calculate the gradient with respect to the first parameter (w)
@partial(jax.jit, static_argnums=[0])
def grad_mse_wrt_param(param_idx, params, x, target):
    # Wrapper to fix all parameters except the one at `param_idx`
    @jax.jit
    def mse_with_fixed_params(updated_param):
        new_params = params[:param_idx] + (updated_param,) + params[param_idx+1:]
        jax.debug.print("params={params}\nnew_params={new_params}", params=new_params, new_params=new_params)
        preds = pred_func(new_params, x)
        return mse(target, preds)
    
    return jax.grad(mse_with_fixed_params)(params[param_idx])

# Compute gradients
grad_w = grad_mse_wrt_param(0, params, x, target)  # Gradient wrt w
grad_b = grad_mse_wrt_param(1, params, x, target)  # Gradient wrt b

print("Gradient of MSE with respect to w:", grad_w)
print("Gradient of MSE with respect to b:", grad_b)
