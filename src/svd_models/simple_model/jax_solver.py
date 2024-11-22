import jax
from jax import numpy as jnp

import numpy as np

@jax.jit
def run_epoch(X, U, V, lr, reg):
    
    def update_step(carry, ranting_tuple):
        U, V = carry
        user, item, rating = ranting_tuple
        
        user = jnp.int32(user)
        item = jnp.int32(item)
        
        pred = jnp.dot(U[user], V[item])
        
        error = rating - pred
        
        # Update latent factors
        U = U.at[user].add(lr * (error * V[item] - reg * U[user]))
        V = V.at[item].add(lr * (error * U[user] - reg * V[item]))
        
        return (U, V), None
    
    (U, V), _ = jax.lax.scan(update_step, (U, V), X)
    
    return U, V

@jax.jit
def compute_metrics(X_val, U, V):
    def compute_residual(carry, ranting_tuple):
        U, V = carry
        user, item, rating = ranting_tuple
        
        user = jnp.int32(user)
        item = jnp.int32(item)
        
        pred = jnp.dot(U[user], V[item])
        
        residual = rating - pred
        return carry, residual
    
    _, residuals = jax.lax.scan(compute_residual, (U, V), X_val)
    
    loss = jnp.mean(jnp.square(residuals))
    rmse = jnp.sqrt(loss)
    mae = jnp.mean(jnp.abs(residuals))
    
    return loss, rmse, mae