import jax
from jax import numpy as jnp

# Define the Mean Squared Error (MSE)
@jax.jit
def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))

# Prediction function for multi-linear regression
@jax.jit
def predict(X, params):
    W, b = params
    return jnp.dot(X, W) + b

# Gradient calculation function
@jax.jit
def compute_gradients(params, X, y):
    def loss_fn(params):
        preds = predict(X, params)
        return mse(y, preds)

    # Compute gradients of the loss with respect to the parameters
    return jax.grad(loss_fn)(params)

# Gradient descent update function
@jax.jit
def update_params(params, grads, lr):
    W, b = params
    grad_W, grad_b = grads
    W = W - lr * grad_W
    b = b - lr * grad_b
    return W, b

# Multi-Linear Regression training
def train(X, y, params, lr=0.01, num_steps=100):
    for step in range(num_steps):
        # Compute gradients
        grads = compute_gradients(params, X, y)
        # Update parameters
        params = update_params(params, grads, lr)

        # Optionally, print loss
        if step % 10 == 0 or step == num_steps - 1:
            loss = mse(y, predict(X, params))
            print(f"Step {step}, Loss: {loss}")
    return params

# Example data
X = jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 4 samples, 2 features
y = jnp.array([5, 7, 9, 11])  # Target values

# Initialize parameters
W_init = jnp.zeros(X.shape[1])  # Initialize weights (2 features)
b_init = 0.0  # Initialize bias
params = (W_init, b_init)

# Train the model
learning_rate = 0.01
num_iterations = 100
optimized_params = train(X, y, params, lr=learning_rate, num_steps=num_iterations)

# Output the optimized parameters
W_opt, b_opt = optimized_params
print("Optimized weights:", W_opt)
print("Optimized bias:", b_opt)
