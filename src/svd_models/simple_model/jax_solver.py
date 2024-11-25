import jax
from jax import numpy as jnp


@jax.jit
def mse(target, pred):
    return jnp.mean(jnp.square(pred - target))

@jax.jit
def grad_loss(target, pred, loss_fn=mse):    
    return jax.grad(loss_fn, argnums=[0])(target, pred)

@jax.jit
def _predict(U, V, user, item):
    return jnp.dot(U[user], V[item])


@jax.jit
def _apply_update_V(U, V, user, item, error, lr, reg):
    V_new = V.at[item].add(lr * (error * U[user] - reg * V[item]))
    return V_new


@jax.jit
def _apply_update_U(U, V, user, item, error, lr, reg):
    U_new = U.at[user].add(lr * (error * V[item] - reg * U[user]))
    return U_new


@jax.jit
def run_epoch_sgd_batched(batches, U, V, lr, reg):

    def update_step(carry, batch):
        U, V = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, 0, 0])(U, V, users, items)

        # Compute errors
        errors = ratings - preds

        # Update latent factors for users and items in the batch
        V_updates = jax.vmap(
            _apply_update_V, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        U_updates = jax.vmap(
            _apply_update_U, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)

        V_deltas = V_updates - V[jnp.newaxis, :, :]
        U_deltas = U_updates - U[jnp.newaxis, :, :]

        V = V + jnp.sum(V_deltas, axis=0)
        U = U + jnp.sum(U_deltas, axis=0)

        return (U, V), None

    # Apply updates for each batch
    (U, V), _ = jax.lax.scan(update_step, (U, V), batches)

    return U, V


@jax.jit
def run_epoch_bcd_batched(batches, U, V, lr, reg):

    def update_step_V(carry, batch):
        U, V = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, 0, 0])(U, V, users, items)

        errors = ratings - preds

        # Update latent factors
        V_updates = jax.vmap(
            _apply_update_V, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        V_deltas = V_updates - V[jnp.newaxis, :, :]
        V = V + jnp.sum(V_deltas, axis=0)

        return (U, V), None

    def update_step_U(carry, batch):
        U, V = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, 0, 0])(U, V, users, items)

        errors = ratings - preds

        # Update latent factors
        U_updates = jax.vmap(
            _apply_update_U, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        U_deltas = U_updates - U[jnp.newaxis, :, :]
        U = U + jnp.sum(U_deltas, axis=0)

        return (U, V), None

    (U, V), _ = jax.lax.scan(update_step_V, (U, V), batches)
    (U, V), _ = jax.lax.scan(update_step_U, (U, V), batches)

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


# -------------- OLD METHODS ---------------


@jax.jit
def run_epoch_sgd(X, U, V, lr, reg):

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
def run_epoch_bcd(X, U, V, lr, reg):

    def update_step_V(carry, ranting_tuple):
        U, V = carry
        user, item, rating = ranting_tuple

        user = jnp.int32(user)
        item = jnp.int32(item)

        pred = jnp.dot(U[user], V[item])

        error = rating - pred

        V = V.at[item].add(lr * (error * U[user] - reg * V[item]))

        return (U, V), None

    def update_step_U(carry, ranting_tuple):
        U, V = carry
        user, item, rating = ranting_tuple

        user = jnp.int32(user)
        item = jnp.int32(item)

        pred = jnp.dot(U[user], V[item])

        error = rating - pred

        # Update latent factors
        U = U.at[user].add(lr * (error * V[item] - reg * U[user]))

        return (U, V), None

    (U, V), _ = jax.lax.scan(update_step_V, (U, V), X)
    (U, V), _ = jax.lax.scan(update_step_U, (U, V), X)

    return U, V


@jax.jit
def run_epoch_bcd_wolfe(X, U, V, lr, reg):

    def _predict_all(X, U, V):
        users, items, ratings = X[:, 0], X[:, 1], X[:, 2]
        users, items = jnp.int32(users), jnp.int32(items)

        def _predict_scan(carry, user_item):
            U, V = carry
            user, item = user_item

            pred = _predict(U, V, user, item)

            return carry, pred

        _, preds = jax.lax.scan(_predict_scan, (U, V), (users, items))

        return preds

    def _wolfe_line_search():
        # while both conditions are not met
        # check armijo condition
        # check curvature condition
        # if armijo condition fails lr_ = gamma * lr
        # if curvature condition fails lr_ = (lr + lr_)/2

        def _check_armijo_condition():
            # IF f' <= f + c_1 * lr * grad(f).T @ descent_dir THEN true ELSE false
            # where f is the prev loss and
            # f' is the loss with the current lr
            ...

        def _check_curvature_condition():
            # IF grad(f').T @ descent_dir >= c_2 * grad(f).T @ descent_dir THEN true ELSE false
            ...

        def _check_conditions_wolfe_not_satisfied():
            return not (_check_armijo_condition() & _check_curvature_condition())

        def _apply_armijo_cond_update(lr, gamma):
            return lr * gamma

        def _apply_curvature_cond_update(lr, lr_new):
            return (lr + lr_new) / 2

        def _search_lr_wolfe():
            # update if armijo condition failed
            armijo_cond_satisfied = _check_armijo_condition()
            lr_new = jax.lax.cond(
                pred=_apply_curvature_cond_update,
                true_fun=lambda _: lr,
                false_fun=lambda _: _apply_armijo_cond_update(lr),
                operands=None,
            )
            # update if curvature condition failed
            curvature_cond_satisfied = _check_curvature_condition()
            lr_new = jax.lax.cond(
                pred=_apply_curvature_cond_update,
                true_fun=lambda _: lr,
                false_fun=lambda _: _apply_curvature_cond_update(lr, lr_new),
                operands=None,
            )
            # return new LR
            return ((), lr_new)

        # shape input

        jax.lax.while_loop(
            cond_fun=_check_conditions_wolfe_not_satisfied,
            body_fun=_search_lr_wolfe,
            init_val=None,
        )

        lr_new = None

        return lr_new

    def _update(X, U, V, update_target, update_fn):
        users, items, ratings = X[:, 0], X[:, 1], X[:, 2]
        users, items = jnp.int32(users), jnp.int32(items)
        
        # predict all
        preds = _predict_all(X, U, V)
        
        # errors
        descent_dir = grad_loss(target=ratings, pred=preds, loss_fn=mse)
        
        # determine LR
        lr = _wolfe_line_search()
        
        # Update latent factors
        def apply_update(target):
            updates = jax.vmap(update_fn, in_axes=[None, None, 0, 0, 0, None, None])(
                U, V, users, items, descent_dir, lr, reg
            )
            deltas = updates - target[jnp.newaxis, :, :]

            return target + jnp.sum(deltas, axis=0)

        V = jax.lax.cond(update_target == "V", lambda: apply_update(V), lambda: V)
        U = jax.lax.cond(update_target == "U", lambda: apply_update(U), lambda: U)

        return (U, V), None

    (U, V), _ = _update(X, U, V, "V", _apply_update_V)
    (U, V), _ = _update(X, U, V, "U", _apply_update_U)

    return U, V
