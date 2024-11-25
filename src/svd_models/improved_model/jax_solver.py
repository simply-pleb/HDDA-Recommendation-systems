import jax
from jax import numpy as jnp


@jax.jit
def _predict(U, V, bu, bi, mu, user, item):
    return jnp.dot(U[user], V[item]) + bu[user] + bi[item] + mu


@jax.jit
def _apply_update_V(U, V, user, item, error, lr, reg):
    V_new = V.at[item].add(lr * (error * U[user] - reg * V[item]))
    return V_new


@jax.jit
def _apply_update_U(U, V, user, item, error, lr, reg):
    U_new = U.at[user].add(lr * (error * V[item] - reg * U[user]))
    return U_new


@jax.jit
def _apply_update_bu(bu, user, error, lr, reg):
    return bu.at[user].add(lr * (error - reg * bu[user]))


@jax.jit
def _apply_update_bi(bi, item, error, lr, reg):
    return bi.at[item].add(lr * (error - reg * bi[item]))


@jax.jit
def run_epoch_sgd_batched(batches, U, V, bu, bi, mu, lr, reg):
    def update_step(carry, batch):
        U, V, bu, bi, mu = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
            U, V, bu, bi, mu, users, items
        )

        # Compute errors
        errors = ratings - preds

        # Update latent factors for users and items in the batch
        V_updates = jax.vmap(
            _apply_update_V, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        U_updates = jax.vmap(
            _apply_update_U, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        bu_updates = jax.vmap(_apply_update_bu, in_axes=[None, 0, 0, None, None])(
            bu, users, errors, lr, reg
        )
        bi_updates = jax.vmap(_apply_update_bi, in_axes=[None, 0, 0, None, None])(
            bi, users, errors, lr, reg
        )

        # add new axis to account for batch size
        V_deltas = V_updates - V[jnp.newaxis, :, :]
        U_deltas = U_updates - U[jnp.newaxis, :, :]
        bu_deltas = bu_updates - bu[jnp.newaxis, :]
        bi_deltas = bi_updates - bi[jnp.newaxis, :]

        V = V + jnp.sum(V_deltas, axis=0)
        U = U + jnp.sum(U_deltas, axis=0)
        bu = bu + jnp.sum(bu_deltas, axis=0)
        bi = bi + jnp.sum(bi_deltas, axis=0)

        return (U, V, bu, bi, mu), None

    # Apply updates for each batch
    (U, V, bu, bi, mu), _ = jax.lax.scan(update_step, (U, V, bu, bi, mu), batches)

    return U, V, bu, bi


@jax.jit
def run_epoch_bcd_batched(batches, U, V, bu, bi, mu, lr, reg):
    def update_step_V(carry, batch):
        U, V, bu, bi, mu = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
            U, V, bu, bi, mu, users, items
        )

        errors = ratings - preds

        # Update latent factors
        V_updates = jax.vmap(
            _apply_update_V, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        V_deltas = V_updates - V[jnp.newaxis, :, :]
        V = V + jnp.sum(V_deltas, axis=0)

        return (U, V, bu, bi, mu), None

    def update_step_U(carry, batch):
        U, V, bu, bi, mu = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
            U, V, bu, bi, mu, users, items
        )

        errors = ratings - preds

        # Update latent factors
        U_updates = jax.vmap(
            _apply_update_U, in_axes=[None, None, 0, 0, 0, None, None]
        )(U, V, users, items, errors, lr, reg)
        U_deltas = U_updates - U[jnp.newaxis, :, :]
        U = U + jnp.sum(U_deltas, axis=0)

        return (U, V, bu, bi, mu), None

    def update_step_bu(carry, batch):
        U, V, bu, bi, mu = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
            U, V, bu, bi, mu, users, items
        )

        errors = ratings - preds

        # Update latent factors
        bu_updates = jax.vmap(_apply_update_bu, in_axes=[None, 0, 0, None, None])(
            bu, users, errors, lr, reg
        )
        bu_deltas = bu_updates - bu[jnp.newaxis, :]
        bu = bu + jnp.sum(bu_deltas, axis=0)

        return (U, V, bu, bi, mu), None

    def update_step_bi(carry, batch):
        U, V, bu, bi, mu = carry
        users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
        # Convert to int32 for indexing
        users, items = jnp.int32(users), jnp.int32(items)

        preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
            U, V, bu, bi, mu, users, items
        )

        errors = ratings - preds

        # Update latent factors
        bi_updates = jax.vmap(_apply_update_bi, in_axes=[None, 0, 0, None, None])(
            bi, items, errors, lr, reg
        )
        bi_deltas = bi_updates - bi[jnp.newaxis, :]
        bi = bi + jnp.sum(bi_deltas, axis=0)

        return (U, V, bu, bi, mu), None

    (U, V, bu, bi, mu), _ = jax.lax.scan(update_step_V, (U, V, bu, bi, mu), batches)
    (U, V, bu, bi, mu), _ = jax.lax.scan(update_step_U, (U, V, bu, bi, mu), batches)
    (U, V, bu, bi, mu), _ = jax.lax.scan(update_step_bu, (U, V, bu, bi, mu), batches)
    (U, V, bu, bi, mu), _ = jax.lax.scan(update_step_bi, (U, V, bu, bi, mu), batches)

    return U, V, bu, bi


# @jax.jit
# def run_epoch_bcd_batched(batches, U, V, bu, bi, mu, lr, reg):
#     def update_step(carry, batch, update_fn, update_target, delta_shape):
#         U, V, bu, bi, mu = carry
#         users, items, ratings = batch[:, 0], batch[:, 1], batch[:, 2]
#         # Convert to int32 for indexing
#         users, items = jnp.int32(users), jnp.int32(items)

#         # Compute predictions
#         preds = jax.vmap(_predict, in_axes=[None, None, None, None, None, 0, 0])(
#             U, V, bu, bi, mu, users, items
#         )
#         errors = ratings - preds

#         # Perform updates
#         def apply_update(target):
#             indices = jax.lax.cond(update_target == "bu", lambda: users, lambda: items)
#             updates = jax.lax.cond(
#                 delta_shape == 3,
#                 lambda: jax.vmap(update_fn, in_axes=[None, None, 0, 0, 0, None, None])(
#                     U, V, users, items, errors, lr, reg
#                 ),
#                 lambda: jax.vmap(update_fn, in_axes=[None, 0, 0, None, None])(
#                     target, indices, errors, lr, reg
#                 ),
#             )
#             deltas = jax.lax.cond(
#                 delta_shape == 3,
#                 lambda: updates - target[jnp.newaxis, :, :],
#                 lambda: updates - target[jnp.newaxis, :]
#             )
#             return target + jnp.sum(deltas, axis=0)

#         # Update the respective variable
#         U = jax.lax.cond(update_target == "U", lambda: apply_update(U), lambda: U)
#         V = jax.lax.cond(update_target == "V", lambda: apply_update(V), lambda: V)
#         bu = jax.lax.cond(update_target == "bu", lambda: apply_update(bu), lambda: bu)
#         bi = jax.lax.cond(update_target == "bi", lambda: apply_update(bi), lambda: bi)

#         return (U, V, bu, bi, mu), None

#     # Perform scans for each target
#     (U, V, bu, bi, mu), _ = jax.lax.scan(
#         lambda carry, batch: update_step(carry, batch, _apply_update_V, "V", 3),
#         (U, V, bu, bi, mu),
#         batches,
#     )
#     (U, V, bu, bi, mu), _ = jax.lax.scan(
#         lambda carry, batch: update_step(carry, batch, _apply_update_U, "U", 3),
#         (U, V, bu, bi, mu),
#         batches,
#     )
#     (U, V, bu, bi, mu), _ = jax.lax.scan(
#         lambda carry, batch: update_step(carry, batch, _apply_update_bu, "bu", 1),
#         (U, V, bu, bi, mu),
#         batches,
#     )
#     (U, V, bu, bi, mu), _ = jax.lax.scan(
#         lambda carry, batch: update_step(carry, batch, _apply_update_bi, "bi", 1),
#         (U, V, bu, bi, mu),
#         batches,
#     )

#     return U, V, bu, bi


@jax.jit
def compute_metrics(X_val, U, V, bu, bi, mu):
    def compute_residual(carry, ranting_tuple):
        U, V, bu, bi, mu = carry
        user, item, rating = ranting_tuple

        user = jnp.int32(user)
        item = jnp.int32(item)

        pred = _predict(U, V, bu, bi, mu, user, item)

        residual = rating - pred
        return carry, residual

    _, residuals = jax.lax.scan(compute_residual, (U, V, bu, bi, mu), X_val)

    loss = jnp.mean(jnp.square(residuals))
    rmse = jnp.sqrt(loss)
    mae = jnp.mean(jnp.abs(residuals))

    return loss, rmse, mae
