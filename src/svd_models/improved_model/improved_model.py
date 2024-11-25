from svd_models import SVDModelABC
from svd_models import jax_utils
from svd_models.improved_model import jax_solver
import jax
from jax import numpy as jnp


class ImprovedSVDModel(SVDModelABC):
    def predict_pair(self, u_id, i_id, clip=True):
        user_known, item_known = False, False
        pred = self.mu

        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]
            pred += self.bu[u_ix]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]
            pred += self.bi[i_ix]

        if user_known and item_known:
            pred += jnp.dot(self.U[u_ix], self.V[i_ix])

        if clip:
            pred = jnp.clip(pred, self.min_rating, self.max_rating)

        return float(pred)

    def _run_optimizer(self, X, X_val=None):
        optimizer_f = None
        if self.optimizer_name == "BCD":
            optimizer_f = jax_solver.run_epoch_bcd_batched
        elif self.optimizer_name == "ALS":
            optimizer_f = jax_solver.run_epoch_bcd_batched
        elif self.optimizer_name == "SGD":
            optimizer_f = jax_solver.run_epoch_sgd_batched
        else:
            raise NotImplementedError(
                f"The optimizer '{self.optimizer_name}' is not implemented. "
                "Supported optimizers are: 'SGD', 'BCD', 'ALS'."
            )

        if X_val is not None:
            X_val = jnp.array(X_val, dtype=jnp.float32)

        n_users = len(jnp.unique(X[:, 0]))
        n_items = len(jnp.unique(X[:, 1]))

        key = jax.random.PRNGKey(0)  # Initialize a random key
        key_U, key_V, key_shuffle = jax.random.split(
            key, num=3
        )  # Split the key for reproducibility

        U = jax.random.normal(key_U, (n_users, self.n_factors)) * 0.1
        V = jax.random.normal(key_V, (n_items, self.n_factors)) * 0.1
        mu = jnp.mean(X[:, 2])
        bu = jnp.zeros(n_users)
        bi = jnp.zeros(n_items)

        for epoch in range(self.n_epochs):
            start = self._on_epoch_begin(epoch)

            if self.shuffle:
                X = jax_utils.shuffle(X=X, key=key_shuffle)

            batches = jax_utils.create_batches(X, self.batch_size)

            U, V, bu, bi = optimizer_f(
                batches=batches, U=U, V=V, bu=bu, bi=bi, mu=mu, lr=self.lr, reg=self.reg
            )

            if X_val is not None:
                self.metrics_.loc[epoch, :] = jax_solver.compute_metrics(
                    X_val, U, V, bu, bi, mu
                )
                self._on_epoch_end(
                    start,
                    self.metrics_.loc[epoch, "Loss"],
                    self.metrics_.loc[epoch, "RMSE"],
                    self.metrics_.loc[epoch, "MAE"],
                )
                if self.early_stopping:
                    val_rmse = self.metrics_["RMSE"].tolist()
                    if self._early_stopping(val_rmse, epoch, self.min_delta):
                        break
            else:
                self._on_epoch_end(start)

        # update model params
        self.U = U
        self.V = V
        self.bu = bu
        self.bi = bi
        self.mu = mu
