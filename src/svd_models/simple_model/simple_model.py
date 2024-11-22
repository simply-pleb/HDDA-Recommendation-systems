from svd_models import SVDModelABC
from svd_models import jax_utils
from svd_models.simple_model import jax_solver
import jax
from jax import numpy as jnp


class SimpleSVDModel(SVDModelABC):
    def predict_pair(self, u_id, i_id, clip=True):
        user_known, item_known = False, False
        pred = 0

        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]

        if user_known and item_known:
            pred += jnp.dot(self.U[u_ix], self.V[i_ix])

        if clip:
            pred = jnp.clip(pred, self.min_rating, self.max_rating)

        return float(pred)

    def _run_optimizer(self, X, X_val=None):
        optimizer_f = None
        if self.optimizer_name == "SGD":
            optimizer_f = self._run_sgd
        elif self.optimizer_name == "BCD":
            optimizer_f = self._run_bcd
        elif self.optimizer_name == "ALS":
            optimizer_f = self._run_als
        else:
            raise NotImplementedError(
                f"The optimizer '{self.optimizer_name}' is not implemented. "
                "Supported optimizers are: 'SGD', 'BCD', 'ALS'."
            )

        if X_val is not None:
            X_val = jnp.array(X_val, dtype=jnp.float32)
        optimizer_f(X=jnp.array(X, dtype=jnp.float32), X_val=X_val)

    def _run_sgd(self, X, X_val):
        n_users = len(jnp.unique(X[:, 0]))
        n_items = len(jnp.unique(X[:, 1]))

        key = jax.random.PRNGKey(0)  # Initialize a random key
        key_U, key_V = jax.random.split(key)  # Split the key for reproducibility

        U = jax.random.normal(key_U, (n_users, self.n_factors)) * 0.1
        V = jax.random.normal(key_V, (n_items, self.n_factors)) * 0.1

        for epoch in range(self.n_epochs):
            start = self._on_epoch_begin(epoch)

            if self.shuffle:
                X = jax_utils.shuffle(X=X)

            U, V = jax_solver.run_epoch(X=X, U=U, V=V, lr=self.lr, reg=self.reg)

            if X_val is not None:
                self.metrics_.loc[epoch, :] = jax_solver.compute_metrics(X_val, U, V)
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

        self.U = U
        self.V = V

    def _run_bcd(X, X_val):
        pass

    def _run_als(X, X_val):
        pass
