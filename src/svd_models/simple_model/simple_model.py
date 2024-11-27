import numpy as np
import pandas as pd
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

        else:
            pred += self.mu

        if clip:
            pred = jnp.clip(pred, self.min_rating, self.max_rating)

        return float(pred)

    def _run_optimizer(self, X, X_val=None):
        optimizer_f = None
        if self.optimizer_name == "BCD":
            optimizer_f = jax_solver.run_epoch_bcd_batched
        elif self.optimizer_name == "ALS":
            optimizer_f = jax_solver.run_epoch_bcd_batched
        elif self.optimizer_name == "ALS_WOLFE":
            optimizer_f = jax_solver.run_epoch_bcd_wolfe
        elif self.optimizer_name == "SGD":
            optimizer_f = jax_solver.run_epoch_sgd_batched
        else:
            raise NotImplementedError(
                f"The optimizer '{self.optimizer_name}' is not implemented. "
                "Supported optimizers are: 'SGD', 'BCD', 'ALS', 'ALS_WOLFE'."
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

        for epoch in range(self.n_epochs):
            start = self._on_epoch_begin(epoch)

            if self.shuffle:
                X = jax_utils.shuffle(X=X, key=key_shuffle)

            # Wolfe function does not use batching
            if self.optimizer_name == "ALS_WOLFE":
                batches = X
            else:
                batches = jax_utils.create_batches(X, self.batch_size)

            U, V = optimizer_f(batches, U=U, V=V, lr=self.lr, reg=self.reg)

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
            else:
                self._on_epoch_end(start)

        # update model params
        self.U = U
        self.V = V
        self.mu = mu

    def _predict(self, X: pd.DataFrame, clip=True):

        # Check that required columns are in the DataFrame
        if not {"u_id", "i_id"}.issubset(X.columns):
            raise ValueError("Input DataFrame must contain 'u_id' and 'i_id' columns.")

        # return [
        #     self.predict_pair(u_id, i_id, clip)
        #     for u_id, i_id in zip(X["u_id"], X["i_id"])
        # ]

        X_ = X.copy()

        X_["u_ix"] = X_["u_id"].map(self.user_mapping_).fillna(-1).astype(int)
        X_["i_ix"] = X_["i_id"].map(self.item_mapping_).fillna(-1).astype(int)

        X_valid_users_items = X_[(X_["u_ix"] != -1) & (X_["i_ix"] != -1)].copy()

        user_item_matrix = X_valid_users_items[["u_ix", "i_ix"]].to_numpy()
        user_item_jnp = jnp.array(user_item_matrix)

        preds_jnp = jax.vmap(jax_solver._predict, in_axes=[None, None, 0, 0])(
            self.U, self.V, user_item_jnp[:, 0], user_item_jnp[:, 1]
        )
        
        preds_series = pd.Series(np.array(preds_jnp), index=X_valid_users_items.index)
        X_valid_users_items["ratings"] = preds_series

        X_["ratings"] = self.mu
      
        X_.loc[X_valid_users_items.index, "ratings"] = X_valid_users_items["ratings"]

        if clip:
            X_["ratings"] = X_["ratings"].clip(
                lower=self.min_rating, upper=self.max_rating
            )  

        return X_["ratings"]
