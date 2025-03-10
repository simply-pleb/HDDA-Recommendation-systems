import numpy as np
import pandas as pd
from recommendex.svd_models import SVDModelABC
from recommendex.svd_models import jax_utils
from recommendex.svd_models.improved_model import jax_solver
import jax
from jax import numpy as jnp


class ImprovedSVDModel(SVDModelABC):
    def __init__(self, *args, **kwargs):
        # Call the parent class __init__ method with all arguments
        super().__init__(*args, **kwargs)

        # Specify beforehand all the properties. This is used for serialization
        self.V = None
        self.U = None
        self.bu = None
        self.bi = None
        self.mu = None
        self.global_mean_ = None  # TODO: why do we have both mu and global mean?
        self.item_mapping_ = None
        self.user_mapping_ = None

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

    def _predict(self, X, clip=True):
        # Check that required columns are in the DataFrame
        if not {"u_id", "i_id"}.issubset(X.columns):
            raise ValueError("Input DataFrame must contain 'u_id' and 'i_id' columns.")

        # return [
        #     self.predict_pair(u_id, i_id, clip)
        #     for u_id, i_id in zip(X["u_id"], X["i_id"])
        # ]

        X_ = X.copy()

        # Map user and item IDs to indices
        X_["u_ix"] = X_["u_id"].map(self.user_mapping_).fillna(-1).astype(int)
        X_["i_ix"] = X_["i_id"].map(self.item_mapping_).fillna(-1).astype(int)

        # Create indicator columns for valid indices
        X_["u_ic"] = (X_["u_ix"] != -1).astype(int)
        X_["i_ic"] = (X_["i_ix"] != -1).astype(int)

        # Set u_ix and i_ix values of -1 to 0
        # These values will be multiplied by 0 and thus canceled
        # Done in order to not use conditionals
        X_.loc[X_["u_ix"] == -1, "u_ix"] = 0
        X_.loc[X_["i_ix"] == -1, "i_ix"] = 0

        # Define prediction function with indicator values
        pred_fn = (
            lambda U, V, bu, bi, mu, u_ic, i_ic, u_ix, i_ix: (u_ic * i_ic)
            * jnp.dot(U[u_ix], V[i_ix])
            + u_ic * bu[u_ix]
            + i_ic * bi[i_ix]
            + mu
        )

        # Compute predictions
        user_item_matrix = X_[["u_ic", "i_ic", "u_ix", "i_ix"]].to_numpy()
        user_item_jnp = jnp.array(user_item_matrix)
        preds_jnp = jax.vmap(
            pred_fn, in_axes=[None, None, None, None, None, 0, 0, 0, 0]
        )(
            self.U,
            self.V,
            self.bu,
            self.bi,
            self.mu,
            user_item_jnp[:, 0],
            user_item_jnp[:, 1],
            user_item_jnp[:, 2],
            user_item_jnp[:, 3],
        )
        preds_series = pd.Series(np.array(preds_jnp), index=X_.index)

        X_["ratings"] = preds_series

        if clip:
            X_["ratings"] = X_["ratings"].clip(
                lower=self.min_rating, upper=self.max_rating
            )

        return X_["ratings"]
