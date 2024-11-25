import numpy as np
import pandas as pd
from svd_models import SVDModelABC
from svd_models import jax_utils
from svd_models.improved_model import jax_solver
import jax
from jax import numpy as jnp

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class ExperimentalSVDModel(SVDModelABC):

    def predict_pair(self, u_id, i_id, clip=True):
        user_known, item_known = False, False
        pred = self.mu

        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]
            pred += self.bu[u_ix]
            used_data = self.np_users_metadata[u_ix]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]
            pred += self.bi[i_ix]
            item_data = self.np_items_metadata[i_ix]

        if user_known and item_known:
            pred += jnp.dot(self.U[u_ix], self.V[i_ix])

            user_item_feature = np.concatenate([used_data, item_data]).reshape(1, -1)
            res_pred = float(self.xgb_model.predict(user_item_feature)[0])
            pred += res_pred

        if clip:
            pred = jnp.clip(pred, self.min_rating, self.max_rating)

        return float(pred)

    @staticmethod
    def _predict_pair(carry, data):
        U, V, bu, bi, mu = carry
        user, item = data
        user, item = jnp.int32(user), jnp.int32(item)
        pred = jnp.dot(U[user], V[item]) + bu[user] + bi[item] + mu

        return carry, pred

    def _predict(self, X):
        _, preds = jax.lax.scan(
            self._predict_pair, (self.U, self.V, self.bu, self.bi, self.mu), X[:, :2]
        )
        return preds

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

    def _map_ids_metadata(
        self, users_metadata: pd.DataFrame, items_metadata: pd.DataFrame
    ):
        # Map IDs using the user and item mappings
        users_metadata["u_id"] = users_metadata["u_id"].map(self.user_mapping_)
        items_metadata["i_id"] = items_metadata["i_id"].map(self.item_mapping_)

        # Drop rows with unmapped IDs (NaN after mapping) if necessary
        users_metadata = users_metadata.dropna(subset=["u_id"])
        items_metadata = items_metadata.dropna(subset=["i_id"])

        # Ensure the indices are integers for consistent indexing
        users_metadata.loc[:, "u_id"] = users_metadata["u_id"].astype(int)
        items_metadata.loc[:, "i_id"] = items_metadata["i_id"].astype(int)

        return users_metadata, items_metadata

    def _to_numpy_metadata(
        self, users_metadata: pd.DataFrame, items_metadata: pd.DataFrame
    ):
        # Sort by ID for proper alignment
        users_metadata_ = users_metadata.sort_values(by="u_id")
        items_metadata_ = items_metadata.sort_values(by="i_id")

        # Remove the index column for NumPy conversion
        user_array = users_metadata_.drop(columns=["u_id"]).to_numpy()
        item_array = items_metadata_.drop(columns=["i_id"]).to_numpy()

        return user_array, item_array

    # this will fix a XGBoost to minimize the residual of
    def fit_residual(self, X, users_metadata, items_metadata):

        users_metadata, items_metadata = self._map_ids_metadata(
            users_metadata.copy(), items_metadata.copy()
        )

        self.np_users_metadata, self.np_items_metadata = self._to_numpy_metadata(
            users_metadata.copy(), items_metadata.copy()
        )

        X = self._preprocess_data(X)

        X_features = pd.DataFrame(X, columns=["u_id", "i_id", "rating"])
        X_features = pd.merge(X_features, users_metadata, on="u_id", how="left")
        X_features = pd.merge(X_features, items_metadata, on="i_id", how="left")
        X_features = X_features.drop(columns=["u_id", "i_id", "rating"]).values

        y_res = X[:, 2] - np.array(self._predict(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_res, test_size=0.1, random_state=42
        )
        # import matplotlib.pyplot as plt
        # plt.hist(y_train, bins=50)
        # plt.title("Target Variable Distribution")
        # plt.show()

        # Define XGBoost regressor
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,  # Number of boosting rounds
            learning_rate=0.05,  # Step size shrinkage
            max_depth=10,  # Max depth of a tree
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
            random_state=42,
            reg_alpha=0.5,  # L1 regularization
            reg_lambda=1.5,  # L2 regularization
        )

        # Train
        print("Fitting XGBoost...")
        self.xgb_model.fit(X_train, y_train)

        # Predict
        y_pred = self.xgb_model.predict(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
