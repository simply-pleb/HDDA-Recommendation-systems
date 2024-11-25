# Code based on https://github.com/gbolmier/funk-svd

from abc import ABC, abstractmethod
import time
import pandas as pd
import numpy as np


class SVDModelABC(ABC):
    """
    Abstract base class for SVD-based models.
    Defines the interface for implementing Singular Value Decomposition
    algorithms for collaborative filtering tasks.
    """

    def __init__(
        self,
        lr=0.005,
        reg=0.02,
        n_epochs=20,
        n_factors=100,
        early_stopping=False,
        shuffle=False,
        min_delta=0.001,
        min_rating=1,
        max_rating=5,
        optimizer="BCD",
        batch_size=128
    ):
        """
        Initializes the SVD model parameters.

        Parameters
        ----------
        lr : float
            Learning rate.
        reg : float
            L2 regularization factor.
        n_epochs : int
            Number of SGD iterations.
        n_factors : int
            Number of latent factors.
        early_stopping : bool
            Whether or not to stop training based on validation monitoring.
        shuffle : bool
            Whether or not to shuffle the training set before each epoch.
        min_delta : float
            Minimum delta to argue for an improvement.
        min_rating : int
            Minimum value a rating should be clipped to at inference time.
        max_rating : int
            Maximum value a rating should be clipped to at inference time.
        """
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta = min_delta
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.optimizer_name = optimizer
        self.batch_size = batch_size

    def fit(self, X, X_val=None):
        """
        Learns model weights from input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training set with 'u_id', 'i_id', and 'rating' columns.
        X_val : pandas.DataFrame, optional
            Validation set with the same structure as X.

        Returns
        -------
        self : SVDModelABC
            Fitted model.
        """
        X = self._preprocess_data(X)

        if X_val is not None:
            X_val = self._preprocess_data(X_val, train=False)
            self._init_metrics()

        self.global_mean_ = np.mean(X[:, 2])
        self._run_optimizer(X, X_val)

    def predict(self, X, clip=True):
        """
        Predicts ratings for given user-item pairs.

        Parameters
        ----------
        X : pandas.DataFrame
            Data with 'u_id' and 'i_id' columns.
        clip : bool, default=True
            Whether to clip predictions to min_rating and max_rating.

        Returns
        -------
        list
            Predicted ratings.
        """
        return [
            self.predict_pair(u_id, i_id, clip)
            for u_id, i_id in zip(X["u_id"], X["i_id"])
        ]

    @abstractmethod
    def predict_pair(self, u_id, i_id, clip=True):
        """
        Predicts the rating for a single user-item pair.

        Parameters
        ----------
        u_id : int
            User ID.
        i_id : int
            Item ID.
        clip : bool, default=True
            Whether to clip predictions to min_rating and max_rating.

        Returns
        -------
        float
            Predicted rating.
        """
        pass

    def _preprocess_data(self, X, train=True):
        """
        Preprocesses the dataset by mapping user/item IDs to indices.

        Parameters
        ----------
        X : pandas.DataFrame
            Data with 'u_id', 'i_id', and 'rating' columns.
        train : bool, default=True
            Whether the dataset is for training or validation.

        Returns
        -------
        np.ndarray
            Preprocessed dataset.
        """
        print("Preprocessing data...\n")
        X = X.copy()

        if train:  # Mappings have to be created
            user_ids = X["u_id"].unique().tolist()
            item_ids = X["i_id"].unique().tolist()

            n_users = len(user_ids)
            n_items = len(item_ids)

            user_idx = range(n_users)
            item_idx = range(n_items)

            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))

        X["u_id"] = X["u_id"].map(self.user_mapping_)
        X["i_id"] = X["i_id"].map(self.item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)

        X["u_id"] = X["u_id"].astype(np.int32)
        X["i_id"] = X["i_id"].astype(np.int32)

        return X[["u_id", "i_id", "rating"]].values

    def _init_metrics(self):
        metrics = np.zeros((self.n_epochs, 3), dtype=float)
        self.metrics_ = pd.DataFrame(metrics, columns=["Loss", "RMSE", "MAE"])

    def _early_stopping(self, val_rmse, epoch_idx, min_delta):
        """Returns True if validation rmse is not improving.

        Last rmse (plus `min_delta`) is compared with the second to last.

        Parameters
        ----------
        val_rmse : list
            Validation RMSEs.
        min_delta : float
            Minimun delta to argue for an improvement.

        Returns
        -------
        early_stopping : bool
            Whether to stop training or not.
        """
        if epoch_idx > 0:
            if val_rmse[epoch_idx] + min_delta > val_rmse[epoch_idx - 1]:
                self.metrics_ = self.metrics_.loc[: (epoch_idx + 1), :]
                return True
        return False

    @abstractmethod
    def _run_optimizer(self, X, X_val=None):
        """
        Runs the optimization algorithm set in self.optimizer_name to update model weights.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        X_val : np.ndarray, optional
            Validation data.
        """
        pass
    
    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Parameters
        ----------
        epoch_ix : int
            Epoch index.

        Returns
        -------
        start : float
            Starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, val_loss=None, val_rmse=None, val_mae=None):
        """Displays epoch ending log.

        If self.verbose, computes and displays validation metrics (loss, rmse,
        and mae).

        Parameters
        ----------
        start : float
            Starting time of the current epoch.
        val_loss : float, default=None
            Validation loss.
        val_rmse : float, default=None
            Validation rmse.
        val_mae : float, default=None
            Validation mae.
        """
        end = time.time()

        if val_loss is not None:
            print(f'val_loss: {val_loss:.2f}', end=' - ')
            print(f'val_rmse: {val_rmse:.2f}', end=' - ')
            print(f'val_mae: {val_mae:.2f}', end=' - ')

        print(f'took {end - start:.1f} sec')
