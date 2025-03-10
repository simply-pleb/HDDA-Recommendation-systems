import itertools
import numpy as np
from recommendex import SimpleSVDModel, ImprovedSVDModel

import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split


VAL_METRICS_DIR = "data/validations/"
SUBMISSION_DIR = "data/predictions/"


def dense_to_sparse(data: pd.DataFrame):
    sparse_matrix = csr_matrix(data.values)

    # Convert sparse matrix to DataFrame with columns u_id, i_id, rating
    rows, cols = (
        sparse_matrix.nonzero()
    )  # Get row and column indices of non-zero elements
    ratings = sparse_matrix.data  # Get the corresponding ratings (non-zero values)

    # Create a DataFrame
    sparse_df = pd.DataFrame(
        {
            "u_id": data.index[rows].values,  # Map rows to user ids
            "i_id": data.columns[cols].values,  # Map columns to item ids
            "rating": ratings,
        }
    )

    return sparse_df


def run_simple_example():
    data_dir = "data/csv/simple_user_movie_ratings.csv"

    data = pd.read_csv(data_dir)
    data.set_index("User name", inplace=True)
    data_no_none = data.fillna(0)

    print(data.head())

    sparse_df = dense_to_sparse(data_no_none)

    model = SimpleSVDModel(
        n_factors=1, min_rating=1, max_rating=5, batch_size=1, n_epochs=50, lr=0.05
    )

    model.fit(X=sparse_df)

    test_data = dense_to_sparse(data)
    pred = model.predict(X=dense_to_sparse(data))

    test_data["pred"] = np.array(pred)
    print(test_data)


def run_simple_model():
    data_dir = "data/csv/ratings_given.csv"
    data_pred_dir = "data/csv/ratings_test_eval.csv"

    data = pd.read_csv(data_dir, names=["u_id", "i_id", "rating"])
    data_pred = pd.read_csv(data_pred_dir, names=["ID", "u_id", "i_id"])

    data_train, data_val = train_test_split(data, test_size=0.01, shuffle=True)
    # data_train, data_val = data, None
    # data_val = None

    print(data_train)
    print(data_val)

    n_factors = 10
    optimizer = "BCD"
    lr = 0.005
    reg = 0.02
    n_epochs = 20
    batch_size = 128

    model = SimpleSVDModel(
        n_factors=n_factors,
        shuffle=True,
        optimizer=optimizer,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        reg=reg,
    )

    model.fit(data_train, data_val)
    print(model.U)
    print(model.V)
    validation_name = f"baseline_n_factors={n_factors}_optimizer={optimizer}_lr={lr}_reg={reg}_batch_size={batch_size}_n_epochs={n_epochs}.csv"
    model.metrics_.to_csv(VAL_METRICS_DIR + validation_name, index=False)

    # pred = model.predict(data_val)
    # print(data_val)
    # print(pred)

    pred = model.predict(data_pred)
    submission = pd.concat([data_pred["ID"], pd.Series(pred, name="Rating")], axis=1)
    submission.columns = ["ID", "Rating"]

    submission_name = f"baseline_n_factors={n_factors}_optimizer={optimizer}_lr={lr}_reg={reg}_batch_size={batch_size}_n_epochs={n_epochs}.csv"

    submission.to_csv(SUBMISSION_DIR + submission_name, index=False)


def run_improved_model():
    data_dir = "data/csv/ratings_given.csv"
    data_pred_dir = "data/csv/ratings_test_eval.csv"

    data = pd.read_csv(data_dir, names=["u_id", "i_id", "rating"])
    data_pred = pd.read_csv(data_pred_dir, names=["ID", "u_id", "i_id"])

    data_train, data_val = train_test_split(data, test_size=0.01, shuffle=True)

    print(data_train)
    print(data_val)

    n_factors = 10
    optimizer = "BCD"
    lr = 0.01
    reg = 0.02
    n_epochs = 20

    model = ImprovedSVDModel(
        n_factors=n_factors, shuffle=True, optimizer=optimizer, n_epochs=n_epochs, lr=lr
    )

    model.fit(data_train, data_val)

    pred = model.predict(data_val[:5])
    print(data_val[:5])
    print(pred)

    pred = model.predict(data_pred)
    submission = pd.concat([data_pred["ID"], pd.Series(pred, name="Rating")], axis=1)
    submission.columns = ["ID", "Rating"]

    submission_name = f"improved_n_factors={n_factors}_optimizer={optimizer}.csv"

    submission.to_csv(SUBMISSION_DIR + submission_name, index=False)


def run_model(
    data, data_pred, model_name, n_factors, optimizer, lr, reg, batch_size, n_epochs
):
    data_train, data_val = train_test_split(data, test_size=0.01, shuffle=True)
    # data_train, data_val = data, None
    # data_val = None

    print(data_train)
    print(data_val)

    if model_name == "baseline":
        model = SimpleSVDModel(
            n_factors=n_factors,
            shuffle=True,
            optimizer=optimizer,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            reg=reg,
        )
    elif model_name == "improved":
        model = ImprovedSVDModel(
            n_factors=n_factors,
            shuffle=True,
            optimizer=optimizer,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            reg=reg,
        )

    model.fit(data_train, data_val)
    print(model.U)
    print(model.V)
    # save validation data
    validation_name = f"{model_name}_n_factors={n_factors}_optimizer={optimizer}_lr={lr}_reg={reg}_batch_size={batch_size}_n_epochs={n_epochs}.csv"
    model.metrics_.to_csv(VAL_METRICS_DIR + validation_name, index=False)

    # save predictions
    pred = model.predict(data_pred)
    submission = pd.concat([data_pred["ID"], pd.Series(pred, name="Rating")], axis=1)
    submission.columns = ["ID", "Rating"]

    submission_name = f"{model_name}_n_factors={n_factors}_optimizer={optimizer}_lr={lr}_reg={reg}_batch_size={batch_size}_n_epochs={n_epochs}.csv"

    submission.to_csv(SUBMISSION_DIR + submission_name, index=False)


grid_ = {
    "model_name": ["baseline"],
    # "n_factors": [1, 2, 3, 5, 7, 11, 13, 17, 23, 29],
    "n_factors": [1, 2, 4, 8, 16, 32],
    "optimizer": ["SGD", "BCD"],
    "lr": [0.001, 0.0025, 0.005, 0.01],
    "reg": [0.02],
    "batch_size": [128],
    "n_epochs": [20],
}


def run_grid_search(data, data_pred, grid_dict):
    # Generate all parameter combinations
    param_keys = list(grid_dict.keys())
    param_combinations = list(itertools.product(*grid_dict.values()))

    # Iterate over all combinations
    for combination in param_combinations:
        params = dict(
            zip(param_keys, combination)
        )  # Map the keys to the combination values
        # Run the model with the current set of parameters
        print("Running", params)
        run_model(
            data,
            data_pred,
            model_name=params["model_name"],
            n_factors=params["n_factors"],
            optimizer=params["optimizer"],
            lr=params["lr"],
            reg=params["reg"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
        )


if __name__ == "__main__":

    # run_simple_model()

    data_dir = "data/csv/ratings_given.csv"
    data_pred_dir = "data/csv/ratings_test_eval.csv"

    data = pd.read_csv(data_dir, names=["u_id", "i_id", "rating"])
    data_pred = pd.read_csv(data_pred_dir, names=["ID", "u_id", "i_id"])

    run_grid_search(data, data_pred, grid_)
