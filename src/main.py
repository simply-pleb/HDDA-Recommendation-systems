from svd_models import SimpleSVDModel

import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split


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

    model = SimpleSVDModel(n_factors=1, min_rating=1, max_rating=4, lr=0.05)

    model.fit(X=sparse_df)

    pred = model.predict(X=dense_to_sparse(data))

    print(sparse_df)
    print(pred)
    
    
def run_full():
    data_dir = "data/csv/ratings_given.csv"
    data_pred_dir = "data/csv/ratings_test_eval.csv"
    
    data = pd.read_csv(data_dir, names=['u_id', 'i_id', 'rating'])
    data_pred = pd.read_csv(data_pred_dir, names=['u_id', 'i_id', 'rating'])
    
    data_train, data_val = train_test_split(data[:5000], test_size=0.05, shuffle=True)
    
    print(data_train)
    print(data_val)
    
    model = SimpleSVDModel(reg=0)
    
    model.fit(data_train, data_val)
    
    pred = model.predict(data_val)
    print(data_val)
    print(pred)

    
if __name__ == '__main__':
    run_full()