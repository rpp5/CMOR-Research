import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def orient_dataset_by_instance(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort and clean the node-level dataset so that:
      - rows are ordered by (instance_id, visit_idx)
      - id columns have consistent types
    """
    df = full_df.copy()

    # Ensure instance_id is int
    df["instance_id"] = df["instance_id"].astype(int)

    # node_id as int
    df["node_id"] = df["node_id"].astype(int)

    # parent_id: allow NA for root
    df["parent_id"] = df["parent_id"].astype("Int64")

    # Sort by instance then visit order
    df = df.sort_values(["instance_id", "visit_idx"]).reset_index(drop=True)

    return df

def build_xy_vectors(full_df: pd.DataFrame):
    """
    From the oriented node-level dataframe, build feature blocks and target.

    Returns:
        X_node   (DataFrame): node-local features
        X_path   (DataFrame): path / branching features
        X_global (DataFrame): global search-state features
        y        (ndarray):   has_optimal_subtree labels (0/1)
    """
    # --- feature blocks ---

    node_cols = [
        "depth",
        "lp_objective",
        "slack",
        "num_fixed",
        "num_fixed_1",
        "num_fixed_0",
        "fix_ratio",
        "total_weight_fixed_1",
        "total_value_fixed_1",
        "frac_var_index",
        "frac_var_normalized",
        "frac_var_weight",
        "frac_var_value",
        "frac_var_ratio",
        "frac_var_fraction",
    ]

    path_cols = [
        "branch_parent_var",
        "branch_parent_dir",
        "branch_parent_weight",
        "branch_parent_value",
        "branch_parent_ratio",
    ]

    global_cols = [
        "visit_idx",
        "open_nodes_count",
        "incumbent_val",
        "has_incumbent",
        "node_gap",
        "node_rel_gap",
    ]

    X_node   = full_df[node_cols].copy()
    X_path   = full_df[path_cols].copy()
    X_global = full_df[global_cols].copy()

    y = full_df["has_optimal_subtree"].astype(int).values
    instance_ids = full_df["instance_id"].values

    return X_node, X_path, X_global, y, instance_ids

def train_test_split_by_instance(
    X: pd.DataFrame,
    y: np.ndarray,
    instance_ids: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
):
    """
    Split data into train and test sets by instance_id so that
    all nodes from the same instance end up in the same split.

    Args:
        X           : full feature matrix (DataFrame or array) for all nodes.
        y           : labels (N,)
        instance_ids: instance_id for each row (N,)
        test_size   : fraction of instances to use as test
        random_state: random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    unique_instances = np.unique(instance_ids)

    train_inst, test_inst = train_test_split(
        unique_instances,
        test_size=test_size,
        random_state=random_state,
    )

    train_mask = np.isin(instance_ids, train_inst)
    test_mask  = np.isin(instance_ids, test_inst)

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    return X_train, X_test, y_train, y_test

def create_logreg_pipeline() -> Pipeline:
    """
    Create a sklearn pipeline that:
      - imputes missing values with the mean,
      - standardizes features,
      - fits a class-weighted logistic regression (weighted cross-entropy).
    """
    pipe = Pipeline([
        # replaces NaN w/ the mean value
        ("imputer", SimpleImputer(strategy="mean")),
        # transforms each feature to have mean 0 and std dev 1
        ("scaler", StandardScaler()),
        # calls logreg model
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # <-- weighted cross-entropy
            penalty="l2",
            solver="lbfgs",
        )),
    ])
    return pipe

def train_logreg(full_df: pd.DataFrame, test_size: float = 0.3, random_state: int = 0):
    """
    Complete pipeline:
      - orient dataset by (instance_id, visit_idx)
      - build feature blocks and labels
      - concatenate feature blocks into a single X
      - split train/test by instance_id
      - fit class-weighted logistic regression pipeline
      - return fitted model and basic evaluation metrics

    Args:
        full_df    : raw concatenated dataframe from build_training_set
        test_size  : fraction of instances to use as test
        random_state: random seed

    Returns:
        clf     : fitted sklearn Pipeline
        metrics : dict with basic metrics (ROC AUC, accuracy, F1)
    """
    # 1. Orient dataset
    df = orient_dataset_by_instance(full_df)

    # 2. Build feature blocks + y + instance_ids
    X_node, X_path, X_global, y, instance_ids = build_xy_vectors(df)

    # 3. Concatenate feature blocks for modeling
    X = pd.concat([X_node, X_path, X_global], axis=1)

    # 4. Train/test split by instance
    X_train, X_test, y_train, y_test = train_test_split_by_instance(
        X, y, instance_ids, test_size=test_size, random_state=random_state
    )

    # 5. Create and fit logistic regression pipeline
    clf = create_logreg_pipeline()
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return clf, metrics



