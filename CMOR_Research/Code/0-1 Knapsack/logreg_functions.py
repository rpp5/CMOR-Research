# logreg_functions.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def orient_dataset_by_instance(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort and clean the node-level dataset so that:
      - rows are ordered by (instance_id, visit_idx) when possible
      - id columns have consistent types (when present)
    """
    df = full_df.copy()

    # Ensure instance_id is int (if present)
    if "instance_id" in df.columns:
        df["instance_id"] = df["instance_id"].astype(int)

    # node_id as int (if present)
    if "node_id" in df.columns:
        df["node_id"] = df["node_id"].astype(int)

    # parent_id: allow NA for root (if present)
    if "parent_id" in df.columns:
        df["parent_id"] = df["parent_id"].astype("Int64")

    # Sort by instance then visit order (if visit_idx exists)
    sort_cols = [c for c in ["instance_id", "visit_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def build_xy_vectors(full_df: pd.DataFrame, strict: bool = False, verbose: bool = True):
    """
    Build feature blocks and target.

    Option A behavior: only keep columns that exist in full_df.
    If strict=True, raise on missing expected columns. Otherwise drop and warn.
    """
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

    def _select(cols, block_name):
        existing = [c for c in cols if c in full_df.columns]
        missing = [c for c in cols if c not in full_df.columns]

        if missing:
            msg = f"[{block_name}] missing cols dropped ({len(missing)}): {missing}"
            if strict:
                raise KeyError(msg)
            if verbose:
                print(msg)

        if not existing and verbose:
            print(f"[{block_name}] WARNING: no columns available from requested list.")

        return full_df[existing].copy()

    X_node = _select(node_cols, "node")
    X_path = _select(path_cols, "path")
    X_global = _select(global_cols, "global")

    # required label + instance id
    required = ["has_optimal_subtree", "instance_id"]
    missing_req = [c for c in required if c not in full_df.columns]
    if missing_req:
        raise KeyError(f"Missing required columns: {missing_req}")

    y = full_df["has_optimal_subtree"].astype(int).to_numpy()
    instance_ids = full_df["instance_id"].to_numpy()

    return X_node, X_path, X_global, y, instance_ids


def train_test_split_by_instance(
    X: pd.DataFrame,
    y: np.ndarray,
    instance_ids: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
):
    """
    Split by instance_id so all nodes from the same instance are in the same split.
    NOTE: Uses .loc[...] for row masking.
    """
    unique_instances = np.unique(instance_ids)

    train_inst, test_inst = train_test_split(
        unique_instances,
        test_size=test_size,
        random_state=random_state,
    )

    train_mask = np.isin(instance_ids, train_inst)
    test_mask = np.isin(instance_ids, test_inst)

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test


def create_logreg_pipeline(X_example: pd.DataFrame) -> Pipeline:
    """
    Create a sklearn pipeline that:
      - imputes numeric features with mean, standardizes them
      - imputes categorical features with most_frequent, one-hot encodes them
      - fits a class-weighted logistic regression
    """
    numeric_features = X_example.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_example.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    # IMPORTANT: bools can be treated as categorical or numeric.
    # We include bool in categorical_features to one-hot encode True/False robustly.
    # If you'd rather treat bool as numeric, remove "bool" above and add:
    # X_example = X_example.copy(); X_example[bool_cols] = X_example[bool_cols].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = Pipeline([
        ("pre", pre),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
        )),
    ])
    return clf


def train_logreg(
    full_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 0,
    strict: bool = False,
    verbose: bool = True,
):
    """
    Complete pipeline:
      - orient dataset by (instance_id, visit_idx) when possible
      - build feature blocks and labels (drops missing expected cols unless strict=True)
      - concatenate feature blocks into a single X
      - split train/test by instance_id
      - fit class-weighted logistic regression pipeline (numeric + categorical handled)
      - return fitted model, metrics, and train/test split objects
    """
    # 1) Orient dataset
    df = orient_dataset_by_instance(full_df)

    # 2) Build feature blocks + y + instance_ids
    X_node, X_path, X_global, y, instance_ids = build_xy_vectors(df, strict=strict, verbose=verbose)

    # 3) Concatenate feature blocks for modeling
    X = pd.concat([X_node, X_path, X_global], axis=1)

    if verbose:
        print("Final X shape:", X.shape)
        print("Numeric cols:", X.select_dtypes(include=[np.number]).columns.tolist())
        print("Categorical cols:", X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist())

    if X.shape[1] == 0:
        raise ValueError("No features available after dropping missing columns. Check your schema/feature lists.")

    # 4) Train/test split by instance
    X_train, X_test, y_train, y_test = train_test_split_by_instance(
        X, y, instance_ids, test_size=test_size, random_state=random_state
    )

    # 5) Create and fit pipeline (needs an example df to detect dtypes)
    clf = create_logreg_pipeline(X_train)
    clf.fit(X_train, y_train)

    # 6) Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # ROC AUC undefined if only one class in y_test
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan

    metrics = {
        "roc_auc": roc,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return clf, metrics, (X_train, X_test, y_train, y_test)


def get_feature_names(clf: Pipeline, X_example: pd.DataFrame):
    """
    Utility: returns the expanded feature names after preprocessing (numeric + OHE).
    Useful for interpreting coefficients.

    Works for sklearn >= 1.0. If it errors, you're on an older sklearn.
    """
    pre = clf.named_steps["pre"]

    numeric_features = X_example.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_example.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    names = []

    # Numeric names passthrough (after scaler)
    names.extend(numeric_features)

    # Categorical one-hot names
    if categorical_features:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
        names.extend(ohe_names)

    return np.array(names)


def inspect_logreg_coefficients(clf: Pipeline, X_example: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """
    Print/return sorted coefficient table (by absolute value).
    For Fix 2, you MUST pass an X_example with the same schema as training data,
    so we can recover the expanded feature names.
    """
    logreg = clf.named_steps["logreg"]
    coefs = logreg.coef_.ravel()

    feat_names = get_feature_names(clf, X_example)

    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    if top_k is not None:
        coef_df = coef_df.head(top_k)

    print(coef_df)
    return coef_df


def prediction_distribution(clf: Pipeline, X: pd.DataFrame):
    probs = clf.predict_proba(X)[:, 1]
    preds = clf.predict(X)
    print("Proportion predicted 1:", preds.mean())
    return probs, preds


# ---------------------------
# Example usage:
#
# from logreg_functions import train_logreg, inspect_logreg_coefficients
#
# clf, metrics, (X_train, X_test, y_train, y_test) = train_logreg(mconstr_df, verbose=True)
# print(metrics)
# inspect_logreg_coefficients(clf, X_train, top_k=30)
# ---------------------------
