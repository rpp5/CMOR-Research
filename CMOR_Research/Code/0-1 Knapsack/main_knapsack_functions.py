import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum

# ============================================================
# OVERARCHING: Generic 0–1 knapsack core (all problem types)
# ============================================================

@dataclass
class KnapsackInstance:
    """
    Generic 0–1 knapsack (possibly m-constraint).

    Attributes
    ----------
    W : np.ndarray, shape (m, n)
        Weight matrix. Each row corresponds to one knapsack constraint.
    values : np.ndarray, shape (n,)
        Item values.
    capacities : np.ndarray, shape (m,)
        Capacity for each constraint.
    """
    W: np.ndarray
    values: np.ndarray
    capacities: np.ndarray

    @property
    def m(self) -> int:
        return int(self.W.shape[0])

    @property
    def n(self) -> int:
        return int(self.W.shape[1])


def solve_knapsack(instance: KnapsackInstance) -> Tuple[Dict[int, int], float]:
    """
    Solve the integer 0–1 knapsack problem (possibly m-constraint).

    Returns
    -------
    sol : dict[int, int]
        Integer solution x_j in {0,1}.
    total_value : float
        Objective value of the integer solution.
    """
    W, v, caps = instance.W, instance.values, instance.capacities
    m, n = instance.m, instance.n

    model = Model("Knapsack_Integer")
    model.hideOutput()

    x = {j: model.addVar(vtype="B", name=f"x_{j}") for j in range(n)}

    model.setObjective(quicksum(v[j] * x[j] for j in range(n)), "maximize")

    for k in range(m):
        model.addCons(quicksum(W[k, j] * x[j] for j in range(n)) <= caps[k])

    model.optimize()

    sol = {j: int(round(model.getVal(x[j]))) for j in range(n)}
    total_value = sum(v[j] * sol[j] for j in range(n))
    return sol, float(total_value)


def solve_knapsack_lp_relaxation(
    instance: KnapsackInstance,
) -> Tuple[Optional[Dict[int, float]], Optional[float]]:
    """
    Solve the LP relaxation at the root (no additional constraints).

    Returns
    -------
    sol : dict[int, float] or None
        Fractional solution in [0,1] if feasible, else None.
    total_value : float or None
        Objective value, or None if infeasible.
    """
    return solve_model_w_constraints(instance, constraints={})[:2]


def solve_model_w_constraints(
    instance: KnapsackInstance,
    constraints: Dict[int, int],
) -> Tuple[Optional[Dict[int, float]], Optional[float], Optional[np.ndarray]]:
    """
    Solve the LP relaxation of the knapsack problem with
    additional fixed-variable constraints used for branch-and-bound nodes.

    Parameters
    ----------
    instance : KnapsackInstance
        The knapsack instance.
    constraints : dict[int, int]
        Fixed-value constraints of the form {j: 0 or 1} saying x_j == value.

    Returns
    -------
    sol : dict[int, float] or None
        Solution of the LP relaxation at this node, or None if infeasible.
    total_value : float or None
        LP objective value, or None if infeasible.
    usage : np.ndarray or None, shape (m,)
        For each constraint k, the used capacity sum_j W[k,j] x_j, or None if infeasible.
    """
    W, v, caps = instance.W, instance.values, instance.capacities
    m, n = instance.m, instance.n

    model = Model("Knapsack_LP_Node")
    model.hideOutput()

    x = {j: model.addVar(lb=0.0, ub=1.0, vtype="C", name=f"x_{j}") for j in range(n)}

    model.setObjective(quicksum(v[j] * x[j] for j in range(n)), "maximize")

    for k in range(m):
        model.addCons(quicksum(W[k, j] * x[j] for j in range(n)) <= caps[k])

    # Apply node fixings
    for j, val in constraints.items():
        model.addCons(x[j] == val)

    model.optimize()

    status = model.getStatus()
    if status not in ("optimal", "solutionlimit"):
        return None, None, None

    sol = {j: model.getVal(x[j]) for j in range(n)}
    total_value = sum(v[j] * sol[j] for j in range(n))

    usage = np.zeros(m, dtype=float)
    for k in range(m):
        usage[k] = sum(W[k, j] * sol[j] for j in range(n))

    return sol, float(total_value), usage


def find_optimal_value(instance: KnapsackInstance) -> float:
    """
    Compute the optimal integer objective value by solving the full MIP.
    """
    _, best_val = solve_knapsack(instance)
    return best_val


def find_optimal_value_with_dataset(
    instance: KnapsackInstance,
    branching_tol: float = 1e-6,
) -> Tuple[float, List[Dict]]:
    """
    Run a simple branch-and-bound using LP relaxations and collect a dataset
    of node-level features.

    Parameters
    ----------
    instance : KnapsackInstance
    branching_tol : float
        Tolerance used to decide whether a variable is fractional.

    Returns
    -------
    best_val : float
        Optimal integer objective value found.
    dataset : list[dict]
        One record per explored node containing:
          - node_id, parent_id, depth
          - constraints (fixed variables)
          - visit_idx, open_nodes_count
          - branch_parent_var, branch_parent_dir, branch_parent_value
          - lp_objective
          - num_frac, branch_var
          - incumbent_at_visit
          - status ("fractional", "integer", "infeasible", "pruned")
    """
    n = instance.n

    # node stack
    A: List[Dict] = []

    # root node
    next_node_id = 0
    root = {
        "node_id": next_node_id,
        "parent_id": None,
        "depth": 0,
        "constraints": {},
        "branch_parent_var": None,
        "branch_parent_dir": None,
        "branch_parent_value": None,
    }
    A.append(root)
    next_node_id += 1

    dataset: List[Dict] = []
    incumbent_val = -float("inf")
    visit_idx = 0

    while A:
        node = A.pop()  # DFS
        constraints = node["constraints"]
        depth = node["depth"]
        node_id = node["node_id"]
        parent_id = node["parent_id"]

        open_nodes_count = len(A)
        incumbent_at_visit = incumbent_val

        sol, lp_obj, usage = solve_model_w_constraints(instance, constraints)

        base_row: Dict = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "constraints": constraints.copy(),
            "visit_idx": visit_idx,
            "open_nodes_count": open_nodes_count,
            "branch_parent_var": node.get("branch_parent_var"),
            "branch_parent_dir": node.get("branch_parent_dir"),
            "branch_parent_value": node.get("branch_parent_value"),
            "lp_objective": None,
            "num_frac": None,
            "branch_var": None,
            "incumbent_at_visit": incumbent_at_visit,
            "status": None,
        }

        if sol is None or lp_obj is None:
            base_row["status"] = "infeasible"
            dataset.append(base_row)
            visit_idx += 1
            continue

        # record LP objective
        base_row["lp_objective"] = lp_obj

        # determine fractionality
        xs = np.array([sol[j] for j in range(n)], dtype=float)
        frac_mask = (xs > branching_tol) & (xs < 1.0 - branching_tol)
        num_frac = int(frac_mask.sum())
        base_row["num_frac"] = num_frac

        if num_frac == 0:
            # integer solution at this node
            base_row["status"] = "integer"
            dataset.append(base_row)
            visit_idx += 1

            if lp_obj > incumbent_val:
                incumbent_val = lp_obj
            continue

        # bound pruning
        if lp_obj <= incumbent_val + 1e-9:
            base_row["status"] = "pruned"
            dataset.append(base_row)
            visit_idx += 1
            continue

        # choose branching variable: closest to 0.5
        frac_indices = np.where(frac_mask)[0]
        center_distance = np.abs(xs[frac_indices] - 0.5)
        best_idx = int(frac_indices[np.argmin(center_distance)])
        base_row["branch_var"] = best_idx
        base_row["status"] = "fractional"
        dataset.append(base_row)
        visit_idx += 1

        # create child nodes
        for direction, val in (("down", 0), ("up", 1)):
            new_constraints = constraints.copy()
            new_constraints[best_idx] = val
            child = {
                "node_id": next_node_id,
                "parent_id": node_id,
                "depth": depth + 1,
                "constraints": new_constraints,
                "branch_parent_var": best_idx,
                "branch_parent_dir": direction,
                "branch_parent_value": float(xs[best_idx]),
            }
            next_node_id += 1
            A.append(child)

    best_val = incumbent_val
    return float(best_val), dataset


def build_node_dataframe(dataset: List[Dict], num_vars: int) -> pd.DataFrame:
    """
    Convert the list of node dictionaries into a wide pandas DataFrame.

    Parameters
    ----------
    dataset : list[dict]
        Records produced by `find_optimal_value_with_dataset`.
    num_vars : int
        Number of decision variables (items).

    Returns
    -------
    df : pandas.DataFrame
    """
    scalar_keys = [
        "node_id",
        "parent_id",
        "depth",
        "visit_idx",
        "open_nodes_count",
        "branch_parent_var",
        "branch_parent_dir",
        "branch_parent_value",
        "lp_objective",
        "num_frac",
        "branch_var",
        "incumbent_at_visit",
        "status",
        "has_optimal_subtree",
    ]

    rows: List[Dict] = []

    for record in dataset:
        row: Dict = {}
        for key in scalar_keys:
            row[key] = record.get(key, None)

        # variable-fixing columns x0, x1, ..., x_{n-1}
        for j in range(num_vars):
            row[f"x{j}"] = np.nan

        constraints = record.get("constraints", {})
        for var_idx, val in constraints.items():
            if 0 <= var_idx < num_vars:
                row[f"x{var_idx}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_optimal_subtree_flags(
    dataset: List[Dict],
    best_val: float,
    tol: float = 1e-6,
) -> List[Dict]:
    """
    For each node in the dataset, determine whether its subtree
    contains at least one optimal integer solution.

    This updates each record in-place by adding:
        record["has_optimal_subtree"] = bool
    and also returns the updated list.
    """
    from collections import defaultdict

    children = defaultdict(list)
    node_map: Dict[int, Dict] = {}

    for rec in dataset:
        nid = rec["node_id"]
        pid = rec.get("parent_id")
        node_map[nid] = rec
        if pid is not None:
            children[pid].append(nid)

    def dfs(nid: int) -> bool:
        node = node_map[nid]
        lp_obj = node.get("lp_objective")
        status = node.get("status")

        is_opt_integer = (
            status == "integer"
            and lp_obj is not None
            and abs(lp_obj - best_val) <= tol
        )

        has_opt = is_opt_integer
        for cid in children.get(nid, []):
            if dfs(cid):
                has_opt = True

        node["has_optimal_subtree"] = has_opt
        return has_opt

    roots = [rec["node_id"] for rec in dataset if rec.get("parent_id") is None]
    for r in roots:
        dfs(r)

    return dataset


def build_training_instance(
    instance_id: int,
    instance_generator,
    num_items: int = 5,
    seed: Optional[int] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Create one knapsack instance, run B&B, and return a DataFrame of node features.
    """
    instance: KnapsackInstance = instance_generator(
        num_items=num_items, seed=seed, **gen_kwargs
    )

    best_val, dataset = find_optimal_value_with_dataset(instance)
    dataset = compute_optimal_subtree_flags(dataset, best_val)

    df_nodes = build_node_dataframe(dataset, num_vars=instance.n)
    df_nodes["instance_id"] = instance_id
    df_nodes["num_items"] = instance.n
    df_nodes["m_constraints"] = instance.m

    return df_nodes


def build_training_set(
    num_instances: int,
    num_items: Optional[int] = None,
    instance_generator=None,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Build a training set by sampling multiple instances of a given type.
    """
    if instance_generator is None:
        instance_generator = generate_knapsack_1constraint

    all_dfs: List[pd.DataFrame] = []

    for i in range(num_instances):
        if num_items is None:
            n_i = int(np.random.randint(5, 9))
        else:
            n_i = num_items
        df_i = build_training_instance(
            instance_id=i,
            instance_generator=instance_generator,
            num_items=n_i,
            seed=i,
            **gen_kwargs,
        )
        all_dfs.append(df_i)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


# ============================================================
# PROBLEM-TYPE SPECIFIC GENERATORS
# ============================================================

def generate_knapsack_1constraint(
    num_items: int = 5,
    max_weight: int = 20,
    max_value: int = 50,
    seed: Optional[int] = None,
) -> KnapsackInstance:
    """
    Standard single-constraint 0–1 knapsack instance.

    Returns
    -------
    KnapsackInstance with m = 1.
    """
    rng = np.random.default_rng(seed)

    weights = rng.integers(1, max_weight + 1, size=num_items)
    values = rng.integers(1, max_value + 1, size=num_items)

    total_weight = weights.sum()
    capacity = rng.integers(int(0.4 * total_weight), int(0.6 * total_weight) + 1)

    W = weights.reshape(1, -1)
    caps = np.array([capacity], dtype=float)
    return KnapsackInstance(W=W.astype(float), values=values.astype(float), capacities=caps)


def generate_knapsack_mconstraint(
    num_items: int = 5,
    m: int = 2,
    max_weight: int = 20,
    max_value: int = 50,
    seed: Optional[int] = None,
) -> KnapsackInstance:
    """
    Multi-constraint (m-dimensional) 0–1 knapsack instance.
    """
    rng = np.random.default_rng(seed)

    W = rng.integers(1, max_weight + 1, size=(m, num_items))
    values = rng.integers(1, max_value + 1, size=num_items)

    capacities = []
    for k in range(m):
        row_sum = W[k].sum()
        cap_k = rng.integers(int(0.4 * row_sum), int(0.6 * row_sum) + 1)
        capacities.append(cap_k)
    caps = np.array(capacities, dtype=float)

    return KnapsackInstance(W=W.astype(float), values=values.astype(float), capacities=caps)


def generate_knapsack_correlated(
    num_items: int = 5,
    max_weight: int = 20,
    alpha: float = 10.0,
    noise_std: float = 2.0,
    seed: Optional[int] = None,
) -> KnapsackInstance:
    """
    Example 'correlated' single-constraint knapsack where
    values are positively correlated with weights.

    v_j = alpha * w_j + epsilon_j, epsilon_j ~ N(0, noise_std^2)

    Returns
    -------
    KnapsackInstance with m = 1.
    """
    rng = np.random.default_rng(seed)

    weights = rng.integers(1, max_weight + 1, size=num_items)
    noise = rng.normal(loc=0.0, scale=noise_std, size=num_items)
    values = alpha * weights + noise
    values = np.maximum(values, 1.0)  # avoid non-positive values

    total_weight = weights.sum()
    capacity = rng.integers(int(0.4 * total_weight), int(0.6 * total_weight) + 1)

    W = weights.reshape(1, -1).astype(float)
    caps = np.array([capacity], dtype=float)

    return KnapsackInstance(W=W, values=values.astype(float), capacities=caps)


# Backwards-compatible alias if you were calling `generate_knapsack_instance`
def generate_knapsack_instance(
    num_items: int = 5,
    max_weight: int = 20,
    max_value: int = 50,
    seed: Optional[int] = None,
) -> KnapsackInstance:
    """
    Backwards-compatible wrapper for the standard 1-constraint generator.
    Note: This now returns a KnapsackInstance instead of (weights, values, capacity).
    """
    return generate_knapsack_1constraint(
        num_items=num_items,
        max_weight=max_weight,
        max_value=max_value,
        seed=seed,
    )
