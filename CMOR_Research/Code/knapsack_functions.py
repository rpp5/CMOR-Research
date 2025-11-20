import random
import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum

# -----------------------------
# 1. Generate Random Instance
# -----------------------------
def generate_knapsack_instance(num_items=5, max_weight=20, max_value=50, seed=40):

    """
    Generate a random 0–1 knapsack instance.

    Args:
        num_items (int): Number of items to generate.
        max_weight (int): Maximum possible weight per item.
        max_value (int): Maximum possible value per item.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            weights (list[int]): List of item weights.
            values (list[int]): List of item values.
            capacity (int): Knapsack capacity set to 60% of total weight.
    """

    random.seed(seed)
    weights = [random.randint(1, max_weight) for _ in range(num_items)]
    values = [random.randint(1, max_value) for _ in range(num_items)]
    capacity = int(sum(weights) * 0.6)  # set capacity to 60% of total weight
    return weights, values, capacity

def solve_knapsack(weights, values, capacity):

    """
    Solve the 0 - 1 knapsack problem using binary variables.

    Args:
        weights (list[int]): Item weights.
        values (list[int]): Item values.
        capacity (int): Maximum weight capacity of the knapsack.

    Returns:
        tuple:
            solution (dict[int, float]): Optimal x[i] values (0 or 1).
            total_value (float): Total value of selected items.
            total_weight (float): Total weight of selected items.
    """

    n = len(weights)
    model = Model("0-1_Knapsack")
    model.hideOutput()

    # Decision variables: x[i] = 1 if item i is chosen
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(n)}

    # Objective: maximize value
    model.setObjective(quicksum(values[i] * x[i] for i in range(n)), "maximize")

    # Weight constraint
    model.addCons(quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Solve
    model.optimize()

    # Extract solution
    solution = {i: model.getVal(x[i]) for i in range(n)}
    total_value = sum(values[i] * solution[i] for i in range(n))
    total_weight = sum(weights[i] * solution[i] for i in range(n))
    
    return solution, total_value, total_weight

def solve_knapsack_lp_relaxation(weights, values, capacity):

    """
    Solve the LP relaxation of the knapsack problem 
    (x[i] allowed to be fractional).

    Args:
        weights (list[int]): Item weights.
        values (list[int]): Item values.
        capacity (int): Maximum weight capacity.

    Returns:
        tuple:
            sol (dict[int, float]): Fractional LP solution for x[i].
            total_value (float): LP objective value.
            total_weight (float): Total fractional weight used.
    """

    n = len(weights)
    model = Model("Knapsack_LP_Relax")
    model.hideOutput()

    # LP variables: 0 <= x[i] <= 1 (continuous)
    x = {i: model.addVar(lb=0.0, ub=1.0, vtype="C", name=f"x_{i}") for i in range(n)}

    # Maximize total value
    model.setObjective(quicksum(values[i] * x[i] for i in range(n)), "maximize")

    # Capacity constraint
    model.addCons(quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Solve
    model.optimize()

    # Extract (possibly fractional) solution
    sol = {i: model.getVal(x[i]) for i in range(n)}
    total_value = sum(values[i] * sol[i] for i in range(n))
    total_weight = sum(weights[i] * sol[i] for i in range(n))
    return sol, total_value, total_weight

def solve_model_w_constraints(weights, values, capacity, constraints):

    """
    Solve the LP relaxation of the knapsack problem with 
    additional fixed-variable constraints used for branch-and-bound nodes.

    Args:
        weights (list[int]): Item weights.
        values (list[int]): Item values.
        capacity (int): Knapsack capacity.
        constraints (dict[int, int]): Fixed-value constraints of the form 
                                      {var_index: 0 or 1}.

    Returns:
        tuple:
            sol (dict[int, float] or None): LP solution, or None if infeasible.
            total_value (float or None): LP objective value (None if infeasible).
            total_weight (float or None): Weight of fractional solution.
    """
    
    n = len(weights)
    model = Model("Knapsack_LP_Relax")
    model.hideOutput()

    # LP variables: 0 <= x[i] <= 1 (continuous)
    x = {i: model.addVar(lb=0.0, ub=1.0, vtype="C", name=f"x_{i}") for i in range(n)}

    # Maximize total value
    model.setObjective(quicksum(values[i] * x[i] for i in range(n)), "maximize")

    # Capacity constraint
    model.addCons(quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    #Additional constraints:
    for var, constraint_val in constraints.items():
        model.addCons(x[var] == constraint_val)

    # Solve
    model.optimize()
    
    status = model.getStatus()
    
    if status == "optimal":
        # Extract (possibly fractional) solution
        sol = {i: model.getVal(x[i]) for i in range(n)}
        total_value = sum(values[i] * sol[i] for i in range(n))
        total_weight = sum(weights[i] * sol[i] for i in range(n))
        return sol, total_value, total_weight
    else:
        return None, None, None

def find_optimal_value(weights, values, capacity):

    """
    Compute the optimal integer knapsack objective value using 
    a basic branch-and-bound procedure (no bounding, full enumeration).

    Args:
        weights (list[int]): Item weights.
        values (list[int]): Item values.
        capacity (int): Knapsack capacity.

    Returns:
        float: Optimal integer objective value.
    """

    V = []
    A = []
    A.append({}) #add root node
    while len(A) > 0:
        node = A.pop() #node is just the constraints of that node
        sol, total_value, total_weight = solve_model_w_constraints(weights, values, capacity, node) #solve LP relaxation of node
        if sol == None: #pop next node if this one is infeasible
            continue
        elif all(float(sol_val).is_integer() for sol_val in sol.values()): #if integer feasible
                V.append(total_value)
        else: #if fractional relaxation
            
            #identify which var to branch on
            fractional_keys = [key for key, v in sol.items() if abs(v - round(v)) > 1e-9]
            branch_var = fractional_keys[0]
            
            #add two new nodes to A
            node1 = node.copy()
            node2 = node.copy()
            node1[branch_var] = 1
            node2[branch_var] = 0
            A.append(node1)
            A.append(node2)
    
    print(f"{max(V)} IS THE OPTIMAL VALUE ")
    return max(V)
         
def find_optimal_value_with_dataset(weights, values, capacity):
    """
    Branch-and-bound for 0-1 knapsack, logging node-level data.

    Returns:
        best_val (float or None): best integer objective found.
        dataset (list[dict]): one record per processed node, each containing
            both search/LP info and online features that could be used
            for a logistic model.
    """
    n = len(weights)

    V = []          # store integer-feasible objective values
    A = []          # stack of active nodes

    dataset = []    # list of per-node records (dicts)
    next_node_id = 0
    visit_idx = 0   # global node-visit counter (online search order)
    incumbent = None  # best integer solution found so far (online incumbent)

    # root node
    root = {
        "constraints": {},      # fixed x[i] assignments at this node
        "depth": 0,
        "parent_id": None,
        "node_id": next_node_id,
        # branching info for how we got here (root has none)
        "branch_parent_var": None,
        "branch_parent_dir": None,
        "branch_parent_weight": None,
        "branch_parent_value": None,
        "branch_parent_ratio": None,
    }
    next_node_id += 1
    A.append(root)

    while len(A) > 0:
        node = A.pop()

        constraints = node["constraints"]
        depth = node["depth"]
        parent_id = node["parent_id"]
        node_id = node["node_id"]

        # --- ONLINE SEARCH-STATE FEATURES ---
        open_nodes_count = len(A)     # how many nodes are currently waiting
        cur_incumbent = incumbent     # incumbent at the time we visit this node

        # solve LP relaxation at this node
        sol, total_value, total_weight = solve_model_w_constraints(
            weights, values, capacity, constraints
        )

        # base row for this node
        row = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "constraints": constraints.copy(),   # fixed variables at this node

            # NEW: online search-order index (0,1,2,...) in which we processed nodes
            "visit_idx": visit_idx,

            # NEW: number of open nodes at the moment we popped this node
            "open_nodes_count": open_nodes_count,

            # NEW: how we got here from the parent branch
            "branch_parent_var": node.get("branch_parent_var", None),
            "branch_parent_dir": node.get("branch_parent_dir", None),
            "branch_parent_weight": node.get("branch_parent_weight", None),
            "branch_parent_value": node.get("branch_parent_value", None),
            "branch_parent_ratio": node.get("branch_parent_ratio", None),
        }
        visit_idx += 1

        # --- INFEASIBLE NODE CASE ---
        if sol is None:
            row["status"] = "infeasible"
            row["objective"] = None
            row["lp_objective"] = None

            # NEW: incumbent info available at this node
            row["incumbent_val"] = cur_incumbent       # best integer value so far
            row["has_incumbent"] = int(cur_incumbent is not None)
            row["node_gap"] = None                     # no LP bound, so no gap
            row["node_rel_gap"] = None

            # NEW: fixings summary at this node (known even if LP infeasible)
            num_fixed = len(constraints)
            num_fixed_1 = sum(v == 1 for v in constraints.values())
            num_fixed_0 = sum(v == 0 for v in constraints.values())
            row["num_fixed"] = num_fixed
            row["num_fixed_1"] = num_fixed_1
            row["num_fixed_0"] = num_fixed_0
            row["fix_ratio"] = num_fixed / n if n > 0 else 0.0

            # no LP-based features for infeasible nodes
            row["slack"] = None
            row["total_weight_fixed_1"] = sum(
                weights[i] for i, v in constraints.items() if v == 1
            )
            row["total_value_fixed_1"] = sum(
                values[i] for i, v in constraints.items() if v == 1
            )

            # no fractional-variable features for infeasible nodes
            row["frac_var_index"] = None
            row["frac_var_normalized"] = None
            row["frac_var_weight"] = None
            row["frac_var_value"] = None
            row["frac_var_ratio"] = None
            row["frac_var_fraction"] = None

            dataset.append(row)
            continue

        # --- FEASIBLE LP RELAXATION ---
        row["lp_objective"] = total_value

        # NEW: LP slack – how much capacity is left under the LP solution
        # (tightness of the capacity constraint, computed online)
        slack = capacity - total_weight
        row["slack"] = slack

        # NEW: fixings features – fully known from the constraint dict
        num_fixed = len(constraints)
        num_fixed_1 = sum(v == 1 for v in constraints.values())
        num_fixed_0 = sum(v == 0 for v in constraints.values())
        row["num_fixed"] = num_fixed
        row["num_fixed_1"] = num_fixed_1
        row["num_fixed_0"] = num_fixed_0
        row["fix_ratio"] = num_fixed / n if n > 0 else 0.0

        # NEW: cumulative weight/value of items already fixed to 1 at this node
        total_weight_fixed_1 = sum(
            weights[i] for i, v in constraints.items() if v == 1
        )
        total_value_fixed_1 = sum(
            values[i] for i, v in constraints.items() if v == 1
        )
        row["total_weight_fixed_1"] = total_weight_fixed_1
        row["total_value_fixed_1"] = total_value_fixed_1

        # NEW: incumbent-based features (all online)
        row["incumbent_val"] = cur_incumbent
        row["has_incumbent"] = int(cur_incumbent is not None)
        if cur_incumbent is not None:
            # Gap between incumbent and this node's LP objective
            node_gap = cur_incumbent - total_value
            row["node_gap"] = node_gap
            row["node_rel_gap"] = node_gap / (abs(cur_incumbent) + 1e-9)
        else:
            row["node_gap"] = None
            row["node_rel_gap"] = None

        # --- FRACTIONALITY & BRANCHING ---
        # list of fractional vars (in knapsack LP, this is usually 0 or 1 variable)
        fractional_keys = [
            key for key, v in sol.items()
            if abs(v - round(v)) > 1e-9
        ]

        # NEW: fractional-variable features (only if there is a fractional var)
        if fractional_keys:
            frac_var = fractional_keys[0]
            row["frac_var_index"] = frac_var
            row["frac_var_normalized"] = frac_var / n
            row["frac_var_weight"] = weights[frac_var]
            row["frac_var_value"] = values[frac_var]
            row["frac_var_ratio"] = values[frac_var] / weights[frac_var]
            row["frac_var_fraction"] = sol[frac_var]
        else:
            row["frac_var_index"] = None
            row["frac_var_normalized"] = None
            row["frac_var_weight"] = None
            row["frac_var_value"] = None
            row["frac_var_ratio"] = None
            row["frac_var_fraction"] = None

        # integrality check (online, from this node's LP solution)
        is_integer = (len(fractional_keys) == 0)

        if is_integer:
            # Case: integer-feasible node
            row["status"] = "integer"
            row["objective"] = total_value
            dataset.append(row)

            V.append(total_value)

            # update incumbent AFTER logging the row (so features use previous incumbent)
            if (incumbent is None) or (total_value > incumbent):
                incumbent = total_value

        else:
            # Case: fractional node → branch on fractional variable
            row["status"] = "fractional"
            row["objective"] = None
            dataset.append(row)

            branch_var = fractional_keys[0]

            # add two children (x[branch_var] = 1 and x[branch_var] = 0)
            for fixed_value in [1, 0]:
                new_constraints = constraints.copy()
                new_constraints[branch_var] = fixed_value

                child = {
                    "constraints": new_constraints,
                    "depth": depth + 1,
                    "parent_id": node_id,
                    "node_id": next_node_id,

                    # NEW: record how the parent branched to create this child
                    "branch_parent_var": branch_var,
                    "branch_parent_dir": fixed_value,
                    "branch_parent_weight": weights[branch_var],
                    "branch_parent_value": values[branch_var],
                    "branch_parent_ratio": values[branch_var] / weights[branch_var],
                }
                next_node_id += 1

                A.append(child)

    # sanity: handle case where no integer solution was found
    if not V:
        print("No integer-feasible solution found.")
        best_val = None
    else:
        best_val = max(V)
        print(f"{best_val} IS THE OPTIMAL VALUE")

    return best_val, dataset

    
def build_node_dataframe(dataset, num_vars):
    """
    Convert the branch-and-bound node records into a structured pandas DataFrame.

    Each element of `dataset` is a dict produced by `find_optimal_value_with_dataset`,
    containing both search-tree information (e.g. depth, parent_id, status) and
    online features (e.g. visit index, slack, incumbent gap, fractional-variable
    characteristics).

    Args:
        dataset (list[dict]): Node records returned by `find_optimal_value_with_dataset`.
        num_vars (int): Number of decision variables in the knapsack model,
                        used to create one column per variable: x0, x1, ..., x{num_vars-1}.

    Returns:
        pandas.DataFrame:
            A table where each row corresponds to a node and includes:
                - Basic structure: node_id, parent_id, depth, status, lp_objective
                - Online search-state features: visit_idx, open_nodes_count
                - Branching info: branch_parent_var, branch_parent_dir, ...
                - Fixing summaries: num_fixed, num_fixed_1, num_fixed_0, fix_ratio,
                  total_weight_fixed_1, total_value_fixed_1
                - Incumbent / bound info: incumbent_val, has_incumbent, node_gap, node_rel_gap
                - Fractional-variable features: frac_var_index, frac_var_weight, etc.
                - Variable fixings: x0, x1, ..., x{num_vars-1}
    """
    rows = []

    # list of scalar/node-level fields we want to carry over directly
    scalar_keys = [
        "node_id",
        "parent_id",
        "depth",
        "status",
        "lp_objective",
        "objective",
        "visit_idx",
        "open_nodes_count",
        "branch_parent_var",
        "branch_parent_dir",
        "branch_parent_weight",
        "branch_parent_value",
        "branch_parent_ratio",
        "slack",
        "num_fixed",
        "num_fixed_1",
        "num_fixed_0",
        "fix_ratio",
        "total_weight_fixed_1",
        "total_value_fixed_1",
        "incumbent_val",
        "has_incumbent",
        "node_gap",
        "node_rel_gap",
        "frac_var_index",
        "frac_var_normalized",
        "frac_var_weight",
        "frac_var_value",
        "frac_var_ratio",
        "frac_var_fraction",
    ]

    for record in dataset:
        row = {}

        # Copy scalar features safely (default to NaN/None if missing)
        for key in scalar_keys:
            # use np.nan for numeric-like missing fields, None for others is also ok;
            # here we'll just default to None and let pandas handle it
            row[key] = record.get(key, None)

        # Initialize all variable columns to NaN (these represent fixed values of x_j)
        for j in range(num_vars):
            row[f"x{j}"] = np.nan

        # Fill in constraints for this node
        # constraints is something like {1: 0, 2: 1}
        constraints = record.get("constraints", {})
        for var_idx, val in constraints.items():
            row[f"x{var_idx}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def compute_optimal_subtree_flags(dataset, best_val):
    
    """
    Determine which branch-and-bound nodes lie on (or above) 
    a subtree containing at least one optimal integer solution.

    Args:
        dataset (list[dict]): Node records from branch-and-bound.
        best_val (float): Optimal integer objective value.

    Returns:
        dict[int, bool]:
            Mapping from node_id -> True/False indicating whether that node's 
            subtree contains an optimal integer solution.
    """

    # Build a parent map: node_id -> parent_id
    parent = {rec["node_id"]: rec["parent_id"] for rec in dataset}

    # Identify which nodes are optimal integer leaves
    is_opt_leaf = {
        rec["node_id"]: (
            rec.get("status") == "integer" and rec.get("objective") == best_val
        )
        for rec in dataset
    }

    # Initialize all flags to False
    has_optimal_in_subtree = {nid: False for nid in parent.keys()}

    # For each optimal leaf, mark it and all its ancestors
    for nid, is_leaf in is_opt_leaf.items():
        if not is_leaf:
            continue

        cur = nid
        # Walk up until root (parent_id = None) or already marked
        while cur is not None and not has_optimal_in_subtree[cur]:
            has_optimal_in_subtree[cur] = True
            cur = parent[cur]

    return has_optimal_in_subtree

def build_training_instance(instance_id, num_items=5, seed=None):

    """
    Runs one knapsack instance + branch-and-bound
    and returns a node dataframe with labels.

    Args:
        instance_id (int): Identifier for this knapsack instance.
        num_items (int): Number of knapsack items.
        seed (int or None): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: node_df with label column and metadata.
    """
    # 1) Create instance
    if seed is None:
        seed = np.random.randint(1_000_000)

    weights, values, capacity = generate_knapsack_instance(
        num_items=num_items,
        seed=seed
    )

    # 2) Run branch-and-bound + collect dataset
    best_val, dataset = find_optimal_value_with_dataset(weights, values, capacity)

    # 3) Build node dataframe
    df_nodes = build_node_dataframe(dataset, num_vars=len(weights))

    # 4) Compute subtree label (1 if subtree contains an optimal solution)
    flags = compute_optimal_subtree_flags(dataset, best_val)
    df_nodes["has_optimal_subtree"] = df_nodes["node_id"].map(flags).astype(int)

    # 5) Add metadata
    df_nodes["instance_id"] = instance_id
    df_nodes["best_val"] = best_val
    df_nodes["num_items"] = num_items
    df_nodes["seed"] = seed

    return df_nodes

def build_training_set(num_instances, num_items = None):

    """
    Generate a unified training dataset by sampling multiple random knapsack
    instances and collecting all branch-and-bound node data into a single
    concatenated DataFrame.

    Each instance:
        • Randomly selects a number of items in the range [5, 8]
        • Runs a full branch-and-bound search via `build_training_instance`
        • Produces a node-level DataFrame with labels indicating whether each
          node's subtree contains an optimal solution

    Args:
        num_instances (int): Number of random knapsack instances to generate
                             and include in the final dataset.

    Returns:
        pandas.DataFrame:
            A concatenated DataFrame containing node-level records from all
            generated knapsack instances. Includes fields such as:
                - node_id
                - parent_id
                - depth
                - lp_objective
                - status
                - fixed variable indicators
                - has_optimal_subtree (training label)
                - instance_id, seed, num_items (metadata)
    """

    all_dfs = []

    for i in range(num_instances):

        if num_items is None:
            num_items = np.random.randint(5, 9)
        df_i = build_training_instance(instance_id = i, num_items = num_items)
        all_dfs.append(df_i)

    full_df = pd.concat(all_dfs, ignore_index=True)

    return full_df