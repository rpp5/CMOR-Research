from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING

INT_TOL = 1e-6

def _is_integral_from_curr_lp(model, vars_):
    for v in vars_:
        if v.vtype() in ("BINARY", "INTEGER", "IMPLINT"):
            val = model.getSolVal(None, v)
            if abs(val - round(val)) > INT_TOL:
                return False
    return True

def _node_details(n):
    try:
        pid = n.getParent().getNumber() if n.getParent() else None
    except Exception:
        pid = None
    try:
        lb = n.getLowerbound()
    except Exception:
        lb = None
    return {
        "node_id": n.getNumber(),
        "parent_id": pid,
        "depth": n.getDepth(),
        "lower_bound": lb,
    }

class FullEnumTracker(Eventhdlr):
    """
    Logs:
      - integer_solutions: each LP-integral node (with node details + integer solution)
      - infeasible_nodes: node details when LP is infeasible
    """
    def __init__(self):
        super().__init__()
        self.integer_solutions = []
        self.infeasible_nodes = []
        self.allvars = None

    def eventinit(self):
        for et in [
            SCIP_EVENTTYPE.NODEFOCUSED,
            SCIP_EVENTTYPE.LPSOLVED,
            SCIP_EVENTTYPE.NODEINFEASIBLE,
            SCIP_EVENTTYPE.NODESOLVED,
        ]:
            try:
                self.model.catchEvent(et, self)
            except:
                pass
        self.allvars = self.model.getVars()

        # Keep cutoff bound at +inf so SCIP never prunes by bound after an incumbent appears.
        try:
            self.model.setCutoffbound(float('inf'))
        except:
            pass

    def eventexec(self, event):
        et = event.getType()

        if et == SCIP_EVENTTYPE.LPSOLVED:
            node = self.model.getCurrentNode()
            if not node:
                return

            # maintain +inf cutoff continuously
            try:
                self.model.setCutoffbound(float('inf'))
            except:
                pass

            # if LP solution is integral, capture and we're done with this node
            if _is_integral_from_curr_lp(self.model, self.allvars):
                vals = {}
                for v in self.allvars:
                    if v.vtype() in ("BINARY", "INTEGER", "IMPLINT"):
                        vals[v.name] = int(round(self.model.getSolVal(None, v)))
                try:
                    obj = self.model.getLPObjVal()
                except Exception:
                    obj = None

                nd = _node_details(node)
                nd.update({"obj": obj, "values": vals})
                self.integer_solutions.append(nd)

        elif et == SCIP_EVENTTYPE.NODEINFEASIBLE:
            node = getattr(event, "getNode", lambda: None)() or self.model.getCurrentNode()
            if node:
                self.infeasible_nodes.append(_node_details(node))

        elif et == SCIP_EVENTTYPE.NODESOLVED:
            # no-op; enumeration continues automatically until no open nodes remain
            pass

def solve_ip_full_enumeration(build_model_fn, time_limit=None, force_branch=True):
    """
    Run B&B until the tree is exhausted, with NO node limit and NO bound-based pruning.
    Leaves must end as either LP-infeasible or LP-integral.
    Integer nodes are logged with node details and integer solution.

    Returns:
      {
        status, nodes_processed, solving_time,
        integer_solutions: [ {node_id, parent_id, depth, lower_bound, obj, values{var:int}} ... ],
        infeasible_nodes:  [ {node_id, parent_id, depth, lower_bound} ... ],
        open_nodes_final:  [ ... ]   # should be empty if tree exhausted
      }
    """
    model = build_model_fn()

    if force_branch:
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setSeparating(SCIP_PARAMSETTING.OFF)
        # optional: reduce extra pruning/propagation “cleverness”
        try: model.setParam("propagating/maxrounds", 0)
        except: pass

    # NO node limit
    # (do not set limits/nodes)
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))

    # Make sure no cutoff pruning ever happens (even after an incumbent is found)
    try:
        model.setCutoffbound(float('inf'))
    except:
        pass

    tracker = FullEnumTracker()
    model.includeEventhdlr(tracker, "FullEnumTracker", "Enumerate: only LP-infeasible or LP-integral leaves")

    model.optimize()

    # gather final open nodes (should be empty if fully exhausted)
    leaves, children, siblings = model.getOpenNodes()
    def _ni(n):
        return {
            "node_id": n.getNumber(),
            "parent_id": (n.getParent().getNumber() if n.getParent() else None),
            "depth": n.getDepth(),
            "lower_bound": (n.getLowerbound() if hasattr(n, "getLowerbound") else None),
            "is_active": n.isActive(),
        }
    open_nodes_final = [_ni(n) for n in (leaves + children + siblings)]

    out = {
        "status": model.getStatus(),
        "nodes_processed": model.getNNodes(),
        "solving_time": model.getSolvingTime(),
        "integer_solutions": tracker.integer_solutions,
        "infeasible_nodes": tracker.infeasible_nodes,
        "open_nodes_final": open_nodes_final,
    }

    try: model.free()
    except: pass
    return out
