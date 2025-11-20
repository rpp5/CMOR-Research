from pyscipopt import Model, Eventhdlr, quicksum, SCIP_EVENTTYPE
import numpy as np

from pyscipopt import Model, Eventhdlr, quicksum, SCIP_EVENTTYPE

INT_TOL = 1e-6

def _safe_set_param(model, name, value):
    try:
        if hasattr(model, "hasParam") and model.hasParam(name):
            if isinstance(value, (bool, int)): model.setIntParam(name, int(value)); return
            if isinstance(value, float):        model.setRealParam(name, float(value)); return
            if isinstance(value, str):          model.setStringParam(name, value); return
        try: model.setIntParam(name, int(value)); return
        except: pass
        try: model.setRealParam(name, float(value)); return
        except: pass
        try: model.setStringParam(name, str(value)); return
        except: pass
    except:
        pass

def _is_integral_from_curr_lp(model, vars_):
    for v in vars_:
        if v.vtype() in ("BINARY", "INTEGER", "IMPLINT"):
            val = model.getSolVal(None, v)
            if abs(val - round(val)) > INT_TOL:
                return False
    return True

class NodeTracker(Eventhdlr):
    def __init__(self):
        super().__init__()
        self.active = {}
        self.sols = []
        self.allvars = None

    def eventinit(self):
        for et in [
            SCIP_EVENTTYPE.NODEBRANCHED,
            SCIP_EVENTTYPE.NODEFOCUSED,
            SCIP_EVENTTYPE.LPSOLVED,
            SCIP_EVENTTYPE.NODEINFEASIBLE,
            SCIP_EVENTTYPE.NODEFEASIBLE,
            SCIP_EVENTTYPE.NODESOLVED,
            SCIP_EVENTTYPE.BESTSOLFOUND,
        ]:
            try: self.model.catchEvent(et, self)
            except: pass
        self.allvars = self.model.getVars()

    def _mark_open(self, node):
        try:
            nid = node.getNumber(); depth = self.model.getDepth()
        except: 
            return
        if nid not in self.active:
            self.active[nid] = {"status": "open", "depth": depth, "lp_bound": None, "integral": None}

    def _close(self, node, reason):
        try: nid = node.getNumber()
        except: return
        if nid in self.active: self.active[nid]["status"] = reason

    def eventexec(self, event):
        et = event.getType()

        if et == SCIP_EVENTTYPE.NODEBRANCHED:
            node = getattr(event, "getNode", lambda: None)() or self.model.getCurrentNode()
            if node: self._mark_open(node)

        elif et == SCIP_EVENTTYPE.NODEFOCUSED:
            node = self.model.getCurrentNode()
            if node: self._mark_open(node)

        elif et == SCIP_EVENTTYPE.LPSOLVED:
            node = self.model.getCurrentNode()
            if not node: 
                return
            self._mark_open(node)
            try: 
                lb = self.model.getLPObjVal()
            except: 
                lb = None
            nid = node.getNumber()
            self.active[nid]["lp_bound"] = lb

            integral = _is_integral_from_curr_lp(self.model, self.allvars)
            self.active[nid]["integral"] = integral
            if integral:
                vals = {}
                for v in self.allvars:
                    if v.vtype() in ("BINARY", "INTEGER", "IMPLINT"):
                        vals[v.name] = int(round(self.model.getSolVal(None, v)))
                try: obj = self.model.getLPObjVal()
                except: obj = None
                self.sols.append({"obj": obj, "values": vals})
                self._close(node, "integer")

        elif et == SCIP_EVENTTYPE.NODEINFEASIBLE:
            node = getattr(event, "getNode", lambda: None)() or self.model.getCurrentNode()
            if node: self._close(node, "infeasible")

        elif et == SCIP_EVENTTYPE.BESTSOLFOUND:
            sol = self.model.getBestSol()
            if sol:
                try: obj = self.model.getSolObjVal(sol)
                except: obj = None
                vals = {}
                for v in self.allvars:
                    if v.vtype() in ("BINARY", "INTEGER", "IMPLINT"):
                        vals[v.name] = int(round(self.model.getSolVal(sol, v)))
                self.sols.append({"obj": obj, "values": vals})

from pyscipopt import SCIP_PARAMSETTING

def solve_ip_enumerate_like(build_model_fn, node_limit, time_limit=None, force_branch=True):
    """
    Run B&B with a node limit, log integer-feasible solutions via NodeTracker,
    and return the true active node list using model.getOpenNodes().

    Set force_branch=True to disable presolve/heuristics/separating so root doesn't close.
    """
    # build random model
    model = build_model_fn()

    # disable heuristics/presolve/separating to avoid root closing
    if force_branch:
        model.setHeuristics(SCIP_PARAMSETTING.OFF)   # disable primal heuristics
        model.setPresolve(SCIP_PARAMSETTING.OFF)     # disable presolve
        model.setSeparating(SCIP_PARAMSETTING.OFF)   # disable cut separation

    # Stop rules 
    model.setLongintParam("limits/nodes", node_limit)
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))

    # attach tracker
    tracker = NodeTracker()
    model.includeEventhdlr(tracker, "NodeTracker", "Log integer nodes")

    model.optimize()

    # get open nodes
    leaves, children, siblings = model.getOpenNodes()
    def _node_info(n):
        try:
            lb = n.getLowerbound()
        except Exception:
            lb = None
        return {
            "node_id": n.getNumber(),
            "parent_id": (n.getParent().getNumber() if n.getParent() else None),
            "depth": n.getDepth(),
            "lower_bound": lb,
            "is_active": n.isActive(),
        }
    active_nodes = [_node_info(n) for n in (leaves + children + siblings)]

    try:
        nodes_remaining = model.getNNodesLeft()
    except AttributeError:
        nodes_remaining = None

    out = {
        "status": model.getStatus(),
        "nodes_processed": model.getNNodes(),
        "nodes_remaining": nodes_remaining,
        "solving_time": model.getSolvingTime(),
        "integer_solutions": tracker.sols,   # keep only tracker to avoid double-counting
        "active_nodes": active_nodes,
    }

    try:
        model.free()
    except Exception:
        pass
    return out
