"""
Browser View of an AICON Graph

Shows the sensors, actions, recursive estimators, active interconnections and goals of a graph, and highlights,
while the graph runs, the gradient path each action component currently follows (its steepest path, from the goal
to the action), and dimmed, the other gradient paths that currently reach it. On the page, nodes can be dragged,
arranged automatically with few crossing edges and hidden (only in the view), and a watch panel shows the current
values of chosen sensors, estimators, actions (also the gradients that reach them), goals or single quantities.

    viz = WebGraph(component_builders, connection_builders)   # serves http://localhost:8765
    while ...:
        run_component_sequence(components, t)
        viz.update(components)

Only uses the standard library: a small HTTP server in a background thread that streams the paths and values to
the page (web_graph.html) as server-sent events.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import json
import math
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Tuple

import torch

from aicon_base.components.action_components import ActionComponent
from aicon_base.components.util import collect_derivatives
from aicon_base.timestamps import as_timestamp, stamps_tensor
from aicon_base.visualization.network_graphs import obtain_code_based_structure
from ..sensor_action_components import SensorActionComponent

PAGE = os.path.join(os.path.dirname(__file__), "web_graph.html")


def plain(tensor: torch.Tensor):
    """A tensor as (nested) lists of numbers for JSON; non-finite numbers as the strings "nan", "inf", "-inf"."""

    def finite(x):
        return [finite(v) for v in x] if isinstance(x, list) else x if math.isfinite(x) else str(x)

    return finite(tensor.detach().cpu().tolist())


def graph_structure(component_builders: Dict[str, Callable], connection_builders: Dict[str, Callable]) -> Dict:
    """Sensors, actions, estimators, the interconnections between them and the goals, with the quantity names needed
    to map gradient traces onto the graph."""
    _, connections, goals, node_types, related_quantities = obtain_code_based_structure(
        component_builders, connection_builders
    )
    shown_types = {"Sensor": "sensor", "Action": "action", "Estimator": "estimator"}
    components = [k for k in component_builders if node_types.get(k) in shown_types]
    owner = {q: k for k in components for q in related_quantities[k]}
    nodes, edges = [], []
    for k in components:
        nodes.append({"id": k, "type": shown_types[node_types[k]]})
        for g in goals[k]:
            nodes.append({"id": g, "type": "goal"})
            edges.append({"source": g, "target": k})
    for c, quantities in connections.items():
        attached = sorted({owner[q] for q in quantities if q in owner})
        nodes.append({"id": c, "type": "connection"})
        edges.extend({"source": c, "target": k} for k in attached)
    return {
        "nodes": nodes,
        "edges": edges,
        "owner": owner,
        "connections": connections,
        "goals": [g for k in components for g in goals[k]],
    }


def trace_to_path(trace: List[str], structure: Dict) -> Dict[str, List[List[str]]]:
    """Map a gradient trace (goal, then the quantities it passed backwards, ending at the action quantity) to the
    nodes it passes (`steps`: goal, estimator, interconnection(s), estimator, ..., interconnection(s), action) and the
    directed edges between them.

    The trace only names quantities, so a hop between two quantities passes every interconnection that connects
    both (they all enter the same estimator update)."""
    quantities = [q for q in trace if not q.startswith("(")]  # "(...)": paths mixed in by projection
    if len(quantities) < 2 or quantities[0] not in structure["goals"]:
        return {"steps": [], "edges": []}
    owner, connections = structure["owner"], structure["connections"]
    steps, edges = [[quantities[0]], [owner[quantities[1]]]], [[quantities[0], owner[quantities[1]]]]
    for q1, q2 in zip(quantities[1:], quantities[2:]):
        c1, c2 = owner.get(q1), owner.get(q2)
        if c1 is None or c2 is None or c1 == c2:
            continue
        via = [name for name, connected in connections.items() if q1 in connected and q2 in connected]
        steps.extend([via, [c2]])
        edges.extend([c1, name] for name in via)
        edges.extend([name, c2] for name in via)
    return {"steps": steps, "edges": edges}


class WebGraph:
    """Serves the graph view and pushes the currently followed gradient paths to all open pages."""

    def __init__(
        self,
        component_builders,
        connection_builders,
        port: int = 8765,
        max_age: float = 0.5,
        values_every: float = 0.1,
        verbose: bool = True,
    ):
        """`max_age`: gradient paths older than this [s] are not shown; `values_every`: the least wall time [s]
        between two updates of the values."""
        self.structure = graph_structure(component_builders, connection_builders)
        self.max_age = max_age
        self.values_every = values_every
        self.clients: List[queue.Queue] = []
        self.last_message = None
        self.last_values = None
        self.last_values_time = -math.inf
        viz = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/graph":
                    self._send(200, "application/json", json.dumps(viz.structure).encode())
                elif self.path == "/events":
                    viz._stream(self)
                elif self.path in ("/", "/index.html"):
                    with open(PAGE, "rb") as f:
                        self._send(200, "text/html; charset=utf-8", f.read())
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("localhost", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        if verbose:
            print(f"AICON graph view at http://localhost:{self.server.server_address[1]}")

    def _stream(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        q = queue.Queue()
        for message in (self.last_message, self.last_values):
            if message is not None:
                q.put(message)
        self.clients.append(q)
        try:
            while True:
                handler.wfile.write(f"data: {q.get()}\n\n".encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.clients.remove(q)

    def computed_gradients(self, action: ActionComponent) -> List[Tuple[List[str], torch.Tensor]]:
        """The gradient paths that currently reach the action's quantities (recent and non-zero) with their
        gradients."""
        _, backward = collect_derivatives(action.connections)
        gradients = []
        # a sensor-action component also holds measurements, which are no actions
        if isinstance(action, SensorActionComponent):
            action_quantities = action.action_quantities()
        else:
            action_quantities = list(action.quantities.keys())
        for quantity in action_quantities:
            if quantity not in backward:
                continue
            block = backward[quantity]
            flat = block.derivatives_tensor.reshape(len(block.quantity_traces), -1)
            recent = stamps_tensor(block.oldest_stamps()) >= as_timestamp(action.timestamp - self.max_age)
            gradients.extend((t, g) for t, g, r in zip(block.quantity_traces, flat, recent) if r and torch.norm(g) > 0)
        return gradients

    def update(self, components, hidden=()):
        """Send the path each action component followed in its last update and the other gradient paths that reach
        it (when they changed), and the values of the components (at most every `values_every` seconds). The
        actions in `hidden` are shown without paths and gradients."""
        paths, gradients = {}, {}
        for name, c in components.items():
            if not isinstance(c, ActionComponent):
                continue
            used = c.used_gradients[0][0] if c.used_gradients[0] and name not in hidden else []
            computed = self.computed_gradients(c) if name not in hidden else []
            path = trace_to_path(used, self.structure)
            others = [trace_to_path(t, self.structure) for t, _ in computed]
            path["others"] = [o for o in others if o["edges"] and o["edges"] != path["edges"]]
            paths[name] = path
            followed = [q for q in used if not q.startswith("(")]
            gradients[name] = [
                {"path": self._components_passed(t), "value": plain(g), "followed": t == followed} for t, g in computed
            ]
        message = json.dumps({"paths": paths})
        if message != self.last_message:
            self.last_message = message
            self._send(message)
        if time.monotonic() - self.last_values_time >= self.values_every:
            self.last_values_time = time.monotonic()
            self.last_values = json.dumps({"values": self.values(components, gradients)})
            self._send(self.last_values)

    def _components_passed(self, trace: List[str]) -> List[str]:
        """The goal and the estimators a gradient trace passes, without the interconnections and the action."""
        steps = trace_to_path(trace, self.structure)["steps"]
        return [s[0] for s in steps[:-1] if s and s[0] not in self.structure["connections"]]

    def values(self, components, gradients: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """The quantities of the shown components, the costs of their goals and the gradients of the actions."""
        shown = {n["id"] for n in self.structure["nodes"] if n["type"] in ("sensor", "action", "estimator")}
        values = {}
        for name, c in components.items():
            if name not in shown:
                continue
            values[name] = {"quantities": {k: plain(c.quantities[k]) for k in c.quantities.keys()}}
            if name in gradients:
                values[name]["gradients"] = gradients[name]
            costs = getattr(c, "goal_values", None)
            for g, goal in c.goals.items():
                cost = costs[g] if costs is not None and g in costs.keys() else None
                values[g] = {"cost": None if cost is None else plain(cost), "active": bool(goal.is_active)}
        return values

    def _send(self, message: str):
        for q in list(self.clients):
            q.put(message)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
