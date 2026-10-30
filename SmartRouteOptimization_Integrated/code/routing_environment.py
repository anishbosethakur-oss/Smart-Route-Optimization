import random
import numpy as np
import networkx as nx


class RoutingEnvironment:
    """
    Shared routing environment: based on Member 1's topology, extended to six routers,
    extended with the interface required by Member 2's DQN.

    Router IDs used by the DQN:
        0=R1, 1=R2, 2=R3, 3=R4, 4=R5, 5=R6

    State (28 floats):
        current-router one-hot (6)
        destination-router one-hot (6)
        normalized effective link delays (8)
        link utilization (8)

    Actions are router IDs. Only direct neighbors are valid.
    """

    ROUTERS = ["R1", "R2", "R3", "R4", "R5", "R6"]
    LINKS = [
        ("R1", "R2", 10.0, 100.0),
        ("R1", "R3", 15.0, 100.0),
        ("R2", "R4", 10.0, 80.0),
        ("R3", "R4", 20.0, 100.0),
        ("R3", "R5", 10.0, 50.0),
        ("R4", "R5", 5.0, 100.0),
        # New router creates additional alternative routes.
        ("R2", "R6", 12.0, 70.0),
        ("R6", "R5", 8.0, 80.0),
    ]
    MAX_STEPS = 10
    MAX_NORMALIZED_DELAY = 40.0

    def __init__(self, seed=None, max_steps=MAX_STEPS):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps)
        self.network = nx.Graph()
        self.network.add_nodes_from(self.ROUTERS)
        for a, b, delay, bandwidth in self.LINKS:
            self.network.add_edge(a, b, delay=delay, bandwidth=bandwidth)

        self.router_to_id = {name: i for i, name in enumerate(self.ROUTERS)}
        self.id_to_router = {i: name for i, name in enumerate(self.ROUTERS)}
        # Explicit edge order keeps the DQN's state layout stable.
        self.edge_order = [(a, b) for a, b, _, _ in self.LINKS]
        self.current_router = 0
        self.source = 0
        self.destination = 4
        self.steps = 0
        self.visited = set()
        self.last_path = []
        self.total_delay = 0.0
        self.utilization = np.zeros(len(self.edge_order), dtype=np.float32)
        self._update_network()

    @property
    def state_size(self):
        return 2 * len(self.ROUTERS) + 2 * len(self.edge_order)

    @property
    def action_size(self):
        return len(self.ROUTERS)

    def _edge_key(self, a, b):
        return frozenset((a, b))

    def _edge_index(self, a, b):
        target = self._edge_key(a, b)
        for i, (u, v) in enumerate(self.edge_order):
            if self._edge_key(u, v) == target:
                return i
        return None

    def _update_network(self):
        for i, (a, b) in enumerate(self.edge_order):
            data = self.network[a][b]
            utilization = self.rng.uniform(0.20, 0.90)
            data["utilization"] = utilization
            data["traffic"] = data["bandwidth"] * utilization
            data["effective_delay"] = data["delay"] * (1.0 + utilization)
            if utilization >= 0.80:
                data["congestion"] = "HIGH"
            elif utilization >= 0.60:
                data["congestion"] = "MEDIUM"
            else:
                data["congestion"] = "LOW"
            self.utilization[i] = utilization

    def _get_state(self):
        current = np.zeros(len(self.ROUTERS), dtype=np.float32)
        current[self.current_router] = 1.0
        destination = np.zeros(len(self.ROUTERS), dtype=np.float32)
        destination[self.destination] = 1.0

        delays = []
        utils = []
        for a, b in self.edge_order:
            data = self.network[a][b]
            delays.append(min(data["effective_delay"] / self.MAX_NORMALIZED_DELAY, 1.0))
            utils.append(data["utilization"])
        return np.concatenate([
            current, destination,
            np.asarray(delays, dtype=np.float32),
            np.asarray(utils, dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, source=0, destination=4):
        # Accept either router IDs (0-5) or names ("R1"-"R6").
        if isinstance(source, str):
            source = self.router_to_id[source]
        if isinstance(destination, str):
            destination = self.router_to_id[destination]
        self.source = int(source)
        self.destination = int(destination)
        if not (0 <= self.source < len(self.ROUTERS)):
            raise ValueError("Source must be a router ID 0-5 or a router name R1-R6.")
        if not (0 <= self.destination < len(self.ROUTERS)):
            raise ValueError("Destination must be a router ID 0-5 or a router name R1-R6.")
        if self.source == self.destination:
            raise ValueError("Source and destination must be different routers.")

        self.current_router = self.source
        self.steps = 0
        self.visited = {self.source}
        self.last_path = [self.source]
        self.total_delay = 0.0
        self._update_network()
        return self._get_state()

    def valid_actions(self, router=None):
        if router is None:
            router = self.current_router
        if isinstance(router, str):
            router = self.router_to_id[router]
        name = self.id_to_router[int(router)]
        return sorted(self.router_to_id[n] for n in self.network.neighbors(name))

    def step(self, action):
        action = int(action)
        self.steps += 1
        valid = self.valid_actions()
        if action not in valid:
            done = True
            return self._get_state(), -10.0, done, {
                "valid": False,
                "success": False,
                "reason": "invalid_action",
                "path": self.path_names(),
                "total_delay": self.total_delay,
            }

        current_name = self.id_to_router[self.current_router]
        next_name = self.id_to_router[action]
        data = self.network[current_name][next_name]
        delay = float(data["effective_delay"])
        utilization = float(data["utilization"])
        selected_congestion = data["congestion"]
        self.total_delay += delay

        # Penalize latency and congestion. Reward successful delivery.
        reward = -0.1 * delay
        if utilization >= 0.80:
            reward -= 1.0
        elif utilization >= 0.60:
            reward -= 0.4
        if action in self.visited:
            reward -= 2.0

        self.current_router = action
        self.last_path.append(action)
        self.visited.add(action)

        success = self.current_router == self.destination
        done = success or self.steps >= self.max_steps
        if success:
            reward += 10.0
        elif self.steps >= self.max_steps:
            reward -= 3.0

        self._update_network()
        info = {
            "valid": True,
            "success": success,
            "path": self.path_names(),
            "delay": delay,
            "total_delay": self.total_delay,
            "link_utilization": utilization,
            "congestion": selected_congestion,
        }
        return self._get_state(), float(reward), bool(done), info

    def path_names(self, path=None):
        if path is None:
            path = self.last_path
        return [self.id_to_router[int(router)] for router in path]
