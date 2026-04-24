"""
environment.py — Módulo 3: Entorno de Reinforcement Learning (Custom Gymnasium).

Define SantiagoUrbanEnv, un entorno Gymnasium con action masking para
que una IA aprenda a navegar la red peatonal de Santiago de Chile.

Compatible con MaskablePPO de sb3-contrib.

Autor: MR_ROBOT Pipeline
"""

import math
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from data_handler import (
    get_forbidden_highways,
    get_avenue_highways,
    create_utm_to_latlon_transformer,
    utm_to_latlon,
)

logger = logging.getLogger(__name__)


class SantiagoUrbanEnv(gym.Env):
    """
    Entorno de navegación urbana sobre el grafo peatonal de Santiago.

    El agente se posiciona en un nodo del grafo y debe alcanzar un
    nodo objetivo seleccionando entre los vecinos disponibles.

    Usa action masking (compatible con MaskablePPO) para manejar
    el número variable de vecinos en cada nodo.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, G_projected, G_latlon, start_node, end_node,
                 max_degree, max_steps=5000):
        """
        Args:
            G_projected: Grafo proyectado a UTM 19S (metros)
            G_latlon: Grafo en WGS84 (para exportar coordenadas)
            start_node: ID del nodo de inicio
            end_node: ID del nodo objetivo
            max_degree: Grado máximo del grafo (define action space)
            max_steps: Máximo de pasos por episodio
        """
        super().__init__()

        self.G = G_projected
        self.G_latlon = G_latlon
        self.start_node = start_node
        self.end_node = end_node
        self.max_steps = max_steps

        # Precachear datos del nodo objetivo
        end_data = self.G.nodes[end_node]
        self.end_x = end_data["x"]
        self.end_y = end_data["y"]

        # ── Spaces ──
        # State: [distancia_relativa, ángulo_normalizado, grado_normalizado, indicador_avenida]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Action: índice discreto entre los vecinos (tamaño fijo = max_degree)
        self.max_degree = max_degree
        self.action_space = spaces.Discrete(max_degree)

        # ── Highway sets ──
        self.forbidden_highways = get_forbidden_highways()
        self.avenue_highways = get_avenue_highways()

        # ── Transformer UTM → WGS84 para logging de trayectoria ──
        self.transformer = create_utm_to_latlon_transformer()

        # ── Episode state ──
        self.current_node = None
        self.current_step = 0
        self.initial_distance = 0.0
        self.prev_distance = 0.0
        self.neighbors_list = []
        self.visited_nodes = set()

        # ── Trajectory logging ──
        self.trajectory = []  # Lista de (lat, lon) del episodio actual
        self.episode_reward = 0.0
        self.episode_result = None  # "success", "fail_risk", "fail_timeout", "fail_deadend"

    def reset(self, seed=None, options=None):
        """Reinicia el episodio: posiciona el robot en el nodo de inicio."""
        super().reset(seed=seed)

        self.current_node = self.start_node
        self.current_step = 0
        self.visited_nodes = {self.start_node}

        # Calcular distancia inicial al objetivo
        self.initial_distance = self._distance_to_goal(self.start_node)
        self.prev_distance = self.initial_distance

        # Actualizar lista de vecinos
        self._update_neighbors()

        # Iniciar trayectoria
        self.trajectory = [self._node_to_latlon(self.start_node)]
        self.episode_reward = 0.0
        self.episode_result = None

        obs = self._get_obs()
        info = {"node_id": self.current_node, "distance": self.initial_distance}
        return obs, info

    def step(self, action):
        """
        Ejecuta una acción (moverse al vecino con índice `action`).

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        terminated = False
        truncated = False
        reward = 0.0

        # ── Validar acción con máscara ──
        mask = self.action_masks()
        if not mask[action]:
            # Acción inválida (slot enmascarado) → penalización severa
            reward = -500.0
            terminated = True
            self.episode_result = "fail_invalid_action"
            self.episode_reward += reward
            obs = self._get_obs()
            return obs, reward, terminated, truncated, self._get_info()

        # ── Mover al vecino seleccionado ──
        next_node = self.neighbors_list[action]

        # ── Verificar vía prohibida ──
        if self._is_forbidden_edge(self.current_node, next_node):
            reward = -500.0
            terminated = True
            self.episode_result = "fail_risk"
            self.episode_reward += reward
            obs = self._get_obs()
            return obs, reward, terminated, truncated, self._get_info()

        # ── Actualizar posición ──
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        self.trajectory.append(self._node_to_latlon(next_node))

        # ── Verificar objetivo alcanzado ──
        if self.current_node == self.end_node:
            reward = 2000.0
            terminated = True
            self.episode_result = "success"
            self.episode_reward += reward
            obs = self._get_obs()
            return obs, reward, terminated, truncated, self._get_info()

        # ── Actualizar vecinos del nuevo nodo ──
        self._update_neighbors()

        # ── Verificar callejón sin salida ──
        if len(self.neighbors_list) == 0:
            reward = -500.0
            terminated = True
            self.episode_result = "fail_deadend"
            self.episode_reward += reward
            obs = self._get_obs()
            return obs, reward, terminated, truncated, self._get_info()

        # ── Calcular recompensa de progreso ──
        current_distance = self._distance_to_goal(self.current_node)
        progress = self.prev_distance - current_distance  # metros de avance
        reward = progress * 1.0 - 0.1  # progreso + fricción temporal

        # Penalización leve por revisitar nodos (evitar ciclos)
        if next_node in self.visited_nodes:
            reward -= 1.0

        self.prev_distance = current_distance
        self.episode_reward += reward

        # ── Truncar si se excede max_steps ──
        if self.current_step >= self.max_steps:
            truncated = True
            self.episode_result = "fail_timeout"

        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._get_info()

    def action_masks(self):
        """
        Retorna máscara booleana de acciones válidas.
        Los primeros len(neighbors) slots son True, el resto False.

        Requerido por MaskablePPO de sb3-contrib.
        """
        mask = np.zeros(self.max_degree, dtype=bool)
        n_neighbors = len(self.neighbors_list)
        if n_neighbors > 0:
            mask[:n_neighbors] = True
        return mask

    # ──────────────── Métodos internos ────────────────

    def _get_obs(self):
        """Construye el vector de observación normalizado."""
        node_data = self.G.nodes[self.current_node]
        node_x, node_y = node_data["x"], node_data["y"]

        # [0] Distancia relativa al objetivo (normalizada por distancia inicial)
        dist = self._distance_to_goal(self.current_node)
        if self.initial_distance > 0:
            dist_rel = min(dist / self.initial_distance, 1.0)
        else:
            dist_rel = 0.0

        # [1] Ángulo hacia el objetivo (normalizado 0-1)
        dx = self.end_x - node_x
        dy = self.end_y - node_y
        angle = math.atan2(dy, dx)  # radianes [-π, π]
        angle_norm = (angle + math.pi) / (2 * math.pi)  # normalizar a [0, 1]

        # [2] Grado del nodo actual (normalizado)
        degree = self.G.degree(self.current_node)
        degree_norm = min(degree / self.max_degree, 1.0)

        # [3] Indicador de cercanía a avenida
        avenue_indicator = 1.0 if self._near_avenue(self.current_node) else 0.0

        obs = np.array([dist_rel, angle_norm, degree_norm, avenue_indicator],
                       dtype=np.float32)
        return obs

    def _get_info(self):
        """Retorna info dict del paso actual."""
        info = {
            "node_id": self.current_node,
            "step": self.current_step,
            "distance_to_goal": self._distance_to_goal(self.current_node),
            "episode_reward": self.episode_reward,
            "result": self.episode_result,
        }
        if self.episode_result is not None:
            info["trajectory_data"] = self.get_trajectory_data()
        return info

    def _distance_to_goal(self, node):
        """Distancia euclidiana en metros (UTM) al nodo objetivo."""
        nd = self.G.nodes[node]
        dx = nd["x"] - self.end_x
        dy = nd["y"] - self.end_y
        return math.sqrt(dx * dx + dy * dy)

    def _update_neighbors(self):
        """Actualiza la lista ordenada de vecinos del nodo actual."""
        self.neighbors_list = sorted(list(self.G.neighbors(self.current_node)))

    def _is_forbidden_edge(self, u, v):
        """Verifica si la arista u→v tiene un highway tag prohibido."""
        edge_data = self.G.get_edge_data(u, v)
        if edge_data is None:
            return True  # Edge no existe
        # MultiDiGraph: edge_data es dict de keys
        for key, data in edge_data.items():
            highway = data.get("highway", "")
            if isinstance(highway, list):
                hw_set = set(highway)
            else:
                hw_set = {highway}
            if hw_set & self.forbidden_highways:
                return True
        return False

    def _near_avenue(self, node):
        """Verifica si alguna arista saliente del nodo es una avenida."""
        for _, _, data in self.G.edges(node, data=True):
            highway = data.get("highway", "")
            if isinstance(highway, list):
                hw_set = set(highway)
            else:
                hw_set = {highway}
            if hw_set & self.avenue_highways:
                return True
        return False

    def _node_to_latlon(self, node):
        """Convierte un nodo del grafo UTM a (lat, lon)."""
        nd = self.G.nodes[node]
        lat, lon = utm_to_latlon(nd["x"], nd["y"], self.transformer)
        return [lat, lon]

    def get_trajectory_data(self):
        """
        Retorna los datos del episodio actual para logging.

        Returns:
            dict con path, result, reward, steps
        """
        return {
            "path": self.trajectory.copy(),
            "result": self.episode_result or "incomplete",
            "total_reward": round(self.episode_reward, 2),
            "steps": self.current_step,
        }
