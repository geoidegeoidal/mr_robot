"""
trainer.py — Módulo 4: Entrenamiento PPO y Exportación de Logs.

Entrena una IA con MaskablePPO (sb3-contrib) para navegar la red peatonal
de Santiago. Exporta training_viz.json para el visualizador D3.js.

Ruta: Bellavista → Plaza de Maipú (de noche, a las 3 AM — la red peatonal
no cambia con la hora, pero el storytelling sí 🌃).

Autor: MR_ROBOT Pipeline
"""

import os
import json
import time
import logging
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from data_handler import (
    load_or_download_graph,
    compute_max_degree,
    get_route_nodes,
    export_network_geojson,
)
from environment import SantiagoUrbanEnv

# ─────────────────────── Config ───────────────────────────

# Ruta: Bellavista → Plaza de Maipú
START_POI = "Barrio Bellavista, Santiago, Chile"
END_POI = "Plaza de Maipú, Maipú, Chile"

# Hiperparámetros de entrenamiento
TOTAL_TIMESTEPS = 500_000       # Con el nuevo "GPS", 500K es suficiente para llegar
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.995                   # Descuento alto para rutas largas
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05                 # ¡MUTACIÓN! Aumentado a 0.05 para forzar exploración aleatoria
MAX_STEPS_PER_EPISODE = 8000    # Permitir rutas largas en RM completa

# Output
OUTPUT_DIR = os.path.dirname(__file__)
VIZ_JSON_PATH = os.path.join(OUTPUT_DIR, "training_viz.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "mr_robot_ppo")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────── Callback: Trajectory Logger ─────────────────

class TrajectoryLoggerCallback(BaseCallback):
    """
    Callback que registra la trayectoria de cada episodio completado.
    Guarda los datos en una lista para exportar al final como JSON.
    """

    def __init__(self, env, max_episodes_to_log=2000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.max_episodes_to_log = max_episodes_to_log
        self.episodes = []
        self.episode_count = 0
        self.successes = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Acceder a la info del environment
        # En SB3, cuando el episodio termina, los dones se reportan
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if done:
                # Obtener datos del episodio desde el info dict (para evitar data de env reset)
                info = infos[i]
                traj_data = info.get("trajectory_data")
                
                # Fallback por si acaso (aunque DummyVecEnv preserva el info terminal)
                if not traj_data:
                    traj_data = self.env.envs[i].get_trajectory_data()

                if traj_data["result"] == "success":
                    self.successes += 1

                # Registrar episodio
                if self.episode_count < self.max_episodes_to_log:
                    episode_record = {
                        "id": self.episode_count,
                        **traj_data
                    }
                    self.episodes.append(episode_record)

                self.episode_count += 1

                # Log periódico
                if self.episode_count % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.successes / self.episode_count * 100 if self.episode_count > 0 else 0
                    logger.info(
                        f"Episodio {self.episode_count} | "
                        f"Éxitos: {self.successes} ({rate:.1f}%) | "
                        f"Timestep: {self.num_timesteps} | "
                        f"Tiempo: {elapsed:.0f}s"
                    )

        return True

    def get_results(self):
        """Retorna los resultados acumulados del entrenamiento."""
        return {
            "total_episodes": self.episode_count,
            "successful_episodes": self.successes,
            "success_rate": round(self.successes / max(self.episode_count, 1) * 100, 2),
            "episodes_logged": len(self.episodes),
        }


# ──────────────── Pipeline Principal ──────────────────────────

def train():
    """Pipeline completo de entrenamiento."""

    logger.info("=" * 60)
    logger.info("MR_ROBOT — Pipeline de Entrenamiento")
    logger.info("=" * 60)

    # ── 1. Cargar/descargar grafo ──
    logger.info("\n[1/5] Cargando grafo de la RM...")
    G_projected, G_latlon = load_or_download_graph()

    # ── 2. Resolver POIs ──
    logger.info("\n[2/5] Resolviendo POIs...")
    start_node, end_node = get_route_nodes(START_POI, END_POI, G_projected, G_latlon)

    # ── 3. Calcular max_degree y crear environment ──
    logger.info("\n[3/5] Configurando entorno...")
    max_degree = compute_max_degree(G_projected)

    env = SantiagoUrbanEnv(
        G_projected=G_projected,
        G_latlon=G_latlon,
        start_node=start_node,
        end_node=end_node,
        max_degree=max_degree,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # ── 4. Entrenar ──
    logger.info("\n[4/5] Iniciando entrenamiento MaskablePPO...")
    logger.info(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    logger.info(f"  Ruta: {START_POI} → {END_POI}")
    logger.info(f"  Max degree (action space): {max_degree}")
    logger.info(f"  Max steps/episodio: {MAX_STEPS_PER_EPISODE}")

    trajectory_callback = TrajectoryLoggerCallback(
        env=env,
        max_episodes_to_log=2000,
        verbose=1,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=trajectory_callback,
        progress_bar=True,
    )

    # ── 5. Guardar modelo y exportar resultados ──
    logger.info("\n[5/5] Exportando resultados...")

    # Guardar modelo
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Modelo guardado: {MODEL_PATH}")

    # Exportar training_viz.json
    results = trajectory_callback.get_results()
    viz_data = {
        "metadata": {
            "start_poi": START_POI,
            "end_poi": END_POI,
            "start_node": int(start_node),
            "end_node": int(end_node),
            "total_timesteps": TOTAL_TIMESTEPS,
            **results,
        },
        "episodes": trajectory_callback.episodes,
    }

    with open(VIZ_JSON_PATH, "w") as f:
        json.dump(viz_data, f, indent=2)
    logger.info(f"Visualización exportada: {VIZ_JSON_PATH}")

    # Exportar red de calles para D3.js (subset alrededor de la ruta)
    # Calcular bbox que cubra la ruta con margen
    start_data = G_latlon.nodes[start_node]
    end_data = G_latlon.nodes[end_node]
    margin = 0.02  # ~2km de margen
    bbox = (
        min(start_data["y"], end_data["y"]) - margin,
        min(start_data["x"], end_data["x"]) - margin,
        max(start_data["y"], end_data["y"]) + margin,
        max(start_data["x"], end_data["x"]) + margin,
    )
    export_network_geojson(G_latlon, bbox=bbox)

    # ── Resumen ──
    logger.info("\n" + "=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"  Episodios: {results['total_episodes']}")
    logger.info(f"  Éxitos: {results['successful_episodes']} ({results['success_rate']}%)")
    logger.info(f"  Episodios logueados: {results['episodes_logged']}")
    logger.info("=" * 60)

    return model, viz_data


# ──────────────── Modo evaluación (post-training) ─────────────

def evaluate(model_path=None, n_episodes=10):
    """
    Carga un modelo entrenado y ejecuta n episodios de evaluación.
    Exporta solo los episodios exitosos al viz JSON.
    """
    if model_path is None:
        model_path = MODEL_PATH

    logger.info("Cargando modelo para evaluación...")
    G_projected, G_latlon = load_or_download_graph()
    start_node, end_node = get_route_nodes(START_POI, END_POI, G_projected, G_latlon)
    max_degree = compute_max_degree(G_projected)

    env = SantiagoUrbanEnv(
        G_projected=G_projected,
        G_latlon=G_latlon,
        start_node=start_node,
        end_node=end_node,
        max_degree=max_degree,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    model = MaskablePPO.load(model_path)

    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        traj = env.get_trajectory_data()
        traj["id"] = ep
        episodes.append(traj)
        logger.info(f"Episodio {ep}: {traj['result']} | {traj['steps']} pasos | reward={traj['total_reward']}")

    # Exportar evaluación
    eval_data = {
        "metadata": {
            "start_poi": START_POI,
            "end_poi": END_POI,
            "mode": "evaluation",
            "total_episodes": n_episodes,
            "successful_episodes": sum(1 for e in episodes if e["result"] == "success"),
        },
        "episodes": episodes,
    }

    eval_path = os.path.join(OUTPUT_DIR, "eval_viz.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    logger.info(f"Evaluación exportada: {eval_path}")

    return eval_data


# ──────────────────────── Entry Point ─────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        train()
