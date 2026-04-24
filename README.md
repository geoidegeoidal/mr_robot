# 🤖 MR_ROBOT — Santiago Urban Navigator (Evolutionary Edition)

Pipeline de Reinforcement Learning para entrenar una IA que navegue de forma autónoma por la red peatonal de la Región Metropolitana de Santiago, Chile.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![D3.js](https://img.shields.io/badge/D3.js-v7-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data_handler.py│────▶│  environment.py  │────▶│   trainer.py    │
│  OSMnx + Grafo  │     │  Gymnasium Env   │     │  MaskablePPO    │
│  Caché RM UTM   │     │  GPS (Dijkstra)  │     │  Mutaciones RL  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │ training_viz.json│
                                               │ streets.geojson  │
                                               └────────┬────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │ visualizer.html  │
                                               │ D3.js Cyberpunk  │
                                               │ 🏆 Top 3 Filter  │
                                               └─────────────────┘
```

## 🧠 Características Avanzadas (The "Secret Sauce")

### 🛰️ Navegación por "GPS" de Grafo (Dense Reward)
A diferencia de otros modelos que usan línea recta, MR_ROBOT precalcula el **árbol de distancias reales por calles (Dijkstra)** desde cada nodo hasta el objetivo. Esto le da un "sentido del olfato" perfecto: el robot sabe exactamente si doblar en una esquina lo acerca o lo aleja del destino caminando, no volando.

### 🧬 Mutaciones Evolutivas (Entropy Exploration)
Usamos un coeficiente de entropía elevado (`ENT_COEF = 0.05`) para forzar al robot a probar "mutaciones" constantes en su ruta. Esto evita que se quede estancado en parques o rotondas y lo obliga a descubrir nuevas variantes de la ruta óptima.

### 🌃 Storytelling: Misión Nocturna
El entrenamiento está configurado por defecto para la ruta **Barrio Bellavista → Plaza de Maipú** (~15 km). Una navegación crítica a través de la red peatonal nocturna de Santiago.

## ⚡ Setup Rápido

```bash
# 1. Clonar
git clone https://github.com/geoidegeoidal/mr_robot.git
cd mr_robot

# 2. Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Ejecución del Pipeline

```bash
# Entrenar (Carga el mapa y precalcula el GPS de grafo)
python trainer.py

# Evaluar modelo entrenado
python trainer.py eval
```

### ¿Qué pasa durante el entrenamiento?
1. **Carga del Grafo:** Lee `santiago_master_walk.graphml` (RM completa).
2. **Precalculo GPS:** Calcula las distancias Dijkstra hacia el destino (tarda ~2-5 segundos).
3. **Rollout PPO:** El robot lanza múltiples agentes al mapa que aprenden por ensayo y error.
4. **Exportación:** Genera `training_viz.json` con toda la telemetría de los intentos.

## 🎮 Visualización Profesional

Para ver los resultados, lanza un servidor local y abre el visualizador cyberpunk:

```bash
# Iniciar servidor
python -m http.server 8080
# Abrir: http://localhost:8080/visualizer.html
```

### Controles HUD
| Botón | Función |
|-------|---------|
| ▶ PLAY | Reproducir/pausar la cinemática del robot. |
| ◎ FOLLOW | Cámara de seguimiento LERP (suave) para grabación. |
| ★ SUCCESS | Filtrar solo los episodios que llegaron a la meta. |
| 🏆 TOP 3 | **Nuevo:** Muestra automáticamente los 3 mejores intentos (mayor recompensa). |

## 🧠 Hiperparámetros de Élite

- **Algorithm:** MaskablePPO (handling variable neighbors).
- **Gamma:** 0.995 (priorizando el largo plazo).
- **Ent_Coef:** 0.05 (forzando exploración/mutación).
- **Max Steps:** 8000 por episodio.

---
*Built with 🧠 and 🦾 by MR_ROBOT Pipeline*
