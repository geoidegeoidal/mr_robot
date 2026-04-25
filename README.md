# 🤖 MR_ROBOT — Santiago Urban Navigator (Swarm Edition)

Pipeline de Reinforcement Learning para entrenar una IA que navegue de forma autónoma por la red peatonal de la Región Metropolitana de Santiago, Chile.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)
![D3.js](https://img.shields.io/badge/D3.js-v7-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data_handler.py│────▶│  environment.py  │────▶│   trainer.py    │
│  OSMnx + Grafo  │     │  Gymnasium Env   │     │  MaskablePPO    │
│  Smart Parsing  │     │  GPS (Dijkstra)  │     │  15 Swarm Envs  │
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

### 🧬 Inteligencia de Enjambre (Vectorized Training)
Hemos implementado **15 entornos en paralelo** (`N_ENVS = 15`). Esto significa que 15 robots exploran la ciudad simultáneamente, compartiendo sus éxitos y fracasos para estabilizar y acelerar el aprendizaje del modelo central.

### 📍 Smart Parsing (Texto o Coordenadas)
El sistema resuelve automáticamente si le pasas una dirección textual ("Plaza Italia") o coordenadas geográficas directas ("-33.43, -70.63").

### ⚡ Optimización de Cómputo (Shared Map)
Para manejar 15 robots simultáneos, el sistema ahora precalcula el mapa de distancias una sola vez al inicio y lo comparte entre todos los hilos, reduciendo el consumo de RAM y el tiempo de arranque en un 90%.

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
# Entrenar con ruta por defecto (Bellavista -> Maipú)
python trainer.py

# Entrenar con ruta personalizada (Texto)
python trainer.py --start "Cerro Santa Lucia, Santiago" --end "Costanera Center, Providencia"

# Entrenar con coordenadas exactas
python trainer.py --start "-33.4372, -70.6342" --end "-33.4025, -70.5802"

# Evaluar modelo entrenado
python trainer.py eval
```

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
| 🏆 TOP 3 | Muestra los 3 mejores intentos con mayor recompensa acumulada. |

## 🧠 Hiperparámetros de Élite

- **Algorithm:** MaskablePPO (sb3-contrib).
- **Parallel Envs:** 15 (Vectorized).
- **Gamma:** 0.995 (visión a largo plazo).
- **Ent_Coef:** 0.05 (exploración/mutación agresiva).
- **Max Steps:** 12,000 por episodio.

---
*Built with 🧠 and 🦾 by MR_ROBOT Pipeline*
