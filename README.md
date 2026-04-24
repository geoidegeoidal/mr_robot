# 🤖 MR_ROBOT — Santiago Urban Navigator

Pipeline de Reinforcement Learning para entrenar una IA que navegue de forma autónoma por la red peatonal de la Región Metropolitana de Santiago, Chile.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![D3.js](https://img.shields.io/badge/D3.js-v7-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data_handler.py│────▶│  environment.py  │────▶│   trainer.py    │
│  OSMnx + Grafo  │     │  Gymnasium Env   │     │  MaskablePPO    │
│  POI Geocoding  │     │  Action Masking  │     │  Callback Logs  │
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
                                               │ Camera Follow    │
                                               └─────────────────┘
```

## ⚡ Setup Rápido

```bash
# 1. Clonar
git clone https://github.com/geoidegeoidal/mr_robot.git
cd mr_robot

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install osmnx stable-baselines3 sb3-contrib gymnasium networkx numpy pyproj

# O usando requirements.txt:
pip install -r requirements.txt
```

## 🚀 Ejecución

```bash
# Entrenar (primera vez descarga el grafo de la RM ~5-15 min)
python trainer.py

# Evaluar modelo entrenado
python trainer.py eval
```

### ¿Qué pasa la primera vez?
1. **Descarga** la red peatonal completa de la RM desde OpenStreetMap (~300-500 MB)
2. **Filtra** autopistas y aristas prohibidas
3. **Guarda** caché en `santiago_master_walk.graphml`
4. Las siguientes ejecuciones cargan desde caché (instantáneo)

## 🎮 Visualización

Después del entrenamiento, abre `visualizer.html` en un navegador:

```bash
# Opción 1: Servidor local (recomendado para CORS)
python -m http.server 8080
# Luego abrir http://localhost:8080/visualizer.html

# Opción 2: Abrir directamente (puede fallar por CORS)
start visualizer.html
```

### Controles
| Tecla/Botón | Acción |
|-------------|--------|
| ▶ PLAY | Reproducir/pausar episodio |
| ◀ / ▶ | Episodio anterior/siguiente |
| EPISODE slider | Saltar a episodio |
| SPEED slider | Velocidad de reproducción |
| ◎ FOLLOW | Toggle cámara de seguimiento |
| ★ SUCCESS | Filtrar solo episodios exitosos |

## 📁 Estructura

| Archivo | Descripción |
|---------|-------------|
| `data_handler.py` | Descarga, filtra y cachea el grafo OSMnx. Geocodifica POIs. |
| `environment.py` | Entorno Gymnasium con action masking para navegación urbana. |
| `trainer.py` | Pipeline MaskablePPO con callback de trayectorias. |
| `visualizer.html` | Motor D3.js cyberpunk con camera follow y glow effects. |
| `requirements.txt` | Dependencias Python. |

## 🧠 Diseño Técnico

### Ruta
**Bellavista → Plaza de Maipú** (~15 km a pie cruzando Santiago)

### Action Masking
SB3 requiere action space fijo. Usamos `MaskablePPO` de `sb3-contrib` con máscara booleana que habilita solo los vecinos disponibles del nodo actual.

### Recompensa
- **+1.0** por metro de acercamiento al objetivo
- **-0.1** por cada paso (fricción temporal)
- **-500** por vías prohibidas o callejones sin salida
- **+2000** al llegar al destino

### State Space
Vector normalizado `[0,1]⁴`:
1. Distancia relativa al objetivo
2. Ángulo hacia el objetivo
3. Grado del nodo actual
4. Indicador de cercanía a avenida

## 🎬 Tips para Storytelling

- Usa el botón **★ SUCCESS** para filtrar solo los episodios exitosos
- La cámara **◎ FOLLOW** crea movimientos orgánicos ideales para grabar
- Graba la pantalla directamente — el efecto cyberpunk es cinemático
- Los primeros episodios son caóticos, los últimos son quirúrgicos

## 📚 Stack

- **[OSMnx](https://github.com/gboeing/osmnx)** — Extracción de redes viales de OpenStreetMap
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** + `sb3-contrib` — RL con MaskablePPO
- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** — API estándar de entornos RL
- **[D3.js v7](https://d3js.org/)** — Visualización interactiva

---

*Built with 🧠 by MR_ROBOT Pipeline*
