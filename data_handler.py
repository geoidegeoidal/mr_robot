"""
data_handler.py — Módulo 1-2: Gestión de Datos, Grafo de Persistencia y Resolución de POI.

Descarga, filtra, cachea y proyecta la red peatonal de la Región Metropolitana
de Santiago (Chile) usando OSMnx. Proporciona geocodificación de POIs y
resolución a nodos del grafo.

Autor: MR_ROBOT Pipeline
"""

import os
import json
import logging
import osmnx as ox
import networkx as nx
import numpy as np
from pyproj import Transformer

# ─────────────────────────── Config ───────────────────────────

GRAPH_FILENAME = "santiago_master_walk.graphml"
STREETS_FILENAME = "streets.geojson"
PLACE_NAME = "Región Metropolitana de Santiago, Chile"
TARGET_CRS = "EPSG:32719"  # UTM 19S

# Highway types prohibidos para peatones (autopistas)
FORBIDDEN_HIGHWAYS = {"motorway", "trunk", "motorway_link", "trunk_link"}

# Highway types que indican avenidas/calles principales
AVENUE_HIGHWAYS = {"primary", "secondary", "tertiary",
                   "primary_link", "secondary_link", "tertiary_link"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────── Módulo 1: Grafo ──────────────────────

def _filter_forbidden_edges(G):
    """Elimina aristas con highway tags de tipo autopista/trunk."""
    edges_to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get("highway", "")
        # highway puede ser string o lista
        if isinstance(highway, list):
            hw_set = set(highway)
        else:
            hw_set = {highway}
        if hw_set & FORBIDDEN_HIGHWAYS:
            edges_to_remove.append((u, v, k))

    G.remove_edges_from(edges_to_remove)
    logger.info(f"Eliminadas {len(edges_to_remove)} aristas de autopistas/trunk.")
    return G


def load_or_download_graph(filepath=None):
    """
    Busca el archivo .graphml en disco. Si no existe, descarga la red
    peatonal de la RM completa, la filtra y la guarda en caché.

    Returns:
        G_projected: MultiDiGraph proyectado a UTM 19S (metros)
        G_latlon: MultiDiGraph en WGS84 (para exportar coords lat/lon)
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), GRAPH_FILENAME)

    if os.path.exists(filepath):
        logger.info(f"Cargando grafo desde caché: {filepath}")
        G = ox.load_graphml(filepath)
    else:
        logger.info(f"Descargando red peatonal de '{PLACE_NAME}'...")
        logger.info("Esto puede tardar 5-15 minutos la primera vez.")
        G = ox.graph_from_place(PLACE_NAME, network_type="walk")
        logger.info(f"Grafo descargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")

        # Filtrar autopistas
        G = _filter_forbidden_edges(G)

        # Conservar solo el componente conectado más grande
        G = ox.truncate.largest_component(G, strongly=False)
        logger.info(f"Componente conectado más grande: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")

        # Guardar caché
        ox.save_graphml(G, filepath=filepath)
        logger.info(f"Grafo guardado en: {filepath}")

    # Mantener copia en lat/lon para exportar coordenadas
    G_latlon = G.copy()

    # Proyectar a UTM 19S
    G_projected = ox.project_graph(G, to_crs=TARGET_CRS)
    logger.info(f"Grafo proyectado a {TARGET_CRS}.")

    return G_projected, G_latlon


def compute_max_degree(G):
    """Calcula el grado máximo del grafo (para definir action space fijo)."""
    max_deg = max(dict(G.degree()).values())
    logger.info(f"Grado máximo del grafo: {max_deg}")
    return max_deg


def get_forbidden_highways():
    """Retorna el set de highway types prohibidos."""
    return FORBIDDEN_HIGHWAYS.copy()


def get_avenue_highways():
    """Retorna el set de highway types de avenidas."""
    return AVENUE_HIGHWAYS.copy()


# ────────────────────── Módulo 2: POI ─────────────────────────

def get_route_nodes(start_poi, end_poi, G_projected, G_latlon):
    """
    Geocodifica dos POIs y encuentra los nodos más cercanos en el grafo.

    Args:
        start_poi: Nombre del lugar de origen (str)
        end_poi: Nombre del lugar de destino (str)
        G_projected: Grafo proyectado a UTM (para cálculos de distancia)
        G_latlon: Grafo en WGS84 (para geocoding con nearest_nodes)

    Returns:
        (start_node, end_node): Tupla de node IDs
    """
    # Geocodificar POIs
    logger.info(f"Geocodificando origen: '{start_poi}'")
    start_coords = ox.geocoder.geocode(start_poi)
    logger.info(f"  → Coordenadas: {start_coords}")

    logger.info(f"Geocodificando destino: '{end_poi}'")
    end_coords = ox.geocoder.geocode(end_poi)
    logger.info(f"  → Coordenadas: {end_coords}")

    # Encontrar nodos más cercanos en el grafo lat/lon
    start_node = ox.distance.nearest_nodes(G_latlon, X=start_coords[1], Y=start_coords[0])
    end_node = ox.distance.nearest_nodes(G_latlon, X=end_coords[1], Y=end_coords[0])

    logger.info(f"Nodo origen: {start_node}")
    logger.info(f"Nodo destino: {end_node}")

    # Validar conectividad
    if not nx.has_path(G_projected, start_node, end_node):
        raise ValueError(
            f"No existe ruta entre {start_poi} (nodo {start_node}) "
            f"y {end_poi} (nodo {end_node}) en el grafo."
        )
    logger.info("✓ Ruta alcanzable confirmada.")

    return start_node, end_node


# ──────────────── Export para Visualización ────────────────────

def export_network_geojson(G_latlon, filepath=None, bbox=None):
    """
    Exporta las aristas del grafo como GeoJSON para el visualizador D3.js.

    Args:
        G_latlon: Grafo en WGS84
        filepath: Ruta de salida
        bbox: Opcional (min_lat, min_lon, max_lat, max_lon) para recortar
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), STREETS_FILENAME)

    logger.info("Exportando red vial como GeoJSON...")

    # Obtener GeoDataFrame de aristas
    gdf_edges = ox.graph_to_gdfs(G_latlon, nodes=False, edges=True)

    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        gdf_edges = gdf_edges.cx[min_lon:max_lon, min_lat:max_lat]
        logger.info(f"Filtrado por bbox: {len(gdf_edges)} aristas.")

    # Simplificar: solo geometría y highway tag
    cols_to_keep = ["geometry"]
    if "highway" in gdf_edges.columns:
        cols_to_keep.append("highway")
    gdf_edges = gdf_edges[cols_to_keep].copy()

    # Exportar
    gdf_edges.to_file(filepath, driver="GeoJSON")
    logger.info(f"GeoJSON exportado: {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")
    return filepath


def export_route_geojson(G_latlon, route_nodes, filepath=None):
    """
    Exporta los nodos de una ruta como GeoJSON con sus coordenadas lat/lon
    para usarlos en el visualizador. Útil para exportar el camino óptimo
    calculado por nx.shortest_path como referencia.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "optimal_route.geojson")

    coords = []
    for node_id in route_nodes:
        node_data = G_latlon.nodes[node_id]
        coords.append([node_data["x"], node_data["y"]])  # lon, lat

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {"type": "optimal_route", "nodes": len(route_nodes)}
        }]
    }

    with open(filepath, "w") as f:
        json.dump(geojson, f)

    logger.info(f"Ruta óptima exportada: {filepath} ({len(route_nodes)} nodos)")
    return filepath


# ──────────────── Utilidades de coordenadas ────────────────────

def create_utm_to_latlon_transformer():
    """Crea un transformer de UTM 19S a WGS84."""
    return Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


def utm_to_latlon(x, y, transformer):
    """Convierte coordenadas UTM 19S a (lat, lon)."""
    lon, lat = transformer.transform(x, y)
    return lat, lon


# ──────────────────────── CLI Test ─────────────────────────────

if __name__ == "__main__":
    logger.info("=== Test de data_handler.py ===")
    G_proj, G_ll = load_or_download_graph()
    max_deg = compute_max_degree(G_proj)
    logger.info(f"Nodos: {G_proj.number_of_nodes()}, Aristas: {G_proj.number_of_edges()}")
    logger.info(f"Max degree: {max_deg}")
