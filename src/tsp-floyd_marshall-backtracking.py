import time
import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush
from pyproj import Transformer
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
from itertools import permutations

inicio_preprocesamiento = time.time()
# Definir el lugar y descargar el gráfico de red
place_name = "Lince, Lima, Peru"
tags = {'amenity': 'place_of_worship','religion': 'christian'}  # Tags para buscar iglesias

graph = ox.graph_from_place(place_name, network_type='drive')   # Grafo de una red
points = ox.features_from_place(place_name, tags)    # Puntos de interés

######################################################################################################
# PREPROCESAMIENTO DE LOS DATOS
######################################################################################################
points = points.dropna(subset=['amenity']) #se eliminan los registros que no tienen valor en la columna amenity
points = points.dropna(subset=['religion']) #se eliminan los registros que no tienen valor en la columna religion
points = points.dropna(subset=['name']) #se eliminan los registros que no tienen valor en la columna name

# Encontrar y eliminar nodos sin salida
dead_ends = [node for node, degree in graph.degree() if degree == 1]
graph.remove_nodes_from(dead_ends)

# Encontrar y eliminar nodos aislados
isolated_nodes = list(nx.isolates(graph))
graph.remove_nodes_from(isolated_nodes)

# Encontrar el GRAFO principal (excluye los componentes desconectados)
largest_component = max(nx.strongly_connected_components(graph), key=len)
graph = graph.subgraph(largest_component).copy()

# Encontrar y eliminar las rutas cerradas (se originan en una ubicación y terminan en la misma ubicación)
nodos_autoreferenciados = list(nx.selfloop_edges(graph))
graph.remove_nodes_from(nodos_autoreferenciados)

#/***********************************************************************/

# Coordenadas de las iglesias
coords = []
for point in points.itertuples():
    if point.geometry.geom_type == 'Point':
        coords.append((point.geometry.y, point.geometry.x))
    elif point.geometry.geom_type in ['Polygon', 'MultiPolygon']:
        coords.append((point.geometry.centroid.y, point.geometry.centroid.x))

# Nodo más cercano a cada iglesia
nodes = [ox.distance.nearest_nodes(graph, point[1], point[0]) for point in coords]
nodes = [node for node in nodes if graph.degree[node] > 1]  # Excluir nodos con solo un camino

# Extraer un nodo como punto de partida
all_nodes = set(graph.nodes)    # Nodos del mapa
#print("all_nodos:",all_nodes)
iglesia_nodes = set(nodes)      # Nodos de iglesias
#print("iglesia_nodes:",iglesia_nodes)
no_iglesia_nodes = list(all_nodes - iglesia_nodes)  # Solo nodos donde no sea iglesia
#print("no_iglesia_nodes",no_iglesia_nodes)


random_node = 263112175 #random.choice(no_iglesia_nodes) #263112013

######################################################################################################
# Funciones utilitarias
######################################################1################################################
# Aplicar el algoritmo de Floyd-Warshall utilizando el atributo 'length' como peso
dist = dict(nx.floyd_warshall(graph, weight='length'))

# Asegurarse de que random_node esté en nodes
if random_node not in nodes:
    nodes.append(random_node)

# Crear un índice para cada nodo en nodes
node_indices = {node: i for i, node in enumerate(nodes)}
n = len(nodes)

# Crear una matriz de distancias específica para los nodos de interés
dist_matrix = [[0] * n for _ in range(n)]

for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i != j:
            dist_matrix[i][j] = dist[node1][node2]

# Función para resolver el problema del viajante utilizando búsqueda en el espacio de estados
def tsp_space_search(dist_matrix, start_index):
    n = len(dist_matrix)
    visited_all = (1 << n) - 1
    
    # Cache para la memoización
    cache = {}
    path_cache = {}
    
    def search(mask, pos):
        # Si todos los nodos han sido visitados, no necesitamos regresar al inicio
        if mask == visited_all:
            return 0, []
        
        # Si el resultado ya está en el cache
        if (mask, pos) in cache:
            return cache[(mask, pos)], path_cache[(mask, pos)]
        
        ans = float('inf')
        best_path = []
        # Probar ir a cada nodo no visitado
        for nxt in range(n):
            if mask & (1 << nxt) == 0:
                new_dist, new_path = search(mask | (1 << nxt), nxt)
                new_dist += dist_matrix[pos][nxt]
                if new_dist < ans:
                    ans = new_dist
                    best_path = [nxt] + new_path
        
        # Guardar en el cache
        cache[(mask, pos)] = ans
        path_cache[(mask, pos)] = best_path
        return ans, best_path
    
    total_dist, path = search(1 << start_index, start_index)
    optimal_path = [start_index] + path
    return total_dist, optimal_path

# Encontrar el índice del nodo random_node en la lista de nodes
start_index = node_indices[random_node]

# Resolver el TSP comenzando desde random_node
best_distance, best_path = tsp_space_search(dist_matrix, start_index)

# Convertir los índices de vuelta a nodos
best_path_nodes = [nodes[i] for i in best_path]

print("La mejor ruta para visitar todos los nodos de interés es:", best_path_nodes)
print("La distancia total de esta ruta es:", best_distance)

optimal_order = best_path_nodes



######################################################################################################
# Visualización de los datos
######################################################################################################
# Visualizar la ruta óptima en el grafo
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, show=False, close=False)

# Dibujar la ruta óptima
for i in range(len(optimal_order) - 1):
    path = nx.shortest_path(graph, optimal_order[i], optimal_order[i + 1], weight='length')
    path_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in path]
    x_coords, y_coords = zip(*path_coords)
    ax.plot(x_coords, y_coords, color='r', linewidth=2, alpha=0.7)
    ax.annotate('', xy=(x_coords[-1], y_coords[-1]), xytext=(x_coords[-2], y_coords[-2]),
                arrowprops=dict(arrowstyle='->', color='r', lw=2))

# Añadir etiquetas a los nodos en el orden de recorrido
for idx, node in enumerate(optimal_order):
    label = str(idx)
    ax.annotate(label, xy=(graph.nodes[node]['x'], graph.nodes[node]['y']), xytext=(5, 5),
                textcoords='offset points', color='blue', fontsize=12, fontweight='bold')

plt.show()

