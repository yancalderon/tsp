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

inicio_preprocesamiento = time.time()

# Variables globales
velocidad_feligres = 10 # min/Km
tiempo_descanso_feligres_x_iglesia = 10 #min

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
all_nodes = [random_node] + nodes

fin_preprocesamiento = time.time()
print(f"Tiempo de preprocesamiento: {fin_preprocesamiento - inicio_preprocesamiento:.2f} segundos")

######################################################################################################
# Calculo de la distancia euclidiana (Euristica)
######################################################################################################
inicio_a_star_backtracking = time.time()
# Definir el transformador para proyectar las coordenadas y poder calcular la distancia euclideana en un sistema métrico
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)

# Función heurística basada en la distancia euclidiana en coordenadas proyectadas
def euclidean_distance(node1, node2):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    x1_proj, y1_proj = transformer.transform(x1, y1)
    x2_proj, y2_proj = transformer.transform(x2, y2)
    return ((x1_proj - x2_proj) ** 2 + (y1_proj - y2_proj) ** 2) ** 0.5

######################################################################################################
# ALGORTIMO DE BUSQUEDA A* y BACKTRACKING para calculo de la ruta mas corta
######################################################################################################
# Implementar el algoritmo A* para encontrar la ruta más corta entre dos nodos
def astar_path(G, source, target, heuristic=euclidean_distance, weight='length'):
    return nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)

# Crear la matriz de distancias entre los nodos de interés usando el algoritmo A*
dist_matrix = np.zeros((len(all_nodes), len(all_nodes)))
#print(dist_matrix)
for i, node1 in enumerate(all_nodes):
    for j, node2 in enumerate(all_nodes):
        if i != j:
            dist_matrix[i, j] = nx.astar_path_length(graph, node1, node2, heuristic=euclidean_distance, weight='length')
        else:
            dist_matrix[i, j] = np.inf  # Reemplazar ceros de la diagonal con infinito            

# Implementación del algoritmo de búsqueda en el espacio de estados
def tsp_backtracking(curr_node, visited, curr_length, best_length, best_path):
    if len(visited) == len(nodes) + 1:  # Todos los nodos visitados (nodos + nodo inicial)
        if curr_length < best_length[0]:
            best_length[0] = curr_length
            best_path[:] = visited[:]
        return

    for next_node in range(len(all_nodes)):
        if next_node not in visited:
            next_length = curr_length + dist_matrix[curr_node][next_node]
            if next_length < best_length[0]:
                visited.append(next_node)
                tsp_backtracking(next_node, visited, next_length, best_length, best_path)
                visited.pop()

best_length = [float('inf')]
best_path = []
tsp_backtracking(0, [0], 0, best_length, best_path)

# Obtener la ruta óptima en términos de nodos y resultados
optimal_order = [all_nodes[i] for i in best_path]
print(f"Numero de iglesias a visitar: {len(optimal_order)-1}")
print(f"Longitud total del recorrido: {best_length[0]} metros")
print(f"Orden óptimo de nodos: {optimal_order}")
print(f"Velocidad de los feligres: {velocidad_feligres} min/Km")
print(f"Tiempo neto de recorrido: {best_length[0]/1000 * velocidad_feligres} minutos")
print(f"Tiempo neto de recorrido: {(best_length[0]/1000 * velocidad_feligres)/60} horas")
print(f"tiempo de descanso en cada iglesia: {tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Numero de iglesias descanso: {len(optimal_order)-2}")
print(f"Tiempo total de recorrido con descanso: {best_length[0]/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Tiempo total de recorrido con descanso: {(best_length[0]/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia)/60} horas")

fin_a_star_backtracking = time.time()
print(f"Tiempo de ejecución del algortimo A* y Backtracking: {fin_a_star_backtracking - inicio_a_star_backtracking:.2f} segundos")

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

    # Calcular y mostrar la longitud del segmento
    segment_length = nx.astar_path_length(graph, optimal_order[i], optimal_order[i + 1], heuristic=euclidean_distance, weight='length')
    mid_x = (x_coords[0] + x_coords[-1]) / 2
    mid_y = (y_coords[0] + y_coords[-1]) / 2
    ax.text(mid_x, mid_y, f'{segment_length:.1f} m', fontsize=10, color='blue')

# Añadir etiquetas a los nodos en el orden de recorrido
for idx, node in enumerate(optimal_order):
    label = str(idx)
    ax.annotate(label, xy=(graph.nodes[node]['x'], graph.nodes[node]['y']), xytext=(5, 5),
                textcoords='offset points', color='blue', fontsize=12, fontweight='bold')

plt.show()