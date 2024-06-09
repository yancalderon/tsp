# -*- coding: utf-8 -*-
"""A_star-backtracking.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hpLpzR3wQ3HZkwlKqEda-_Lo7xH9Z9Iq

1. Instalar las librerias necesarias
"""

pip install osmnx

"""2. Importar la librerias correspondientes para la ejecución del agente"""

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

"""3. Importación del mapa y puntos de interes (Para el ejemplo usaremos el distrito de Lince y sus iglesias afiliadas)"""

# Variables globales
velocidad_feligres = 10 # min/Km
tiempo_descanso_feligres_x_iglesia = 10 #min

# Definir el lugar y descargar el gráfico de red
place_name = "Lince, Lima, Peru"
tags = {'amenity': 'place_of_worship','religion': 'christian'}  # Tags para buscar iglesias

graph = ox.graph_from_place(place_name, network_type='drive')   # Grafo de una red
points = ox.features_from_place(place_name, tags)    # Puntos de interés

"""4. PRE-PROCESAMIENTO DE LOS DATOS"""

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
# Crear la matriz de distancias entre los nodos de interés usando el algoritmo A*
all_nodes = [random_node] + nodes
print("ID de Nodos de interes(Punto inicial + Iglesias):",all_nodes)

"""PRESENTAR ESTADISTICAS
Luego de finalizado el pre-procesamiento de los datos, se presentan las estadisticas.
"""

stats=ox.stats.basic_stats(graph)
for key, value in stats.items():
    print(f"{key}: {value}")
# Calcular el grado de cada nodo en el grafo
degrees = [graph.degree[node] for node in graph.nodes]

# Mostrar el histograma del grado de cada nodo
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=range(1, max(degrees) + 1), edgecolor='black')
plt.title('Histograma del grado de cada nodo')
plt.xlabel('Grado')
plt.ylabel('Número de nodos')
plt.show()

# Convertir el grafo dirigido en uno no dirigido
G_undirected = graph.to_undirected()

# Calcular el diámetro del grafo
diameter = nx.diameter(G_undirected, e=None, usebounds=False)

# Calcular el radio del grafo
radius = nx.radius(G_undirected)

# Imprimir el diámetro y el radio del grafo
print(f"El diámetro del grafo es: {diameter}")
print(f"El radio del grafo es: {radius}")

"""5. Definición de Función auxiliar para calculo de la distancia euclidiana (Heuristica)"""

# Definir el transformador para proyectar las coordenadas y poder calcular la distancia euclideana en un sistema métrico
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)

# Función heurística basada en la distancia euclidiana en coordenadas proyectadas
def euclidean_distance(node1, node2):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    x1_proj, y1_proj = transformer.transform(x1, y1)
    x2_proj, y2_proj = transformer.transform(x2, y2)
    return ((x1_proj - x2_proj) ** 2 + (y1_proj - y2_proj) ** 2) ** 0.5

"""6. ALGORITMO DE BUSQUEDA EN ESPACIO DE ESTADO: (A* + BACKTRACKING)


---


Cálculo de la distancia mas corta entre los puntos de interes (Nodo inicial + Iglesias) usando A*:

*   Definición de función A_star y uso de la libreria `networkx.a_astar_path` para calcular la ditancia mas corta entre los nodos de interes
*   Creación de matriz con distancias mas cortas entre pares de nodos de interes `dist_matrix`


"""

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
print("Matriz de distancias mas cortas entre los nodos de interes:")
print(np.array2string(dist_matrix))

"""6. ALGORITMO DE BUSQUEDA EN ESPACIO DE ESTADO: (A* + BACKTRACKING)


---


Cálculo de la distancia mas corta entre los puntos de interes (Nodo inicial + Iglesias) usando A*:

*   Calculadas los las distancias mas cortas entre los nodos de interes, procedemos a encontrar el recorrido mas corto, visitando cada punto de interes una sola vez y partiendo del punto `random_node`. Para ello usamos backtracking y expandimos todas las rutas posibles.
*   Calculo del tamaño del camino mas corto `best_length`
*   Calculo del la ruta mas corta `best_path`
"""

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

"""7. VISUALIZACION DE LOS DATOS"""

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

"""**8**. ALGORITMO DE BUSQUEDA EN ESPACIO DE ESTADO: (BFS)"""

def bfs_path(graph, start, goal):
    """

    Implementacion del algoritmo BFS.

    """

    queue = [(start, [start])]
    visited = set()
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex in visited:
            continue
        for next_vertex in set(graph[vertex]) - visited:
            if next_vertex == goal:
                return path + [next_vertex]
            else:
                queue.append((next_vertex, path + [next_vertex]))
        visited.add(vertex)

    return []

def longitud_de_ruta(graph, path):
    """

    Calculo de la longitud de una ruta.

    """
    total_distance = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        # Sumar la longitud de la arista (u, v)
        edge_data = graph.get_edge_data(u, v)
        if edge_data:  # Verificar si existe la arista (u, v)
            total_distance += edge_data[0]['length']  # Sumamos la longitud de la arista
    return total_distance

ruta = 1
total_path_bfs = list()

posicion_inicial = 263112175
posicion_inicial_base = posicion_inicial
longitud_total = 0

# implementacion de BFS

for dest_node in iglesia_nodes:
    ruta_bfs = bfs_path(graph,posicion_inicial, dest_node)
    posicion_inicial = dest_node
    total_path_bfs.append(ruta_bfs)
    len_bfs =len (ruta_bfs) -1
    ruta_longitud = longitud_de_ruta(graph,ruta_bfs)
    print(f"Ruta {ruta}: numero de aristas: {len_bfs}")
    print(f"Longitud de ruta: {ruta_longitud} m")
    ruta += 1
    longitud_total += ruta_longitud

print(f"Longitud total de ruta: {longitud_total} m")

"""8.1. Visualizacion de los datos"""

nodes_base = list(iglesia_nodes)
optimal_order = [posicion_inicial_base] + nodes_base
#optimal_order = optimal_order[0:1]
print(optimal_order)
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, show=False, close=False)



for i in range(len(optimal_order) - 1):
    path = total_path_bfs[i]
    path_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in path]
    x_coords, y_coords = zip(*path_coords)
    ax.plot(x_coords, y_coords, color='r', linewidth=2, alpha=0.7)
    ax.annotate('', xy=(x_coords[-1], y_coords[-1]), xytext=(x_coords[-2], y_coords[-2]),
                arrowprops=dict(arrowstyle='->', color='r', lw=2))

    # Calcular y mostrar la longitud del segmento
    segment_length = longitud_de_ruta(graph,path)
    mid_x = (x_coords[0] + x_coords[-1]) / 2
    mid_y = (y_coords[0] + y_coords[-1]) / 2
    ax.text(mid_x, mid_y, f'{segment_length:.1f} m', fontsize=10, color='blue')


# Añadir etiquetas a los nodos en el orden de recorrido
for idx, node in enumerate(optimal_order):
    label = str(idx)
    ax.annotate(label, xy=(graph.nodes[node]['x'], graph.nodes[node]['y']), xytext=(5, 5),
                textcoords='offset points', color='blue', fontsize=12, fontweight='bold')



plt.show()

# Obtener la ruta óptima en términos de nodos y resultados
optimal_order = [all_nodes[i] for i in best_path]
print(f"Numero de iglesias a visitar: {len(optimal_order)-1}")
print(f"Longitud total del recorrido: {longitud_total} metros")
print(f"Orden óptimo de nodos: {optimal_order}")
print(f"Velocidad de los feligres: {velocidad_feligres} min/Km")
print(f"Tiempo neto de recorrido: {longitud_total/1000 * velocidad_feligres} minutos")
print(f"Tiempo neto de recorrido: {(longitud_total/1000 * velocidad_feligres)/60} horas")
print(f"tiempo de descanso en cada iglesia: {tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Numero de iglesias descanso: {len(optimal_order)-2}")
print(f"Tiempo total de recorrido con descanso: {longitud_total/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Tiempo total de recorrido con descanso: {(longitud_total/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia)/60} horas")

"""9. ALGORITMO DE BUSQUEDA EN ESPACIO DE ESTADO: (DFS)"""

def dfs(graph, start, goal, visited=None, path=None):
    """

    Implementacion del algoritmos DFS.

    """
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == goal:
        return path.copy()

    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            result_path = dfs(graph, neighbor, goal, visited, path)
            if result_path:
                return result_path

    path.pop()
    return None

ruta = 1
total_path_dfs = list()

longitud_total = 0

for dest_node in iglesia_nodes:
    ruta_dfs = dfs(graph,posicion_inicial, dest_node)
    posicion_inicial = dest_node
    total_path_dfs.append(ruta_dfs)
    len_dfs =len (ruta_dfs) -1
    ruta_longitud = longitud_de_ruta(graph,ruta_dfs)
    print(f"Ruta {ruta}: numero de aristas: {len_dfs}")
    print(f"Longitud de ruta: {ruta_longitud} m")
    ruta += 1
    longitud_total += ruta_longitud

print(f"Longitud total de ruta: {longitud_total} m")

"""9.1 Representacion de los datos"""

nodes_base = list(iglesia_nodes)
optimal_order = [posicion_inicial_base] + nodes_base

fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(graph, ax=ax, show=False, close=False)



for i in range(len(optimal_order) - 1):
    path = total_path_dfs[i]
    path_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in path]
    x_coords, y_coords = zip(*path_coords)
    ax.plot(x_coords, y_coords, color='r', linewidth=2, alpha=0.7)
    ax.annotate('', xy=(x_coords[-1], y_coords[-1]), xytext=(x_coords[-2], y_coords[-2]),
                arrowprops=dict(arrowstyle='->', color='r', lw=2))
    segment_length = longitud_de_ruta(graph,path)
    mid_x = (x_coords[0] + x_coords[-1]) / 2
    mid_y = (y_coords[0] + y_coords[-1]) / 2
    ax.text(mid_x, mid_y, f'{segment_length:.1f} m', fontsize=10, color='blue')

# Añadir etiquetas a los nodos en el orden de recorrido
for idx, node in enumerate(optimal_order):
    label = str(idx)
    ax.annotate(label, xy=(graph.nodes[node]['x'], graph.nodes[node]['y']), xytext=(5, 5),
                textcoords='offset points', color='blue', fontsize=12, fontweight='bold')



plt.show()

# Obtener la ruta óptima en términos de nodos y resultados
optimal_order = [all_nodes[i] for i in best_path]
print(f"Numero de iglesias a visitar: {len(optimal_order)-1}")
print(f"Longitud total del recorrido: {longitud_total} metros")
print(f"Orden óptimo de nodos: {optimal_order}")
print(f"Velocidad de los feligres: {velocidad_feligres} min/Km")
print(f"Tiempo neto de recorrido: {longitud_total/1000 * velocidad_feligres} minutos")
print(f"Tiempo neto de recorrido: {(longitud_total/1000 * velocidad_feligres)/60} horas")
print(f"tiempo de descanso en cada iglesia: {tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Numero de iglesias descanso: {len(optimal_order)-2}")
print(f"Tiempo total de recorrido con descanso: {longitud_total/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Tiempo total de recorrido con descanso: {(longitud_total/1000 * velocidad_feligres + (len(optimal_order)-2) * tiempo_descanso_feligres_x_iglesia)/60} horas")

"""### 10. ALGORITMO DE BUSQUEDA EN ESPACIO DE ESTADO: (AVARA - Greedy Best First Search)"""

# Función heurística basada en la distancia euclidiana en coordenadas proyectadas
def euclidean_distance_2(node1, node2, graph):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    x1_proj, y1_proj = transformer.transform(x1, y1)
    x2_proj, y2_proj = transformer.transform(x2, y2)
    return ((x1_proj - x2_proj) ** 2 + (y1_proj - y2_proj) ** 2) ** 0.5

# Implementación del algoritmo Greedy Best First Search
def greedy_bfs(graph, start, goal, heuristic):
    queue = [(0, start)]
    visited = set()
    came_from = {start: None}

    while queue:
        _, current = heappop(queue)

        if current == goal:
            break

        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            priority = heuristic(neighbor, goal, graph)
            heappush(queue, (priority, neighbor))
            came_from[neighbor] = current

    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# Encontrar la iglesia más cercana en términos de distancia euclidiana
def find_closest_node(graph, current_node, target_nodes):
    closest_node = min(target_nodes, key=lambda node: euclidean_distance_2(current_node, node, graph))
    #print("***************************************")
    return closest_node

# Recorrer todas las iglesias utilizando el algoritmo Greedy Best First Search
def visit_all_iglesias(graph, start_node, iglesia_nodes):
    visited_path = []
    current_node = start_node
    remaining_iglesia_nodes = iglesia_nodes.copy()

    while remaining_iglesia_nodes:
        closest_node = find_closest_node(graph, current_node, remaining_iglesia_nodes)
        path = greedy_bfs(graph, current_node, closest_node, euclidean_distance_2)
        #print("*********************************************************************************")
        visited_path.extend(path[:-1])  # Agregar el camino sin el último nodo para evitar duplicados
        current_node = closest_node
        remaining_iglesia_nodes.remove(closest_node)

    visited_path.append(current_node)  # Agregar el último nodo visitado
    return visited_path

# Obtener el camino para visitar todas las iglesias
path_to_all_iglesias = visit_all_iglesias(graph, random_node, list(iglesia_nodes))

# Graficar el grafo, los nodos de las iglesias y la ruta
fig, ax = ox.plot_graph(graph, show=False, close=False)

# Graficar los nodos de las iglesias
iglesia_xs = [graph.nodes[node]['x'] for node in iglesia_nodes]
iglesia_ys = [graph.nodes[node]['y'] for node in iglesia_nodes]
ax.scatter(iglesia_xs, iglesia_ys, c='blue', s=100, zorder=5, label='Iglesias')

# Graficar el nodo aleatorio inicial
ax.scatter(graph.nodes[random_node]['x'], graph.nodes[random_node]['y'], c='red', s=100, zorder=5, label='Nodo inicial')

# Graficar la ruta para visitar todas las iglesias
ox.plot_graph_route(graph, path_to_all_iglesias, route_linewidth=2, node_size=0, bgcolor='w', ax=ax, color='green', label='Ruta para visitar todas las iglesias')

# Calcular la distancia total de la ruta
total_distance = 0
for i in range(len(path_to_all_iglesias) - 1):
    u = path_to_all_iglesias[i]
    v = path_to_all_iglesias[i + 1]
    # Sumar la longitud de la arista (u, v)
    edge_data = graph.get_edge_data(u, v)
    if edge_data:  # Verificar si existe la arista (u, v)
        total_distance += edge_data[0]['length']  # Sumamos la longitud de la arista

# Imprimir la distancia total de la ruta
print(f"La distancia total de la ruta es: {total_distance} metros")
print(f"Ruta de vistia a las iglesias: {path_to_all_iglesias}")
plt.show()

# Obtener la ruta óptima en términos de nodos y resultados
path_to_all_iglesias = [all_nodes[i] for i in best_path]
print(f"Numero de iglesias a visitar: {len(path_to_all_iglesias)-1}")
print(f"Longitud total del recorrido: {total_distance} metros")
print(f"Orden óptimo de nodos: {path_to_all_iglesias}")
print(f"Velocidad de los feligres: {velocidad_feligres} min/Km")
print(f"Tiempo neto de recorrido: {total_distance/1000 * velocidad_feligres} minutos")
print(f"Tiempo neto de recorrido: {(total_distance/1000 * velocidad_feligres)/60} horas")
print(f"tiempo de descanso en cada iglesia: {tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Numero de iglesias descanso: {len(path_to_all_iglesias)-2}")
print(f"Tiempo total de recorrido con descanso: {total_distance/1000 * velocidad_feligres + (len(path_to_all_iglesias)-2) * tiempo_descanso_feligres_x_iglesia} minutos")
print(f"Tiempo total de recorrido con descanso: {(total_distance/1000 * velocidad_feligres + (len(path_to_all_iglesias)-2) * tiempo_descanso_feligres_x_iglesia)/60} horas")