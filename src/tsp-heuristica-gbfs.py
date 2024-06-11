import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush
from pyproj import Transformer

# Definir el lugar y descargar el gráfico de red
place_name = "Lince, Lima, Peru"
tags = {'amenity': 'place_of_worship','religion': 'christian'}  # Tags para buscar iglesias

graph = ox.graph_from_place(place_name, network_type='drive')   # Grafo de una red
points = ox.features_from_place(place_name, tags)    # Puntos de interés

#/*********************PREPROCESAMIENTO DE LOS DATOS**************************************************/
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
iglesia_nodes = set(nodes)      # Nodos de iglesias
no_iglesia_nodes = list(all_nodes - iglesia_nodes)  # Solo nodos donde no sea iglesia

random_node = 263112175 #random.choice(no_iglesia_nodes) #263112013 

# Definir el transformador para proyectar las coordenadas y poder calcular la distancia euclideana en un sistema métrico
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)

# Función heurística basada en la distancia euclidiana en coordenadas proyectadas
def euclidean_distance(node1, node2, graph):
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
    closest_node = min(target_nodes, key=lambda node: euclidean_distance(current_node, node, graph))
    #print("***************************************")
    return closest_node

# Recorrer todas las iglesias utilizando el algoritmo Greedy Best First Search
def visit_all_iglesias(graph, start_node, iglesia_nodes):
    visited_path = []
    current_node = start_node
    remaining_iglesia_nodes = iglesia_nodes.copy()
    
    while remaining_iglesia_nodes:
        closest_node = find_closest_node(graph, current_node, remaining_iglesia_nodes)
        path = greedy_bfs(graph, current_node, closest_node, euclidean_distance)
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

stats=ox.stats.basic_stats(graph)
print(stats)

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

plt.show()
