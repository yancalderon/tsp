import osmnx as ox
import numpy as np
import cx_Oracle

##################################################################
# Obtener los objetos del grafo planar
##################################################################
# 1. Definir el lugar y descargar el grafo de la red de carreteras
place_name = "Lince, Lima, Peru"
G = ox.graph_from_place(place_name, network_type='drive')

# Definir los tags para buscar iglesias
tags = {'amenity': 'place_of_worship', 'religion': 'christian'}

# Obtener los puntos de interés (iglesias) que coinciden con los tags
points = ox.features_from_place(place_name, tags)

# Convertir los nodos a un DataFrame
nodes, edges = ox.graph_to_gdfs(G)

# Obtener la lista de nodos y crear un diccionario para mapear nodos a índices
nodes_from_list = list(G.nodes)
nodes_from_list_indices = {node: i for i, node in enumerate(nodes)}

##################################################################
# Conexion a la base de datos Oracle
##################################################################
# Conectar a la base de datos Oracle
dsn_tns = cx_Oracle.makedsn('localhost', '1521', service_name='xe')
conn = cx_Oracle.connect(user='usrpreprod', password='usrpreprod', dsn=dsn_tns)
cursor = conn.cursor()

##################################################################
# Carga objetos a la base de datos Oracle
##################################################################
# Función para convertir un DataFrame a una cadena con campos separados por '|'
def df_to_string(df):
    return df.apply(lambda row: '|'.join(row.astype(str)), axis=1).tolist()

points_data = df_to_string(points)
for row in points_data:
    cursor.execute("""
        INSERT INTO usrpreprod.ef_graph_objects (entity_type, entity_data)
        VALUES (:1, :2)
    """, ("points", row))

# Insertar datos de nodes
nodes_data = df_to_string(nodes)
for row in nodes_data:
    cursor.execute("""
        INSERT INTO usrpreprod.ef_graph_objects (entity_type, entity_data)
        VALUES (:1, :2)
    """, ("nodes", row))

# Insertar datos de edges
edges_data = df_to_string(edges)
for row in edges_data:
    cursor.execute("""
        INSERT INTO ef_graph_objects (entity_type, entity_data)
        VALUES (:1, :2)
    """, ("edges", row))

# Confirmar los cambios
conn.commit()

# Cerrar cursor y conexión
cursor.close()
conn.close()