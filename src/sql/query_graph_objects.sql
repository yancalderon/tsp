-- Nodes
with a as(
SELECT 
    entity_type,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 1) AS y,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 2) AS x,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 3) AS highway,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 4) AS street_count,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 5) AS ref,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 6) AS geometry
FROM 
    ef_graph_objects
WHERE
    entity_type = 'nodes')

-- Edges
SELECT 
    entity_type,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 1) AS osmid,-->
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 2) AS oneway,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 3) AS lanes,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 4) AS name,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 5) AS highway,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 6) AS maxspeed,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 7) AS reversed,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 8) AS length,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 9) AS geometry,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 10) AS access_
FROM 
    ef_graph_objects
WHERE
    entity_type = 'edges';

-- Points
SELECT 
    entity_type,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 1) AS amenity,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 2) AS name,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 3) AS religion,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 4) AS geometry,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 5) AS addr_city,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 6) AS addr_street,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 7) AS website,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 8) AS addr_housenumber,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 9) AS source,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 10) AS addr_postcode,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 11) AS nodes,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 12) AS addr_district,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 13) AS building,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 14) AS contact_facebook,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 15) AS email,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 16) AS phone,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 17) AS addr_state,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 18) AS denomination,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 19) AS source_1,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 20) AS building_levels,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 21) AS name_es,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 22) AS roof_shape,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 23) AS wikidata,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 24) AS wikimedia_commons,
    REGEXP_SUBSTR(entity_data, '[^|]+', 1, 25) AS wikipedia
FROM 
    ef_graph_objects
WHERE
    entity_type = 'points';