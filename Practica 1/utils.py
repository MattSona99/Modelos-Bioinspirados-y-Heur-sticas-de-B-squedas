import math

def distancia_manhattan_km(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radio de la Tierra en km
    dlat_rad = math.radians(abs(lat2 - lat1))
    dlon_rad = math.radians(abs(lon2 - lon1))
    lat_media_rad = math.radians((lat1 + lat2) / 2.0)
    dist_norte_sur = R * dlat_rad
    dist_este_oeste = R * dlon_rad * math.cos(lat_media_rad)
    
    distancia_total = dist_norte_sur + dist_este_oeste
    return distancia_total

def calcular_entropia_total(b, c):
    # b: número de bicicletas en cada estación
    # c: capacidad máxima de cada estación
    entropia_total = 0.0
    for bi, ci in zip(b, c):
        # Si la estación está vacía o llena, la entropía es 0
        if bi == 0 or bi == ci:
            continue
        p = bi / ci
        h_i = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        entropia_total += h_i
    return entropia_total