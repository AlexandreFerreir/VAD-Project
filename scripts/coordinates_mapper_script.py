#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geocodificador Completo de Acidentes Aéreos
Desenvolvido para: JohnyPeters
Data: 2025-04-18 18:11:54
"""

import pandas as pd
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Configurações
INPUT_FILE = "plane_crash_data.csv"  # Nome do arquivo de entrada
CACHE_FILE = "coordinates_cache.csv"  # Arquivo de cache de coordenadas
LOG_FILE = "geocoding_log.txt"  # Arquivo de log
METADATA_FILE = "geocoding_metadata.json"  # Arquivo de metadados
USER_NAME = "JohnyPeters"  # Nome do usuário
TIMESTAMP = "2025-04-18 18:11:54"  # Timestamp UTC
DELAY = 1.1  # Delay entre chamadas à API (segundos)

# Configurar logging para arquivo e console
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GeocodingScript")

def geocode_location(location, geolocator, location_dict, attempts=3, delay=DELAY):
    """
    Geocodifica uma localização com estratégia de remoção progressiva de palavras.
    Tenta remover palavras do início até conseguir geocodificar com sucesso.
    """
    if pd.isna(location) or location == "?":
        return None
    
    # Limpa o texto da localização
    location_clean = str(location).replace('"', '').strip()
    
    # Verifica se já está no cache
    if location_clean in location_dict and location_dict[location_clean] not in [None, 'None']:
        # Converte string de tupla para tupla real se necessário
        if isinstance(location_dict[location_clean], str):
            coord_str = location_dict[location_clean].replace('(', '').replace(')', '')
            if ',' in coord_str:
                try:
                    lat, lon = map(float, coord_str.split(','))
                    return (lat, lon)
                except:
                    pass
        else:
            return location_dict[location_clean]
    
    # Cria variações removendo palavras do início
    location_variations = []
    
    # Adiciona a localização original
    location_variations.append(location_clean)
    
    # Divide a localização em palavras
    words = location_clean.split()
    
    # Gera variações removendo palavras do início (mantém pelo menos 2 palavras)
    for i in range(1, len(words)-1):
        variation = ' '.join(words[i:])
        location_variations.append(variation)
    
    # Se tiver vírgula, tenta também apenas a primeira parte principal
    if "," in location_clean:
        main_part = location_clean.split(",")[0].strip()
        words_main = main_part.split()
        for i in range(1, len(words_main)):
            variation = ' '.join(words_main[i:])
            if len(variation) > 3 and variation not in location_variations:
                location_variations.append(variation)
    
    # Tenta geocodificar cada variação
    for loc_variant in location_variations:
        logger.info(f"Tentando geocodificar: '{loc_variant}' (variação de '{location_clean}')")
        
        for attempt in range(attempts):
            try:
                geocode_result = geolocator.geocode(loc_variant, exactly_one=True)
                if geocode_result:
                    coords = (geocode_result.latitude, geocode_result.longitude)
                    # Salva no cache usando a localização original
                    location_dict[location_clean] = coords
                    logger.info(f"Sucesso! '{loc_variant}' -> {coords}")
                    time.sleep(delay)
                    return coords
                time.sleep(delay)
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                logger.warning(f"Erro na tentativa {attempt+1} para '{loc_variant}': {str(e)}")
                time.sleep(delay * 2)
            except Exception as e:
                logger.error(f"Erro inesperado na geocodificação de '{loc_variant}': {str(e)}")
                time.sleep(delay)
    
    logger.warning(f"Falha em todas as variações para '{location_clean}'")
    location_dict[location_clean] = None
    return None

def parse_route_with_stops(route):
    """Extrai todos os pontos de uma rota, incluindo escalas."""
    if pd.isna(route) or route == "?" or route in ["Demonstration", "Test flight", "Air show"]:
        return None
    
    # Remove aspas e limpa
    route = str(route).replace('"', '').strip()
    
    # Verifica se há barras para rotas alternativas
    if '/' in route:
        # Pega apenas a primeira rota mencionada
        route = route.split('/')[0].strip()
    
    # Separa pontos da rota pelo hífen
    waypoints = [point.strip() for point in route.split('-')]
    
    # Filtra pontos vazios
    waypoints = [wp for wp in waypoints if wp]
    
    # Se tiver pelo menos dois pontos, é uma rota válida
    if len(waypoints) >= 2:
        return waypoints
    
    return None

def save_cache(location_dict, filename=CACHE_FILE):
    """Salva o dicionário de cache em um arquivo CSV."""
    logger.info(f"Salvando cache com {len(location_dict)} localizações em {filename}...")
    
    # Converte o dicionário para DataFrame
    cache_df = pd.DataFrame({
        'Location': list(location_dict.keys()),
        'Coordinates': list(location_dict.values())
    })
    
    # Salva no arquivo
    cache_df.to_csv(filename, index=False)
    logger.info(f"Cache salvo com sucesso!")

def main():
    # Registra horário de início
    start_time = datetime.now()
    logger.info(f"=== Iniciando processo de geocodificação em {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"Usuário: {USER_NAME}")
    logger.info(f"Timestamp de referência: {TIMESTAMP}")
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(INPUT_FILE):
        logger.error(f"ERRO: Arquivo {INPUT_FILE} não encontrado!")
        return
    
    # Inicializa o geocodificador
    user_agent = f"accident_map_geocoder_{USER_NAME.replace(' ', '_')}_{start_time.strftime('%Y%m%d')}"
    geolocator = Nominatim(user_agent=user_agent)
    logger.info(f"Geocodificador inicializado com user-agent: {user_agent}")
    
    # Carrega dados
    logger.info(f"Carregando dados de {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.replace('?', np.nan)
    logger.info(f"Carregados {len(df)} registros de acidentes")
    
    # Verifica se já temos cache de coordenadas
    location_dict = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache_df = pd.read_csv(CACHE_FILE)
            for _, row in cache_df.iterrows():
                location_dict[row['Location']] = row['Coordinates']
            logger.info(f"Cache carregado de {CACHE_FILE} com {len(location_dict)} localizações")
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")
            logger.info("Criando novo cache...")
    
    # Geocodificar localizações de acidentes
    logger.info("Geocodificando localizações de acidentes...")
    locations = []
    new_locations = False
    
    for i, location in enumerate(df['Location']):
        if i % 10 == 0 or i == len(df) - 1:
            logger.info(f"Progresso: {i+1}/{len(df)} ({((i+1)/len(df)*100):.1f}%)")
        
        if pd.isna(location) or location == "?":
            locations.append(None)
            continue
            
        location_clean = str(location).replace('"', '').strip()
        
        if location_clean in location_dict and location_dict[location_clean] not in [None, 'None']:
            # Já está no cache
            coords = location_dict[location_clean]
            if isinstance(coords, str):
                # Converter string para tupla se necessário
                coord_str = coords.replace('(', '').replace(')', '')
                if ',' in coord_str:
                    try:
                        lat, lon = map(float, coord_str.split(','))
                        locations.append((lat, lon))
                    except:
                        locations.append(None)
                else:
                    locations.append(None)
            else:
                locations.append(coords)
        else:
            # Geocodificar novo local
            coords = geocode_location(location_clean, geolocator, location_dict)
            locations.append(coords)
            new_locations = True
        
        # Salva o cache a cada 20 itens para evitar perda de dados
        if (i + 1) % 20 == 0 and new_locations:
            save_cache(location_dict)
    
    # Adiciona coordenadas à DataFrame
    df['Coordinates'] = locations
    
    # Processar rotas completas (incluindo escalas)
    logger.info("Processando rotas (incluindo escalas)...")
    routes_info = []
    routes_with_stops = 0
    all_waypoints_count = 0
    
    for i, route in enumerate(df['Route']):
        if i % 10 == 0 or i == len(df) - 1:
            logger.info(f"Progresso: {i+1}/{len(df)} ({((i+1)/len(df)*100):.1f}%)")
        
        # Extrair todos os pontos da rota
        waypoints = parse_route_with_stops(route)
        
        if waypoints:
            # Conta rotas com escalas
            if len(waypoints) > 2:
                routes_with_stops += 1
                logger.info(f"Rota com escalas ({len(waypoints)} pontos): {route}")
            
            all_waypoints_count += len(waypoints)
            
            # Geocodifica cada ponto da rota
            waypoint_coords = []
            for point in waypoints:
                coords = geocode_location(point, geolocator, location_dict)
                waypoint_coords.append(coords)
                new_locations = True
            
            # Armazena informações da rota
            routes_info.append({
                'origin': waypoints[0],
                'destination': waypoints[-1],
                'waypoints': waypoints,
                'waypoint_coords': waypoint_coords
            })
            
            # Salva o cache a cada 10 rotas
            if (i + 1) % 10 == 0 and new_locations:
                save_cache(location_dict)
        else:
            routes_info.append(None)
    
    # Salva o cache final
    if new_locations:
        save_cache(location_dict)
    
    # Salva metadados
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60.0  # em minutos
    
    metadata = {
        "process_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "process_end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_minutes": round(duration, 2),
        "user": USER_NAME,
        "timestamp": TIMESTAMP,
        "total_accidents": len(df),
        "accidents_with_coordinates": sum(1 for loc in locations if loc is not None),
        "total_routes_processed": len(routes_info),
        "routes_with_waypoints": sum(1 for r in routes_info if r is not None),
        "routes_with_stops": routes_with_stops,
        "total_waypoints": all_waypoints_count,
        "unique_locations_geocoded": len(location_dict),
        "valid_locations_geocoded": sum(1 for loc in location_dict.values() if loc is not None),
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Metadados salvos em {METADATA_FILE}")
    
    # Estatísticas finais
    valid_locations = sum(1 for loc in location_dict.values() if loc is not None)
    logger.info(f"=== Processo concluído ===")
    logger.info(f"Total de acidentes: {len(df)}")
    logger.info(f"Acidentes com coordenadas: {sum(1 for loc in locations if loc is not None)} ({((sum(1 for loc in locations if loc is not None)/len(df))*100):.1f}%)")
    logger.info(f"Rotas processadas: {sum(1 for r in routes_info if r is not None)} de {len(df)}")
    logger.info(f"Rotas com escalas: {routes_with_stops}")
    logger.info(f"Total de waypoints: {all_waypoints_count}")
    logger.info(f"Localizações únicas processadas: {len(location_dict)}")
    logger.info(f"Localizações válidas: {valid_locations} ({(valid_locations/len(location_dict)*100):.1f}%)")
    logger.info(f"Tempo total: {(end_time - start_time).total_seconds() / 60.0:.2f} minutos")
    logger.info(f"Cache salvo em: {CACHE_FILE}")
    logger.info(f"Log completo em: {LOG_FILE}")
    logger.info(f"Metadados em: {METADATA_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Processo interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())