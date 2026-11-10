import pandas as pd
import heapq
import timeit
import itertools

# 1. Crear el DataFrame con los datos de los jugadores
data = [
    {"name": "Ortega", "pos": "POR", "price": 9, "points": 5.8},
    {"name": "Lunin", "pos": "POR", "price": 6, "points": 4.6},
    {"name": "Remiro", "pos": "POR", "price": 4, "points": 3.9},
    {"name": "Koundé", "pos": "DEF", "price": 12, "points": 6.4},
    {"name": "Mingueza", "pos": "DEF", "price": 7, "points": 5.1},
    {"name": "Vivian", "pos": "DEF", "price": 5, "points": 4.2},
    {"name": "Huijsen", "pos": "DEF", "price": 9, "points": 5.4},
    {"name": "Cucurella", "pos": "DEF", "price": 11, "points": 5.9},
    {"name": "Lejeune", "pos": "DEF", "price": 3, "points": 3.1},
    {"name": "Mosquera", "pos": "DEF", "price": 6, "points": 4.5},
    {"name": "Pedri", "pos": "MED", "price": 18, "points": 8.1},
    {"name": "Baena", "pos": "MED", "price": 13, "points": 6.9},
    {"name": "Sučić", "pos": "MED", "price": 8, "points": 5.2},
    {"name": "Barrios", "pos": "MED", "price": 10, "points": 5.9},
    {"name": "Fermín", "pos": "MED", "price": 11, "points": 6.2},
    {"name": "Gaya", "pos": "MED", "price": 4, "points": 3.3},
    {"name": "Arambarri", "pos": "MED", "price": 6, "points": 4.4},
    {"name": "Mbappé", "pos": "DEL", "price": 24, "points": 9.6},
    {"name": "Budimir", "pos": "DEL", "price": 9, "points": 6.0},
    {"name": "Oyarzabal", "pos": "DEL", "price": 14, "points": 7.2},
    {"name": "Iglesias", "pos": "DEL", "price": 7, "points": 4.8},
]

df = pd.DataFrame(data)

# Parámetros del problema
quotas = {"POR": 1, "DEF": 4, "MED": 4, "DEL": 2}
MAX_BUDGET = 100
TOTAL_PLAYERS = 11

def calculate_upper_bound(df_players, index, current_points, counts):
    """Calcula la cota superior optimista para un nodo usando el DataFrame."""
    bound = current_points
    needed = {pos: quotas[pos] - counts.get(pos, 0) for pos in quotas}
    remaining_df = df_players.iloc[index:]
    
    for pos, n in needed.items():
        if n > 0:
            top_points = remaining_df[remaining_df["pos"] == pos]["points"].nlargest(n)
            bound += top_points.sum()
            
    return bound

def branch_and_bound(df_players):
    initial_counts = {pos: 0 for pos in quotas}
    initial_bound = calculate_upper_bound(df_players, 0, 0, initial_counts)
    
    # Contador único para evitar comparaciones entre diccionarios en el heap
    counter = itertools.count()
    
    # Estructura: (-bound, -current_points, counter, index, cost, points, counts, team_indices)
    pq = [(-initial_bound, 0, next(counter), 0, 0, 0, initial_counts, [])]
    
    best_points = -1.0
    best_team_indices = []
    nodes_opened = 0
    total_rows = len(df_players)
    
    while pq:
        neg_bound, neg_pts, _, index, cost, points, counts, team_indices = heapq.heappop(pq)
        current_bound = -neg_bound
        
        nodes_opened += 1
        
        # Poda por cota
        if current_bound <= best_points:
            continue
            
        # Si ya completamos el once ideal (11 jugadores)
        if len(team_indices) == TOTAL_PLAYERS:
            if points > best_points:
                best_points = points
                best_team_indices = team_indices
            continue
            
        # Si se acabaron los jugadores a evaluar
        if index >= total_rows:
            continue
            
        player = df_players.iloc[index]
        pos = player["pos"]
        price = player["price"]
        pts = player["points"]
        
        # RAMA 1: Fichar al jugador actual
        if counts.get(pos, 0) < quotas[pos] and cost + price <= MAX_BUDGET:
            new_counts = counts.copy()
            new_counts[pos] += 1
            new_cost = cost + price
            new_points = points + pts
            new_team = team_indices + [index]
            
            new_bound = calculate_upper_bound(df_players, index + 1, new_points, new_counts)
            if new_bound > best_points:
                heapq.heappush(
                    pq, 
                    (-new_bound, -new_points, next(counter), index + 1, new_cost, new_points, new_counts, new_team)
                )
                
        # RAMA 2: No fichar al jugador actual
        remaining_slots = TOTAL_PLAYERS - len(team_indices)
        remaining_available = total_rows - (index + 1)
        
        if remaining_available >= remaining_slots:
            new_bound = calculate_upper_bound(df_players, index + 1, points, counts)
            if new_bound > best_points:
                heapq.heappush(
                    pq, 
                    (-new_bound, -points, next(counter), index + 1, cost, points, counts, team_indices)
                )
                
    best_team_df = df_players.iloc[best_team_indices].copy()
    return best_team_df, best_points, nodes_opened

# 3. Ejecución y medición de rendimiento
if __name__ == "__main__":
    start_time = timeit.default_timer()
    best_team_df, max_pts, nodes = branch_and_bound(df)
    elapsed_time = timeit.default_timer() - start_time
    
    print("=== ONCE IDEAL SELECCIONADO (BRANCH AND BOUND) ===")
    print(best_team_df[["name", "pos", "price", "points"]].to_string(index=False))
    print("\n--- RESUMEN ---")
    print(f"Gasto Total: {best_team_df['price'].sum()} M€ / {MAX_BUDGET} M€")
    print(f"Puntos Totales Esperados: {max_pts:.2f}")
    print(f"Nodos Abiertos: {nodes}")
    print(f"Tiempo de Ejecución: {elapsed_time:.6f} segundos")