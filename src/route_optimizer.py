"""
Route Optimization Module
Implements various algorithms for solving Vehicle Routing Problem (VRP) and Traveling Salesman Problem (TSP).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import networkx as nx
from pulp import *
import itertools
from dataclasses import dataclass
import heapq
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

@dataclass
class Route:
    """Represents a delivery route for a vehicle."""
    vehicle_id: str
    stops: List[int]
    total_distance: float
    total_time: float
    total_cost: float
    load_weight: float
    load_volume: float

class RouteOptimizer:
    def __init__(self, distance_matrix: np.ndarray, delivery_points: pd.DataFrame, 
                 vehicles: pd.DataFrame, traffic_data: Dict = None):
        """Initialize the route optimizer with data."""
        self.distance_matrix = distance_matrix
        self.delivery_points = delivery_points
        self.vehicles = vehicles
        self.traffic_data = traffic_data or {}
        self.n_points = len(delivery_points)
        self.n_vehicles = len(vehicles)
        
    def dijkstra_shortest_path(self, start: int, end: int, traffic_period: str = 'midday') -> Tuple[List[int], float]:
        """
        Find shortest path between two points using Dijkstra's algorithm.
        Considers traffic conditions if available.
        """
        # Create weighted adjacency matrix
        weights = self.distance_matrix.copy()
        
        if traffic_period in self.traffic_data:
            weights = weights * self.traffic_data[traffic_period]
        
        # Create graph
        G = nx.from_numpy_array(weights)
        
        try:
            path = nx.shortest_path(G, start, end, weight='weight')
            distance = nx.shortest_path_length(G, start, end, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return [start, end], weights[start][end]
    
    def solve_tsp_nearest_neighbor(self, points_subset: List[int]) -> Tuple[List[int], float]:
        """
        Solve TSP using nearest neighbor heuristic.
        Good for small to medium sized problems.
        """
        if len(points_subset) <= 1:
            return points_subset, 0.0
        
        unvisited = set(points_subset[1:])  # Start from first point
        current = points_subset[0]
        tour = [current]
        total_distance = 0.0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            total_distance += self.distance_matrix[current][nearest]
            current = nearest
            tour.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_distance += self.distance_matrix[current][points_subset[0]]
        
        return tour, total_distance
    
    def solve_tsp_genetic_algorithm(self, points_subset: List[int], 
                                  population_size: int = 100, generations: int = 500) -> Tuple[List[int], float]:
        """
        Solve TSP using genetic algorithm for better solutions.
        """
        if len(points_subset) <= 3:
            return self.solve_tsp_nearest_neighbor(points_subset)
        
        def calculate_tour_distance(tour):
            distance = 0
            for i in range(len(tour)):
                distance += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            return distance
        
        def create_individual():
            tour = points_subset.copy()
            np.random.shuffle(tour)
            return tour
        
        def crossover(parent1, parent2):
            # Order crossover (OX)
            size = len(parent1)
            start, end = sorted(np.random.choice(size, 2, replace=False))
            
            child = [-1] * size
            child[start:end] = parent1[start:end]
            
            pointer = end
            for city in parent2[end:] + parent2[:end]:
                if city not in child:
                    child[pointer % size] = city
                    pointer += 1
            
            return child
        
        def mutate(individual, mutation_rate=0.01):
            if np.random.random() < mutation_rate:
                i, j = np.random.choice(len(individual), 2, replace=False)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        best_tour = None
        best_distance = float('inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(tour, calculate_tour_distance(tour)) for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            # Track best solution
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0].copy()
            
            # Selection (top 50%)
            survivors = [tour for tour, _ in fitness_scores[:population_size // 2]]
            
            # Generate new population
            new_population = survivors.copy()
            
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(survivors), 2, replace=False)
                child = crossover(survivors[parent1], survivors[parent2])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_tour, best_distance
    
    def solve_vrp_clarke_wright(self) -> List[Route]:
        """
        Solve Vehicle Routing Problem using Clarke-Wright Savings Algorithm.
        """
        # Calculate savings matrix
        depot = 0  # Assume first point is depot
        n = self.n_points
        savings = np.zeros((n, n))
        
        for i in range(1, n):
            for j in range(i + 1, n):
                savings[i][j] = (self.distance_matrix[depot][i] + 
                               self.distance_matrix[depot][j] - 
                               self.distance_matrix[i][j])
                savings[j][i] = savings[i][j]
        
        # Create list of savings with customer pairs
        savings_list = []
        for i in range(1, n):
            for j in range(i + 1, n):
                if savings[i][j] > 0:
                    savings_list.append((savings[i][j], i, j))
        
        # Sort by savings in descending order
        savings_list.sort(reverse=True)
        
        # Initialize routes (each customer in its own route)
        routes = {i: [i] for i in range(1, n)}
        route_loads_weight = {i: self.delivery_points.iloc[i]['package_weight'] for i in range(1, n)}
        route_loads_volume = {i: self.delivery_points.iloc[i]['package_volume'] for i in range(1, n)}
        
        # Merge routes based on savings
        for saving, i, j in savings_list:
            route_i = None
            route_j = None
            
            # Find which routes contain customers i and j
            for route_id, route in routes.items():
                if i in route:
                    route_i = route_id
                if j in route:
                    route_j = route_id
            
            # If customers are in different routes, try to merge
            if route_i != route_j and route_i is not None and route_j is not None:
                # Check capacity constraints
                combined_weight = route_loads_weight[route_i] + route_loads_weight[route_j]
                combined_volume = route_loads_volume[route_i] + route_loads_volume[route_j]
                
                # Find a vehicle that can handle this route
                suitable_vehicle = None
                for _, vehicle in self.vehicles.iterrows():
                    if (combined_weight <= vehicle['capacity_weight'] and 
                        combined_volume <= vehicle['capacity_volume']):
                        suitable_vehicle = vehicle
                        break
                
                if suitable_vehicle is not None:
                    # Merge routes
                    if (routes[route_i][-1] == i and routes[route_j][0] == j):
                        # i is at end of route_i, j is at start of route_j
                        routes[route_i].extend(routes[route_j])
                    elif (routes[route_i][-1] == j and routes[route_j][0] == i):
                        # j is at end of route_i, i is at start of route_j
                        routes[route_i].extend(routes[route_j])
                    elif (routes[route_i][0] == i and routes[route_j][-1] == j):
                        # i is at start of route_i, j is at end of route_j
                        routes[route_j].extend(routes[route_i])
                        routes[route_i] = routes[route_j]
                    elif (routes[route_i][0] == j and routes[route_j][-1] == i):
                        # j is at start of route_i, i is at end of route_j
                        routes[route_j].extend(routes[route_i])
                        routes[route_i] = routes[route_j]
                    else:
                        continue  # Cannot merge efficiently
                    
                    # Update loads
                    route_loads_weight[route_i] = combined_weight
                    route_loads_volume[route_i] = combined_volume
                    
                    # Remove the merged route
                    del routes[route_j]
                    del route_loads_weight[route_j]
                    del route_loads_volume[route_j]
        
        # Convert to Route objects
        final_routes = []
        vehicle_idx = 0
        
        for route_id, stops in routes.items():
            if vehicle_idx < len(self.vehicles):
                vehicle = self.vehicles.iloc[vehicle_idx]
                
                # Calculate route metrics
                full_route = [0] + stops + [0]  # Add depot at start and end
                total_distance = sum(self.distance_matrix[full_route[i]][full_route[i+1]] 
                                   for i in range(len(full_route)-1))
                
                # Estimate time (distance / avg_speed)
                total_time = total_distance / vehicle['avg_speed'] * 60  # minutes
                
                # Add service times
                service_time = sum(self.delivery_points.iloc[stop]['service_time'] for stop in stops)
                total_time += service_time
                
                total_cost = total_distance * vehicle['cost_per_km']
                
                route = Route(
                    vehicle_id=vehicle['vehicle_id'],
                    stops=stops,
                    total_distance=total_distance,
                    total_time=total_time,
                    total_cost=total_cost,
                    load_weight=route_loads_weight[route_id],
                    load_volume=route_loads_volume[route_id]
                )
                
                final_routes.append(route)
                vehicle_idx += 1
        
        return final_routes
    
    def solve_vrp_ortools(self) -> List[Route]:
        """
        Solve VRP using Google OR-Tools for optimal solutions.
        """
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(
            self.n_points, self.n_vehicles, 0)  # depot at index 0
        
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node] * 100)  # Scale for integer
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(self.delivery_points.iloc[from_node]['package_weight'] * 10)  # Scale
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Add capacity constraints for each vehicle
        capacities = [int(cap * 10) for cap in self.vehicles['capacity_weight'].values]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(30)
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_routes_from_ortools_solution(
                manager, routing, solution)
        else:
            # Fallback to Clarke-Wright if OR-Tools fails
            return self.solve_vrp_clarke_wright()
    
    def _extract_routes_from_ortools_solution(self, manager, routing, solution) -> List[Route]:
        """Extract routes from OR-Tools solution."""
        routes = []
        
        for vehicle_id in range(self.n_vehicles):
            index = routing.Start(vehicle_id)
            route_stops = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:  # Skip depot
                    route_stops.append(node_index)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id) / 100  # Unscale
            
            if route_stops:  # Only add non-empty routes
                vehicle = self.vehicles.iloc[vehicle_id]
                
                # Calculate metrics
                total_time = route_distance / vehicle['avg_speed'] * 60
                service_time = sum(self.delivery_points.iloc[stop]['service_time'] 
                                 for stop in route_stops)
                total_time += service_time
                
                total_cost = route_distance * vehicle['cost_per_km']
                
                load_weight = sum(self.delivery_points.iloc[stop]['package_weight'] 
                                for stop in route_stops)
                load_volume = sum(self.delivery_points.iloc[stop]['package_volume'] 
                                for stop in route_stops)
                
                route = Route(
                    vehicle_id=vehicle['vehicle_id'],
                    stops=route_stops,
                    total_distance=route_distance,
                    total_time=total_time,
                    total_cost=total_cost,
                    load_weight=load_weight,
                    load_volume=load_volume
                )
                
                routes.append(route)
        
        return routes
    
    def optimize_routes(self, method: str = 'ortools') -> List[Route]:
        """
        Main method to optimize routes using specified algorithm.
        
        Args:
            method: 'ortools', 'clarke_wright', or 'genetic'
        """
        if method == 'ortools':
            return self.solve_vrp_ortools()
        elif method == 'clarke_wright':
            return self.solve_vrp_clarke_wright()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def calculate_route_metrics(self, routes: List[Route]) -> Dict:
        """Calculate overall metrics for the route solution."""
        total_distance = sum(route.total_distance for route in routes)
        total_time = sum(route.total_time for route in routes)
        total_cost = sum(route.total_cost for route in routes)
        
        vehicles_used = len(routes)
        deliveries_covered = sum(len(route.stops) for route in routes)
        
        avg_route_distance = total_distance / vehicles_used if vehicles_used > 0 else 0
        avg_route_time = total_time / vehicles_used if vehicles_used > 0 else 0
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'total_cost': total_cost,
            'vehicles_used': vehicles_used,
            'deliveries_covered': deliveries_covered,
            'avg_route_distance': avg_route_distance,
            'avg_route_time': avg_route_time,
            'coverage_percentage': (deliveries_covered / (self.n_points - 1)) * 100  # Exclude depot
        }
