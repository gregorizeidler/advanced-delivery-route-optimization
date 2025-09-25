#!/usr/bin/env python3
"""
⚡ Real-Time Route Optimization System
Advanced Delivery Route Optimization

Dynamic optimization system that continuously adapts to:
- New incoming orders
- Traffic condition changes  
- Vehicle breakdowns
- Weather updates
- Driver availability

Author: IBM Data Analyst Professional Certificate Student
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import queue
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import websockets
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Order:
    """Real-time order data structure."""
    order_id: str
    customer_id: str
    latitude: float
    longitude: float
    package_weight: float
    package_volume: float
    service_time: float
    priority: str
    time_window_start: int
    time_window_end: int
    created_at: datetime
    status: str = "pending"

@dataclass
class Vehicle:
    """Real-time vehicle data structure."""
    vehicle_id: str
    driver_name: str
    current_latitude: float
    current_longitude: float
    capacity_weight: float
    capacity_volume: float
    current_load_weight: float
    current_load_volume: float
    status: str  # "available", "en_route", "delivering", "maintenance"
    last_update: datetime
    estimated_availability: datetime

@dataclass
class RouteUpdate:
    """Route update event."""
    event_id: str
    event_type: str  # "new_order", "traffic_update", "vehicle_breakdown", etc.
    data: Dict
    timestamp: datetime
    priority: int  # 1 = highest, 5 = lowest


class RealTimeOptimizer:
    """
    🚀 Real-Time Route Optimization Engine
    
    Continuously optimizes routes based on real-time events and conditions.
    """
    
    def __init__(self):
        self.orders = {}  # order_id -> Order
        self.vehicles = {}  # vehicle_id -> Vehicle
        self.current_routes = {}  # vehicle_id -> List[order_id]
        self.event_queue = queue.PriorityQueue()
        self.optimization_lock = threading.Lock()
        self.is_running = False
        self.optimization_thread = None
        self.websocket_server = None
        
        # Performance tracking
        self.metrics = {
            'total_orders_processed': 0,
            'optimization_cycles': 0,
            'average_response_time': 0,
            'total_distance_saved': 0,
            'total_time_saved': 0
        }
        
        logger.info("Real-time optimizer initialized")
    
    def start(self):
        """Start the real-time optimization system."""
        if self.is_running:
            logger.warning("Optimizer already running")
            return
        
        self.is_running = True
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        # Start websocket server for real-time updates
        self._start_websocket_server()
        
        logger.info("🚀 Real-time optimizer started")
    
    def stop(self):
        """Stop the real-time optimization system."""
        self.is_running = False
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        logger.info("⏹️ Real-time optimizer stopped")
    
    def add_order(self, order_data: Dict) -> str:
        """Add new order to the system."""
        order = Order(
            order_id=str(uuid.uuid4()),
            customer_id=order_data.get('customer_id', 'unknown'),
            latitude=order_data['latitude'],
            longitude=order_data['longitude'],
            package_weight=order_data['package_weight'],
            package_volume=order_data['package_volume'],
            service_time=order_data.get('service_time', 10),
            priority=order_data.get('priority', 'medium'),
            time_window_start=order_data.get('time_window_start', 8),
            time_window_end=order_data.get('time_window_end', 18),
            created_at=datetime.now()
        )
        
        self.orders[order.order_id] = order
        
        # Add optimization event
        event = RouteUpdate(
            event_id=str(uuid.uuid4()),
            event_type="new_order",
            data={"order_id": order.order_id},
            timestamp=datetime.now(),
            priority=self._get_event_priority(order.priority)
        )
        
        self.event_queue.put((event.priority, event))
        
        logger.info(f"📦 New order added: {order.order_id}")
        return order.order_id
    
    def update_vehicle_status(self, vehicle_id: str, status_data: Dict):
        """Update vehicle status and location."""
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            
            # Update fields
            for key, value in status_data.items():
                if hasattr(vehicle, key):
                    setattr(vehicle, key, value)
            
            vehicle.last_update = datetime.now()
            
            # Add optimization event if significant change
            if status_data.get('status') in ['available', 'maintenance']:
                event = RouteUpdate(
                    event_id=str(uuid.uuid4()),
                    event_type="vehicle_status_update",
                    data={"vehicle_id": vehicle_id, "status": status_data.get('status')},
                    timestamp=datetime.now(),
                    priority=2
                )
                self.event_queue.put((event.priority, event))
            
            logger.info(f"🚛 Vehicle {vehicle_id} status updated")
    
    def update_traffic_conditions(self, area_data: Dict):
        """Update traffic conditions for an area."""
        event = RouteUpdate(
            event_id=str(uuid.uuid4()),
            event_type="traffic_update",
            data=area_data,
            timestamp=datetime.now(),
            priority=3
        )
        
        self.event_queue.put((event.priority, event))
        logger.info("🚦 Traffic conditions updated")
    
    def update_weather_conditions(self, weather_data: Dict):
        """Update weather conditions."""
        # Only trigger re-optimization for severe weather
        if weather_data.get('severity', 'low') in ['high', 'severe']:
            event = RouteUpdate(
                event_id=str(uuid.uuid4()),
                event_type="weather_update",
                data=weather_data,
                timestamp=datetime.now(),
                priority=2
            )
            
            self.event_queue.put((event.priority, event))
            logger.info(f"🌤️ Weather update: {weather_data.get('condition', 'unknown')}")
    
    def _optimization_loop(self):
        """Main optimization loop running in separate thread."""
        logger.info("🔄 Optimization loop started")
        
        while self.is_running:
            try:
                # Check for events (with timeout)
                try:
                    priority, event = self.event_queue.get(timeout=1.0)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    # No events, continue loop
                    continue
                
                # Periodic optimization (every 5 minutes)
                if self.metrics['optimization_cycles'] % 300 == 0:
                    self._run_periodic_optimization()
                
                self.metrics['optimization_cycles'] += 1
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(1)
        
        logger.info("🔄 Optimization loop stopped")
    
    def _process_event(self, event: RouteUpdate):
        """Process a single optimization event."""
        start_time = time.time()
        
        logger.info(f"⚡ Processing event: {event.event_type}")
        
        with self.optimization_lock:
            if event.event_type == "new_order":
                self._optimize_for_new_order(event.data['order_id'])
            elif event.event_type == "vehicle_status_update":
                self._optimize_for_vehicle_change(event.data['vehicle_id'])
            elif event.event_type == "traffic_update":
                self._optimize_for_traffic_change(event.data)
            elif event.event_type == "weather_update":
                self._optimize_for_weather_change(event.data)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_response_time_metric(processing_time)
        
        # Broadcast update via websocket
        self._broadcast_route_update(event)
    
    def _optimize_for_new_order(self, order_id: str):
        """Optimize routes when a new order arrives."""
        order = self.orders[order_id]
        
        # Find best vehicle for this order
        best_vehicle = self._find_best_vehicle_for_order(order)
        
        if best_vehicle:
            # Add order to vehicle's route
            if best_vehicle.vehicle_id not in self.current_routes:
                self.current_routes[best_vehicle.vehicle_id] = []
            
            # Insert order at optimal position in route
            optimal_position = self._find_optimal_insertion_point(
                best_vehicle.vehicle_id, order_id
            )
            
            self.current_routes[best_vehicle.vehicle_id].insert(optimal_position, order_id)
            order.status = "assigned"
            
            logger.info(f"📦 Order {order_id} assigned to vehicle {best_vehicle.vehicle_id}")
            
            # Update metrics
            self.metrics['total_orders_processed'] += 1
        else:
            logger.warning(f"⚠️ No suitable vehicle found for order {order_id}")
            order.status = "pending"
    
    def _optimize_for_vehicle_change(self, vehicle_id: str):
        """Optimize when vehicle status changes."""
        vehicle = self.vehicles.get(vehicle_id)
        
        if not vehicle:
            return
        
        if vehicle.status == "maintenance":
            # Reassign orders from this vehicle
            if vehicle_id in self.current_routes:
                orders_to_reassign = self.current_routes[vehicle_id].copy()
                self.current_routes[vehicle_id] = []
                
                for order_id in orders_to_reassign:
                    self.orders[order_id].status = "pending"
                    # Add back to queue for reassignment
                    event = RouteUpdate(
                        event_id=str(uuid.uuid4()),
                        event_type="new_order",
                        data={"order_id": order_id},
                        timestamp=datetime.now(),
                        priority=1  # High priority for reassignment
                    )
                    self.event_queue.put((event.priority, event))
                
                logger.info(f"🔧 Reassigned {len(orders_to_reassign)} orders from vehicle {vehicle_id}")
        
        elif vehicle.status == "available":
            # Check if this vehicle can take on pending orders
            self._assign_pending_orders_to_vehicle(vehicle_id)
    
    def _optimize_for_traffic_change(self, traffic_data: Dict):
        """Optimize routes based on traffic updates."""
        affected_area = traffic_data.get('area', {})
        congestion_factor = traffic_data.get('congestion_factor', 1.0)
        
        if congestion_factor > 1.5:  # Significant traffic
            # Re-optimize routes passing through this area
            affected_routes = self._find_routes_in_area(affected_area)
            
            for vehicle_id in affected_routes:
                self._reoptimize_vehicle_route(vehicle_id)
            
            logger.info(f"🚦 Re-optimized {len(affected_routes)} routes due to traffic")
    
    def _optimize_for_weather_change(self, weather_data: Dict):
        """Optimize routes based on weather conditions."""
        severity = weather_data.get('severity', 'low')
        
        if severity in ['high', 'severe']:
            # Increase service times and add buffer
            time_factor = 1.5 if severity == 'high' else 2.0
            
            # Update all current routes with weather adjustment
            for vehicle_id in self.current_routes:
                self._adjust_route_for_weather(vehicle_id, time_factor)
            
            logger.info(f"🌤️ Adjusted all routes for {severity} weather conditions")
    
    def _find_best_vehicle_for_order(self, order: Order) -> Optional[Vehicle]:
        """Find the best vehicle for a given order."""
        available_vehicles = [
            v for v in self.vehicles.values() 
            if v.status == "available" and self._can_vehicle_handle_order(v, order)
        ]
        
        if not available_vehicles:
            return None
        
        # Score vehicles based on multiple factors
        best_vehicle = None
        best_score = float('inf')
        
        for vehicle in available_vehicles:
            score = self._calculate_vehicle_score(vehicle, order)
            if score < best_score:
                best_score = score
                best_vehicle = vehicle
        
        return best_vehicle
    
    def _can_vehicle_handle_order(self, vehicle: Vehicle, order: Order) -> bool:
        """Check if vehicle can handle the order."""
        # Check capacity constraints
        if (vehicle.current_load_weight + order.package_weight > vehicle.capacity_weight or
            vehicle.current_load_volume + order.package_volume > vehicle.capacity_volume):
            return False
        
        # Check time window constraints (simplified)
        current_hour = datetime.now().hour
        if current_hour < order.time_window_start or current_hour > order.time_window_end:
            return False
        
        return True
    
    def _calculate_vehicle_score(self, vehicle: Vehicle, order: Order) -> float:
        """Calculate score for vehicle-order assignment."""
        # Distance factor
        distance = self._calculate_distance(
            vehicle.current_latitude, vehicle.current_longitude,
            order.latitude, order.longitude
        )
        
        # Capacity utilization factor
        weight_util = vehicle.current_load_weight / vehicle.capacity_weight
        volume_util = vehicle.current_load_volume / vehicle.capacity_volume
        capacity_factor = max(weight_util, volume_util)
        
        # Priority factor
        priority_factor = 1.0
        if order.priority == "high":
            priority_factor = 0.5
        elif order.priority == "low":
            priority_factor = 1.5
        
        # Combined score (lower is better)
        score = distance * priority_factor * (1 + capacity_factor)
        
        return score
    
    def _find_optimal_insertion_point(self, vehicle_id: str, order_id: str) -> int:
        """Find optimal position to insert order in vehicle's route."""
        current_route = self.current_routes.get(vehicle_id, [])
        
        if not current_route:
            return 0
        
        best_position = 0
        best_cost_increase = float('inf')
        
        for pos in range(len(current_route) + 1):
            # Calculate cost increase for inserting at this position
            cost_increase = self._calculate_insertion_cost(vehicle_id, order_id, pos)
            
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_position = pos
        
        return best_position
    
    def _calculate_insertion_cost(self, vehicle_id: str, order_id: str, position: int) -> float:
        """Calculate the cost increase for inserting order at given position."""
        # Simplified cost calculation based on distance increase
        current_route = self.current_routes.get(vehicle_id, [])
        order = self.orders[order_id]
        
        if not current_route:
            return 0
        
        # Get coordinates for cost calculation
        if position == 0:
            # Insert at beginning
            next_order = self.orders[current_route[0]]
            cost_increase = self._calculate_distance(
                order.latitude, order.longitude,
                next_order.latitude, next_order.longitude
            )
        elif position == len(current_route):
            # Insert at end
            prev_order = self.orders[current_route[-1]]
            cost_increase = self._calculate_distance(
                prev_order.latitude, prev_order.longitude,
                order.latitude, order.longitude
            )
        else:
            # Insert in middle
            prev_order = self.orders[current_route[position - 1]]
            next_order = self.orders[current_route[position]]
            
            old_distance = self._calculate_distance(
                prev_order.latitude, prev_order.longitude,
                next_order.latitude, next_order.longitude
            )
            
            new_distance = (
                self._calculate_distance(
                    prev_order.latitude, prev_order.longitude,
                    order.latitude, order.longitude
                ) +
                self._calculate_distance(
                    order.latitude, order.longitude,
                    next_order.latitude, next_order.longitude
                )
            )
            
            cost_increase = new_distance - old_distance
        
        return cost_increase
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _run_periodic_optimization(self):
        """Run periodic full route optimization."""
        logger.info("🔄 Running periodic optimization")
        
        # Implement periodic optimization logic here
        # This could involve more sophisticated algorithms like genetic algorithms
        # or simulated annealing for global optimization
        
        pass
    
    def _get_event_priority(self, order_priority: str) -> int:
        """Convert order priority to event priority."""
        priority_map = {
            'high': 1,
            'medium': 3,
            'low': 5
        }
        return priority_map.get(order_priority, 3)
    
    def _update_response_time_metric(self, processing_time: float):
        """Update average response time metric."""
        current_avg = self.metrics['average_response_time']
        cycles = self.metrics['optimization_cycles']
        
        # Running average
        self.metrics['average_response_time'] = (
            (current_avg * cycles + processing_time) / (cycles + 1)
        )
    
    def _broadcast_route_update(self, event: RouteUpdate):
        """Broadcast route updates via websocket."""
        # Implementation for websocket broadcasting
        # This would send updates to connected clients (drivers, dispatchers)
        pass
    
    def _start_websocket_server(self):
        """Start websocket server for real-time communication."""
        # Implementation for websocket server
        # This would allow real-time communication with drivers and dispatchers
        pass
    
    def get_current_status(self) -> Dict:
        """Get current system status."""
        pending_orders = len([o for o in self.orders.values() if o.status == "pending"])
        assigned_orders = len([o for o in self.orders.values() if o.status == "assigned"])
        available_vehicles = len([v for v in self.vehicles.values() if v.status == "available"])
        
        return {
            'system_status': 'running' if self.is_running else 'stopped',
            'orders': {
                'total': len(self.orders),
                'pending': pending_orders,
                'assigned': assigned_orders
            },
            'vehicles': {
                'total': len(self.vehicles),
                'available': available_vehicles
            },
            'metrics': self.metrics,
            'queue_size': self.event_queue.qsize()
        }
    
    def export_routes_json(self) -> str:
        """Export current routes as JSON."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'routes': {},
            'orders': {},
            'vehicles': {}
        }
        
        # Export routes
        for vehicle_id, order_ids in self.current_routes.items():
            export_data['routes'][vehicle_id] = {
                'order_ids': order_ids,
                'total_orders': len(order_ids)
            }
        
        # Export orders
        for order_id, order in self.orders.items():
            export_data['orders'][order_id] = asdict(order)
            # Convert datetime to string
            export_data['orders'][order_id]['created_at'] = order.created_at.isoformat()
        
        # Export vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            export_data['vehicles'][vehicle_id] = asdict(vehicle)
            export_data['vehicles'][vehicle_id]['last_update'] = vehicle.last_update.isoformat()
            export_data['vehicles'][vehicle_id]['estimated_availability'] = vehicle.estimated_availability.isoformat()
        
        return json.dumps(export_data, indent=2)


class SimulatedEnvironment:
    """
    🎮 Simulated Environment for Testing Real-Time Optimizer
    
    Generates realistic events for testing the optimization system.
    """
    
    def __init__(self, optimizer: RealTimeOptimizer):
        self.optimizer = optimizer
        self.is_running = False
        self.simulation_thread = None
        
        # Initialize some vehicles
        self._initialize_vehicles()
    
    def _initialize_vehicles(self):
        """Initialize test vehicles."""
        vehicles_data = [
            {
                'vehicle_id': 'V001',
                'driver_name': 'João Silva',
                'current_latitude': -23.5505,
                'current_longitude': -46.6333,
                'capacity_weight': 1000,
                'capacity_volume': 10,
                'current_load_weight': 0,
                'current_load_volume': 0,
                'status': 'available'
            },
            {
                'vehicle_id': 'V002',
                'driver_name': 'Maria Santos',
                'current_latitude': -23.5600,
                'current_longitude': -46.6400,
                'capacity_weight': 800,
                'capacity_volume': 8,
                'current_load_weight': 200,
                'current_load_volume': 2,
                'status': 'available'
            },
            {
                'vehicle_id': 'V003',
                'driver_name': 'Pedro Costa',
                'current_latitude': -23.5400,
                'current_longitude': -46.6200,
                'capacity_weight': 1200,
                'capacity_volume': 12,
                'current_load_weight': 0,
                'current_load_volume': 0,
                'status': 'available'
            }
        ]
        
        for vehicle_data in vehicles_data:
            vehicle = Vehicle(
                **vehicle_data,
                last_update=datetime.now(),
                estimated_availability=datetime.now()
            )
            self.optimizer.vehicles[vehicle.vehicle_id] = vehicle
    
    def start_simulation(self, duration_minutes: int = 60):
        """Start simulated environment."""
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop, 
            args=(duration_minutes,), 
            daemon=True
        )
        self.simulation_thread.start()
        
        logger.info(f"🎮 Simulation started for {duration_minutes} minutes")
    
    def stop_simulation(self):
        """Stop simulated environment."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
        
        logger.info("🎮 Simulation stopped")
    
    def _simulation_loop(self, duration_minutes: int):
        """Main simulation loop."""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while self.is_running and time.time() < end_time:
            # Generate random events
            event_type = np.random.choice([
                'new_order', 'vehicle_update', 'traffic_change', 'weather_change'
            ], p=[0.6, 0.2, 0.15, 0.05])
            
            if event_type == 'new_order':
                self._generate_random_order()
            elif event_type == 'vehicle_update':
                self._generate_vehicle_update()
            elif event_type == 'traffic_change':
                self._generate_traffic_update()
            elif event_type == 'weather_change':
                self._generate_weather_update()
            
            # Wait before next event
            wait_time = np.random.exponential(10)  # Average 10 seconds between events
            time.sleep(min(wait_time, 30))  # Cap at 30 seconds
        
        logger.info("🎮 Simulation loop completed")
    
    def _generate_random_order(self):
        """Generate a random order."""
        order_data = {
            'customer_id': f'CUST_{np.random.randint(1000, 9999)}',
            'latitude': np.random.uniform(-23.6, -23.5),
            'longitude': np.random.uniform(-46.7, -46.6),
            'package_weight': np.random.uniform(1, 50),
            'package_volume': np.random.uniform(0.1, 5),
            'service_time': np.random.uniform(5, 30),
            'priority': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.4, 0.1]),
            'time_window_start': np.random.randint(8, 16),
            'time_window_end': np.random.randint(16, 20)
        }
        
        order_id = self.optimizer.add_order(order_data)
        logger.info(f"🎮 Generated order: {order_id}")
    
    def _generate_vehicle_update(self):
        """Generate a random vehicle status update."""
        if not self.optimizer.vehicles:
            return
        
        vehicle_id = np.random.choice(list(self.optimizer.vehicles.keys()))
        
        # Simulate location update
        current_vehicle = self.optimizer.vehicles[vehicle_id]
        lat_change = np.random.normal(0, 0.001)
        lon_change = np.random.normal(0, 0.001)
        
        update_data = {
            'current_latitude': current_vehicle.current_latitude + lat_change,
            'current_longitude': current_vehicle.current_longitude + lon_change,
            'last_update': datetime.now()
        }
        
        # Occasionally change status
        if np.random.random() < 0.1:
            new_status = np.random.choice(['available', 'en_route', 'delivering'])
            update_data['status'] = new_status
        
        self.optimizer.update_vehicle_status(vehicle_id, update_data)
    
    def _generate_traffic_update(self):
        """Generate a random traffic update."""
        traffic_data = {
            'area': {
                'center_lat': np.random.uniform(-23.6, -23.5),
                'center_lon': np.random.uniform(-46.7, -46.6),
                'radius': np.random.uniform(0.5, 2.0)
            },
            'congestion_factor': np.random.uniform(1.0, 2.5),
            'duration_minutes': np.random.randint(15, 120)
        }
        
        self.optimizer.update_traffic_conditions(traffic_data)
    
    def _generate_weather_update(self):
        """Generate a random weather update."""
        weather_conditions = ['clear', 'rain', 'heavy_rain', 'snow']
        severities = ['low', 'medium', 'high', 'severe']
        
        weather_data = {
            'condition': np.random.choice(weather_conditions, p=[0.6, 0.25, 0.1, 0.05]),
            'severity': np.random.choice(severities, p=[0.6, 0.25, 0.1, 0.05]),
            'visibility': np.random.uniform(1, 10),
            'temperature': np.random.uniform(-5, 35)
        }
        
        self.optimizer.update_weather_conditions(weather_data)


# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Real-Time Optimization System...")
    print("=" * 60)
    
    # Create optimizer
    optimizer = RealTimeOptimizer()
    
    # Start optimizer
    optimizer.start()
    
    # Create simulated environment
    sim_env = SimulatedEnvironment(optimizer)
    
    try:
        # Run simulation for 2 minutes
        sim_env.start_simulation(duration_minutes=2)
        
        # Monitor status
        for i in range(12):  # Check every 10 seconds for 2 minutes
            time.sleep(10)
            status = optimizer.get_current_status()
            
            print(f"\n📊 Status Update {i+1}:")
            print(f"  Orders: {status['orders']['total']} total, {status['orders']['pending']} pending")
            print(f"  Vehicles: {status['vehicles']['available']} available")
            print(f"  Queue size: {status['queue_size']}")
            print(f"  Avg response time: {status['metrics']['average_response_time']:.3f}s")
        
        # Export final routes
        print("\n📋 Final Routes:")
        routes_json = optimizer.export_routes_json()
        print(routes_json[:500] + "..." if len(routes_json) > 500 else routes_json)
        
    finally:
        # Cleanup
        sim_env.stop_simulation()
        optimizer.stop()
        
        print("\n✅ Real-time optimization testing complete!")
        
        final_status = optimizer.get_current_status()
        print(f"\n📈 Final Metrics:")
        print(f"  Total orders processed: {final_status['metrics']['total_orders_processed']}")
        print(f"  Optimization cycles: {final_status['metrics']['optimization_cycles']}")
        print(f"  Average response time: {final_status['metrics']['average_response_time']:.3f}s")
