#!/usr/bin/env python3
"""
⚡ Real-Time Route Optimization System
IBM Data Analyst Professional Certificate Project

True real-time capabilities for route optimization and monitoring.
"""

import asyncio
import websockets
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import numpy as np
from dataclasses import dataclass, asdict
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LiveUpdate:
    """Real-time update data structure."""
    timestamp: str
    update_type: str
    data: Dict[str, Any]
    vehicle_id: Optional[str] = None

@dataclass
class VehicleStatus:
    """Real-time vehicle status."""
    vehicle_id: str
    current_lat: float
    current_lon: float
    next_stop_id: str
    eta_minutes: int
    status: str  # 'driving', 'delivering', 'idle'
    last_update: str

class RealTimeOptimizer:
    """Real-time route optimization engine."""
    
    def __init__(self):
        """Initialize real-time optimizer."""
        self.active_routes = {}
        self.vehicle_statuses = {}
        self.update_queue = queue.Queue()
        self.websocket_clients = set()
        self.is_running = False
        
        # Performance monitoring
        self.metrics = {
            'updates_processed': 0,
            'route_recalculations': 0,
            'average_response_time': 0.0,
            'active_vehicles': 0
        }
    
    async def start_server(self, host='localhost', port=8765):
        """Start the real-time WebSocket server."""
        logger.info(f"🚀 Starting real-time server on {host}:{port}")
        
        self.is_running = True
        
        # Start background processing
        asyncio.create_task(self.process_updates())
        asyncio.create_task(self.simulate_vehicle_updates())
        
        # Start WebSocket server
        async with websockets.serve(self.handle_websocket, host, port):
            logger.info("✅ Real-time server started!")
            await asyncio.Future()  # Run forever
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        logger.info(f"🔗 New client connected: {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        
        try:
            # Send initial data
            await self.send_initial_data(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"❌ Client disconnected: {websocket.remote_address}")
        finally:
            self.websocket_clients.discard(websocket)
    
    async def send_initial_data(self, websocket):
        """Send initial data to new client."""
        initial_data = {
            'type': 'initial_data',
            'timestamp': datetime.now().isoformat(),
            'vehicle_statuses': {vid: asdict(status) for vid, status in self.vehicle_statuses.items()},
            'active_routes': len(self.active_routes),
            'metrics': self.metrics
        }
        
        await websocket.send(json.dumps(initial_data))
    
    async def handle_client_message(self, websocket, message):
        """Handle messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'request_route_update':
                await self.handle_route_update_request(websocket, data)
            elif message_type == 'emergency_reroute':
                await self.handle_emergency_reroute(websocket, data)
            elif message_type == 'vehicle_status_update':
                await self.handle_vehicle_status_update(data)
                
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON received from client")
        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")
    
    async def handle_route_update_request(self, websocket, data):
        """Handle route update requests."""
        vehicle_id = data.get('vehicle_id')
        
        if vehicle_id in self.active_routes:
            response = {
                'type': 'route_update_response',
                'vehicle_id': vehicle_id,
                'route': self.active_routes[vehicle_id],
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
    
    async def handle_emergency_reroute(self, websocket, data):
        """Handle emergency rerouting requests."""
        vehicle_id = data.get('vehicle_id')
        blocked_location = data.get('blocked_location')
        
        logger.info(f"🚨 Emergency reroute requested for {vehicle_id}")
        
        # Simulate route recalculation
        await asyncio.sleep(0.5)  # Simulate processing time
        
        new_route = await self.calculate_emergency_route(vehicle_id, blocked_location)
        
        # Update active routes
        self.active_routes[vehicle_id] = new_route
        self.metrics['route_recalculations'] += 1
        
        # Broadcast update to all clients
        await self.broadcast_update({
            'type': 'emergency_reroute_complete',
            'vehicle_id': vehicle_id,
            'new_route': new_route,
            'timestamp': datetime.now().isoformat()
        })
    
    async def calculate_emergency_route(self, vehicle_id: str, blocked_location: Dict) -> Dict:
        """Calculate new route avoiding blocked location."""
        # Simulate advanced route calculation
        current_status = self.vehicle_statuses.get(vehicle_id)
        
        if not current_status:
            return {}
        
        # Generate alternative route
        new_route = {
            'route_id': f"emergency_{vehicle_id}_{int(time.time())}",
            'waypoints': [
                {'lat': current_status.current_lat, 'lon': current_status.current_lon},
                {'lat': current_status.current_lat + 0.01, 'lon': current_status.current_lon + 0.01},
                {'lat': current_status.current_lat + 0.02, 'lon': current_status.current_lon - 0.01}
            ],
            'estimated_time': 25,
            'distance': 15.5,
            'avoids_blocked': True
        }
        
        return new_route
    
    async def handle_vehicle_status_update(self, data):
        """Handle vehicle status updates."""
        vehicle_id = data.get('vehicle_id')
        
        if vehicle_id:
            # Update vehicle status
            self.vehicle_statuses[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                current_lat=data.get('lat', 0.0),
                current_lon=data.get('lon', 0.0),
                next_stop_id=data.get('next_stop_id', ''),
                eta_minutes=data.get('eta_minutes', 0),
                status=data.get('status', 'unknown'),
                last_update=datetime.now().isoformat()
            )
            
            # Broadcast update
            await self.broadcast_update({
                'type': 'vehicle_status_update',
                'vehicle_id': vehicle_id,
                'status': asdict(self.vehicle_statuses[vehicle_id]),
                'timestamp': datetime.now().isoformat()
            })
    
    async def process_updates(self):
        """Process queued updates in real-time."""
        while self.is_running:
            try:
                # Process updates from queue
                if not self.update_queue.empty():
                    update = self.update_queue.get_nowait()
                    await self.handle_update(update)
                    self.metrics['updates_processed'] += 1
                
                # Update metrics
                self.metrics['active_vehicles'] = len(self.vehicle_statuses)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                logger.error(f"❌ Error processing updates: {e}")
    
    async def handle_update(self, update: LiveUpdate):
        """Handle individual live updates."""
        start_time = time.time()
        
        if update.update_type == 'vehicle_position':
            await self.handle_vehicle_position_update(update)
        elif update.update_type == 'traffic_incident':
            await self.handle_traffic_incident(update)
        elif update.update_type == 'delivery_complete':
            await self.handle_delivery_complete(update)
        
        # Update response time metric
        response_time = time.time() - start_time
        self.metrics['average_response_time'] = (
            self.metrics['average_response_time'] * 0.9 + response_time * 0.1
        )
    
    async def handle_vehicle_position_update(self, update: LiveUpdate):
        """Handle vehicle position updates."""
        vehicle_id = update.vehicle_id
        position_data = update.data
        
        # Check if route optimization is needed
        if self.should_reoptimize_route(vehicle_id, position_data):
            await self.trigger_route_reoptimization(vehicle_id)
    
    def should_reoptimize_route(self, vehicle_id: str, position_data: Dict) -> bool:
        """Determine if route should be reoptimized."""
        # Simple logic: reoptimize if significantly off planned route
        if vehicle_id not in self.active_routes:
            return False
        
        # Simulate route deviation check
        deviation = np.random.uniform(0, 1)
        return deviation > 0.8  # 20% chance of needing reoptimization
    
    async def trigger_route_reoptimization(self, vehicle_id: str):
        """Trigger route reoptimization for a vehicle."""
        logger.info(f"🔄 Reoptimizing route for {vehicle_id}")
        
        # Simulate route calculation
        await asyncio.sleep(0.3)
        
        # Generate new optimized route
        new_route = await self.calculate_optimized_route(vehicle_id)
        
        self.active_routes[vehicle_id] = new_route
        self.metrics['route_recalculations'] += 1
        
        # Broadcast update
        await self.broadcast_update({
            'type': 'route_optimized',
            'vehicle_id': vehicle_id,
            'new_route': new_route,
            'timestamp': datetime.now().isoformat()
        })
    
    async def calculate_optimized_route(self, vehicle_id: str) -> Dict:
        """Calculate optimized route for vehicle."""
        # Simulate advanced optimization
        return {
            'route_id': f"opt_{vehicle_id}_{int(time.time())}",
            'waypoints': [
                {'lat': -23.5505, 'lon': -46.6333},
                {'lat': -23.5515, 'lon': -46.6343},
                {'lat': -23.5525, 'lon': -46.6353}
            ],
            'estimated_time': 30,
            'distance': 18.2,
            'optimization_score': 0.92
        }
    
    async def simulate_vehicle_updates(self):
        """Simulate realistic vehicle updates for demonstration."""
        vehicle_ids = ['VEH001', 'VEH002', 'VEH003', 'VEH004', 'VEH005']
        
        # Initialize vehicles
        for i, vehicle_id in enumerate(vehicle_ids):
            self.vehicle_statuses[vehicle_id] = VehicleStatus(
                vehicle_id=vehicle_id,
                current_lat=-23.5505 + i * 0.01,
                current_lon=-46.6333 + i * 0.01,
                next_stop_id=f"STOP_{i+1}",
                eta_minutes=np.random.randint(10, 45),
                status='driving',
                last_update=datetime.now().isoformat()
            )
            
            # Create initial route
            self.active_routes[vehicle_id] = {
                'route_id': f"route_{vehicle_id}",
                'waypoints': [
                    {'lat': -23.5505 + i * 0.01, 'lon': -46.6333 + i * 0.01},
                    {'lat': -23.5515 + i * 0.01, 'lon': -46.6343 + i * 0.01}
                ],
                'estimated_time': 35,
                'distance': 20.0
            }
        
        while self.is_running:
            # Simulate random vehicle updates
            vehicle_id = np.random.choice(vehicle_ids)
            
            # Update position
            current_status = self.vehicle_statuses[vehicle_id]
            new_lat = current_status.current_lat + np.random.uniform(-0.001, 0.001)
            new_lon = current_status.current_lon + np.random.uniform(-0.001, 0.001)
            
            # Create update
            update = LiveUpdate(
                timestamp=datetime.now().isoformat(),
                update_type='vehicle_position',
                vehicle_id=vehicle_id,
                data={'lat': new_lat, 'lon': new_lon, 'speed': np.random.uniform(20, 60)}
            )
            
            self.update_queue.put(update)
            
            # Occasionally simulate delivery completion
            if np.random.random() < 0.1:  # 10% chance
                delivery_update = LiveUpdate(
                    timestamp=datetime.now().isoformat(),
                    update_type='delivery_complete',
                    vehicle_id=vehicle_id,
                    data={'stop_id': f"STOP_{np.random.randint(1, 10)}"}
                )
                self.update_queue.put(delivery_update)
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    async def broadcast_update(self, update_data: Dict):
        """Broadcast update to all connected clients."""
        if self.websocket_clients:
            message = json.dumps(update_data)
            disconnected_clients = set()
            
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    def add_update(self, update: LiveUpdate):
        """Add update to processing queue."""
        self.update_queue.put(update)
    
    def get_metrics(self) -> Dict:
        """Get real-time performance metrics."""
        return {
            **self.metrics,
            'queue_size': self.update_queue.qsize(),
            'connected_clients': len(self.websocket_clients),
            'timestamp': datetime.now().isoformat()
        }

class RealTimeMonitor:
    """Real-time monitoring and analytics."""
    
    def __init__(self):
        """Initialize monitor."""
        self.performance_history = []
        self.alert_thresholds = {
            'response_time': 1.0,  # seconds
            'queue_backlog': 100,   # updates
            'vehicle_offline': 300  # seconds
        }
    
    def analyze_performance(self, metrics: Dict) -> Dict:
        """Analyze real-time performance."""
        alerts = []
        
        # Check response time
        if metrics['average_response_time'] > self.alert_thresholds['response_time']:
            alerts.append({
                'type': 'performance',
                'message': f"High response time: {metrics['average_response_time']:.3f}s"
            })
        
        # Check queue backlog
        if metrics['queue_size'] > self.alert_thresholds['queue_backlog']:
            alerts.append({
                'type': 'capacity',
                'message': f"Queue backlog: {metrics['queue_size']} updates"
            })
        
        return {
            'alerts': alerts,
            'performance_score': self.calculate_performance_score(metrics),
            'recommendations': self.generate_recommendations(metrics)
        }
    
    def calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score."""
        # Simple scoring based on key metrics
        response_score = max(0, 1 - metrics['average_response_time'])
        queue_score = max(0, 1 - metrics['queue_size'] / 100)
        
        return (response_score + queue_score) / 2
    
    def generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics['average_response_time'] > 0.5:
            recommendations.append("Consider optimizing update processing algorithms")
        
        if metrics['queue_size'] > 50:
            recommendations.append("Increase processing capacity or reduce update frequency")
        
        if metrics['connected_clients'] > 10:
            recommendations.append("Consider implementing client connection pooling")
        
        return recommendations

async def main():
    """Main function to start real-time system."""
    print("⚡ Starting Real-Time Route Optimization System...")
    
    optimizer = RealTimeOptimizer()
    monitor = RealTimeMonitor()
    
    # Start the server
    try:
        await optimizer.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down real-time system...")
        optimizer.is_running = False

if __name__ == "__main__":
    asyncio.run(main())
