#!/usr/bin/env python3
"""
🌐 Real-World API Integrations
Advanced Route Optimization - Enterprise Edition

Integrates with real-world APIs for:
- Google Maps for routing and traffic
- OpenWeatherMap for weather conditions  
- Real-time traffic data
- Economic indicators

Author: IBM Data Analyst Professional Certificate Student
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleMapsAPI:
    """Integration with Google Maps API for real routing and traffic data."""
    
    def __init__(self, api_key: str = None):
        """Initialize Google Maps API client."""
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        self.base_url = "https://maps.googleapis.com/maps/api"
        
        if not self.api_key:
            logger.warning("Google Maps API key not provided. Using simulated data.")
            self.use_simulation = True
        else:
            self.use_simulation = False
    
    def get_distance_matrix(self, origins: List[Tuple[float, float]], 
                           destinations: List[Tuple[float, float]],
                           mode: str = "driving") -> Dict:
        """Get real distance matrix from Google Maps."""
        
        if self.use_simulation:
            return self._simulate_distance_matrix(origins, destinations)
        
        # Prepare origins and destinations
        origins_str = "|".join([f"{lat},{lon}" for lat, lon in origins])
        destinations_str = "|".join([f"{lat},{lon}" for lat, lon in destinations])
        
        url = f"{self.base_url}/distancematrix/json"
        params = {
            'origins': origins_str,
            'destinations': destinations_str,
            'mode': mode,
            'units': 'metric',
            'departure_time': 'now',
            'traffic_model': 'best_guess',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK':
                return self._parse_distance_matrix(data)
            else:
                logger.error(f"Google Maps API error: {data['status']}")
                return self._simulate_distance_matrix(origins, destinations)
                
        except Exception as e:
            logger.error(f"Error calling Google Maps API: {e}")
            return self._simulate_distance_matrix(origins, destinations)
    
    def get_route_optimization(self, waypoints: List[Tuple[float, float]], 
                              optimize: bool = True) -> Dict:
        """Get optimized route from Google Maps."""
        
        if self.use_simulation:
            return self._simulate_route_optimization(waypoints)
        
        # Prepare waypoints
        waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in waypoints[1:-1]])
        
        url = f"{self.base_url}/directions/json"
        params = {
            'origin': f"{waypoints[0][0]},{waypoints[0][1]}",
            'destination': f"{waypoints[-1][0]},{waypoints[-1][1]}",
            'waypoints': f"optimize:{optimize}|{waypoints_str}" if waypoints_str else "",
            'mode': 'driving',
            'departure_time': 'now',
            'traffic_model': 'best_guess',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK':
                return self._parse_route_data(data)
            else:
                logger.error(f"Google Directions API error: {data['status']}")
                return self._simulate_route_optimization(waypoints)
                
        except Exception as e:
            logger.error(f"Error calling Google Directions API: {e}")
            return self._simulate_route_optimization(waypoints)
    
    def _simulate_distance_matrix(self, origins: List[Tuple[float, float]], 
                                 destinations: List[Tuple[float, float]]) -> Dict:
        """Simulate distance matrix with realistic data."""
        n_origins = len(origins)
        n_destinations = len(destinations)
        
        distances = np.zeros((n_origins, n_destinations))
        durations = np.zeros((n_origins, n_destinations))
        
        for i, origin in enumerate(origins):
            for j, destination in enumerate(destinations):
                # Haversine distance
                R = 6371  # Earth's radius in km
                lat1, lon1 = np.radians(origin)
                lat2, lon2 = np.radians(destination)
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c
                
                # Add traffic factor (1.2-1.8x)
                traffic_factor = np.random.uniform(1.2, 1.8)
                duration = (distance / 40) * 60 * traffic_factor  # Assuming 40 km/h average
                
                distances[i, j] = distance
                durations[i, j] = duration
        
        return {
            'distances': distances,
            'durations': durations,
            'status': 'simulated'
        }
    
    def _simulate_route_optimization(self, waypoints: List[Tuple[float, float]]) -> Dict:
        """Simulate route optimization."""
        total_distance = 0
        total_duration = 0
        
        for i in range(len(waypoints) - 1):
            # Calculate distance between consecutive points
            result = self._simulate_distance_matrix([waypoints[i]], [waypoints[i+1]])
            total_distance += result['distances'][0, 0]
            total_duration += result['durations'][0, 0]
        
        return {
            'total_distance': total_distance,
            'total_duration': total_duration,
            'waypoint_order': list(range(len(waypoints))),
            'status': 'simulated'
        }
    
    def _parse_distance_matrix(self, data: Dict) -> Dict:
        """Parse Google Maps distance matrix response."""
        rows = data['rows']
        n_origins = len(rows)
        n_destinations = len(rows[0]['elements']) if rows else 0
        
        distances = np.zeros((n_origins, n_destinations))
        durations = np.zeros((n_origins, n_destinations))
        
        for i, row in enumerate(rows):
            for j, element in enumerate(row['elements']):
                if element['status'] == 'OK':
                    distances[i, j] = element['distance']['value'] / 1000  # Convert to km
                    durations[i, j] = element['duration']['value'] / 60  # Convert to minutes
                else:
                    distances[i, j] = np.inf
                    durations[i, j] = np.inf
        
        return {
            'distances': distances,
            'durations': durations,
            'status': 'real'
        }
    
    def _parse_route_data(self, data: Dict) -> Dict:
        """Parse Google Directions API response."""
        route = data['routes'][0]
        leg = route['legs'][0]
        
        return {
            'total_distance': sum(leg['distance']['value'] for leg in route['legs']) / 1000,
            'total_duration': sum(leg['duration']['value'] for leg in route['legs']) / 60,
            'waypoint_order': route.get('waypoint_order', []),
            'status': 'real'
        }


class WeatherAPI:
    """Integration with OpenWeatherMap API for weather conditions."""
    
    def __init__(self, api_key: str = None):
        """Initialize Weather API client."""
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not provided. Using simulated data.")
            self.use_simulation = True
        else:
            self.use_simulation = False
    
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather conditions."""
        
        if self.use_simulation:
            return self._simulate_weather()
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except Exception as e:
            logger.error(f"Error calling Weather API: {e}")
            return self._simulate_weather()
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> List[Dict]:
        """Get weather forecast."""
        
        if self.use_simulation:
            return [self._simulate_weather() for _ in range(days)]
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return [self._parse_weather_data(item) for item in data['list']]
            
        except Exception as e:
            logger.error(f"Error calling Weather Forecast API: {e}")
            return [self._simulate_weather() for _ in range(days)]
    
    def _simulate_weather(self) -> Dict:
        """Simulate weather data."""
        conditions = ['Clear', 'Clouds', 'Rain', 'Snow']
        condition = np.random.choice(conditions, p=[0.4, 0.3, 0.2, 0.1])
        
        base_temp = 20
        if condition == 'Snow':
            base_temp = 0
        elif condition == 'Rain':
            base_temp = 15
        
        return {
            'temperature': base_temp + np.random.normal(0, 5),
            'humidity': np.random.uniform(30, 90),
            'condition': condition,
            'wind_speed': np.random.uniform(0, 20),
            'visibility': np.random.uniform(1, 10) if condition in ['Rain', 'Snow'] else 10,
            'delivery_impact': self._calculate_delivery_impact(condition),
            'status': 'simulated'
        }
    
    def _parse_weather_data(self, data: Dict) -> Dict:
        """Parse weather API response."""
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'condition': data['weather'][0]['main'],
            'wind_speed': data['wind']['speed'],
            'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
            'delivery_impact': self._calculate_delivery_impact(data['weather'][0]['main']),
            'status': 'real'
        }
    
    def _calculate_delivery_impact(self, condition: str) -> Dict:
        """Calculate weather impact on delivery operations."""
        impact_factors = {
            'Clear': {'time_factor': 1.0, 'risk_level': 'low'},
            'Clouds': {'time_factor': 1.05, 'risk_level': 'low'},
            'Rain': {'time_factor': 1.3, 'risk_level': 'medium'},
            'Snow': {'time_factor': 1.6, 'risk_level': 'high'},
            'Fog': {'time_factor': 1.4, 'risk_level': 'high'}
        }
        
        return impact_factors.get(condition, {'time_factor': 1.2, 'risk_level': 'medium'})


class TrafficAPI:
    """Integration with real-time traffic APIs."""
    
    def __init__(self):
        """Initialize Traffic API client."""
        self.apis = {
            'tomtom': os.getenv('TOMTOM_API_KEY'),
            'here': os.getenv('HERE_API_KEY')
        }
        
        # Check if any API keys are available
        self.use_simulation = not any(self.apis.values())
        
        if self.use_simulation:
            logger.warning("Traffic API keys not provided. Using simulated data.")
    
    def get_traffic_flow(self, lat: float, lon: float, radius: float = 1000) -> Dict:
        """Get real-time traffic flow data."""
        
        if self.use_simulation:
            return self._simulate_traffic_flow()
        
        # Try TomTom API first
        if self.apis['tomtom']:
            return self._get_tomtom_traffic(lat, lon, radius)
        
        # Fallback to simulation
        return self._simulate_traffic_flow()
    
    def get_traffic_incidents(self, lat: float, lon: float, radius: float = 5000) -> List[Dict]:
        """Get traffic incidents in the area."""
        
        if self.use_simulation:
            return self._simulate_traffic_incidents()
        
        # Try TomTom API
        if self.apis['tomtom']:
            return self._get_tomtom_incidents(lat, lon, radius)
        
        return self._simulate_traffic_incidents()
    
    def _simulate_traffic_flow(self) -> Dict:
        """Simulate traffic flow data."""
        current_hour = datetime.now().hour
        
        # Traffic patterns based on time of day
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            # Rush hour
            flow_speed = np.random.uniform(15, 35)
            congestion_level = np.random.choice(['moderate', 'heavy'], p=[0.3, 0.7])
        elif 10 <= current_hour <= 16:
            # Midday
            flow_speed = np.random.uniform(35, 55)
            congestion_level = np.random.choice(['light', 'moderate'], p=[0.7, 0.3])
        else:
            # Off-peak
            flow_speed = np.random.uniform(45, 65)
            congestion_level = 'light'
        
        return {
            'current_speed': flow_speed,
            'free_flow_speed': 60,
            'congestion_level': congestion_level,
            'travel_time_factor': 60 / flow_speed,
            'status': 'simulated'
        }
    
    def _simulate_traffic_incidents(self) -> List[Dict]:
        """Simulate traffic incidents."""
        n_incidents = np.random.poisson(2)  # Average 2 incidents
        incidents = []
        
        incident_types = ['accident', 'construction', 'road_closure', 'weather']
        severities = ['minor', 'moderate', 'severe']
        
        for _ in range(n_incidents):
            incident = {
                'type': np.random.choice(incident_types),
                'severity': np.random.choice(severities, p=[0.5, 0.3, 0.2]),
                'description': f"Simulated {np.random.choice(incident_types)} incident",
                'impact_factor': np.random.uniform(1.2, 2.5),
                'duration_minutes': np.random.randint(30, 180),
                'status': 'simulated'
            }
            incidents.append(incident)
        
        return incidents


class EconomicDataAPI:
    """Integration with economic indicators APIs."""
    
    def __init__(self):
        """Initialize Economic Data API client."""
        self.use_simulation = True  # Most economic APIs require registration
        logger.warning("Economic data APIs require registration. Using simulated data.")
    
    def get_fuel_prices(self) -> Dict:
        """Get current fuel price data."""
        if self.use_simulation:
            return self._simulate_fuel_prices()
        
        # Implementation for real fuel price APIs would go here
        return self._simulate_fuel_prices()
    
    def get_economic_indicators(self) -> Dict:
        """Get relevant economic indicators."""
        if self.use_simulation:
            return self._simulate_economic_indicators()
        
        return self._simulate_economic_indicators()
    
    def _simulate_fuel_prices(self) -> Dict:
        """Simulate fuel price data."""
        base_price = 1.50  # Base price per liter
        variation = np.random.normal(0, 0.1)
        
        return {
            'gasoline_price': base_price + variation,
            'diesel_price': base_price - 0.1 + variation,
            'price_trend': np.random.choice(['increasing', 'stable', 'decreasing'], p=[0.3, 0.4, 0.3]),
            'last_updated': datetime.now().isoformat(),
            'status': 'simulated'
        }
    
    def _simulate_economic_indicators(self) -> Dict:
        """Simulate economic indicators."""
        return {
            'inflation_rate': np.random.uniform(2, 6),
            'unemployment_rate': np.random.uniform(4, 10),
            'consumer_confidence': np.random.uniform(80, 120),
            'logistics_cost_index': np.random.uniform(95, 115),
            'seasonal_factor': np.random.uniform(0.8, 1.3),
            'status': 'simulated'
        }


class APIIntegrationManager:
    """Centralized manager for all API integrations."""
    
    def __init__(self):
        """Initialize all API clients."""
        self.google_maps = GoogleMapsAPI()
        self.weather = WeatherAPI()
        self.traffic = TrafficAPI()
        self.economic = EconomicDataAPI()
        
        logger.info("API Integration Manager initialized")
    
    def get_comprehensive_data(self, lat: float, lon: float) -> Dict:
        """Get comprehensive data from all APIs."""
        
        logger.info(f"Fetching comprehensive data for location: {lat}, {lon}")
        
        # Gather data from all sources
        data = {
            'location': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().isoformat(),
            'weather': self.weather.get_current_weather(lat, lon),
            'traffic': self.traffic.get_traffic_flow(lat, lon),
            'incidents': self.traffic.get_traffic_incidents(lat, lon),
            'fuel_prices': self.economic.get_fuel_prices(),
            'economic_indicators': self.economic.get_economic_indicators()
        }
        
        # Calculate composite delivery conditions
        data['delivery_conditions'] = self._calculate_delivery_conditions(data)
        
        return data
    
    def _calculate_delivery_conditions(self, data: Dict) -> Dict:
        """Calculate overall delivery conditions based on all factors."""
        
        # Weather impact
        weather_impact = data['weather']['delivery_impact']['time_factor']
        
        # Traffic impact
        traffic_impact = data['traffic']['travel_time_factor']
        
        # Economic impact (fuel prices)
        fuel_impact = data['fuel_prices']['gasoline_price'] / 1.50  # Normalized to base price
        
        # Incident impact
        incident_impact = 1.0
        for incident in data['incidents']:
            if incident['severity'] == 'severe':
                incident_impact *= incident['impact_factor']
        
        # Overall condition score
        overall_factor = weather_impact * traffic_impact * incident_impact
        
        if overall_factor <= 1.2:
            condition = 'excellent'
        elif overall_factor <= 1.5:
            condition = 'good'
        elif overall_factor <= 2.0:
            condition = 'moderate'
        else:
            condition = 'challenging'
        
        return {
            'overall_condition': condition,
            'time_factor': overall_factor,
            'weather_impact': weather_impact,
            'traffic_impact': traffic_impact,
            'fuel_impact': fuel_impact,
            'incident_impact': incident_impact,
            'recommendation': self._get_delivery_recommendation(condition, overall_factor)
        }
    
    def _get_delivery_recommendation(self, condition: str, factor: float) -> str:
        """Get delivery recommendation based on conditions."""
        
        recommendations = {
            'excellent': "Optimal conditions for deliveries. Consider increasing route density.",
            'good': "Good conditions. Normal operations recommended.",
            'moderate': "Moderate conditions. Allow extra time and monitor closely.",
            'challenging': "Difficult conditions. Consider delaying non-urgent deliveries."
        }
        
        return recommendations.get(condition, "Monitor conditions closely.")


# Example usage and testing
if __name__ == "__main__":
    # Test the API integration
    manager = APIIntegrationManager()
    
    # Test location (São Paulo)
    lat, lon = -23.5505, -46.6333
    
    print("🌐 Testing API Integrations...")
    print("=" * 50)
    
    # Get comprehensive data
    data = manager.get_comprehensive_data(lat, lon)
    
    print(f"📍 Location: {lat}, {lon}")
    print(f"🌤️  Weather: {data['weather']['condition']} ({data['weather']['temperature']:.1f}°C)")
    print(f"🚦 Traffic: {data['traffic']['congestion_level']} congestion")
    print(f"⛽ Fuel Price: ${data['fuel_prices']['gasoline_price']:.2f}/L")
    print(f"📊 Overall Condition: {data['delivery_conditions']['overall_condition'].upper()}")
    print(f"💡 Recommendation: {data['delivery_conditions']['recommendation']}")
    
    print("\n✅ API Integration testing complete!")
