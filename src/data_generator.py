"""
Data Generator for Delivery Route Optimization Project
Generates synthetic data for vehicles, delivery points, and traffic conditions.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import json

class DeliveryDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Define the operational area (São Paulo city center as example)
        self.center_lat = -23.5505
        self.center_lon = -46.6333
        self.radius = 0.1  # Approximately 10km radius
        
    def generate_delivery_points(self, n_points: int = 50) -> pd.DataFrame:
        """Generate random delivery points within the operational area."""
        delivery_points = []
        
        for i in range(n_points):
            # Generate random coordinates within the radius
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, self.radius)
            
            lat = self.center_lat + r * np.cos(angle)
            lon = self.center_lon + r * np.sin(angle)
            
            delivery_points.append({
                'delivery_id': f'DEL_{i+1:03d}',
                'customer_name': self.fake.company(),
                'address': self.fake.street_address(),
                'latitude': lat,
                'longitude': lon,
                'package_weight': random.uniform(0.5, 25.0),  # kg
                'package_volume': random.uniform(0.1, 2.0),   # m³
                'priority': random.choice(['low', 'medium', 'high']),
                'time_window_start': random.randint(8, 14),   # hour
                'time_window_end': random.randint(15, 18),    # hour
                'service_time': random.randint(5, 30),        # minutes
                'delivery_date': datetime.now().date()
            })
            
        return pd.DataFrame(delivery_points)
    
    def generate_vehicles(self, n_vehicles: int = 5) -> pd.DataFrame:
        """Generate vehicle fleet data."""
        vehicles = []
        
        for i in range(n_vehicles):
            vehicles.append({
                'vehicle_id': f'VEH_{i+1:03d}',
                'driver_name': self.fake.name(),
                'vehicle_type': random.choice(['van', 'truck', 'motorcycle']),
                'capacity_weight': random.uniform(100, 1000),  # kg
                'capacity_volume': random.uniform(5, 20),      # m³
                'fuel_consumption': random.uniform(8, 15),     # km/l
                'avg_speed': random.uniform(25, 45),           # km/h
                'start_latitude': self.center_lat + random.uniform(-0.02, 0.02),
                'start_longitude': self.center_lon + random.uniform(-0.02, 0.02),
                'available_start_time': random.randint(6, 8),  # hour
                'available_end_time': random.randint(17, 20),  # hour
                'cost_per_km': random.uniform(0.5, 2.0)       # currency per km
            })
            
        return pd.DataFrame(vehicles)
    
    def generate_distance_matrix(self, points: pd.DataFrame) -> np.ndarray:
        """Generate distance matrix between all points using Haversine formula."""
        n_points = len(points)
        distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    distance_matrix[i][j] = self._haversine_distance(
                        points.iloc[i]['latitude'], points.iloc[i]['longitude'],
                        points.iloc[j]['latitude'], points.iloc[j]['longitude']
                    )
                    
        return distance_matrix
    
    def generate_traffic_data(self, n_points: int, time_periods: List[str] = None) -> Dict:
        """Generate traffic multiplier data for different time periods."""
        if time_periods is None:
            time_periods = ['morning_rush', 'midday', 'afternoon_rush', 'evening']
        
        traffic_data = {}
        
        for period in time_periods:
            # Traffic multipliers (1.0 = normal, >1.0 = slower, <1.0 = faster)
            if period in ['morning_rush', 'afternoon_rush']:
                multipliers = np.random.uniform(1.2, 2.5, (n_points, n_points))
            elif period == 'midday':
                multipliers = np.random.uniform(0.8, 1.3, (n_points, n_points))
            else:  # evening
                multipliers = np.random.uniform(0.7, 1.2, (n_points, n_points))
            
            # Diagonal should be 0 (same point)
            np.fill_diagonal(multipliers, 0)
            traffic_data[period] = multipliers
            
        return traffic_data
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the Haversine distance between two points in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def generate_enhanced_delivery_orders(self, n_orders: int = 200) -> pd.DataFrame:
        """Generate enhanced delivery orders with customer segments and business metrics."""
        orders = []
        
        # Customer segments with different characteristics
        segments = {
            'high_value': {'weight': 0.2, 'order_min': 150, 'order_max': 500, 'loyalty_min': 0.7, 'loyalty_max': 1.0},
            'medium_value': {'weight': 0.5, 'order_min': 50, 'order_max': 200, 'loyalty_min': 0.4, 'loyalty_max': 0.8},
            'low_value': {'weight': 0.3, 'order_min': 10, 'order_max': 80, 'loyalty_min': 0.1, 'loyalty_max': 0.6}
        }
        
        for i in range(n_orders):
            # Select customer segment based on weights
            segment = np.random.choice(
                list(segments.keys()), 
                p=[segments[s]['weight'] for s in segments.keys()]
            )
            segment_info = segments[segment]
            
            # Generate coordinates within operational area
            lat = self.center_lat + np.random.normal(0, self.radius)
            lon = self.center_lon + np.random.normal(0, self.radius)
            
            # Generate order characteristics based on segment
            order_value = np.random.uniform(segment_info['order_min'], segment_info['order_max'])
            customer_loyalty = np.random.uniform(segment_info['loyalty_min'], segment_info['loyalty_max'])
            
            # Package characteristics (correlated with order value)
            package_weight = max(0.5, np.random.normal(order_value * 0.05, order_value * 0.01))
            package_volume = max(0.1, package_weight * np.random.uniform(0.3, 0.8))
            
            # Service time (correlated with package complexity)
            base_service_time = 5 + (package_weight * 0.5) + (package_volume * 2)
            service_time = max(2, np.random.normal(base_service_time, base_service_time * 0.2))
            
            # Time sensitivity (higher for high-value customers)
            if segment == 'high_value':
                time_sensitivity = np.random.uniform(0.7, 1.0)
            elif segment == 'medium_value':
                time_sensitivity = np.random.uniform(0.4, 0.8)
            else:
                time_sensitivity = np.random.uniform(0.1, 0.6)
            
            # Priority based on segment and time sensitivity
            if segment == 'high_value' and time_sensitivity > 0.8:
                priority = 'high'
            elif segment == 'medium_value' or (segment == 'high_value' and time_sensitivity <= 0.8):
                priority = 'medium'
            else:
                priority = 'low'
            
            # Time windows
            time_window_start = np.random.randint(8, 16)  # 8 AM to 4 PM
            time_window_duration = np.random.choice([2, 3, 4, 6, 8], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            time_window_end = min(18, time_window_start + time_window_duration)
            
            order = {
                'order_id': f'ORD_{i+1:04d}',
                'customer_segment': segment,
                'latitude': lat,
                'longitude': lon,
                'package_weight': round(package_weight, 2),
                'package_volume': round(package_volume, 3),
                'service_time': round(service_time, 1),
                'order_value': round(order_value, 2),
                'customer_loyalty': round(customer_loyalty, 3),
                'time_sensitivity': round(time_sensitivity, 3),
                'priority': priority,
                'time_window_start': time_window_start,
                'time_window_end': time_window_end,
                'delivery_date': self.fake.date_between(start_date='-30d', end_date='+30d')
            }
            orders.append(order)
        
        return pd.DataFrame(orders)
    
    def generate_customers_df(self, n_customers: int = 100) -> pd.DataFrame:
        """Generate customer database with profiles and history."""
        customers = []
        
        for i in range(n_customers):
            # Determine customer segment
            segment_prob = np.random.random()
            if segment_prob < 0.2:
                segment = 'high_value'
                avg_order_value = np.random.uniform(200, 400)
                order_frequency = np.random.uniform(15, 30)  # orders per month
            elif segment_prob < 0.7:
                segment = 'medium_value'
                avg_order_value = np.random.uniform(50, 150)
                order_frequency = np.random.uniform(5, 20)
            else:
                segment = 'low_value'
                avg_order_value = np.random.uniform(10, 60)
                order_frequency = np.random.uniform(1, 8)
            
            customer = {
                'customer_id': f'CUST_{i+1:04d}',
                'customer_name': self.fake.company(),
                'segment': segment,
                'registration_date': self.fake.date_between(start_date='-2y', end_date='-1m'),
                'avg_order_value': round(avg_order_value, 2),
                'order_frequency_monthly': round(order_frequency, 1),
                'total_orders': np.random.randint(int(order_frequency * 0.5), int(order_frequency * 2)),
                'customer_lifetime_value': round(avg_order_value * order_frequency * 12, 2),
                'preferred_delivery_time': np.random.choice(['morning', 'afternoon', 'evening']),
                'location_type': np.random.choice(['residential', 'commercial', 'industrial'], p=[0.4, 0.5, 0.1])
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_demand_time_series(self, days: int = 730) -> np.ndarray:
        """Generate realistic demand time series with multiple patterns."""
        # Base demand with growth trend
        base_demand = 80 + np.linspace(0, 40, days)
        
        # Seasonal components
        yearly_cycle = 25 * np.sin(2 * np.pi * np.arange(days) / 365.25)
        weekly_cycle = 15 * np.sin(2 * np.pi * np.arange(days) / 7)
        monthly_cycle = 10 * np.sin(2 * np.pi * np.arange(days) / 30.44)
        
        # Holiday effects
        holiday_effects = np.zeros(days)
        holiday_dates = [60, 120, 180, 240, 300, 330, 425, 485, 545, 605, 665, 695]
        
        for holiday in holiday_dates:
            if holiday < days:
                for i in range(max(0, holiday-3), min(days, holiday+4)):
                    holiday_effects[i] += 30 * np.exp(-0.5 * ((i - holiday) ** 2))
        
        # Weekend effects
        weekend_effects = np.array([20 if i % 7 >= 5 else 0 for i in range(days)])
        
        # Random noise with seasonal volatility
        volatility = 8 + 4 * np.sin(2 * np.pi * np.arange(days) / 365.25)
        noise = np.random.normal(0, volatility)
        
        # Combine all components
        demand = base_demand + yearly_cycle + weekly_cycle + monthly_cycle + holiday_effects + weekend_effects + noise
        return np.maximum(demand, 10)  # Ensure minimum demand
    
    def save_generated_data(self, output_dir: str = "data"):
        """Generate and save all datasets."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎲 Generating comprehensive datasets...")
        
        # Generate all data
        delivery_points = self.generate_delivery_points(50)
        vehicles = self.generate_vehicles(5)
        distance_matrix = self.generate_distance_matrix(delivery_points)
        traffic_data = self.generate_traffic_data(len(delivery_points))
        
        # Generate enhanced datasets for advanced analysis
        delivery_orders = self.generate_enhanced_delivery_orders(200)
        customers_df = self.generate_customers_df(100)
        demand_series = self.generate_demand_time_series(730)
        
        print("💾 Saving all datasets...")
        
        # Save basic data
        delivery_points.to_csv(f"{output_dir}/delivery_points.csv", index=False)
        vehicles.to_csv(f"{output_dir}/vehicles.csv", index=False)
        np.save(f"{output_dir}/distance_matrix.npy", distance_matrix)
        
        # Save enhanced data
        delivery_orders.to_csv(f"{output_dir}/delivery_orders.csv", index=False)
        customers_df.to_csv(f"{output_dir}/customers.csv", index=False)
        np.save(f"{output_dir}/demand_series.npy", demand_series)
        
        # Save traffic data
        with open(f"{output_dir}/traffic_data.json", 'w') as f:
            traffic_json = {k: v.tolist() for k, v in traffic_data.items()}
            json.dump(traffic_json, f, indent=2)
        
        print(f"✅ All data generated and saved to {output_dir}/")
        print(f"📊 Generated:")
        print(f"  - {len(delivery_points)} delivery points")
        print(f"  - {len(vehicles)} vehicles")
        print(f"  - {len(delivery_orders)} delivery orders")
        print(f"  - {len(customers_df)} customers")
        print(f"  - {len(demand_series)} days of demand data")
        
        return delivery_points, vehicles, distance_matrix, traffic_data, delivery_orders, customers_df, demand_series

if __name__ == "__main__":
    generator = DeliveryDataGenerator()
    results = generator.save_generated_data()
    
    print("\n🎉 Data generation completed successfully!")
    print("📁 Files created in 'data/' directory:")
    print("  - delivery_points.csv")
    print("  - vehicles.csv")
    print("  - delivery_orders.csv")
    print("  - customers.csv")
    print("  - distance_matrix.npy")
    print("  - demand_series.npy")
    print("  - traffic_data.json")
    
    print("\n✅ Ready for analysis in Jupyter notebook!")
