"""
Visualization Module for Delivery Route Optimization
Creates interactive maps and charts to visualize routes, traffic, and optimization results.
"""

import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
import branca.colormap as cm
from folium import plugins
import json

class RouteVisualizer:
    def __init__(self, delivery_points: pd.DataFrame, vehicles: pd.DataFrame, 
                 distance_matrix: np.ndarray = None, traffic_data: Dict = None):
        """Initialize the visualizer with data."""
        self.delivery_points = delivery_points
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.traffic_data = traffic_data or {}
        
        # Color palette for different routes
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 
                      'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                      'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue',
                      'lightgreen', 'gray', 'black', 'lightgray']
    
    def create_base_map(self, zoom_start: int = 12) -> folium.Map:
        """Create a base map centered on the delivery area."""
        center_lat = self.delivery_points['latitude'].mean()
        center_lon = self.delivery_points['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        return m
    
    def add_delivery_points(self, m: folium.Map, show_details: bool = True) -> folium.Map:
        """Add delivery points to the map."""
        for idx, point in self.delivery_points.iterrows():
            # Create popup text
            if show_details:
                popup_text = f"""
                <b>{point['customer_name']}</b><br>
                ID: {point['delivery_id']}<br>
                Address: {point['address']}<br>
                Weight: {point['package_weight']:.1f} kg<br>
                Volume: {point['package_volume']:.1f} m³<br>
                Priority: {point['priority']}<br>
                Time Window: {point['time_window_start']}:00 - {point['time_window_end']}:00<br>
                Service Time: {point['service_time']} min
                """
            else:
                popup_text = f"{point['delivery_id']}<br>{point['customer_name']}"
            
            # Color based on priority
            color = {'high': 'red', 'medium': 'orange', 'low': 'green'}.get(
                point['priority'], 'blue')
            
            # Add marker
            folium.Marker(
                location=[point['latitude'], point['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=point['delivery_id'],
                icon=folium.Icon(color=color, icon='package', prefix='fa')
            ).add_to(m)
        
        return m
    
    def add_depot(self, m: folium.Map, depot_idx: int = 0) -> folium.Map:
        """Add depot marker to the map."""
        depot = self.delivery_points.iloc[depot_idx]
        
        folium.Marker(
            location=[depot['latitude'], depot['longitude']],
            popup=f"<b>DEPOT</b><br>{depot['address']}",
            tooltip="Distribution Center",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa', size='large')
        ).add_to(m)
        
        return m
    
    def visualize_routes(self, routes: List, save_path: str = None, 
                        show_traffic: bool = False, traffic_period: str = 'midday') -> folium.Map:
        """Create an interactive map showing optimized routes."""
        m = self.create_base_map()
        
        # Add depot
        m = self.add_depot(m)
        
        # Add delivery points
        m = self.add_delivery_points(m, show_details=True)
        
        # Add traffic layer if requested
        if show_traffic and self.traffic_data and traffic_period in self.traffic_data:
            self._add_traffic_layer(m, traffic_period)
        
        # Add routes
        for i, route in enumerate(routes):
            color = self.colors[i % len(self.colors)]
            self._add_route_to_map(m, route, color, i)
        
        # Add legend
        self._add_route_legend(m, routes)
        
        # Add route statistics
        self._add_route_statistics(m, routes)
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def _add_route_to_map(self, m: folium.Map, route, color: str, route_idx: int):
        """Add a single route to the map."""
        # Get vehicle info
        vehicle = self.vehicles[self.vehicles['vehicle_id'] == route.vehicle_id].iloc[0]
        
        # Create route coordinates (depot -> stops -> depot)
        depot_coord = [self.delivery_points.iloc[0]['latitude'], 
                      self.delivery_points.iloc[0]['longitude']]
        
        route_coords = [depot_coord]
        
        # Add all stops
        for stop_idx in route.stops:
            stop_coord = [self.delivery_points.iloc[stop_idx]['latitude'],
                         self.delivery_points.iloc[stop_idx]['longitude']]
            route_coords.append(stop_coord)
        
        # Return to depot
        route_coords.append(depot_coord)
        
        # Add route line
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"""
            <b>Route {route_idx + 1}</b><br>
            Vehicle: {route.vehicle_id}<br>
            Driver: {vehicle['driver_name']}<br>
            Distance: {route.total_distance:.1f} km<br>
            Time: {route.total_time:.0f} min<br>
            Cost: ${route.total_cost:.2f}<br>
            Stops: {len(route.stops)}<br>
            Load: {route.load_weight:.1f} kg, {route.load_volume:.1f} m³
            """,
            tooltip=f"Route {route_idx + 1} - {route.vehicle_id}"
        ).add_to(m)
        
        # Add direction arrows
        for i in range(len(route_coords) - 1):
            mid_lat = (route_coords[i][0] + route_coords[i+1][0]) / 2
            mid_lon = (route_coords[i][1] + route_coords[i+1][1]) / 2
            
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.Icon(color=color, icon='arrow-right', prefix='fa'),
                tooltip=f"Route {route_idx + 1} direction"
            ).add_to(m)
    
    def _add_traffic_layer(self, m: folium.Map, traffic_period: str):
        """Add traffic visualization as a heatmap layer."""
        if traffic_period not in self.traffic_data:
            return
        
        traffic_matrix = self.traffic_data[traffic_period]
        
        # Create heatmap data
        heat_data = []
        for i in range(len(self.delivery_points)):
            for j in range(len(self.delivery_points)):
                if i != j and traffic_matrix[i][j] > 1.2:  # Only show congested areas
                    lat = self.delivery_points.iloc[i]['latitude']
                    lon = self.delivery_points.iloc[i]['longitude']
                    intensity = min(traffic_matrix[i][j], 3.0)  # Cap intensity
                    heat_data.append([lat, lon, intensity])
        
        if heat_data:
            plugins.HeatMap(heat_data, name=f'Traffic - {traffic_period}').add_to(m)
    
    def _add_route_legend(self, m: folium.Map, routes: List):
        """Add a legend showing route information."""
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Route Legend</h4>
        '''
        
        for i, route in enumerate(routes):
            color = self.colors[i % len(self.colors)]
            legend_html += f'''
            <p><span style="color:{color};">●</span> Route {i+1}: {route.vehicle_id} 
            ({len(route.stops)} stops, {route.total_distance:.1f}km)</p>
            '''
        
        legend_html += '''
        <hr>
        <p><span style="color:red;">📦</span> High Priority</p>
        <p><span style="color:orange;">📦</span> Medium Priority</p>
        <p><span style="color:green;">📦</span> Low Priority</p>
        <p><span style="color:black;">🏢</span> Depot</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_route_statistics(self, m: folium.Map, routes: List):
        """Add route statistics panel."""
        total_distance = sum(route.total_distance for route in routes)
        total_time = sum(route.total_time for route in routes)
        total_cost = sum(route.total_cost for route in routes)
        total_stops = sum(len(route.stops) for route in routes)
        
        stats_html = f'''
        <div style="position: fixed; 
                    bottom: 10px; left: 10px; width: 250px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Route Statistics</h4>
        <p><b>Total Distance:</b> {total_distance:.1f} km</p>
        <p><b>Total Time:</b> {total_time:.0f} minutes</p>
        <p><b>Total Cost:</b> ${total_cost:.2f}</p>
        <p><b>Vehicles Used:</b> {len(routes)}</p>
        <p><b>Deliveries:</b> {total_stops}</p>
        <p><b>Avg Route Length:</b> {total_distance/len(routes):.1f} km</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(stats_html))
    
    def create_comparison_dashboard(self, route_solutions: Dict, save_path: str = None) -> go.Figure:
        """Create a dashboard comparing different optimization methods."""
        methods = list(route_solutions.keys())
        
        # Extract metrics
        distances = [sum(route.total_distance for route in routes) 
                    for routes in route_solutions.values()]
        times = [sum(route.total_time for route in routes) 
                for routes in route_solutions.values()]
        costs = [sum(route.total_cost for route in routes) 
                for routes in route_solutions.values()]
        vehicles_used = [len(routes) for routes in route_solutions.values()]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Distance (km)', 'Total Time (minutes)', 
                          'Total Cost ($)', 'Vehicles Used'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar charts
        fig.add_trace(go.Bar(x=methods, y=distances, name='Distance', 
                           marker_color='blue'), row=1, col=1)
        fig.add_trace(go.Bar(x=methods, y=times, name='Time', 
                           marker_color='green'), row=1, col=2)
        fig.add_trace(go.Bar(x=methods, y=costs, name='Cost', 
                           marker_color='red'), row=2, col=1)
        fig.add_trace(go.Bar(x=methods, y=vehicles_used, name='Vehicles', 
                           marker_color='orange'), row=2, col=2)
        
        fig.update_layout(
            title_text="Route Optimization Methods Comparison",
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_route_analysis_charts(self, routes: List, save_path: str = None) -> go.Figure:
        """Create detailed analysis charts for the routes."""
        # Prepare data
        route_data = []
        for i, route in enumerate(routes):
            vehicle = self.vehicles[self.vehicles['vehicle_id'] == route.vehicle_id].iloc[0]
            route_data.append({
                'route_id': f'Route {i+1}',
                'vehicle_id': route.vehicle_id,
                'vehicle_type': vehicle['vehicle_type'],
                'distance': route.total_distance,
                'time': route.total_time,
                'cost': route.total_cost,
                'stops': len(route.stops),
                'load_weight': route.load_weight,
                'load_volume': route.load_volume,
                'capacity_weight': vehicle['capacity_weight'],
                'capacity_volume': vehicle['capacity_volume'],
                'weight_utilization': (route.load_weight / vehicle['capacity_weight']) * 100,
                'volume_utilization': (route.load_volume / vehicle['capacity_volume']) * 100
            })
        
        df = pd.DataFrame(route_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Distance vs Stops', 'Capacity Utilization', 
                          'Cost per Route', 'Time per Route',
                          'Vehicle Type Distribution', 'Load Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Distance vs Stops scatter
        fig.add_trace(go.Scatter(
            x=df['stops'], y=df['distance'],
            mode='markers+text',
            text=df['route_id'],
            textposition="top center",
            marker=dict(size=10, color='blue'),
            name='Distance vs Stops'
        ), row=1, col=1)
        
        # Capacity utilization
        fig.add_trace(go.Bar(
            x=df['route_id'],
            y=df['weight_utilization'],
            name='Weight Utilization (%)',
            marker_color='lightblue'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=df['route_id'],
            y=df['volume_utilization'],
            name='Volume Utilization (%)',
            marker_color='lightcoral'
        ), row=1, col=2)
        
        # Cost per route
        fig.add_trace(go.Bar(
            x=df['route_id'],
            y=df['cost'],
            name='Cost',
            marker_color='green'
        ), row=2, col=1)
        
        # Time per route
        fig.add_trace(go.Bar(
            x=df['route_id'],
            y=df['time'],
            name='Time',
            marker_color='orange'
        ), row=2, col=2)
        
        # Vehicle type distribution
        vehicle_counts = df['vehicle_type'].value_counts()
        fig.add_trace(go.Pie(
            labels=vehicle_counts.index,
            values=vehicle_counts.values,
            name='Vehicle Types'
        ), row=3, col=1)
        
        # Load distribution
        fig.add_trace(go.Histogram(
            x=df['load_weight'],
            name='Load Weight Distribution',
            marker_color='purple'
        ), row=3, col=2)
        
        fig.update_layout(
            title_text="Detailed Route Analysis",
            height=1000,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_traffic_heatmap(self, traffic_period: str = 'midday', save_path: str = None) -> go.Figure:
        """Create a heatmap showing traffic conditions."""
        if traffic_period not in self.traffic_data:
            return None
        
        traffic_matrix = np.array(self.traffic_data[traffic_period])
        
        # Create labels for the heatmap
        labels = [f"Point {i}" for i in range(len(traffic_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=traffic_matrix,
            x=labels,
            y=labels,
            colorscale='Reds',
            colorbar=dict(title="Traffic Multiplier"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Traffic Conditions - {traffic_period.title()}',
            xaxis_title='Destination Points',
            yaxis_title='Origin Points',
            width=600,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def export_route_summary(self, routes: List, filename: str = "route_summary.csv"):
        """Export route summary to CSV."""
        summary_data = []
        
        for i, route in enumerate(routes):
            vehicle = self.vehicles[self.vehicles['vehicle_id'] == route.vehicle_id].iloc[0]
            
            summary_data.append({
                'route_number': i + 1,
                'vehicle_id': route.vehicle_id,
                'driver_name': vehicle['driver_name'],
                'vehicle_type': vehicle['vehicle_type'],
                'stops': len(route.stops),
                'stop_ids': ', '.join([self.delivery_points.iloc[stop]['delivery_id'] 
                                     for stop in route.stops]),
                'total_distance_km': round(route.total_distance, 2),
                'total_time_minutes': round(route.total_time, 0),
                'total_cost': round(route.total_cost, 2),
                'load_weight_kg': round(route.load_weight, 2),
                'load_volume_m3': round(route.load_volume, 2),
                'capacity_utilization_weight_pct': round(
                    (route.load_weight / vehicle['capacity_weight']) * 100, 1),
                'capacity_utilization_volume_pct': round(
                    (route.load_volume / vehicle['capacity_volume']) * 100, 1)
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(filename, index=False)
        
        return df_summary
