"""
Delivery Route Optimization Package

This package provides tools for optimizing delivery routes using advanced algorithms
and geospatial data analysis techniques.

Modules:
- data_generator: Generate synthetic delivery and vehicle data
- route_optimizer: Implement various optimization algorithms (VRP, TSP)
- visualizer: Create interactive maps and visualizations

Author: IBM Data Analyst Professional Certificate Project
"""

__version__ = "1.0.0"
__author__ = "IBM Data Analyst Professional Certificate Student"

from .data_generator import DeliveryDataGenerator
from .route_optimizer import RouteOptimizer, Route
from .visualizer import RouteVisualizer

__all__ = [
    'DeliveryDataGenerator',
    'RouteOptimizer', 
    'Route',
    'RouteVisualizer'
]

