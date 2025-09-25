#!/usr/bin/env python3
"""
🧪 Comprehensive Automated Test Suite
IBM Data Analyst Professional Certificate Project

Automated testing for all project components.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules to test
from src.data_generator import DeliveryDataGenerator
from src.route_optimizer import RouteOptimizer
from src.visualizer import RouteVisualizer
from src.advanced_analytics import BayesianAnalyzer, MonteCarloSimulator
from src.api_integrations import GoogleMapsAPI, WeatherAPI

class TestDataGeneration(unittest.TestCase):
    """Test data generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DeliveryDataGenerator()
    
    def test_delivery_points_generation(self):
        """Test delivery points generation."""
        points = self.generator.generate_delivery_points(50)
        
        self.assertEqual(len(points), 50)
        self.assertIn('lat', points[0])
        self.assertIn('lon', points[0])
        self.assertIn('address', points[0])
        
        # Test coordinate bounds (São Paulo area)
        for point in points:
            self.assertTrue(-24.0 <= point['lat'] <= -23.0)
            self.assertTrue(-47.0 <= point['lon'] <= -46.0)
    
    def test_vehicle_generation(self):
        """Test vehicle generation."""
        vehicles = self.generator.generate_vehicles(10)
        
        self.assertEqual(len(vehicles), 10)
        self.assertIn('vehicle_id', vehicles[0])
        self.assertIn('capacity_weight', vehicles[0])
        self.assertIn('driver_name', vehicles[0])
        
        # Test capacity ranges
        for vehicle in vehicles:
            self.assertTrue(500 <= vehicle['capacity_weight'] <= 2000)
    
    def test_distance_matrix_generation(self):
        """Test distance matrix generation."""
        points = self.generator.generate_delivery_points(10)
        matrix = self.generator.generate_distance_matrix(points)
        
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(matrix[0, 0], 0)  # Diagonal should be zero
        self.assertTrue(np.allclose(matrix, matrix.T))  # Should be symmetric

class TestRouteOptimization(unittest.TestCase):
    """Test route optimization algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DeliveryDataGenerator()
        self.points = self.generator.generate_delivery_points(20)
        self.vehicles = self.generator.generate_vehicles(3)
        self.distance_matrix = self.generator.generate_distance_matrix(self.points)
        self.optimizer = RouteOptimizer(self.distance_matrix, self.points, self.vehicles)
    
    def test_clarke_wright_algorithm(self):
        """Test Clarke-Wright algorithm."""
        routes = self.optimizer.solve_vrp_clarke_wright()
        
        self.assertIsInstance(routes, list)
        self.assertTrue(len(routes) > 0)
        
        # Test route structure
        for route in routes:
            self.assertIn('vehicle_id', route.__dict__)
            self.assertIn('stops', route.__dict__)
            self.assertIn('total_distance', route.__dict__)
    
    def test_ortools_optimization(self):
        """Test OR-Tools optimization."""
        try:
            routes = self.optimizer.solve_vrp_ortools()
            
            self.assertIsInstance(routes, list)
            self.assertTrue(len(routes) > 0)
            
            # Calculate metrics
            metrics = self.optimizer.calculate_route_metrics(routes)
            self.assertIn('total_distance', metrics)
            self.assertIn('total_time', metrics)
            self.assertGreater(metrics['total_distance'], 0)
            
        except ImportError:
            self.skipTest("OR-Tools not available")
    
    def test_route_metrics_calculation(self):
        """Test route metrics calculation."""
        routes = self.optimizer.solve_vrp_clarke_wright()
        metrics = self.optimizer.calculate_route_metrics(routes)
        
        required_metrics = ['total_distance', 'total_time', 'total_cost', 
                           'deliveries_covered', 'coverage_percentage']
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bayesian = BayesianAnalyzer()
        self.monte_carlo = MonteCarloSimulator()
    
    def test_bayesian_ab_test(self):
        """Test Bayesian A/B testing."""
        # Generate test data
        group_a = np.random.normal(50, 10, 100)
        group_b = np.random.normal(45, 8, 100)
        
        results = self.bayesian.bayesian_ab_test(group_a, group_b, "test_metric")
        
        self.assertIn('probability_b_better', results)
        self.assertIn('mean_difference', results)
        self.assertIn('credible_interval', results)
        
        # Probability should be between 0 and 1
        self.assertTrue(0 <= results['probability_b_better'] <= 1)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        results = self.monte_carlo.run_route_optimization_simulation(
            base_distance=100,
            traffic_variance=0.2,
            weather_impact=0.1,
            n_simulations=100
        )
        
        self.assertIn('total_distances', results)
        self.assertEqual(len(results['total_distances']), 100)
        self.assertTrue(all(d > 0 for d in results['total_distances']))

class TestAPIIntegrations(unittest.TestCase):
    """Test API integration components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.google_maps = GoogleMapsAPI()
        self.weather = WeatherAPI()
    
    def test_google_maps_simulation(self):
        """Test Google Maps API simulation fallback."""
        origins = [(40.7128, -74.0060), (40.7589, -73.9851)]
        destinations = [(40.6892, -74.0445), (40.7831, -73.9712)]
        
        # Test without API key (should use simulation)
        result = self.google_maps.get_distance_matrix(origins, destinations)
        
        self.assertIn('distances', result)
        self.assertIn('durations', result)
        self.assertEqual(len(result['distances']), len(origins))
    
    def test_weather_simulation(self):
        """Test weather API simulation fallback."""
        # Test without API key (should use simulation)
        result = self.weather.get_current_weather(40.7128, -74.0060)
        
        self.assertIn('temperature', result)
        self.assertIn('condition', result)
        self.assertIn('delivery_impact', result)

class TestVisualization(unittest.TestCase):
    """Test visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DeliveryDataGenerator()
        self.points = self.generator.generate_delivery_points(10)
        self.vehicles = self.generator.generate_vehicles(2)
        self.visualizer = RouteVisualizer(self.points, self.vehicles)
    
    def test_route_map_creation(self):
        """Test route map creation."""
        # Create mock routes
        from src.route_optimizer import Route
        
        route1 = Route("VEH001")
        route1.stops = self.points[:5]
        route1.total_distance = 50.0
        
        routes = [route1]
        
        # Test map creation (should not raise exceptions)
        try:
            route_map = self.visualizer.visualize_routes(routes)
            self.assertIsNotNone(route_map)
        except Exception as e:
            self.fail(f"Route visualization failed: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Generate data
        generator = DeliveryDataGenerator()
        points = generator.generate_delivery_points(15)
        vehicles = generator.generate_vehicles(3)
        distance_matrix = generator.generate_distance_matrix(points)
        
        # Optimize routes
        optimizer = RouteOptimizer(distance_matrix, points, vehicles)
        routes_cw = optimizer.solve_vrp_clarke_wright()
        
        # Calculate metrics
        metrics = optimizer.calculate_route_metrics(routes_cw)
        
        # Validate results
        self.assertGreater(len(routes_cw), 0)
        self.assertGreater(metrics['total_distance'], 0)
        self.assertEqual(metrics['deliveries_covered'], len(points) - 1)  # Excluding depot
    
    def test_data_pipeline_consistency(self):
        """Test data pipeline consistency."""
        generator = DeliveryDataGenerator()
        
        # Generate same data twice with same seed
        np.random.seed(42)
        points1 = generator.generate_delivery_points(20)
        
        np.random.seed(42)
        points2 = generator.generate_delivery_points(20)
        
        # Should be identical
        self.assertEqual(len(points1), len(points2))
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1['lat'], p2['lat'])
            self.assertEqual(p1['lon'], p2['lon'])

class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        import time
        
        generator = DeliveryDataGenerator()
        
        start_time = time.time()
        points = generator.generate_delivery_points(100)
        vehicles = generator.generate_vehicles(10)
        distance_matrix = generator.generate_distance_matrix(points)
        generation_time = time.time() - start_time
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(generation_time, 10.0)
        
        # Test optimization performance
        optimizer = RouteOptimizer(distance_matrix, points, vehicles)
        
        start_time = time.time()
        routes = optimizer.solve_vrp_clarke_wright()
        optimization_time = time.time() - start_time
        
        # Should complete within reasonable time (30 seconds)
        self.assertLess(optimization_time, 30.0)

def run_test_suite():
    """Run the complete test suite."""
    print("🧪 Starting Automated Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestDataGeneration,
        TestRouteOptimization,
        TestStatisticalAnalysis,
        TestAPIIntegrations,
        TestVisualization,
        TestIntegration,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("🧪 Test Suite Results:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    print(f"⏭️ Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
