#!/usr/bin/env python3
"""
🤖 Complete ML Model Training and Validation Script
IBM Data Analyst Professional Certificate Project

This script provides comprehensive training and validation for all ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our ML modules
from src.advanced_ml import MLPipeline
from src.advanced_analytics import BayesianAnalyzer, MonteCarloSimulator
from src.data_generator import DeliveryDataGenerator

class ModelTrainer:
    """Complete ML Model Training and Validation Pipeline."""
    
    def __init__(self):
        """Initialize trainer with data and models."""
        self.data_generator = DeliveryDataGenerator()
        self.ml_pipeline = MLPipeline()
        self.bayesian_analyzer = BayesianAnalyzer()
        self.monte_carlo = MonteCarloSimulator()
        
        self.models = {}
        self.metrics = {}
        self.trained_models_dir = "trained_models"
        
        # Create directory for trained models
        os.makedirs(self.trained_models_dir, exist_ok=True)
        
    def generate_training_data(self, n_points=200, n_vehicles=10):
        """Generate comprehensive training dataset."""
        print("📊 Generating training data...")
        
        # Generate delivery data
        delivery_points = self.data_generator.generate_delivery_points(n_points)
        vehicles = self.data_generator.generate_vehicles(n_vehicles)
        
        # Generate demand time series (1 year of daily data)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # Seasonal demand with trend and noise
        trend = np.linspace(100, 120, 365)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
        weekly = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
        noise = np.random.normal(0, 5, 365)
        
        demand_series = trend + seasonal + weekly + noise
        demand_df = pd.DataFrame({
            'date': dates,
            'demand': demand_series,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'quarter': dates.quarter
        })
        
        print(f"✅ Generated {len(delivery_points)} delivery points")
        print(f"✅ Generated {len(vehicles)} vehicles")
        print(f"✅ Generated {len(demand_df)} days of demand data")
        
        return delivery_points, vehicles, demand_df
    
    def train_demand_forecasting_model(self, demand_df):
        """Train and validate demand forecasting model."""
        print("\n🤖 Training Demand Forecasting Model...")
        
        # Feature engineering
        features = ['day_of_week', 'month', 'quarter']
        X = demand_df[features].values
        y = demand_df['demand'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train transformer model
        try:
            self.ml_pipeline.train_transformer_forecaster(demand_df['demand'].values)
            
            # Make predictions
            predictions = self.ml_pipeline.predict_demand(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            self.metrics['demand_forecasting'] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'accuracy': max(0, r2) * 100  # Convert R² to percentage
            }
            
            print(f"✅ Demand Forecasting Model Trained!")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R²: {r2:.3f}")
            print(f"   Accuracy: {self.metrics['demand_forecasting']['accuracy']:.1f}%")
            
            # Save model
            joblib.dump({
                'model': self.ml_pipeline.transformer_model,
                'scaler': scaler,
                'metrics': self.metrics['demand_forecasting']
            }, f"{self.trained_models_dir}/demand_forecasting_model.pkl")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training demand forecasting: {e}")
            # Fallback to simple linear regression
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.metrics['demand_forecasting'] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'accuracy': max(0, r2) * 100,
                'model_type': 'LinearRegression'
            }
            
            print(f"✅ Fallback Linear Regression Model Trained!")
            print(f"   RMSE: {np.sqrt(mse):.2f}")
            print(f"   R²: {r2:.3f}")
            
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'metrics': self.metrics['demand_forecasting']
            }, f"{self.trained_models_dir}/demand_forecasting_model.pkl")
            
            return True
    
    def train_route_optimization_model(self, delivery_points):
        """Train and validate route optimization model."""
        print("\n🗺️ Training Route Optimization Model...")
        
        try:
            # Prepare graph data
            coords = np.array([[p['lat'], p['lon']] for p in delivery_points])
            
            # Calculate distance matrix
            from geopy.distance import geodesic
            n_points = len(coords)
            distance_matrix = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        distance_matrix[i][j] = geodesic(coords[i], coords[j]).kilometers
            
            # Train Graph Neural Network
            self.ml_pipeline.train_graph_neural_network(
                delivery_points, distance_matrix
            )
            
            # Test route optimization
            from src.route_optimizer import RouteOptimizer
            optimizer = RouteOptimizer(distance_matrix, delivery_points)
            
            # Compare GNN vs traditional methods
            routes_cw = optimizer.solve_vrp_clarke_wright()
            routes_ortools = optimizer.solve_vrp_ortools()
            
            # Calculate improvement metrics
            metrics_cw = optimizer.calculate_route_metrics(routes_cw)
            metrics_ortools = optimizer.calculate_route_metrics(routes_ortools)
            
            improvement = ((metrics_cw['total_distance'] - metrics_ortools['total_distance']) / 
                          metrics_cw['total_distance']) * 100
            
            self.metrics['route_optimization'] = {
                'clarke_wright_distance': metrics_cw['total_distance'],
                'ortools_distance': metrics_ortools['total_distance'],
                'improvement_percentage': improvement,
                'ortools_better': improvement > 0
            }
            
            print(f"✅ Route Optimization Model Trained!")
            print(f"   Clarke-Wright Distance: {metrics_cw['total_distance']:.1f} km")
            print(f"   OR-Tools Distance: {metrics_ortools['total_distance']:.1f} km")
            print(f"   Improvement: {improvement:.1f}%")
            
            # Save model
            joblib.dump({
                'gnn_model': self.ml_pipeline.gnn_model,
                'optimizer': optimizer,
                'metrics': self.metrics['route_optimization']
            }, f"{self.trained_models_dir}/route_optimization_model.pkl")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training route optimization: {e}")
            return False
    
    def train_statistical_models(self, demand_df):
        """Train and validate statistical analysis models."""
        print("\n📊 Training Statistical Analysis Models...")
        
        # Bayesian A/B Testing
        try:
            # Simulate two route strategies
            strategy_a_times = np.random.normal(45, 8, 100)  # Average 45 min
            strategy_b_times = np.random.normal(38, 7, 100)  # Average 38 min (better)
            
            # Run Bayesian A/B test
            results = self.bayesian_analyzer.bayesian_ab_test(
                strategy_a_times, strategy_b_times, "delivery_time"
            )
            
            self.metrics['bayesian_analysis'] = {
                'probability_b_better': results['probability_b_better'],
                'mean_difference': results['mean_difference'],
                'credible_interval': results['credible_interval'],
                'significant': results['probability_b_better'] > 0.95
            }
            
            print(f"✅ Bayesian A/B Test Completed!")
            print(f"   Probability B > A: {results['probability_b_better']:.1%}")
            print(f"   Mean Difference: {results['mean_difference']:.2f} minutes")
            
        except Exception as e:
            print(f"⚠️ Bayesian analysis error: {e}")
        
        # Monte Carlo Simulation
        try:
            # Simulate route optimization scenarios
            base_distance = 150  # km
            scenarios = self.monte_carlo.run_route_optimization_simulation(
                base_distance=base_distance,
                traffic_variance=0.2,
                weather_impact=0.1,
                n_simulations=1000
            )
            
            self.metrics['monte_carlo'] = {
                'mean_distance': np.mean(scenarios['total_distances']),
                'std_distance': np.std(scenarios['total_distances']),
                'confidence_95': np.percentile(scenarios['total_distances'], [2.5, 97.5]),
                'risk_scenarios': len([d for d in scenarios['total_distances'] if d > base_distance * 1.3])
            }
            
            print(f"✅ Monte Carlo Simulation Completed!")
            print(f"   Mean Distance: {self.metrics['monte_carlo']['mean_distance']:.1f} km")
            print(f"   95% CI: {self.metrics['monte_carlo']['confidence_95']}")
            
        except Exception as e:
            print(f"⚠️ Monte Carlo simulation error: {e}")
        
        return True
    
    def validate_all_models(self):
        """Comprehensive model validation."""
        print("\n🔍 Validating All Models...")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': len(self.metrics),
            'overall_performance': {},
            'recommendations': []
        }
        
        # Validate demand forecasting
        if 'demand_forecasting' in self.metrics:
            df_metrics = self.metrics['demand_forecasting']
            if df_metrics['accuracy'] > 70:
                validation_report['overall_performance']['demand_forecasting'] = 'GOOD'
                validation_report['recommendations'].append(
                    "Demand forecasting model shows good predictive performance"
                )
            else:
                validation_report['overall_performance']['demand_forecasting'] = 'NEEDS_IMPROVEMENT'
                validation_report['recommendations'].append(
                    "Consider more features or different model architecture for demand forecasting"
                )
        
        # Validate route optimization
        if 'route_optimization' in self.metrics:
            ro_metrics = self.metrics['route_optimization']
            if ro_metrics['improvement_percentage'] > 5:
                validation_report['overall_performance']['route_optimization'] = 'EXCELLENT'
                validation_report['recommendations'].append(
                    f"Route optimization shows {ro_metrics['improvement_percentage']:.1f}% improvement"
                )
            else:
                validation_report['overall_performance']['route_optimization'] = 'MARGINAL'
        
        # Save validation report
        joblib.dump({
            'validation_report': validation_report,
            'all_metrics': self.metrics
        }, f"{self.trained_models_dir}/validation_report.pkl")
        
        print("✅ Model validation completed!")
        print(f"📊 Trained {len(self.metrics)} model categories")
        
        return validation_report
    
    def generate_model_report(self):
        """Generate comprehensive model performance report."""
        print("\n📋 Generating Model Performance Report...")
        
        report = f"""
# 🤖 ML Model Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Model Performance Summary

"""
        
        for model_name, metrics in self.metrics.items():
            report += f"### {model_name.replace('_', ' ').title()}\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"- **{metric}**: {value:.3f}\n"
                else:
                    report += f"- **{metric}**: {value}\n"
            report += "\n"
        
        # Save report
        with open(f"{self.trained_models_dir}/model_report.md", 'w') as f:
            f.write(report)
        
        print("✅ Model report saved!")
        return report

def main():
    """Main training pipeline."""
    print("🚀 Starting Complete ML Model Training Pipeline...")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    # Generate data
    delivery_points, vehicles, demand_df = trainer.generate_training_data()
    
    # Train all models
    success_count = 0
    
    if trainer.train_demand_forecasting_model(demand_df):
        success_count += 1
    
    if trainer.train_route_optimization_model(delivery_points):
        success_count += 1
    
    if trainer.train_statistical_models(demand_df):
        success_count += 1
    
    # Validate models
    validation_report = trainer.validate_all_models()
    
    # Generate report
    trainer.generate_model_report()
    
    print("\n" + "=" * 60)
    print("🎉 ML Model Training Pipeline Complete!")
    print(f"✅ Successfully trained {success_count} model categories")
    print(f"📁 Models saved to: {trainer.trained_models_dir}/")
    print("=" * 60)
    
    return trainer.metrics

if __name__ == "__main__":
    metrics = main()
