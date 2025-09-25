#!/usr/bin/env python3
"""
Test script to validate the notebook functionality
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test if all data files exist and can be loaded"""
    print("🧪 Testing data loading...")
    
    try:
        # Test basic data
        delivery_points = pd.read_csv("data/delivery_points.csv")
        vehicles = pd.read_csv("data/vehicles.csv")
        distance_matrix = np.load("data/distance_matrix.npy")
        
        with open("data/traffic_data.json", 'r') as f:
            traffic_data = json.load(f)
        
        # Test enhanced data
        delivery_orders = pd.read_csv("data/delivery_orders.csv")
        customers_df = pd.read_csv("data/customers.csv")
        demand_series = np.load("data/demand_series.npy")
        
        print("✅ All data files loaded successfully!")
        print(f"  📊 {len(delivery_points)} delivery points")
        print(f"  🚛 {len(vehicles)} vehicles")
        print(f"  📦 {len(delivery_orders)} delivery orders")
        print(f"  👥 {len(customers_df)} customers")
        print(f"  📈 {len(demand_series)} days of demand data")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_data_structure():
    """Test if data has the expected structure"""
    print("\n🧪 Testing data structure...")
    
    try:
        delivery_orders = pd.read_csv("data/delivery_orders.csv")
        
        # Check required columns
        required_cols = ['customer_segment', 'order_value', 'customer_loyalty', 'time_sensitivity']
        missing_cols = [col for col in required_cols if col not in delivery_orders.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Check data types and ranges
        print("✅ All required columns present!")
        print(f"  Customer segments: {delivery_orders['customer_segment'].unique()}")
        print(f"  Order value range: ${delivery_orders['order_value'].min():.2f} - ${delivery_orders['order_value'].max():.2f}")
        print(f"  Customer loyalty range: {delivery_orders['customer_loyalty'].min():.3f} - {delivery_orders['customer_loyalty'].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_visualizations():
    """Test if visualizations can be created"""
    print("\n🧪 Testing visualization creation...")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        delivery_orders = pd.read_csv("data/delivery_orders.csv")
        
        # Test simple pie chart
        segment_counts = delivery_orders['customer_segment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.4
        )])
        
        print("✅ Plotly visualizations working!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING NOTEBOOK FUNCTIONALITY")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_data_structure,
        test_visualizations
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Notebook should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
