#!/usr/bin/env python3
"""
🧠 Advanced Machine Learning & Deep Learning Models
Delivery Route Optimization - Advanced Analytics

Implements cutting-edge ML techniques:
- Transformer Networks for demand forecasting
- Graph Neural Networks for route optimization  
- Reinforcement Learning for dynamic routing
- Advanced ensemble methods
- Bayesian optimization

Author: IBM Data Analyst Professional Certificate Student
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class TransformerDemandForecaster(nn.Module):
    """
    🤖 Transformer Network for Advanced Demand Forecasting
    
    Uses attention mechanisms to capture complex temporal patterns
    in delivery demand data.
    """
    
    def __init__(self, input_dim=10, d_model=64, nhead=8, num_layers=3, 
                 seq_length=30, forecast_horizon=7):
        super(TransformerDemandForecaster, self).__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, forecast_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Input embedding
        x = self.input_embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use the last time step for forecasting
        last_hidden = transformer_out[:, -1, :]
        
        # Generate forecast
        forecast = self.fc_out(last_hidden)
        
        return forecast


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GraphNeuralNetwork(nn.Module):
    """
    🕸️ Graph Neural Network for Route Optimization
    
    Models the delivery network as a graph and uses GNN to learn
    optimal routing patterns.
    """
    
    def __init__(self, node_features=10, edge_features=5, hidden_dim=64, output_dim=1):
        super(GraphNeuralNetwork, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Node embedding layers
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding layers
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.gconv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gconv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gconv3 = GraphConvLayer(hidden_dim, hidden_dim)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, node_features, edge_features, adjacency_matrix):
        # Embed nodes and edges
        node_embed = self.node_embed(node_features)
        edge_embed = self.edge_embed(edge_features)
        
        # Graph convolutions
        h1 = F.relu(self.gconv1(node_embed, edge_embed, adjacency_matrix))
        h2 = F.relu(self.gconv2(h1, edge_embed, adjacency_matrix))
        h3 = F.relu(self.gconv3(h2, edge_embed, adjacency_matrix))
        
        # Global pooling (mean of all nodes)
        graph_representation = torch.mean(h3, dim=0)
        
        # Output
        output = self.output(graph_representation)
        
        return output, h3


class GraphConvLayer(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, node_features, edge_features, adjacency_matrix):
        # Simple graph convolution: aggregate neighbor features
        # node_features: (num_nodes, in_features)
        # adjacency_matrix: (num_nodes, num_nodes)
        
        # Aggregate neighbor information
        aggregated = torch.matmul(adjacency_matrix, node_features)
        
        # Apply linear transformation
        output = self.linear(aggregated)
        
        return output


class ReinforcementLearningRouter:
    """
    🎯 Reinforcement Learning for Dynamic Route Optimization
    
    Uses Deep Q-Learning to learn optimal routing decisions
    in dynamic environments.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Target network for stability
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # RL parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss and optimization
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class BayesianOptimizer:
    """
    📊 Bayesian Optimization for Hyperparameter Tuning
    
    Uses Gaussian Processes to efficiently search hyperparameter space.
    """
    
    def __init__(self, bounds, acquisition='ei'):
        self.bounds = bounds
        self.acquisition = acquisition
        self.X_sample = []
        self.y_sample = []
    
    def objective_function(self, params):
        """
        Objective function to optimize.
        This should be implemented based on the specific optimization task.
        """
        # Placeholder implementation
        return np.random.random()
    
    def gaussian_process_surrogate(self, X, y, X_test):
        """Simple GP surrogate model."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        gp.fit(X, y)
        mu, sigma = gp.predict(X_test, return_std=True)
        
        return mu, sigma
    
    def acquisition_function(self, X, X_sample, y_sample):
        """Expected improvement acquisition function."""
        mu, sigma = self.gaussian_process_surrogate(X_sample, y_sample, X)
        
        if len(y_sample) == 0:
            return np.zeros(len(X))
        
        mu_sample_opt = np.max(y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def optimize(self, n_iterations=50):
        """Run Bayesian optimization."""
        from scipy.stats import norm
        from scipy.optimize import minimize as scipy_minimize
        
        for i in range(n_iterations):
            # Sample random point if no samples yet
            if len(self.X_sample) == 0:
                X_next = np.random.uniform(self.bounds[0], self.bounds[1], 
                                         size=(1, len(self.bounds)))
            else:
                # Optimize acquisition function
                def neg_acquisition(x):
                    return -self.acquisition_function(x.reshape(1, -1), 
                                                    np.array(self.X_sample), 
                                                    np.array(self.y_sample))
                
                result = scipy_minimize(neg_acquisition, 
                                      x0=np.random.uniform(self.bounds[0], self.bounds[1]),
                                      bounds=list(zip(self.bounds[0], self.bounds[1])),
                                      method='L-BFGS-B')
                
                X_next = result.x.reshape(1, -1)
            
            # Evaluate objective function
            y_next = self.objective_function(X_next[0])
            
            # Add to samples
            self.X_sample.append(X_next[0])
            self.y_sample.append(y_next)
        
        # Return best point
        best_idx = np.argmax(self.y_sample)
        return self.X_sample[best_idx], self.y_sample[best_idx]


class AdvancedEnsembleModel:
    """
    🎭 Advanced Ensemble Methods
    
    Combines multiple models using sophisticated ensemble techniques.
    """
    
    def __init__(self, models, ensemble_method='stacking'):
        self.models = models
        self.ensemble_method = ensemble_method
        self.meta_model = None
        self.weights = None
    
    def fit(self, X, y):
        """Fit ensemble model."""
        if self.ensemble_method == 'stacking':
            self._fit_stacking(X, y)
        elif self.ensemble_method == 'blending':
            self._fit_blending(X, y)
        elif self.ensemble_method == 'bayesian':
            self._fit_bayesian_ensemble(X, y)
    
    def _fit_stacking(self, X, y):
        """Fit stacking ensemble."""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge
        
        # Create meta-features using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict(X_val)
        
        # Train meta-model
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y)
        
        # Refit base models on full data
        for model in self.models:
            model.fit(X, y)
    
    def _fit_blending(self, X, y):
        """Fit blending ensemble."""
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train base models
        blend_features = np.zeros((len(X_blend), len(self.models)))
        
        for i, model in enumerate(self.models):
            model.fit(X_train, y_train)
            blend_features[:, i] = model.predict(X_blend)
        
        # Train meta-model
        from sklearn.linear_model import Ridge
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(blend_features, y_blend)
    
    def _fit_bayesian_ensemble(self, X, y):
        """Fit Bayesian ensemble with uncertainty quantification."""
        # Train base models
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            model.fit(X, y)
            predictions[:, i] = model.predict(X)
        
        # Estimate weights using Bayesian approach
        from scipy.optimize import minimize
        
        def negative_log_likelihood(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            ensemble_pred = np.dot(predictions, weights)
            mse = np.mean((y - ensemble_pred) ** 2)
            
            return mse
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        result = minimize(negative_log_likelihood, initial_weights, 
                         method='SLSQP')
        
        self.weights = np.abs(result.x)
        self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X):
        """Make ensemble predictions."""
        if self.ensemble_method in ['stacking', 'blending']:
            # Get base model predictions
            base_predictions = np.zeros((len(X), len(self.models)))
            for i, model in enumerate(self.models):
                base_predictions[:, i] = model.predict(X)
            
            # Meta-model prediction
            return self.meta_model.predict(base_predictions)
        
        elif self.ensemble_method == 'bayesian':
            # Weighted average
            predictions = np.zeros((len(X), len(self.models)))
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(X)
            
            return np.dot(predictions, self.weights)


class AdvancedMLPipeline:
    """
    🚀 Complete Advanced ML Pipeline
    
    Integrates all advanced ML techniques into a unified pipeline.
    """
    
    def __init__(self):
        self.transformer_model = None
        self.gnn_model = None
        self.rl_router = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
    def prepare_transformer_data(self, demand_series, seq_length=30, forecast_horizon=7):
        """Prepare data for transformer model."""
        # Create sequences
        X, y = [], []
        
        for i in range(len(demand_series) - seq_length - forecast_horizon + 1):
            # Input sequence
            seq = demand_series[i:i + seq_length]
            
            # Add features (day of week, month, etc.)
            features = np.zeros((seq_length, 10))
            features[:, 0] = seq  # Demand values
            
            # Add temporal features
            for j in range(seq_length):
                day_of_week = (i + j) % 7
                features[j, 1:8] = np.eye(7)[day_of_week]  # One-hot day of week
                features[j, 8] = np.sin(2 * np.pi * (i + j) / 365.25)  # Year cycle
                features[j, 9] = np.cos(2 * np.pi * (i + j) / 365.25)
            
            X.append(features)
            y.append(demand_series[i + seq_length:i + seq_length + forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_graph_data(self, delivery_points, distance_matrix):
        """Prepare data for graph neural network."""
        n_points = len(delivery_points)
        
        # Node features (location, demand characteristics)
        node_features = np.zeros((n_points, 10))
        node_features[:, 0] = delivery_points['latitude']
        node_features[:, 1] = delivery_points['longitude']
        node_features[:, 2] = delivery_points['package_weight']
        node_features[:, 3] = delivery_points['package_volume']
        node_features[:, 4] = delivery_points['service_time']
        
        # Add derived features
        center_lat, center_lon = delivery_points['latitude'].mean(), delivery_points['longitude'].mean()
        node_features[:, 5] = np.sqrt((delivery_points['latitude'] - center_lat)**2 + 
                                    (delivery_points['longitude'] - center_lon)**2)
        
        # Edge features (distances, travel times)
        edge_features = np.zeros((n_points, n_points, 5))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    edge_features[i, j, 0] = distance_matrix[i, j]  # Distance
                    edge_features[i, j, 1] = distance_matrix[i, j] / 40 * 60  # Travel time
                    edge_features[i, j, 2] = 1 if distance_matrix[i, j] < 5 else 0  # Close neighbor
                    edge_features[i, j, 3] = np.random.uniform(0.8, 1.5)  # Traffic factor
                    edge_features[i, j, 4] = 1  # Road quality
        
        # Adjacency matrix (connect all points)
        adjacency_matrix = np.ones((n_points, n_points)) - np.eye(n_points)
        
        return (torch.FloatTensor(node_features), 
                torch.FloatTensor(edge_features), 
                torch.FloatTensor(adjacency_matrix))
    
    def train_transformer_forecaster(self, demand_series, epochs=100):
        """Train transformer demand forecaster."""
        print("🤖 Training Transformer Demand Forecaster...")
        
        # Prepare data
        X, y = self.prepare_transformer_data(demand_series)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create model
        self.transformer_model = TransformerDemandForecaster(
            input_dim=10, d_model=64, nhead=8, num_layers=3
        )
        
        # Training setup
        optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        
        for epoch in range(epochs):
            self.transformer_model.train()
            
            optimizer.zero_grad()
            outputs = self.transformer_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate
        self.transformer_model.eval()
        with torch.no_grad():
            test_predictions = self.transformer_model(X_test_tensor)
            test_loss = criterion(test_predictions, y_test_tensor)
            
            print(f"✅ Transformer training complete!")
            print(f"Final test loss: {test_loss.item():.4f}")
        
        return train_losses, test_loss.item()
    
    def train_graph_neural_network(self, delivery_points, distance_matrix, epochs=200):
        """Train graph neural network for route optimization."""
        print("🕸️ Training Graph Neural Network...")
        
        # Prepare data
        node_features, edge_features, adjacency_matrix = self.prepare_graph_data(
            delivery_points, distance_matrix
        )
        
        # Create model
        self.gnn_model = GraphNeuralNetwork(
            node_features=10, edge_features=5, hidden_dim=64
        )
        
        # Training setup
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Generate synthetic targets (total route cost)
        targets = torch.FloatTensor([np.random.uniform(100, 500) for _ in range(epochs)])
        
        # Training loop
        train_losses = []
        
        for epoch in range(epochs):
            self.gnn_model.train()
            
            optimizer.zero_grad()
            output, _ = self.gnn_model(node_features, edge_features.mean(dim=1), adjacency_matrix)
            
            # Create target for this epoch
            target = targets[epoch].unsqueeze(0)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print(f"✅ GNN training complete!")
        
        return train_losses
    
    def get_comprehensive_predictions(self, delivery_points, demand_series, distance_matrix):
        """Get predictions from all models."""
        results = {}
        
        # Transformer predictions
        if self.transformer_model:
            X_latest, _ = self.prepare_transformer_data(demand_series[-60:])  # Use last 60 days
            if len(X_latest) > 0:
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_latest[-1:])  # Latest sequence
                    demand_forecast = self.transformer_model(X_tensor)
                    results['demand_forecast'] = demand_forecast.numpy()[0]
        
        # GNN predictions
        if self.gnn_model:
            node_features, edge_features, adjacency_matrix = self.prepare_graph_data(
                delivery_points, distance_matrix
            )
            with torch.no_grad():
                route_cost, node_embeddings = self.gnn_model(
                    node_features, edge_features.mean(dim=1), adjacency_matrix
                )
                results['predicted_route_cost'] = route_cost.item()
                results['node_embeddings'] = node_embeddings.numpy()
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Advanced ML Pipeline...")
    print("=" * 50)
    
    # Create synthetic data for testing
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sample demand series
    demand_series = 80 + 20 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 5, 365)
    
    # Generate sample delivery points
    delivery_points = pd.DataFrame({
        'latitude': np.random.uniform(-23.6, -23.5, 20),
        'longitude': np.random.uniform(-46.7, -46.6, 20),
        'package_weight': np.random.uniform(1, 20, 20),
        'package_volume': np.random.uniform(0.1, 2, 20),
        'service_time': np.random.uniform(5, 30, 20)
    })
    
    # Generate distance matrix
    n_points = len(delivery_points)
    distance_matrix = np.random.uniform(1, 20, (n_points, n_points))
    np.fill_diagonal(distance_matrix, 0)
    
    # Initialize pipeline
    pipeline = AdvancedMLPipeline()
    
    # Train models
    print("\n1. Training Transformer...")
    transformer_losses, test_loss = pipeline.train_transformer_forecaster(demand_series, epochs=50)
    
    print("\n2. Training Graph Neural Network...")
    gnn_losses = pipeline.train_graph_neural_network(delivery_points, distance_matrix, epochs=100)
    
    print("\n3. Getting comprehensive predictions...")
    predictions = pipeline.get_comprehensive_predictions(delivery_points, demand_series, distance_matrix)
    
    print("\n📊 Results:")
    if 'demand_forecast' in predictions:
        print(f"Next 7-day demand forecast: {predictions['demand_forecast']}")
    if 'predicted_route_cost' in predictions:
        print(f"Predicted route cost: ${predictions['predicted_route_cost']:.2f}")
    
    print("\n✅ Advanced ML pipeline testing complete!")
