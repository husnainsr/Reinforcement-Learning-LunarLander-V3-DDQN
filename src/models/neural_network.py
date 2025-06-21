"""
Neural Network Architecture for DQN Agent
Implements the Q-Network used in Double DQN for Lunar Lander
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Lunar Lander environment
    
    Architecture:
    - Input: State vector (8 dimensions)
    - Hidden layers: Configurable fully connected layers
    - Output: Q-values for each action (4 actions)
    """
    
    def __init__(self, state_size=8, action_size=4, hidden_layers=[128, 128], 
                 activation='relu', dropout_rate=0.1, seed=42):
        """
        Initialize the DQN network
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function type
            dropout_rate (float): Dropout rate for regularization
            seed (int): Random seed for reproducibility
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
            
        # Build network layers
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = state
        
        # Pass through hidden layers with activation and dropout
        for i, layer in enumerate(self.network[:-1]):
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:  # Dropout layer
                x = layer(x)
                
        # Output layer (no activation)
        x = self.network[-1](x)
        
        return x
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state (torch.Tensor): Current state
            epsilon (float): Exploration probability
            
        Returns:
            int: Selected action
        """
        if np.random.random() > epsilon:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state)
                action = torch.argmax(q_values).item()
        else:
            # Random action
            action = np.random.randint(self.action_size)
            
        return action
    
    def get_q_values(self, state):
        """
        Get Q-values for a given state
        
        Args:
            state (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        with torch.no_grad():
            return self.forward(state)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network Architecture
    Separates state value and advantage functions
    """
    
    def __init__(self, state_size=8, action_size=4, hidden_layers=[128, 128],
                 activation='relu', dropout_rate=0.1, seed=42):
        """
        Initialize the Dueling DQN network
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function type
            dropout_rate (float): Dropout rate for regularization
            seed (int): Random seed for reproducibility
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
            
        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        input_size = state_size
        
        for hidden_size in hidden_layers[:-1]:
            self.feature_layers.append(nn.Linear(input_size, hidden_size))
            self.feature_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
            
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in [self.feature_layers, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
                    
    def forward(self, state):
        """
        Forward pass through the dueling network
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Extract features
        x = state
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:  # Dropout layer
                x = layer(x)
                
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state (torch.Tensor): Current state
            epsilon (float): Exploration probability
            
        Returns:
            int: Selected action
        """
        if np.random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(state)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.action_size)
            
        return action 