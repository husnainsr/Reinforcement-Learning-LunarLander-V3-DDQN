"""
Double Deep Q-Network (Double DQN) Agent
Implements the Double DQN algorithm for Lunar Lander environment
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque

from ..models.neural_network import DQNNetwork, DuelingDQNNetwork
from ..utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DoubleDQNAgent:
    """
    Double DQN Agent with Experience Replay
    
    Implements the Double DQN algorithm which reduces overestimation bias
    by decoupling action selection from action evaluation
    """
    
    def __init__(self, state_size, action_size, config, device='cpu'):
        """
        Initialize the Double DQN Agent
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            config (dict): Configuration parameters
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.config = config
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_frequency = config.get('target_update_frequency', 100)
        self.learning_starts = config.get('learning_starts', 1000)
        self.train_frequency = config.get('train_frequency', 4)
        
        # Networks
        network_config = config.get('network', {})
        self.use_dueling = network_config.get('use_dueling', False)
        
        if self.use_dueling:
            self.q_network = DuelingDQNNetwork(
                state_size, action_size,
                hidden_layers=network_config.get('hidden_layers', [128, 128]),
                activation=network_config.get('activation', 'relu'),
                dropout_rate=network_config.get('dropout_rate', 0.1)
            ).to(device)
            
            self.target_network = DuelingDQNNetwork(
                state_size, action_size,
                hidden_layers=network_config.get('hidden_layers', [128, 128]),
                activation=network_config.get('activation', 'relu'),
                dropout_rate=network_config.get('dropout_rate', 0.1)
            ).to(device)
        else:
            self.q_network = DQNNetwork(
                state_size, action_size,
                hidden_layers=network_config.get('hidden_layers', [128, 128]),
                activation=network_config.get('activation', 'relu'),
                dropout_rate=network_config.get('dropout_rate', 0.1)
            ).to(device)
            
            self.target_network = DQNNetwork(
                state_size, action_size,
                hidden_layers=network_config.get('hidden_layers', [128, 128]),
                activation=network_config.get('activation', 'relu'),
                dropout_rate=network_config.get('dropout_rate', 0.1)
            ).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        buffer_config = config.get('buffer', {})
        use_prioritized = buffer_config.get('prioritized', False)
        
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(
                buffer_size=config.get('memory_size', 100000),
                batch_size=self.batch_size,
                alpha=buffer_config.get('alpha', 0.6),
                beta=buffer_config.get('beta', 0.4)
            )
        else:
            self.memory = ReplayBuffer(
                buffer_size=config.get('memory_size', 100000),
                batch_size=self.batch_size
            )
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        self.loss_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)
        
    def act(self, state, epsilon=None):
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            state (np.array): Current state
            epsilon (float): Exploration probability (uses self.epsilon if None)
            
        Returns:
            int: Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() > epsilon:
            # Greedy action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
            self.q_network.train()
            
            # Store Q-values for analysis
            self.q_value_history.append(q_values.cpu().numpy().max())
        else:
            # Random action
            action = random.choice(np.arange(self.action_size))
            
        return action
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience and learn from a batch of experiences
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # Learn from experience
        if (len(self.memory) > self.learning_starts and 
            self.step_count % self.train_frequency == 0):
            self.learn()
            
    def learn(self):
        """
        Learn from a batch of experiences using Double DQN algorithm
        """
        # Sample batch from memory
        if hasattr(self.memory, 'sample') and len(self.memory) >= self.batch_size:
            if isinstance(self.memory, PrioritizedReplayBuffer):
                experiences, indices, weights = self.memory.sample(self.device)
                states, actions, rewards, next_states, dones = experiences
            else:
                states, actions, rewards, next_states, dones = self.memory.sample(self.device)
                weights = torch.ones(self.batch_size).to(self.device)
                indices = None
        else:
            return
            
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Double DQN: Action selection using online network, evaluation using target network
        with torch.no_grad():
            # Get actions from online network
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            
            # Get Q-values from target network for selected actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            # Compute target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            # Weighted loss for prioritized experience replay
            loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # Update priorities
            priorities = td_errors.abs().cpu().data.numpy()
            self.memory.update_priorities(indices, priorities)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Store loss for analysis
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """
        Update target network by copying weights from main network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def soft_update_target_network(self, tau=0.001):
        """
        Soft update target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            tau (float): Interpolation parameter
        """
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
    def get_stats(self):
        """
        Get training statistics
        
        Returns:
            dict: Dictionary containing training statistics
        """
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'avg_q_value': np.mean(self.q_value_history) if self.q_value_history else 0
        }
    
    def reset_episode(self):
        """Reset episode-specific parameters"""
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.target_update_frequency == 0:
            self.update_target_network()
            
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.q_network.eval()
        self.target_network.eval()
        
    def set_train_mode(self):
        """Set networks to training mode"""
        self.q_network.train()
        self.target_network.train() 