"""
Experience Replay Buffer for DQN Agent
Implements prioritized and uniform sampling for experience replay
"""

import random
import numpy as np
from collections import deque, namedtuple
import torch


# Experience tuple for storing transitions
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    
    Stores transitions and provides sampling functionality for training
    """
    
    def __init__(self, buffer_size=100000, batch_size=64, seed=42):
        """
        Initialize the replay buffer
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed for reproducibility
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, device='cpu'):
        """
        Randomly sample a batch of experiences from memory
        
        Args:
            device: PyTorch device for tensor placement
            
        Returns:
            tuple: Batch of experiences as tensors
        """
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return current size of internal memory"""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Samples experiences based on their TD error for more efficient learning
    """
    
    def __init__(self, buffer_size=100000, batch_size=64, alpha=0.6, beta=0.4, seed=42):
        """
        Initialize the prioritized replay buffer
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            alpha (float): Prioritization exponent
            beta (float): Importance sampling exponent
            seed (int): Random seed for reproducibility
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.seed = random.seed(seed)
        
        # Initialize memory and priority arrays
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add a new experience to memory with priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            error: TD error for prioritization
        """
        experience = Experience(state, action, reward, next_state, done)
        
        # Calculate priority
        priority = (abs(error) + 1e-6) ** self.alpha if error is not None else 1.0
        
        if self.size < self.buffer_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self, device='cpu'):
        """
        Sample a batch of experiences based on priorities
        
        Args:
            device: PyTorch device for tensor placement
            
        Returns:
            tuple: Batch of experiences, indices, and importance weights
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, self.batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Convert to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, errors):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices of sampled experiences
            errors: New TD errors for priority calculation
        """
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            
    def __len__(self):
        """Return current size of internal memory"""
        return self.size


class MultiStepBuffer:
    """
    Multi-step experience buffer for n-step learning
    
    Accumulates rewards over multiple steps for more efficient learning
    """
    
    def __init__(self, n_steps=3, gamma=0.99):
        """
        Initialize multi-step buffer
        
        Args:
            n_steps (int): Number of steps to accumulate
            gamma (float): Discount factor
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to multi-step buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def get_transition(self):
        """
        Get n-step transition from buffer
        
        Returns:
            tuple: Multi-step transition or None if buffer not full
        """
        if len(self.buffer) < self.n_steps:
            return None
            
        # Calculate n-step reward
        n_step_reward = 0
        for i in range(self.n_steps):
            n_step_reward += (self.gamma ** i) * self.buffer[i][2]
            
        # Get initial state and action
        state = self.buffer[0][0]
        action = self.buffer[0][1]
        
        # Get final next_state and done
        next_state = self.buffer[-1][3]
        done = self.buffer[-1][4]
        
        return (state, action, n_step_reward, next_state, done)
    
    def reset(self):
        """Reset the buffer"""
        self.buffer.clear() 