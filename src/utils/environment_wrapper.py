"""
Environment wrapper utilities for Lunar Lander
Provides preprocessing, monitoring, and environment modifications
"""

import gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt


class LunarLanderWrapper(gym.Wrapper):
    """
    Custom wrapper for Lunar Lander environment
    
    Provides additional functionality:
    - State normalization
    - Reward shaping
    - Episode monitoring
    - Action logging
    """
    
    def __init__(self, env, normalize_states=True, reward_shaping=False, monitor=True):
        """
        Initialize the wrapper
        
        Args:
            env: The base environment
            normalize_states (bool): Whether to normalize state observations
            reward_shaping (bool): Whether to apply reward shaping
            monitor (bool): Whether to monitor episode statistics
        """
        super().__init__(env)
        
        self.normalize_states = normalize_states
        self.reward_shaping = reward_shaping
        self.monitor = monitor
        
        # State normalization parameters (based on typical value ranges)
        self.state_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state_stds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Monitoring
        if self.monitor:
            self.episode_rewards = []
            self.episode_lengths = []
            self.action_counts = [0, 0, 0, 0]  # Count for each action
            self.current_episode_length = 0
            self.current_episode_reward = 0
            
        # Previous state for reward shaping
        self.prev_state = None
        
    def reset(self, **kwargs):
        """Reset the environment and apply preprocessing"""
        state = self.env.reset(**kwargs)
        
        # Reset monitoring
        if self.monitor:
            if self.current_episode_length > 0:  # Not the first episode
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
            self.current_episode_length = 0
            self.current_episode_reward = 0
            
        self.prev_state = state
        
        if self.normalize_states:
            state = self._normalize_state(state)
            
        return state
    
    def step(self, action):
        """Take a step in the environment with preprocessing"""
        state, reward, done, info = self.env.step(action)
        
        # Apply reward shaping if enabled
        if self.reward_shaping:
            reward = self._shape_reward(state, reward, done)
            
        # Update monitoring
        if self.monitor:
            self.action_counts[action] += 1
            self.current_episode_length += 1
            self.current_episode_reward += reward
            
        self.prev_state = state
        
        if self.normalize_states:
            state = self._normalize_state(state)
            
        return state, reward, done, info
    
    def _normalize_state(self, state):
        """Normalize state observations"""
        # Simple normalization (can be improved with running statistics)
        return (state - self.state_means) / self.state_stds
    
    def _shape_reward(self, state, reward, done):
        """Apply reward shaping to encourage better behavior"""
        shaped_reward = reward
        
        if self.prev_state is not None:
            # Reward for staying close to the landing pad (x position close to 0)
            distance_penalty = -abs(state[0]) * 0.1
            
            # Reward for reducing vertical velocity when close to ground
            if state[1] < 0.5:  # Close to ground
                vertical_velocity_penalty = -abs(state[3]) * 0.1
                shaped_reward += vertical_velocity_penalty
            
            # Reward for reducing angle when close to ground
            if state[1] < 0.5:  # Close to ground
                angle_penalty = -abs(state[4]) * 0.1
                shaped_reward += angle_penalty
            
            shaped_reward += distance_penalty
        
        return shaped_reward
    
    def get_statistics(self):
        """Get monitoring statistics"""
        if not self.monitor:
            return None
            
        stats = {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'action_distribution': {
                'do_nothing': self.action_counts[0],
                'fire_left': self.action_counts[1],
                'fire_main': self.action_counts[2],
                'fire_right': self.action_counts[3]
            },
            'total_actions': sum(self.action_counts),
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }
        
        return stats
    
    def plot_statistics(self, save_path=None):
        """Plot environment statistics"""
        if not self.monitor or not self.episode_rewards:
            print("No monitoring data available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Environment Statistics', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Action distribution
        actions = ['Do Nothing', 'Left', 'Main', 'Right']
        axes[1, 0].bar(actions, self.action_counts)
        axes[1, 0].set_title('Action Distribution')
        axes[1, 0].set_ylabel('Count')
        
        # Reward distribution
        axes[1, 1].hist(self.episode_rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FrameStackWrapper(gym.Wrapper):
    """
    Stack consecutive frames for temporal information
    Useful for environments where history matters
    """
    
    def __init__(self, env, num_frames=4):
        """
        Initialize frame stacking wrapper
        
        Args:
            env: Base environment
            num_frames (int): Number of frames to stack
        """
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        # Update observation space
        low = np.tile(env.observation_space.low, num_frames)
        high = np.tile(env.observation_space.high, num_frames)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset and initialize frame stack"""
        obs = self.env.reset(**kwargs)
        
        # Fill frames with initial observation
        for _ in range(self.num_frames):
            self.frames.append(obs)
            
        return self._get_observation()
    
    def step(self, action):
        """Take step and update frame stack"""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get stacked frames as observation"""
        return np.concatenate(list(self.frames))


class NoiseWrapper(gym.Wrapper):
    """
    Add noise to observations for robustness testing
    """
    
    def __init__(self, env, noise_std=0.01):
        """
        Initialize noise wrapper
        
        Args:
            env: Base environment
            noise_std (float): Standard deviation of Gaussian noise
        """
        super().__init__(env)
        self.noise_std = noise_std
    
    def step(self, action):
        """Add noise to observations"""
        obs, reward, done, info = self.env.step(action)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, obs.shape)
        noisy_obs = obs + noise
        
        return noisy_obs, reward, done, info


class RecordingWrapper(gym.Wrapper):
    """
    Record episodes for video generation
    """
    
    def __init__(self, env, record_frequency=100, video_dir='videos'):
        """
        Initialize recording wrapper
        
        Args:
            env: Base environment
            record_frequency (int): Record every N episodes
            video_dir (str): Directory to save videos
        """
        super().__init__(env)
        self.record_frequency = record_frequency
        self.video_dir = video_dir
        self.episode_count = 0
        self.recording = False
        self.frames = []
        
        # Create video directory
        import os
        os.makedirs(video_dir, exist_ok=True)
    
    def reset(self, **kwargs):
        """Reset and potentially start recording"""
        obs = self.env.reset(**kwargs)
        
        # Save previous recording if exists
        if self.recording and self.frames:
            self._save_video()
        
        # Check if we should record this episode
        self.episode_count += 1
        self.recording = (self.episode_count % self.record_frequency == 0)
        self.frames = []
        
        if self.recording:
            frame = self.env.render(mode='rgb_array')
            self.frames.append(frame)
        
        return obs
    
    def step(self, action):
        """Take step and record frame if needed"""
        obs, reward, done, info = self.env.step(action)
        
        if self.recording:
            frame = self.env.render(mode='rgb_array')
            self.frames.append(frame)
        
        return obs, reward, done, info
    
    def _save_video(self):
        """Save recorded frames as video"""
        if not self.frames:
            return
            
        filename = f"episode_{self.episode_count - 1}.mp4"
        filepath = f"{self.video_dir}/{filename}"
        
        # Save using OpenCV
        try:
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
            
            for frame in self.frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving video: {e}")


def create_env(env_name='LunarLander-v3', wrappers=None, **wrapper_kwargs):
    """
    Create environment with optional wrappers
    
    Args:
        env_name (str): Name of the gym environment
        wrappers (list): List of wrapper classes to apply
        **wrapper_kwargs: Keyword arguments for wrappers
    
    Returns:
        gym.Env: Wrapped environment
    """
    env = gym.make(env_name)
    
    if wrappers:
        for wrapper_class in wrappers:
            if wrapper_class == LunarLanderWrapper:
                env = wrapper_class(env, **wrapper_kwargs.get('lunar_lander', {}))
            elif wrapper_class == FrameStackWrapper:
                env = wrapper_class(env, **wrapper_kwargs.get('frame_stack', {}))
            elif wrapper_class == NoiseWrapper:
                env = wrapper_class(env, **wrapper_kwargs.get('noise', {}))
            elif wrapper_class == RecordingWrapper:
                env = wrapper_class(env, **wrapper_kwargs.get('recording', {}))
            else:
                env = wrapper_class(env)
    
    return env 