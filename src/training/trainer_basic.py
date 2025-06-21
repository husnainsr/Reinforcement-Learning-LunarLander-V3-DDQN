"""
Basic training orchestrator for Double DQN with gymnasium
Works with standard environments without box2d dependencies
"""

import gymnasium as gym
import numpy as np
import torch
import time
import os
from collections import deque
from tqdm import tqdm
import yaml

from ..agent.double_dqn_agent import DoubleDQNAgent
from ..utils.plotting import TrainingPlotter


class BasicTrainer:
    """
    Basic trainer class for Double DQN with standard environments
    
    Uses CartPole-v1 as a fallback when LunarLander is not available
    """
    
    def __init__(self, config_path=None, config_dict=None, device='cpu'):
        """
        Initialize the trainer
        
        Args:
            config_path (str): Path to configuration file
            config_dict (dict): Configuration dictionary (alternative to file)
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
            
        self.device = device
        
        # Try to create environment with fallback
        env_config = self.config['environment']
        env_name = env_config['name']
        
        try:
            # Try LunarLander first
            self.env = gym.make('LunarLander-v3')
            print(f"✅ Using LunarLander-v3 environment")
        except:
            try:
                # Fallback to CartPole
                self.env = gym.make('CartPole-v1')
                print(f"⚠️  LunarLander not available, using CartPole-v1")
                env_name = 'CartPole-v1'
            except:
                raise RuntimeError("Neither LunarLander nor CartPole environments are available")
        
        # Set seed
        self.env.reset(seed=env_config.get('seed', 42))
        
        # Get environment dimensions
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        print(f"Environment: {env_name}")
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        
        # Create agent
        agent_config = self.config['agent']
        self.agent = DoubleDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=agent_config,
            device=device
        )
        
        # Training parameters
        training_config = self.config['training']
        self.max_episodes = training_config['max_episodes']
        self.max_steps_per_episode = training_config['max_steps_per_episode']
        self.save_frequency = training_config['save_frequency']
        self.eval_frequency = training_config['eval_frequency']
        self.early_stopping_patience = training_config['early_stopping_patience']
        
        # Logging parameters
        logging_config = self.config['logging']
        self.log_frequency = logging_config['log_frequency']
        self.plot_frequency = logging_config['plot_frequency']
        
        # Paths
        paths_config = self.config['paths']
        self.models_dir = paths_config['models_dir']
        self.results_dir = paths_config['results_dir']
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize plotter
        self.plotter = TrainingPlotter(save_dir=self.results_dir)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.recent_scores = deque(maxlen=100)
        self.best_avg_score = -np.inf
        self.episodes_without_improvement = 0
        
        # Success criteria (adjusted for CartPole if needed)
        success_config = self.config['success_criteria']
        if env_name == 'CartPole-v1':
            self.target_score = 475  # CartPole target
            print(f"Adjusted target score for CartPole: {self.target_score}")
        else:
            self.target_score = success_config['target_score']
            
        self.success_threshold = success_config['success_threshold']
        self.consecutive_successes = success_config['consecutive_successes']
        
    def train(self):
        """
        Main training loop
        
        Returns:
            dict: Training statistics and final performance
        """
        print("🚀 Starting Double DQN Training")
        print(f"Device: {self.device}")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"Target Score: {self.target_score}")
        print("-" * 50)
        
        start_time = time.time()
        
        for episode in tqdm(range(1, self.max_episodes + 1), desc="Training"):
            episode_reward, episode_length = self._run_episode(episode)
            
            # Update statistics
            self.episode_rewards.append(episode_reward)
            self.recent_scores.append(episode_reward)
            
            # Get agent statistics
            agent_stats = self.agent.get_stats()
            
            # Update plotter
            success = episode_reward >= self.target_score
            self.plotter.update_data(
                episode=episode,
                reward=episode_reward,
                length=episode_length,
                loss=agent_stats['avg_loss'],
                epsilon=agent_stats['epsilon'],
                success=success,
                avg_q_value=agent_stats['avg_q_value']
            )
            
            # Logging
            if episode % self.log_frequency == 0:
                avg_score = np.mean(self.recent_scores)
                print(f"\nEpisode {episode:4d} | "
                      f"Score: {episode_reward:7.2f} | "
                      f"Avg Score: {avg_score:7.2f} | "
                      f"Epsilon: {agent_stats['epsilon']:.3f} | "
                      f"Memory: {agent_stats['memory_size']:5d}")
                
                # Check for improvement
                if avg_score > self.best_avg_score:
                    self.best_avg_score = avg_score
                    self.episodes_without_improvement = 0
                    # Save best model
                    self._save_model(f'best_model_ep{episode}.pth')
                else:
                    self.episodes_without_improvement += 1
            
            # Periodic evaluation
            if episode % self.eval_frequency == 0:
                eval_score = self._evaluate_agent()
                print(f"Evaluation Score: {eval_score:.2f}")
            
            # Periodic plotting
            if episode % self.plot_frequency == 0:
                self.plotter.plot_training_progress(save=True, show=False)
            
            # Periodic model saving
            if episode % self.save_frequency == 0:
                self._save_model(f'checkpoint_ep{episode}.pth')
            
            # Check success criteria
            if self._check_success_criteria():
                print(f"\n🎉 SUCCESS! Target achieved in {episode} episodes!")
                break
                
            # Early stopping
            if self.episodes_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️ Early stopping after {episode} episodes "
                      f"({self.early_stopping_patience} episodes without improvement)")
                break
        
        # Final evaluation and reporting
        training_time = time.time() - start_time
        final_stats = self._generate_final_report(episode, training_time)
        
        # Generate comprehensive plots
        self.plotter.generate_report()
        
        # Save final model
        self._save_model('final_model.pth')
        
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        return final_stats
    
    def _run_episode(self, episode_num):
        """Run a single training episode"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience and learn
            self.agent.step(state, action, reward, next_state, done)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Episode cleanup
        self.agent.reset_episode()
        
        return episode_reward, episode_length
    
    def _evaluate_agent(self, num_episodes=10, render=False):
        """Evaluate the agent's performance"""
        self.agent.set_eval_mode()
        eval_scores = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.max_steps_per_episode):
                action = self.agent.act(state, epsilon=0.0)  # Greedy policy
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
                    
                if terminated or truncated:
                    break
                    
            eval_scores.append(episode_reward)
        
        self.agent.set_train_mode()
        return np.mean(eval_scores)
    
    def _check_success_criteria(self):
        """Check if training success criteria are met"""
        if len(self.recent_scores) < self.consecutive_successes:
            return False
            
        # Check if recent scores meet the target
        recent_success_rate = sum(1 for score in list(self.recent_scores)[-self.consecutive_successes:] 
                                 if score >= self.target_score) / self.consecutive_successes
        
        return recent_success_rate >= self.success_threshold
    
    def _save_model(self, filename):
        """Save the current model"""
        filepath = os.path.join(self.models_dir, filename)
        self.agent.save_model(filepath)
    
    def _generate_final_report(self, final_episode, training_time):
        """Generate final training report"""
        final_eval_score = self._evaluate_agent(num_episodes=20)
        
        stats = {
            'total_episodes': final_episode,
            'training_time_minutes': training_time / 60,
            'final_avg_score': np.mean(self.recent_scores),
            'best_avg_score': self.best_avg_score,
            'final_evaluation_score': final_eval_score,
            'success_rate': sum(1 for score in self.recent_scores if score >= self.target_score) / len(self.recent_scores),
            'agent_stats': self.agent.get_stats()
        }
        
        # Save final report
        report_path = os.path.join(self.results_dir, 'final_report.yaml')
        with open(report_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        return stats 