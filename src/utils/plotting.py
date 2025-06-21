"""
Plotting utilities for visualizing training progress and results
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd
from collections import deque
import os


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingPlotter:
    """
    Real-time plotting class for training visualization
    """
    
    def __init__(self, save_dir='results'):
        """
        Initialize the plotter
        
        Args:
            save_dir (str): Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training data storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_epsilons = []
        self.success_rate_history = []
        self.q_value_history = []
        
        # Moving averages
        self.reward_ma = deque(maxlen=100)
        self.success_ma = deque(maxlen=100)
        
    def update_data(self, episode, reward, length, loss=None, epsilon=None, 
                   success=None, avg_q_value=None):
        """
        Update plotting data with new episode results
        
        Args:
            episode (int): Episode number
            reward (float): Episode reward
            length (int): Episode length
            loss (float): Average loss for the episode
            epsilon (float): Current epsilon value
            success (bool): Whether episode was successful
            avg_q_value (float): Average Q-value
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.episode_losses.append(loss)
        if epsilon is not None:
            self.episode_epsilons.append(epsilon)
        if avg_q_value is not None:
            self.q_value_history.append(avg_q_value)
            
        # Update moving averages
        self.reward_ma.append(reward)
        if success is not None:
            self.success_ma.append(1 if success else 0)
            self.success_rate_history.append(np.mean(self.success_ma))
    
    def plot_training_progress(self, save=True, show=False):
        """
        Create comprehensive training progress plots
        
        Args:
            save (bool): Whether to save the plot
            show (bool): Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Lunar Lander Double DQN Training Progress', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # Episode Rewards
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
        if len(self.reward_ma) > 0:
            ma_rewards = [np.mean(list(self.reward_ma)[:i+1]) for i in range(len(episodes))]
            axes[0, 0].plot(episodes, ma_rewards, color='red', linewidth=2, label=f'MA(100)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Episode Lengths
        axes[0, 1].plot(episodes, self.episode_lengths, alpha=0.7, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success Rate
        if self.success_rate_history:
            axes[0, 2].plot(episodes, self.success_rate_history, color='purple', linewidth=2)
            axes[0, 2].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
            axes[0, 2].set_title('Success Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Success Rate')
            axes[0, 2].set_ylim([0, 1])
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
        else:
            axes[0, 2].text(0.5, 0.5, 'Success Rate\n(Not Available)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Training Loss
        if self.episode_losses:
            axes[1, 0].plot(episodes, self.episode_losses, color='orange', alpha=0.8)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Training Loss\n(Not Available)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Epsilon Decay
        if self.episode_epsilons:
            axes[1, 1].plot(episodes, self.episode_epsilons, color='brown', linewidth=2)
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Epsilon Decay\n(Not Available)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Q-Value Evolution
        if self.q_value_history:
            axes[1, 2].plot(episodes, self.q_value_history, color='teal', alpha=0.8)
            axes[1, 2].set_title('Average Q-Values')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Q-Value')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Q-Values\n(Not Available)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_reward_distribution(self, save=True, show=False):
        """
        Plot reward distribution analysis
        
        Args:
            save (bool): Whether to save the plot
            show (bool): Whether to display the plot
        """
        if not self.episode_rewards:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Reward Distribution Analysis', fontsize=14, fontweight='bold')
        
        # Histogram
        axes[0].hist(self.episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(self.episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        axes[0].axvline(200, color='green', linestyle='--', alpha=0.7, label='Success Threshold: 200')
        axes[0].set_title('Reward Distribution')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by episode chunks
        chunk_size = max(len(self.episode_rewards) // 10, 1)
        chunks = []
        labels = []
        
        for i in range(0, len(self.episode_rewards), chunk_size):
            chunk = self.episode_rewards[i:i+chunk_size]
            if chunk:
                chunks.append(chunk)
                labels.append(f'{i+1}-{min(i+chunk_size, len(self.episode_rewards))}')
        
        if chunks:
            axes[1].boxplot(chunks, labels=labels)
            axes[1].set_title('Reward Evolution (Boxplots)')
            axes[1].set_xlabel('Episode Range')
            axes[1].set_ylabel('Reward')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'reward_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_curve(self, save=True, show=False):
        """
        Plot detailed learning curve with confidence intervals
        
        Args:
            save (bool): Whether to save the plot
            show (bool): Whether to display the plot
        """
        if len(self.episode_rewards) < 10:
            return
            
        # Calculate rolling statistics
        window = min(100, len(self.episode_rewards) // 10)
        df = pd.DataFrame({'reward': self.episode_rewards})
        df['rolling_mean'] = df['reward'].rolling(window=window).mean()
        df['rolling_std'] = df['reward'].rolling(window=window).std()
        df['episode'] = range(1, len(df) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw rewards
        ax.plot(df['episode'], df['reward'], alpha=0.3, color='lightblue', linewidth=0.5, label='Episode Rewards')
        
        # Plot rolling mean
        ax.plot(df['episode'], df['rolling_mean'], color='blue', linewidth=2, label=f'Rolling Mean ({window})')
        
        # Plot confidence interval
        upper_bound = df['rolling_mean'] + df['rolling_std']
        lower_bound = df['rolling_mean'] - df['rolling_std']
        ax.fill_between(df['episode'], lower_bound, upper_bound, alpha=0.2, color='blue', label='±1 Std Dev')
        
        # Add success threshold
        ax.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Success Threshold (200)')
        
        # Formatting
        ax.set_title('Learning Curve with Confidence Interval', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'learning_curve.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_data(self):
        """Save training data to CSV files"""
        # Create DataFrame
        data = {
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'length': self.episode_lengths
        }
        
        if self.episode_losses:
            data['loss'] = self.episode_losses + [None] * (len(self.episode_rewards) - len(self.episode_losses))
        if self.episode_epsilons:
            data['epsilon'] = self.episode_epsilons + [None] * (len(self.episode_rewards) - len(self.episode_epsilons))
        if self.success_rate_history:
            data['success_rate'] = self.success_rate_history + [None] * (len(self.episode_rewards) - len(self.success_rate_history))
        if self.q_value_history:
            data['q_value'] = self.q_value_history + [None] * (len(self.episode_rewards) - len(self.q_value_history))
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_dir, 'training_data.csv'), index=False)
        
    def generate_report(self):
        """Generate a comprehensive training report"""
        if not self.episode_rewards:
            return
            
        self.plot_training_progress(save=True, show=False)
        self.plot_reward_distribution(save=True, show=False)
        self.plot_learning_curve(save=True, show=False)
        self.save_data()
        
        # Generate summary statistics
        stats = {
            'total_episodes': len(self.episode_rewards),
            'final_reward': self.episode_rewards[-1],
            'best_reward': max(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'success_episodes': sum(1 for r in self.episode_rewards if r >= 200),
            'success_rate': sum(1 for r in self.episode_rewards if r >= 200) / len(self.episode_rewards),
            'final_success_rate': self.success_rate_history[-1] if self.success_rate_history else 0
        }
        
        # Save summary
        with open(os.path.join(self.save_dir, 'training_summary.txt'), 'w') as f:
            f.write("Lunar Lander Double DQN Training Summary\n")
            f.write("=" * 40 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                
        return stats


def plot_comparison(results_dict, save_dir='results', save=True, show=False):
    """
    Compare results from multiple training runs
    
    Args:
        results_dict (dict): Dictionary with run names as keys and reward lists as values
        save_dir (str): Directory to save plots
        save (bool): Whether to save the plot
        show (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Comparison', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # Learning curves
    for i, (name, rewards) in enumerate(results_dict.items()):
        episodes = range(1, len(rewards) + 1)
        
        # Raw rewards (transparent)
        axes[0].plot(episodes, rewards, alpha=0.3, color=colors[i], linewidth=0.5)
        
        # Moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            ma_rewards = pd.Series(rewards).rolling(window=window).mean()
            axes[0].plot(episodes, ma_rewards, color=colors[i], linewidth=2, label=name)
    
    axes[0].axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
    axes[0].set_title('Learning Curves')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_rewards = [rewards[-100:] for rewards in results_dict.values()]  # Last 100 episodes
    axes[1].boxplot(final_rewards, labels=list(results_dict.keys()))
    axes[1].set_title('Final Performance (Last 100 Episodes)')
    axes[1].set_ylabel('Reward')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close() 