#!/usr/bin/env python3
"""
Quick start script for Lunar Lander Double DQN
Runs a minimal training session for testing the setup
"""

import sys
import os
import torch
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import LunarLanderTrainer


def quick_test():
    """Run a quick test with minimal episodes"""
    
    # Create a minimal config for testing
    test_config = {
        'environment': {
            'name': 'LunarLander-v3',
            'render_mode': 'rgb_array',
            'seed': 42
        },
        'agent': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'memory_size': 10000,
            'target_update_frequency': 50
        },
        'network': {
            'hidden_layers': [64, 64],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'training': {
            'max_episodes': 50,  # Very short for testing
            'max_steps_per_episode': 500,
            'learning_starts': 100,
            'train_frequency': 4,
            'save_frequency': 25,
            'eval_frequency': 25,
            'early_stopping_patience': 50
        },
        'logging': {
            'log_frequency': 10,
            'use_tensorboard': False,
            'use_wandb': False,
            'project_name': 'lunar_lander_test',
            'save_plots': True,
            'plot_frequency': 25
        },
        'paths': {
            'models_dir': 'test_models',
            'results_dir': 'test_results',
            'logs_dir': 'test_logs',
            'videos_dir': 'test_videos'
        },
        'success_criteria': {
            'target_score': 200,
            'consecutive_successes': 5,
            'success_threshold': 0.8
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🚀 Quick Start Test - Lunar Lander Double DQN")
    print("=" * 50)
    print(f"Device: {device}")
    print("Running minimal training (50 episodes)...")
    print("This will test all components of the system")
    print("=" * 50)
    
    try:
        # Create trainer
        trainer = LunarLanderTrainer(
            config_dict=test_config,
            device=device
        )
        
        # Run training
        stats = trainer.train()
        
        print("\n✅ Quick test completed successfully!")
        print(f"Final average score: {stats['final_avg_score']:.2f}")
        print(f"Training time: {stats['training_time_minutes']:.2f} minutes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during quick test: {str(e)}")
        return False


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 System is working correctly!")
        print("You can now run full training with:")
        print("  python main.py")
    else:
        print("\n⚠️  Please check your installation and dependencies")
        print("  pip install -r requirements.txt") 