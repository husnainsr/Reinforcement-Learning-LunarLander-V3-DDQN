#!/usr/bin/env python3
"""
Test script to verify the setup works with basic environments
Uses CartPole as a fallback when LunarLander is not available
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer_basic import BasicTrainer


def test_setup():
    """Test the basic setup with minimal configuration"""
    
    # Create a minimal test config
    test_config = {
        'environment': {
            'name': 'LunarLander-v3',  # Will fallback to CartPole if not available
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
            'memory_size': 1000,
            'target_update_frequency': 10
        },
        'network': {
            'hidden_layers': [64, 64],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'training': {
            'max_episodes': 20,  # Very short for testing
            'max_steps_per_episode': 200,
            'learning_starts': 50,
            'train_frequency': 4,
            'save_frequency': 10,
            'eval_frequency': 10,
            'early_stopping_patience': 50
        },
        'logging': {
            'log_frequency': 5,
            'use_tensorboard': False,
            'use_wandb': False,
            'project_name': 'test_run',
            'save_plots': True,
            'plot_frequency': 10
        },
        'paths': {
            'models_dir': 'test_models',
            'results_dir': 'test_results',
            'logs_dir': 'test_logs',
            'videos_dir': 'test_videos'
        },
        'success_criteria': {
            'target_score': 200,  # Will be adjusted for CartPole
            'consecutive_successes': 3,
            'success_threshold': 0.8
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🧪 Testing Setup - Double DQN")
    print("=" * 40)
    print(f"Device: {device}")
    print("Running basic test (20 episodes)...")
    print("=" * 40)
    
    try:
        # Create trainer
        trainer = BasicTrainer(
            config_dict=test_config,
            device=device
        )
        
        print(f"✅ Trainer created successfully!")
        print(f"Environment: {trainer.env.spec.id}")
        print(f"State size: {trainer.state_size}")
        print(f"Action size: {trainer.action_size}")
        print(f"Target score: {trainer.target_score}")
        
        # Run a few episodes to test everything works
        print("\n🏃 Running test episodes...")
        
        # Test environment interaction
        state, _ = trainer.env.reset()
        for step in range(10):
            action = trainer.agent.act(state)
            next_state, reward, terminated, truncated, _ = trainer.env.step(action)
            trainer.agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            if terminated or truncated:
                state, _ = trainer.env.reset()
        
        print("✅ Environment interaction works!")
        
        # Test saving/loading
        trainer._save_model('test_model.pth')
        print("✅ Model saving works!")
        
        # Test plotting (basic)
        trainer.plotter.update_data(1, 100, 50, 0.1, 0.5, False, 10.0)
        print("✅ Plotting works!")
        
        print("\n🎉 All basic tests passed!")
        print("Your setup is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Run a very short training session"""
    try:
        # Create the test config for training
        test_config = {
            'environment': {
                'name': 'LunarLander-v3',  # Will fallback to CartPole if not available
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
                'memory_size': 1000,
                'target_update_frequency': 10
            },
            'network': {
                'hidden_layers': [64, 64],
                'activation': 'relu',
                'dropout_rate': 0.1
            },
            'training': {
                'max_episodes': 10,  # Very short for testing
                'max_steps_per_episode': 200,
                'learning_starts': 50,
                'train_frequency': 4,
                'save_frequency': 10,
                'eval_frequency': 10,
                'early_stopping_patience': 50
            },
            'logging': {
                'log_frequency': 5,
                'use_tensorboard': False,
                'use_wandb': False,
                'project_name': 'test_run',
                'save_plots': True,
                'plot_frequency': 10
            },
            'paths': {
                'models_dir': 'test_models',
                'results_dir': 'test_results',
                'logs_dir': 'test_logs',
                'videos_dir': 'test_videos'
            },
            'success_criteria': {
                'target_score': 200,  # Will be adjusted for CartPole
                'consecutive_successes': 3,
                'success_threshold': 0.8
            }
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        trainer = BasicTrainer(config_dict=test_config, device=device)
        
        print("\n🚀 Running mini training session...")
        stats = trainer.train()
        
        print(f"✅ Training completed!")
        print(f"Final score: {stats['final_avg_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔧 Setting up and testing Double DQN environment...")
    
    # Basic setup test
    setup_ok = test_setup()
    
    if setup_ok:
        print("\n" + "=" * 50)
        print("✅ Setup verification complete!")
        print("\nYou can now:")
        print("1. Run full training: python main.py")
        print("2. Use the Jupyter notebook: jupyter notebook notebooks/analysis.ipynb")
        print("3. Run a quick test: python quick_start.py")
        
        # Ask if user wants to run quick training
        try:
            response = input("\nWould you like to run a quick training test? (y/n): ")
            if response.lower() in ['y', 'yes']:
                test_training()
        except KeyboardInterrupt:
            print("\n👋 Setup complete!")
            
    else:
        print("\n⚠️  Please check the error messages above and:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version compatibility")
        print("3. Verify PyTorch installation") 