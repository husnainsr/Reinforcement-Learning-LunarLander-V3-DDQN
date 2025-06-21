#!/usr/bin/env python3
"""
Main training script for Lunar Lander Double DQN
Usage: python main.py [--config path/to/config.yaml] [--device cuda/cpu]
"""

import argparse
import torch
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import LunarLanderTrainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Lunar Lander with Double DQN')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration instead of training')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pre-trained model for demonstration')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🚀 Lunar Lander Double DQN")
    print("=" * 40)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available() and device == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    try:
        # Initialize trainer
        trainer = LunarLanderTrainer(
            config_path=args.config,
            device=device
        )
        
        # Set random seed if provided
        if args.seed is not None:
            torch.manual_seed(args.seed)
            print(f"Random seed set to: {args.seed}")
        
        if args.demo:
            # Demonstration mode
            if args.model is None:
                print("❌ Error: --model path required for demonstration mode")
                return
            
            print(f"Loading model from: {args.model}")
            trainer.load_model(args.model)
            trainer.demo_trained_agent(num_episodes=5, render=True)
            
        else:
            # Training mode
            print("Starting training...")
            
            # Run training
            stats = trainer.train()
            
            # Print final results
            print("\n" + "=" * 50)
            print("🎉 TRAINING COMPLETED!")
            print("=" * 50)
            print(f"Total Episodes: {stats['total_episodes']}")
            print(f"Training Time: {stats['training_time_minutes']:.2f} minutes")
            print(f"Final Average Score: {stats['final_avg_score']:.2f}")
            print(f"Best Average Score: {stats['best_avg_score']:.2f}")
            print(f"Final Evaluation Score: {stats['final_evaluation_score']:.2f}")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            
            # Check if target was reached
            target_score = trainer.target_score
            if stats['final_evaluation_score'] >= target_score:
                print(f"✅ TARGET ACHIEVED! Score >= {target_score}")
            else:
                print(f"⚠️  Target not reached. Score: {stats['final_evaluation_score']:.2f} < {target_score}")
            
            print(f"\nResults saved in: {trainer.results_dir}")
            print(f"Models saved in: {trainer.models_dir}")
            
            # Run a quick demonstration
            print("\n🎮 Running demonstration of trained agent...")
            trainer.demo_trained_agent(num_episodes=3, render=False)
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise
        
    finally:
        print("\nTraining session ended.")


if __name__ == "__main__":
    main() 