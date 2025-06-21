#!/usr/bin/env python3
"""
Evaluation script for trained Lunar Lander Double DQN models
Usage: python evaluate.py --model path/to/model.pth [options]
"""

import argparse
import torch
import numpy as np
import gym
import sys
import os
import time
import cv2
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.double_dqn_agent import DoubleDQNAgent


class ModelEvaluator:
    """
    Evaluates trained Double DQN models
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the evaluator
        
        Args:
            model_path (str): Path to the trained model
            device (str): PyTorch device
        """
        self.model_path = model_path
        self.device = device
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        
        # Create environment
        self.env = gym.make('LunarLander-v3')
        
        # Create agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = DoubleDQNAgent(
            state_size=state_size,
            action_size=action_size,
            config=self.config,
            device=device
        )
        
        # Load model weights
        self.agent.load_model(model_path)
        self.agent.set_eval_mode()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"Model trained for {checkpoint['episode_count']} episodes")
        
    def evaluate(self, num_episodes=100, render=False, save_video=False):
        """
        Evaluate the model performance
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
            save_video (bool): Whether to save evaluation video
            
        Returns:
            dict: Evaluation statistics
        """
        print(f"🔍 Evaluating model for {num_episodes} episodes...")
        
        scores = []
        episode_lengths = []
        successes = 0
        frames = [] if save_video else None
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_score = 0
            episode_length = 0
            
            while True:
                # Select action (greedy policy)
                action = self.agent.act(state, epsilon=0.0)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                episode_score += reward
                episode_length += 1
                state = next_state
                
                # Render if requested
                if render:
                    self.env.render()
                    time.sleep(0.02)
                
                # Save frame for video
                if save_video and episode < 5:  # Only save first 5 episodes
                    frame = self.env.render(mode='rgb_array')
                    frames.append(frame)
                
                if done:
                    break
            
            scores.append(episode_score)
            episode_lengths.append(episode_length)
            
            # Check success (landing with score >= 200)
            if episode_score >= 200:
                successes += 1
            
            # Progress update
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Episode {episode + 1:3d}/{num_episodes} | "
                      f"Score: {episode_score:7.2f} | "
                      f"Avg (last 10): {avg_score:7.2f}")
        
        # Calculate statistics
        stats = {
            'num_episodes': num_episodes,
            'average_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'success_rate': successes / num_episodes,
            'successes': successes,
            'average_length': np.mean(episode_lengths),
            'total_length': np.sum(episode_lengths),
            'scores': scores,
            'episode_lengths': episode_lengths
        }
        
        # Save video if requested
        if save_video and frames:
            self._save_video(frames, 'evaluation_video.mp4')
        
        return stats
    
    def benchmark_performance(self):
        """
        Run comprehensive performance benchmark
        
        Returns:
            dict: Benchmark results
        """
        print("🏁 Running comprehensive benchmark...")
        
        # Standard evaluation
        standard_eval = self.evaluate(num_episodes=100, render=False)
        
        # Longer evaluation for stability
        long_eval = self.evaluate(num_episodes=500, render=False)
        
        # Performance over different episode lengths
        performance_tests = {}
        for max_steps in [200, 500, 1000]:
            print(f"Testing with max {max_steps} steps per episode...")
            scores = []
            
            for _ in range(50):
                state = self.env.reset()
                episode_score = 0
                
                for step in range(max_steps):
                    action = self.agent.act(state, epsilon=0.0)
                    state, reward, done, _ = self.env.step(action)
                    episode_score += reward
                    
                    if done:
                        break
                
                scores.append(episode_score)
            
            performance_tests[f'max_steps_{max_steps}'] = {
                'avg_score': np.mean(scores),
                'success_rate': sum(1 for s in scores if s >= 200) / len(scores)
            }
        
        benchmark_results = {
            'standard_evaluation': standard_eval,
            'long_evaluation': {
                'average_score': long_eval['average_score'],
                'success_rate': long_eval['success_rate'],
                'std_score': long_eval['std_score']
            },
            'performance_tests': performance_tests,
            'model_info': {
                'model_path': self.model_path,
                'device': self.device,
                'config': self.config
            }
        }
        
        return benchmark_results
    
    def _save_video(self, frames, filename):
        """
        Save frames as video
        
        Args:
            frames (list): List of frames
            filename (str): Output filename
        """
        if not frames:
            return
            
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        os.makedirs('videos', exist_ok=True)
        video_path = os.path.join('videos', filename)
        
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"📹 Video saved as: {video_path}")
    
    def interactive_demo(self):
        """
        Run an interactive demonstration
        """
        print("🎮 Interactive Demo - Press 'q' to quit")
        
        episode = 1
        while True:
            print(f"\n--- Episode {episode} ---")
            state = self.env.reset()
            episode_score = 0
            steps = 0
            
            while True:
                self.env.render()
                time.sleep(0.05)
                
                action = self.agent.act(state, epsilon=0.0)
                state, reward, done, _ = self.env.step(action)
                episode_score += reward
                steps += 1
                
                if done:
                    result = "🎯 SUCCESS" if episode_score >= 200 else "💥 CRASH"
                    print(f"{result} | Score: {episode_score:.2f} | Steps: {steps}")
                    break
            
            # Wait for user input
            user_input = input("Press Enter for next episode, 'q' to quit: ").strip().lower()
            if user_input == 'q':
                break
                
            episode += 1
        
        self.env.close()


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained Lunar Lander Double DQN model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during evaluation')
    parser.add_argument('--save-video', action='store_true',
                       help='Save evaluation video')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run comprehensive benchmark')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demonstration')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🔍 Lunar Lander Double DQN Evaluation")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model, device=device)
        
        if args.interactive:
            # Interactive demo
            evaluator.interactive_demo()
            
        elif args.benchmark:
            # Comprehensive benchmark
            results = evaluator.benchmark_performance()
            
            print("\n" + "=" * 50)
            print("📊 BENCHMARK RESULTS")
            print("=" * 50)
            
            std_eval = results['standard_evaluation']
            print(f"Standard Evaluation (100 episodes):")
            print(f"  Average Score: {std_eval['average_score']:.2f} ± {std_eval['std_score']:.2f}")
            print(f"  Success Rate: {std_eval['success_rate']:.2%}")
            print(f"  Score Range: {std_eval['min_score']:.2f} - {std_eval['max_score']:.2f}")
            
            long_eval = results['long_evaluation']
            print(f"\nLong Evaluation (500 episodes):")
            print(f"  Average Score: {long_eval['average_score']:.2f} ± {long_eval['std_score']:.2f}")
            print(f"  Success Rate: {long_eval['success_rate']:.2%}")
            
            print(f"\nPerformance Tests:")
            for test_name, test_results in results['performance_tests'].items():
                print(f"  {test_name}: Score {test_results['avg_score']:.2f}, "
                      f"Success {test_results['success_rate']:.2%}")
            
        else:
            # Standard evaluation
            results = evaluator.evaluate(
                num_episodes=args.episodes,
                render=args.render,
                save_video=args.save_video
            )
            
            print("\n" + "=" * 50)
            print("📊 EVALUATION RESULTS")
            print("=" * 50)
            print(f"Episodes: {results['num_episodes']}")
            print(f"Average Score: {results['average_score']:.2f} ± {results['std_score']:.2f}")
            print(f"Median Score: {results['median_score']:.2f}")
            print(f"Score Range: {results['min_score']:.2f} - {results['max_score']:.2f}")
            print(f"Success Rate: {results['success_rate']:.2%} ({results['successes']}/{results['num_episodes']})")
            print(f"Average Episode Length: {results['average_length']:.1f} steps")
            
            # Performance assessment
            if results['average_score'] >= 200:
                print("✅ EXCELLENT: Model consistently achieves target performance!")
            elif results['average_score'] >= 100:
                print("✅ GOOD: Model shows strong performance")
            elif results['average_score'] >= 0:
                print("⚠️  FAIR: Model shows some learning but needs improvement")
            else:
                print("❌ POOR: Model performance below baseline")
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise
        
    finally:
        print("\nEvaluation session ended.")


if __name__ == "__main__":
    main() 