#!/usr/bin/env python3
"""
Quick start script for MARL Construction Simulation.
This script provides a simple demo and quick training session.
"""

import torch
import numpy as np
import argparse
import time
from marl_construction.envs import ConstructionEnv
from marl_construction.agents import ConstructionAgent
from marl_construction.training import MARLTrainer

def demo_environment():
    """Run a quick demo of the environment with random agents."""
    print("🎮 Running Environment Demo with Random Agents")
    print("-" * 50)
    
    # Create a small environment for quick demo
    env = ConstructionEnv(
        num_agents=2,
        site_width=10,
        site_height=10,
        max_timesteps=100,
        render_mode="human"
    )
    
    observations, _ = env.reset()
    print(f"Environment created with {env.num_agents} agents")
    print(f"Construction site: {env.site_width}x{env.site_height}")
    print("Watch the agents move around randomly!")
    
    total_reward = 0
    
    try:
        for step in range(100):
            # Random actions
            actions = {
                agent_id: env.action_spaces[agent_id].sample()
                for agent_id in env.agents
            }
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            step_reward = sum(rewards.values())
            total_reward += step_reward
            
            if step % 20 == 0:
                completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
                total_tasks = len(env.construction_site.tasks)
                print(f"Step {step}: {completed_tasks}/{total_tasks} tasks, reward: {step_reward:.2f}")
            
            # Add small delay for visualization
            time.sleep(0.1)
            
            if any(terminations.values()) or any(truncations.values()):
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    env.close()
    print(f"Demo completed! Total reward: {total_reward:.2f}")

def quick_training():
    """Run a quick training session (few episodes for demonstration)."""
    print("🚀 Running Quick Training Session")
    print("-" * 50)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment
    env = ConstructionEnv(
        num_agents=2,
        site_width=8,
        site_height=8,
        max_timesteps=150,
        render_mode=None  # No rendering for faster training
    )
    
    # Get dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create agents
    roles = ["builder", "transporter"]
    agents = {}
    
    for i, agent_id in enumerate(env.agents):
        role = roles[i % len(roles)]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,  # Smaller network for quick training
            role=role
        )
        print(f"Created {agent_id} with role: {role}")
    
    # Create trainer
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        lr=5e-4,  # Higher learning rate for quick learning
        log_dir="./logs/quick_start"
    )
    
    print("\nStarting quick training (50 episodes)...")
    
    # Quick training
    episode_rewards = trainer.train(
        num_episodes=50,
        update_frequency=5,
        save_frequency=25
    )
    
    print("Quick training completed!")
    
    # Quick evaluation
    print("\nRunning quick evaluation...")
    eval_stats = trainer.evaluate(num_episodes=3, render=False)
    
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    for agent_id, avg_reward in eval_stats['avg_rewards'].items():
        role = agents[agent_id].role
        print(f"{agent_id} ({role}): {avg_reward:.2f}")
    
    env.close()
    return episode_rewards

def interactive_demo():
    """Run an interactive demo where user can observe trained agents."""
    print("🎯 Running Interactive Demo")
    print("-" * 50)
    
    # Create environment
    env = ConstructionEnv(
        num_agents=2,
        site_width=8,
        site_height=8,
        max_timesteps=200,
        render_mode="human"
    )
    
    # Get dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Create agents
    roles = ["builder", "transporter"]
    agents = {}
    
    for i, agent_id in enumerate(env.agents):
        role = roles[i % len(roles)]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,
            role=role
        )
    
    # Create trainer
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        log_dir="./logs/quick_start"
    )
    
    # Try to load trained models
    try:
        trainer.load_models("final_model")
        print("Loaded trained models for demonstration!")
        use_trained = True
    except:
        print("No trained models found. Using random agents.")
        use_trained = False
    
    print("Starting interactive demo...")
    print("Press Ctrl+C to stop the demo at any time.")
    
    try:
        observations, _ = env.reset()
        if use_trained:
            trainer.multi_agent_system.reset_communication()
        
        for step in range(200):
            env.render()
            
            if use_trained:
                # Use trained agents
                obs_tensors = {
                    agent_id: torch.FloatTensor(obs).unsqueeze(0)
                    for agent_id, obs in observations.items()
                }
                agent_outputs = trainer.multi_agent_system.step(obs_tensors, deterministic=True)
                actions = agent_outputs["actions"]
            else:
                # Random actions
                actions = {
                    agent_id: env.action_spaces[agent_id].sample()
                    for agent_id in env.agents
                }
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            if step % 25 == 0:
                completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
                total_tasks = len(env.construction_site.tasks)
                print(f"Step {step}: {completed_tasks}/{total_tasks} tasks completed")
            
            time.sleep(0.2)  # Slow down for better visualization
            
            if any(terminations.values()) or any(truncations.values()):
                break
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    
    env.close()
    
    completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
    total_tasks = len(env.construction_site.tasks)
    print(f"Final result: {completed_tasks}/{total_tasks} tasks completed")

def main():
    parser = argparse.ArgumentParser(description="MARL Construction Quick Start")
    parser.add_argument("--mode", 
                       choices=["demo", "train", "interactive", "all"],
                       default="all",
                       help="Mode to run")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏗️  MARL Construction Simulation - Quick Start")
    print("=" * 60)
    
    if args.mode in ["demo", "all"]:
        print("\n1️⃣  Environment Demo")
        demo_environment()
        
        if args.mode == "all":
            input("\nPress Enter to continue to training...")
    
    if args.mode in ["train", "all"]:
        print("\n2️⃣  Quick Training")
        episode_rewards = quick_training()
        
        if args.mode == "all":
            input("\nPress Enter to continue to interactive demo...")
    
    if args.mode in ["interactive", "all"]:
        print("\n3️⃣  Interactive Demo")
        interactive_demo()
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("• Run full training: python examples/basic_training.py")
    print("• Evaluate agents: python examples/evaluate_agents.py") 
    print("• Check logs: ./logs/quick_start/")

if __name__ == "__main__":
    main()