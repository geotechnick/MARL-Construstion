#!/usr/bin/env python3
"""
Basic training example for MARL Construction Simulation.
This script demonstrates how to set up and train a multi-agent system.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from marl_construction.envs import ConstructionEnv
from marl_construction.agents import ConstructionAgent
from marl_construction.training import MARLTrainer

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== MARL Construction Simulation Training ===")
    
    # Create environment
    env = ConstructionEnv(
        num_agents=4,
        site_width=15,
        site_height=15,
        max_timesteps=300,
        render_mode="human"
    )
    
    print(f"Created environment with {env.num_agents} agents")
    print(f"Construction site: {env.site_width}x{env.site_height}")
    
    # Get observation and action dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agents with different roles
    roles = ["builder", "transporter", "crane_operator", "supervisor"]
    agents = {}
    
    for i, agent_id in enumerate(env.agents):
        role = roles[i % len(roles)]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            role=role
        )
        print(f"Created {agent_id} with role: {role}")
    
    # Create trainer
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        communication_enabled=True,
        log_dir="./logs/basic_training"
    )
    
    print("\nStarting training...")
    
    # Train the agents
    episode_rewards = trainer.train(
        num_episodes=500,
        update_frequency=10,
        save_frequency=100
    )
    
    print("\nTraining completed!")
    
    # Plot training progress
    trainer.plot_training_progress(episode_rewards)
    
    # Evaluate the trained agents
    print("\nEvaluating trained agents...")
    eval_stats = trainer.evaluate(num_episodes=10, render=True)
    
    print(f"\nFinal Evaluation Results:")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    
    for agent_id, avg_reward in eval_stats['avg_rewards'].items():
        std_reward = eval_stats['std_rewards'][agent_id]
        role = agents[agent_id].role
        print(f"{agent_id} ({role}): {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Close environment
    env.close()
    
    print("\nTraining session completed successfully!")

if __name__ == "__main__":
    main()