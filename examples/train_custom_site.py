#!/usr/bin/env python3
"""
Train MARL agents on a custom construction site.
Load a site created with the site builder and train agents on it.
"""

import torch
import numpy as np
import json
import argparse
from marl_construction.envs import ConstructionEnv, ConstructionSite, CustomConstructionEnv
from marl_construction.agents import ConstructionAgent
from marl_construction.training import MARLTrainer

def load_custom_site(filename: str) -> ConstructionSite:
    """Load a custom site from JSON file."""
    try:
        with open(filename, 'r') as f:
            site_data = json.load(f)
            
        # Create construction site
        site = ConstructionSite(site_data["width"], site_data["height"])
        
        # Load materials
        site.materials = {
            tuple(map(int, pos.split(','))): material 
            for pos, material in site_data["materials"].items()
        }
        
        # Load tasks
        site.tasks = site_data["tasks"]
        
        # Update grid
        for (x, y) in site.materials:
            site.grid[y, x] = 1
            
        print(f"Loaded custom site: {site_data['width']}x{site_data['height']}")
        print(f"  Materials: {len(site.materials)}")
        print(f"  Tasks: {len(site.tasks)}")
        print(f"  Spawn points: {len(site_data['spawn_points'])}")
        
        return site, site_data["spawn_points"]
        
    except FileNotFoundError:
        print(f"Custom site file not found: {filename}")
        print("Create a site using: python examples/site_builder.py")
        return None, None
    except Exception as e:
        print(f"Error loading custom site: {e}")
        return None, None

def train_on_custom_site(site_filename: str, episodes: int = 500):
    """Train agents on a custom construction site."""
    
    # Load custom site
    custom_site, spawn_points = load_custom_site(site_filename)
    if custom_site is None:
        return
        
    # Determine number of agents from spawn points or default
    num_agents = len(spawn_points) if spawn_points else 4
    num_agents = min(max(num_agents, 2), 8)  # Between 2-8 agents
    
    print(f"Training with {num_agents} agents...")
    
    # Create custom environment
    env = CustomConstructionEnv(custom_site)
    
    # Get dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    print(f"Environment created:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Site size: {custom_site.width}x{custom_site.height}")
    
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
    log_dir = f"./logs/custom_site_{custom_site.width}x{custom_site.height}"
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        lr=3e-4,
        gamma=0.99,
        communication_enabled=True,
        log_dir=log_dir
    )
    
    print(f"\nStarting training for {episodes} episodes...")
    print(f"Logs will be saved to: {log_dir}")
    
    # Train the agents
    episode_rewards = trainer.train(
        num_episodes=episodes,
        update_frequency=10,
        save_frequency=100
    )
    
    print("\nTraining completed!")
    
    # Plot training progress
    trainer.plot_training_progress(episode_rewards)
    
    # Evaluate the trained agents
    print("\nEvaluating trained agents on custom site...")
    eval_stats = trainer.evaluate(num_episodes=10, render=True)
    
    print(f"\nFinal Evaluation Results:")
    print(f"Success Rate: {eval_stats['success_rate']:.2%}")
    
    for agent_id, avg_reward in eval_stats['avg_rewards'].items():
        std_reward = eval_stats['std_rewards'][agent_id]
        role = agents[agent_id].role
        print(f"{agent_id} ({role}): {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Close environment
    env.close()
    
    print(f"\nTraining session completed!")
    print(f"Models saved in: {log_dir}/checkpoints/")

def demo_custom_site(site_filename: str):
    """Run a demo on a custom construction site."""
    
    # Load custom site
    custom_site, spawn_points = load_custom_site(site_filename)
    if custom_site is None:
        return
        
    print("Running demo on custom site...")
    
    # Create custom environment
    env = CustomConstructionEnv(custom_site)
    
    # Get dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Create agents
    roles = ["builder", "transporter", "crane_operator", "supervisor"]
    agents = {}
    
    for i, agent_id in enumerate(env.agents):
        role = roles[i % len(roles)]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,
            role=role
        )
    
    print("Running demo with random agents...")
    print("Close the window to stop the demo.")
    
    step = 0
    try:
        while step < 300:
            env.render()
            
            # Get actions (mix of random and agent-based)
            actions = {}
            for agent_id, agent in agents.items():
                if np.random.random() < 0.5:
                    # Random action
                    actions[agent_id] = env.action_spaces[agent_id].sample()
                else:
                    # Agent action
                    obs_tensor = torch.FloatTensor(sample_obs[agent_id]).unsqueeze(0)
                    action, _ = agent.get_action(obs_tensor)
                    actions[agent_id] = action
            
            sample_obs, rewards, terms, truncs, _ = env.step(actions)
            
            if step % 50 == 0:
                completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
                total_tasks = len(env.construction_site.tasks)
                print(f"Step {step}: {completed_tasks}/{total_tasks} tasks completed")
            
            if any(terms.values()) or any(truncs.values()):
                break
                
            step += 1
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    
    env.close()
    print("Demo completed!")

def main():
    parser = argparse.ArgumentParser(description="Train on Custom Construction Site")
    parser.add_argument("--site", type=str, help="Custom site JSON file")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--demo", action="store_true", help="Run demo instead of training")
    parser.add_argument("--width", type=int, default=15, help="Default site width")
    parser.add_argument("--height", type=int, default=15, help="Default site height")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏗️  MARL Custom Construction Site Training")
    print("=" * 60)
    
    # Use provided site or default filename
    if args.site:
        site_filename = args.site
    else:
        site_filename = f"custom_site_{args.width}x{args.height}.json"
    
    if args.demo:
        demo_custom_site(site_filename)
    else:
        train_on_custom_site(site_filename, args.episodes)

if __name__ == "__main__":
    main()