#!/usr/bin/env python3
"""
Graphics demo script - shows the visual interface of MARL Construction.
"""

import torch
import numpy as np
import time
from marl_construction.envs import ConstructionEnv
from marl_construction.agents import ConstructionAgent

def demo_graphics():
    """Run a visual demo showing the graphics interface."""
    print("=== MARL Construction Graphics Demo ===")
    print("Watch the agents collaborate on construction tasks!")
    print("Close the window or press Ctrl+C to stop.\n")
    
    # Create environment with rendering enabled
    env = ConstructionEnv(
        num_agents=4,
        site_width=12,
        site_height=12,
        max_timesteps=200,
        render_mode="human"  # Enable graphics!
    )
    
    print("Environment created with graphics enabled.")
    print("Legend:")
    print("  🔴 Red Circle    = Builder Agent")
    print("  🟢 Green Circle  = Transporter Agent") 
    print("  🔵 Blue Circle   = Crane Operator Agent")
    print("  🟡 Yellow Circle = Supervisor Agent")
    print("  🟤 Brown Square  = Materials (bricks)")
    print("  ⬜ Gray Square   = Completed Structures")
    print("  🟪 Magenta Border = Task Locations")
    print("  ⬜ White Background = Empty Space\n")
    
    # Initialize environment
    observations, _ = env.reset()
    
    # Get dimensions for agents
    obs_dim = len(observations[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Create agents (for intelligent behavior)
    roles = ["builder", "transporter", "crane_operator", "supervisor"]
    agents = {}
    
    for i, agent_id in enumerate(env.agents):
        role = roles[i % len(roles)]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            role=role
        )
    
    print("Agents created. Starting visual simulation...")
    print("Watch how different roles behave:")
    print("- Builders seek construction sites")
    print("- Transporters move materials around")
    print("- Crane operators coordinate placement")
    print("- Supervisors oversee the project\n")
    
    step = 0
    total_rewards = {agent_id: 0 for agent_id in env.agents}
    
    try:
        while step < 200:
            # Render the environment (shows graphics)
            env.render()
            
            # Get actions from agents (mix of smart and random)
            actions = {}
            for agent_id, agent in agents.items():
                if step < 50:
                    # Start with some random exploration
                    if np.random.random() < 0.3:
                        actions[agent_id] = env.action_spaces[agent_id].sample()
                    else:
                        obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0)
                        action, _ = agent.get_action(obs_tensor)
                        actions[agent_id] = action
                else:
                    # More intelligent behavior later
                    obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0)
                    action, _ = agent.get_action(obs_tensor)
                    actions[agent_id] = action
            
            # Step environment
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Track progress
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            # Print progress every 25 steps
            if step % 25 == 0:
                completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
                total_tasks = len(env.construction_site.tasks)
                print(f"Step {step:3d}: {completed_tasks}/{total_tasks} tasks completed")
                
                # Show agent rewards
                for agent_id, total_reward in total_rewards.items():
                    role = agents[agent_id].role
                    print(f"  {agent_id} ({role}): {total_reward:.1f} reward")
                print()
            
            # Small delay for better visualization
            time.sleep(0.1)
            
            step += 1
            
            # Check if done
            if any(terminations.values()) or any(truncations.values()):
                break
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    
    # Final results
    completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
    total_tasks = len(env.construction_site.tasks)
    
    print(f"\n=== Demo Results ===")
    print(f"Tasks Completed: {completed_tasks}/{total_tasks}")
    print(f"Success Rate: {completed_tasks/total_tasks:.1%}")
    print(f"Total Steps: {step}")
    
    print(f"\nFinal Agent Performance:")
    for agent_id, total_reward in total_rewards.items():
        role = agents[agent_id].role
        print(f"  {agent_id} ({role}): {total_reward:.1f} total reward")
    
    env.close()
    print("\nGraphics demo completed!")

if __name__ == "__main__":
    demo_graphics()