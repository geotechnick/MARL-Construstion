#!/usr/bin/env python3
"""
Evaluation script for trained MARL agents.
This script loads trained models and evaluates their performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from marl_construction.envs import ConstructionEnv
from marl_construction.agents import ConstructionAgent
from marl_construction.training import MARLTrainer

def main():
    print("=== MARL Construction Simulation Evaluation ===")
    
    # Create environment (same as training)
    env = ConstructionEnv(
        num_agents=4,
        site_width=15,
        site_height=15,
        max_timesteps=300,
        render_mode="human"
    )
    
    # Get dimensions
    sample_obs, _ = env.reset()
    obs_dim = len(sample_obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Create agents (same architecture as training)
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
    
    # Create trainer (for loading models)
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        log_dir="./logs/basic_training"
    )
    
    # Load trained models
    print("Loading trained models...")
    try:
        trainer.load_models("final_model")
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you have trained models in ./logs/basic_training/checkpoints/")
        return
    
    # Detailed evaluation
    print("\nRunning detailed evaluation...")
    
    # Multiple evaluation runs
    num_eval_runs = 20
    all_results = []
    
    for run in range(num_eval_runs):
        print(f"Evaluation run {run + 1}/{num_eval_runs}")
        
        observations, _ = env.reset()
        trainer.multi_agent_system.reset_communication()
        
        run_data = {
            'episode_length': 0,
            'total_reward': 0,
            'tasks_completed': 0,
            'agent_rewards': {agent_id: 0 for agent_id in env.agents},
            'success': False
        }
        
        for step in range(300):
            # Get actions
            obs_tensors = {
                agent_id: torch.FloatTensor(obs).unsqueeze(0)
                for agent_id, obs in observations.items()
            }
            
            agent_outputs = trainer.multi_agent_system.step(obs_tensors, deterministic=True)
            actions = agent_outputs["actions"]
            
            # Environment step
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            # Update run data
            run_data['episode_length'] = step + 1
            for agent_id, reward in rewards.items():
                run_data['agent_rewards'][agent_id] += reward
                run_data['total_reward'] += reward
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        # Check success
        tasks_completed = len([t for t in env.construction_site.tasks if t["completed"]])
        total_tasks = len(env.construction_site.tasks)
        run_data['tasks_completed'] = tasks_completed
        run_data['success'] = tasks_completed == total_tasks
        
        all_results.append(run_data)
    
    # Analyze results
    analyze_results(all_results, agents)
    
    # Interactive demonstration
    print("\nRunning interactive demonstration...")
    demonstrate_agents(env, trainer)
    
    env.close()

def analyze_results(results, agents):
    print("\n=== Evaluation Analysis ===")
    
    # Success rate
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"Success Rate: {success_rate:.2%}")
    
    # Average metrics
    avg_episode_length = np.mean([r['episode_length'] for r in results])
    avg_total_reward = np.mean([r['total_reward'] for r in results])
    avg_tasks_completed = np.mean([r['tasks_completed'] for r in results])
    
    print(f"Average Episode Length: {avg_episode_length:.1f} steps")
    print(f"Average Total Reward: {avg_total_reward:.2f}")
    print(f"Average Tasks Completed: {avg_tasks_completed:.1f}")
    
    # Agent-specific analysis
    print("\n=== Agent Performance ===")
    for agent_id, agent in agents.items():
        agent_rewards = [r['agent_rewards'][agent_id] for r in results]
        avg_reward = np.mean(agent_rewards)
        std_reward = np.std(agent_rewards)
        print(f"{agent_id} ({agent.role}): {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Create visualization
    create_evaluation_plots(results, agents)

def create_evaluation_plots(results, agents):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Success rate over runs
    ax1 = axes[0, 0]
    success_history = []
    cumulative_success = 0
    for i, result in enumerate(results):
        if result['success']:
            cumulative_success += 1
        success_history.append(cumulative_success / (i + 1))
    
    ax1.plot(success_history)
    ax1.axhline(y=success_history[-1], color='r', linestyle='--', 
                label=f'Final: {success_history[-1]:.2%}')
    ax1.set_title('Cumulative Success Rate')
    ax1.set_xlabel('Evaluation Run')
    ax1.set_ylabel('Success Rate')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Reward distribution by agent
    ax2 = axes[0, 1]
    agent_rewards_data = []
    agent_labels = []
    
    for agent_id, agent in agents.items():
        rewards = [r['agent_rewards'][agent_id] for r in results]
        agent_rewards_data.append(rewards)
        agent_labels.append(f"{agent_id}\n({agent.role})")
    
    ax2.boxplot(agent_rewards_data, labels=agent_labels)
    ax2.set_title('Reward Distribution by Agent')
    ax2.set_ylabel('Total Reward')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Plot 3: Episode length vs Success
    ax3 = axes[1, 0]
    successful_episodes = [r['episode_length'] for r in results if r['success']]
    failed_episodes = [r['episode_length'] for r in results if not r['success']]
    
    ax3.hist([successful_episodes, failed_episodes], bins=20, alpha=0.7, 
             label=['Successful', 'Failed'], color=['green', 'red'])
    ax3.set_title('Episode Length Distribution')
    ax3.set_xlabel('Episode Length (steps)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Plot 4: Tasks completed distribution
    ax4 = axes[1, 1]
    tasks_completed = [r['tasks_completed'] for r in results]
    ax4.hist(tasks_completed, bins=range(max(tasks_completed) + 2), alpha=0.7)
    ax4.set_title('Tasks Completed Distribution')
    ax4.set_xlabel('Number of Tasks Completed')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('./logs/basic_training/evaluation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_agents(env, trainer):
    """Run an interactive demonstration with visualization."""
    print("Starting interactive demonstration...")
    print("Watch the agents collaborate to complete construction tasks!")
    
    observations, _ = env.reset()
    trainer.multi_agent_system.reset_communication()
    
    step_count = 0
    
    try:
        for step in range(300):
            # Render environment
            env.render()
            
            # Get actions
            obs_tensors = {
                agent_id: torch.FloatTensor(obs).unsqueeze(0)
                for agent_id, obs in observations.items()
            }
            
            agent_outputs = trainer.multi_agent_system.step(obs_tensors, deterministic=True)
            actions = agent_outputs["actions"]
            
            # Environment step
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            step_count += 1
            
            # Print progress every 50 steps
            if step % 50 == 0:
                completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
                total_tasks = len(env.construction_site.tasks)
                print(f"Step {step}: {completed_tasks}/{total_tasks} tasks completed")
            
            if any(terminations.values()) or any(truncations.values()):
                break
                
        # Final results
        completed_tasks = len([t for t in env.construction_site.tasks if t["completed"]])
        total_tasks = len(env.construction_site.tasks)
        
        print(f"\nDemonstration completed!")
        print(f"Final result: {completed_tasks}/{total_tasks} tasks completed")
        print(f"Episode length: {step_count} steps")
        
        if completed_tasks == total_tasks:
            print("🎉 All tasks completed successfully!")
        else:
            print(f"Partial completion: {completed_tasks/total_tasks:.1%}")
            
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")

if __name__ == "__main__":
    main()