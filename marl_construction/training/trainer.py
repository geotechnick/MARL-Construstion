import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard import SummaryWriter
import os
import pickle
from tqdm import tqdm

from ..envs import ConstructionEnv
from ..agents import ConstructionAgent, MultiAgentSystem, RoleSpecificReward

class MARLTrainer:
    def __init__(self,
                 env: ConstructionEnv,
                 agents: Dict[str, ConstructionAgent],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 communication_enabled: bool = True,
                 log_dir: str = "./logs"):
        
        self.env = env
        self.agents = agents
        self.multi_agent_system = MultiAgentSystem(agents, communication_enabled)
        
        # Training hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Optimizers
        self.optimizers = {
            agent_id: optim.Adam(agent.parameters(), lr=lr)
            for agent_id, agent in agents.items()
        }
        
        # Experience storage
        self.episode_buffer = defaultdict(list)
        self.training_stats = defaultdict(list)
        
        # Logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Reward calculator
        self.reward_calculator = RoleSpecificReward()
        
    def collect_episode(self, max_steps: int = 500) -> Dict[str, List]:
        observations, infos = self.env.reset()
        self.multi_agent_system.reset_communication()
        
        episode_data = defaultdict(list)
        total_rewards = defaultdict(float)
        
        for step in range(max_steps):
            # Convert observations to tensors
            obs_tensors = {
                agent_id: torch.FloatTensor(obs).unsqueeze(0)
                for agent_id, obs in observations.items()
            }
            
            # Get actions from multi-agent system
            agent_outputs = self.multi_agent_system.step(obs_tensors)
            actions = agent_outputs["actions"]
            values = agent_outputs["values"]
            
            # Store experience
            for agent_id in self.env.agents:
                episode_data[agent_id].append({
                    'observation': observations[agent_id].copy(),
                    'action': actions[agent_id],
                    'value': values[agent_id].item(),
                    'log_prob': self._get_log_prob(agent_id, obs_tensors[agent_id], actions[agent_id])
                })
            
            # Environment step
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Add role-specific rewards
            for agent_id in self.env.agents:
                role = self.agents[agent_id].role
                role_reward = self.reward_calculator.calculate_role_reward(
                    role, actions[agent_id], {}, {}
                )
                rewards[agent_id] += role_reward
                total_rewards[agent_id] += rewards[agent_id]
                
                # Store reward
                episode_data[agent_id][-1]['reward'] = rewards[agent_id]
            
            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                break
                
        # Calculate advantages and returns
        for agent_id in self.env.agents:
            self._calculate_advantages(episode_data[agent_id])
            
        return episode_data, total_rewards
        
    def _get_log_prob(self, agent_id: str, obs: torch.Tensor, action: int) -> float:
        with torch.no_grad():
            action_probs, _, _ = self.agents[agent_id](obs)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            return log_prob.item()
            
    def _calculate_advantages(self, episode_data: List[Dict]):
        rewards = [step['reward'] for step in episode_data]
        values = [step['value'] for step in episode_data]
        
        # Calculate returns
        returns = []
        running_return = 0
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
            
        # Calculate advantages using GAE
        advantages = []
        running_advantage = 0
        
        for i in reversed(range(len(episode_data))):
            if i == len(episode_data) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            running_advantage = delta + self.gamma * self.gae_lambda * running_advantage
            advantages.insert(0, running_advantage)
            
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in episode data
        for i, step in enumerate(episode_data):
            step['return'] = returns[i]
            step['advantage'] = advantages[i]
            
    def update_agent(self, agent_id: str, episode_data: List[Dict], epochs: int = 4):
        agent = self.agents[agent_id]
        optimizer = self.optimizers[agent_id]
        
        # Convert data to tensors
        observations = torch.FloatTensor([step['observation'] for step in episode_data])
        actions = torch.LongTensor([step['action'] for step in episode_data])
        old_log_probs = torch.FloatTensor([step['log_prob'] for step in episode_data])
        returns = torch.FloatTensor([step['return'] for step in episode_data])
        advantages = torch.FloatTensor([step['advantage'] for step in episode_data])
        old_values = torch.FloatTensor([step['value'] for step in episode_data])
        
        for epoch in range(epochs):
            # Forward pass
            action_probs, values, _ = agent(observations)
            values = values.squeeze()
            
            # Calculate new log probabilities
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            
            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy loss (for exploration)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def train(self, num_episodes: int = 1000, update_frequency: int = 10, save_frequency: int = 100):
        print(f"Starting MARL training for {num_episodes} episodes...")
        
        episode_rewards = defaultdict(list)
        episode = 0
        
        with tqdm(total=num_episodes, desc="Training") as pbar:
            while episode < num_episodes:
                # Collect episodes
                batch_data = defaultdict(list)
                batch_rewards = defaultdict(list)
                
                for _ in range(update_frequency):
                    episode_data, total_rewards = self.collect_episode()
                    
                    for agent_id in self.env.agents:
                        batch_data[agent_id].extend(episode_data[agent_id])
                        batch_rewards[agent_id].append(total_rewards[agent_id])
                        episode_rewards[agent_id].append(total_rewards[agent_id])
                        
                    episode += 1
                    pbar.update(1)
                    
                    if episode >= num_episodes:
                        break
                        
                # Update agents
                training_losses = {}
                for agent_id in self.env.agents:
                    if batch_data[agent_id]:  # Only update if we have data
                        losses = self.update_agent(agent_id, batch_data[agent_id])
                        training_losses[agent_id] = losses
                        
                # Logging
                self._log_training_stats(episode, episode_rewards, training_losses)
                
                # Save models
                if episode % save_frequency == 0:
                    self.save_models(f"checkpoint_episode_{episode}")
                    
        print("Training completed!")
        self.save_models("final_model")
        return episode_rewards
        
    def _log_training_stats(self, episode: int, episode_rewards: Dict, training_losses: Dict):
        # Log rewards
        for agent_id, rewards in episode_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards[-10:])  # Last 10 episodes
                self.writer.add_scalar(f"Rewards/{agent_id}", avg_reward, episode)
                
        # Log losses
        for agent_id, losses in training_losses.items():
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f"Losses/{agent_id}/{loss_name}", loss_value, episode)
                
        # Log environment stats
        completed_tasks = len(self.env.construction_site.completed_tasks)
        total_tasks = len(self.env.construction_site.tasks)
        completion_rate = completed_tasks / max(total_tasks, 1)
        self.writer.add_scalar("Environment/TaskCompletionRate", completion_rate, episode)
        
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict:
        print(f"Evaluating for {num_episodes} episodes...")
        
        evaluation_rewards = defaultdict(list)
        success_rate = 0
        
        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            self.multi_agent_system.reset_communication()
            
            episode_reward = defaultdict(float)
            
            for step in range(500):  # Max steps per episode
                if render and episode == 0:  # Render first episode
                    self.env.render()
                    
                # Convert observations to tensors
                obs_tensors = {
                    agent_id: torch.FloatTensor(obs).unsqueeze(0)
                    for agent_id, obs in observations.items()
                }
                
                # Get actions (deterministic)
                agent_outputs = self.multi_agent_system.step(obs_tensors, deterministic=True)
                actions = agent_outputs["actions"]
                
                # Environment step
                observations, rewards, terminations, truncations, _ = self.env.step(actions)
                
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                    
                if any(terminations.values()) or any(truncations.values()):
                    break
                    
            # Check success (all tasks completed)
            if all(task["completed"] for task in self.env.construction_site.tasks):
                success_rate += 1
                
            for agent_id, reward in episode_reward.items():
                evaluation_rewards[agent_id].append(reward)
                
        success_rate /= num_episodes
        
        # Calculate statistics
        eval_stats = {
            "success_rate": success_rate,
            "avg_rewards": {
                agent_id: np.mean(rewards)
                for agent_id, rewards in evaluation_rewards.items()
            },
            "std_rewards": {
                agent_id: np.std(rewards)
                for agent_id, rewards in evaluation_rewards.items()
            }
        }
        
        print(f"Evaluation Results:")
        print(f"Success Rate: {success_rate:.2%}")
        for agent_id, avg_reward in eval_stats["avg_rewards"].items():
            print(f"{agent_id} Average Reward: {avg_reward:.2f} ± {eval_stats['std_rewards'][agent_id]:.2f}")
            
        return eval_stats
        
    def save_models(self, checkpoint_name: str):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': self.optimizers[agent_id].state_dict(),
                'role': agent.role
            }, os.path.join(checkpoint_dir, f"{checkpoint_name}_{agent_id}.pth"))
            
    def load_models(self, checkpoint_name: str):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        
        for agent_id, agent in self.agents.items():
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_{agent_id}.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                agent.load_state_dict(checkpoint['model_state_dict'])
                self.optimizers[agent_id].load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded model for {agent_id}")
            else:
                print(f"No checkpoint found for {agent_id}")
                
    def plot_training_progress(self, episode_rewards: Dict):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Individual agent rewards
        plt.subplot(1, 3, 1)
        for agent_id, rewards in episode_rewards.items():
            plt.plot(rewards, label=agent_id, alpha=0.7)
        plt.title("Training Rewards by Agent")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        
        # Plot 2: Average team reward
        plt.subplot(1, 3, 2)
        team_rewards = []
        max_episodes = max(len(rewards) for rewards in episode_rewards.values())
        for episode in range(max_episodes):
            episode_total = sum(
                rewards[episode] if episode < len(rewards) else 0
                for rewards in episode_rewards.values()
            )
            team_rewards.append(episode_total)
        plt.plot(team_rewards)
        plt.title("Team Performance")
        plt.xlabel("Episode")
        plt.ylabel("Total Team Reward")
        
        # Plot 3: Moving average
        plt.subplot(1, 3, 3)
        window = 50
        for agent_id, rewards in episode_rewards.items():
            if len(rewards) >= window:
                moving_avg = []
                for i in range(window, len(rewards)):
                    moving_avg.append(np.mean(rewards[i-window:i]))
                plt.plot(range(window, len(rewards)), moving_avg, label=f"{agent_id} (MA)")
        plt.title(f"Moving Average ({window} episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_progress.png"))
        plt.show()