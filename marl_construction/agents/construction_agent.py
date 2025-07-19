import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class ConstructionAgent(nn.Module):
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 role: str = "builder"):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.role = role
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Role-specific networks
        if role == "builder":
            self.role_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
        elif role == "transporter":
            self.role_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
        elif role == "crane_operator":
            self.role_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
        else:  # supervisor
            self.role_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Communication network
        self.communication = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
    def forward(self, obs: torch.Tensor, messages: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature extraction
        features = self.feature_net(obs)
        
        # Role-specific processing
        role_features = self.role_net(features)
        
        # Combine features
        combined_features = torch.cat([features, role_features], dim=-1)
        
        # Actor output (action probabilities)
        action_logits = self.actor(combined_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output (value estimate)
        value = self.critic(combined_features)
        
        # Communication message
        message = self.communication(features)
        
        return action_probs, value, message
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        action_probs, value, message = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = torch.multinomial(action_probs, 1)
            
        return action.item(), message
        
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value, _ = self.forward(obs)
        return value

class MultiAgentSystem:
    def __init__(self, 
                 agents: Dict[str, ConstructionAgent],
                 communication_enabled: bool = True):
        self.agents = agents
        self.communication_enabled = communication_enabled
        self.message_buffer = {}
        
    def reset_communication(self):
        self.message_buffer = {agent_id: [] for agent_id in self.agents.keys()}
        
    def step(self, observations: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Any]:
        actions = {}
        values = {}
        messages = {}
        
        # Get messages from previous step
        prev_messages = self._aggregate_messages() if self.communication_enabled else None
        
        # Process each agent
        for agent_id, agent in self.agents.items():
            obs = observations[agent_id]
            action, message = agent.get_action(obs, deterministic)
            value = agent.get_value(obs)
            
            actions[agent_id] = action
            values[agent_id] = value
            messages[agent_id] = message
            
        # Update message buffer
        if self.communication_enabled:
            self.message_buffer = messages
            
        return {
            "actions": actions,
            "values": values,
            "messages": messages
        }
        
    def _aggregate_messages(self) -> torch.Tensor:
        if not self.message_buffer:
            return None
            
        messages = list(self.message_buffer.values())
        return torch.stack(messages).mean(dim=0)

class RoleSpecificReward:
    @staticmethod
    def calculate_role_reward(agent_role: str, 
                            action: int, 
                            env_state: Dict,
                            agent_state: Dict) -> float:
        
        base_reward = 0.0
        
        if agent_role == "builder":
            # Reward for building structures
            if action == 9 and env_state.get("structure_built", False):
                base_reward += 15.0
            # Penalty for being far from construction sites
            if env_state.get("distance_to_task", float('inf')) > 5:
                base_reward -= 1.0
                
        elif agent_role == "transporter":
            # Reward for carrying materials
            if agent_state.get("carrying_material", False):
                base_reward += 2.0
            # Reward for delivering materials
            if action == 9 and env_state.get("material_delivered", False):
                base_reward += 8.0
                
        elif agent_role == "crane_operator":
            # Reward for precise placement
            if action == 9 and env_state.get("precise_placement", False):
                base_reward += 12.0
            # Reward for coordinating with other agents
            if env_state.get("coordinating", False):
                base_reward += 3.0
                
        elif agent_role == "supervisor":
            # Reward for overall project coordination
            project_efficiency = env_state.get("project_efficiency", 0.0)
            base_reward += project_efficiency * 5.0
            # Reward for agent coordination
            coordination_score = env_state.get("coordination_score", 0.0)
            base_reward += coordination_score * 3.0
            
        return base_reward

class CommunicationProtocol:
    def __init__(self, message_dim: int = 16):
        self.message_dim = message_dim
        self.message_types = {
            "request_help": 0,
            "material_available": 1,
            "task_completed": 2,
            "coordination": 3,
            "emergency": 4
        }
        
    def encode_message(self, message_type: str, content: Dict) -> torch.Tensor:
        message = torch.zeros(self.message_dim)
        
        # Message type encoding
        if message_type in self.message_types:
            message[self.message_types[message_type]] = 1.0
            
        # Content encoding (simplified)
        if "urgency" in content:
            message[5] = min(content["urgency"], 1.0)
        if "position" in content:
            message[6:8] = torch.tensor(content["position"][:2])
        if "resource_type" in content:
            message[8] = hash(content["resource_type"]) % 100 / 100.0
            
        return message
        
    def decode_message(self, message: torch.Tensor) -> Dict:
        decoded = {}
        
        # Find message type
        message_type_idx = torch.argmax(message[:5]).item()
        decoded["type"] = list(self.message_types.keys())[message_type_idx]
        
        # Extract content
        decoded["urgency"] = message[5].item()
        decoded["position"] = message[6:8].tolist()
        decoded["resource_info"] = message[8].item()
        
        return decoded