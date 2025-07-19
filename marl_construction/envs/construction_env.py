import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional, Any
from pettingzoo import ParallelEnv
from gymnasium import spaces
import random

class ConstructionSite:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.materials = {}
        self.tasks = []
        self.completed_tasks = []
        
    def add_material(self, x: int, y: int, material_type: str):
        self.materials[(x, y)] = material_type
        self.grid[y, x] = 1
        
    def add_task(self, task: Dict):
        self.tasks.append(task)
        
    def remove_material(self, x: int, y: int):
        if (x, y) in self.materials:
            del self.materials[(x, y)]
            self.grid[y, x] = 0
            
    def place_structure(self, x: int, y: int):
        self.grid[y, x] = 2

class ConstructionEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "construction_v1"}
    
    def __init__(self, 
                 num_agents: int = 4,
                 site_width: int = 20,
                 site_height: int = 20,
                 max_timesteps: int = 500,
                 render_mode: Optional[str] = None):
        
        self.num_agents = num_agents
        self.site_width = site_width
        self.site_height = site_height
        self.max_timesteps = max_timesteps
        self.render_mode = render_mode
        
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        
        self.construction_site = ConstructionSite(site_width, site_height)
        self.agent_positions = {}
        self.agent_inventories = {}
        self.agent_roles = {}
        
        self.timestep = 0
        
        # Define action and observation spaces
        self.action_spaces = {
            agent: spaces.Discrete(9) for agent in self.agents  # 8 directions + stay + special action
        }
        
        obs_size = site_width * site_height + 10  # Grid + agent info
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Pygame for rendering
        self.screen = None
        self.clock = None
        self.cell_size = 30
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.construction_site = ConstructionSite(self.site_width, self.site_height)
        self.timestep = 0
        
        # Reset agent states
        self.agent_positions = {}
        self.agent_inventories = {}
        self.agent_roles = {}
        
        # Randomly place agents
        for i, agent in enumerate(self.agents):
            x, y = random.randint(0, self.site_width-1), random.randint(0, self.site_height-1)
            while (x, y) in self.agent_positions.values():
                x, y = random.randint(0, self.site_width-1), random.randint(0, self.site_height-1)
            self.agent_positions[agent] = (x, y)
            self.agent_inventories[agent] = {"materials": 0, "carrying": None}
            self.agent_roles[agent] = ["builder", "transporter", "crane_operator", "supervisor"][i % 4]
            
        # Add initial materials and tasks
        self._generate_construction_scenario()
        
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
        
    def _generate_construction_scenario(self):
        # Add random materials
        for _ in range(10):
            x, y = random.randint(0, self.site_width-1), random.randint(0, self.site_height-1)
            if self.construction_site.grid[y, x] == 0:
                self.construction_site.add_material(x, y, "brick")
                
        # Add construction tasks
        for i in range(5):
            x, y = random.randint(2, self.site_width-3), random.randint(2, self.site_height-3)
            self.construction_site.add_task({
                "id": i,
                "type": "build_wall",
                "position": (x, y),
                "materials_needed": 2,
                "completed": False,
                "assigned_agents": []
            })
            
    def _get_observation(self, agent: str) -> np.ndarray:
        obs = []
        
        # Grid state (flattened)
        obs.extend(self.construction_site.grid.flatten())
        
        # Agent position
        x, y = self.agent_positions[agent]
        obs.extend([x / self.site_width, y / self.site_height])
        
        # Agent inventory
        obs.append(self.agent_inventories[agent]["materials"] / 10.0)
        obs.append(1.0 if self.agent_inventories[agent]["carrying"] else 0.0)
        
        # Role encoding
        role_encoding = [0, 0, 0, 0]
        role_map = {"builder": 0, "transporter": 1, "crane_operator": 2, "supervisor": 3}
        role_encoding[role_map[self.agent_roles[agent]]] = 1.0
        obs.extend(role_encoding)
        
        # Nearby agents
        nearby_agents = 0
        for other_agent in self.agents:
            if other_agent != agent:
                ox, oy = self.agent_positions[other_agent]
                if abs(ox - x) <= 2 and abs(oy - y) <= 2:
                    nearby_agents += 1
        obs.append(nearby_agents / len(self.agents))
        
        # Tasks info
        incomplete_tasks = len([t for t in self.construction_site.tasks if not t["completed"]])
        obs.append(incomplete_tasks / max(len(self.construction_site.tasks), 1))
        
        return np.array(obs, dtype=np.float32)
        
    def step(self, actions: Dict[str, int]):
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # Process actions
        for agent, action in actions.items():
            self._process_agent_action(agent, action, rewards)
            
        # Update environment state
        self._update_tasks()
        
        # Calculate global rewards
        global_reward = self._calculate_global_reward()
        for agent in self.agents:
            rewards[agent] += global_reward
            
        self.timestep += 1
        
        # Check termination conditions
        all_tasks_completed = all(t["completed"] for t in self.construction_site.tasks)
        time_limit_reached = self.timestep >= self.max_timesteps
        
        if all_tasks_completed or time_limit_reached:
            terminations = {agent: True for agent in self.agents}
            
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
        
    def _process_agent_action(self, agent: str, action: int, rewards: Dict[str, float]):
        x, y = self.agent_positions[agent]
        
        # Movement actions (0-7: 8 directions, 8: stay, 9: special action)
        if action < 8:
            dx, dy = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][action]
            new_x, new_y = max(0, min(self.site_width-1, x + dx)), max(0, min(self.site_height-1, y + dy))
            
            # Check collision with other agents
            if (new_x, new_y) not in [pos for agent_id, pos in self.agent_positions.items() if agent_id != agent]:
                self.agent_positions[agent] = (new_x, new_y)
                rewards[agent] += 0.01  # Small reward for movement
                
        elif action == 9:  # Special action based on role
            self._execute_special_action(agent, rewards)
            
    def _execute_special_action(self, agent: str, rewards: Dict[str, float]):
        x, y = self.agent_positions[agent]
        role = self.agent_roles[agent]
        
        if role == "transporter":
            # Pick up or drop material
            if (x, y) in self.construction_site.materials and not self.agent_inventories[agent]["carrying"]:
                material = self.construction_site.materials[(x, y)]
                self.construction_site.remove_material(x, y)
                self.agent_inventories[agent]["carrying"] = material
                self.agent_inventories[agent]["materials"] += 1
                rewards[agent] += 5.0
            elif self.agent_inventories[agent]["carrying"]:
                # Drop material
                if self.construction_site.grid[y, x] == 0:
                    self.construction_site.add_material(x, y, self.agent_inventories[agent]["carrying"])
                    self.agent_inventories[agent]["carrying"] = None
                    rewards[agent] += 2.0
                    
        elif role == "builder":
            # Build structure if materials available
            nearby_materials = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.site_width and 0 <= ny < self.site_height:
                        if (nx, ny) in self.construction_site.materials:
                            nearby_materials += 1
                            
            if nearby_materials >= 2 and self.construction_site.grid[y, x] == 0:
                self.construction_site.place_structure(x, y)
                rewards[agent] += 10.0
                
    def _update_tasks(self):
        for task in self.construction_site.tasks:
            if not task["completed"]:
                tx, ty = task["position"]
                if self.construction_site.grid[ty, tx] == 2:  # Structure built
                    task["completed"] = True
                    self.construction_site.completed_tasks.append(task)
                    
    def _calculate_global_reward(self) -> float:
        completed_ratio = len(self.construction_site.completed_tasks) / max(len(self.construction_site.tasks), 1)
        return completed_ratio * 5.0
        
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
            
    def _render_human(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((
                self.site_width * self.cell_size, 
                self.site_height * self.cell_size
            ))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.site_width * self.cell_size, self.site_height * self.cell_size))
        canvas.fill((255, 255, 255))
        
        # Draw grid
        for x in range(self.site_width):
            for y in range(self.site_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                if self.construction_site.grid[y, x] == 1:  # Material
                    pygame.draw.rect(canvas, (139, 69, 19), rect)  # Brown
                elif self.construction_site.grid[y, x] == 2:  # Structure
                    pygame.draw.rect(canvas, (128, 128, 128), rect)  # Gray
                    
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)
                
        # Draw agents
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, agent in enumerate(self.agents):
            x, y = self.agent_positions[agent]
            center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(canvas, colors[i % len(colors)], center, self.cell_size // 3)
            
        # Draw tasks
        for task in self.construction_site.tasks:
            if not task["completed"]:
                tx, ty = task["position"]
                rect = pygame.Rect(tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(canvas, (255, 0, 255), rect, 3)  # Magenta border
                
        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(60)
        
    def _render_rgb_array(self):
        # Similar to _render_human but return numpy array
        pass
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None