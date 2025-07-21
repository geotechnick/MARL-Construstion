import pygame
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from .construction_env import ConstructionEnv, ConstructionSite

class ConstructionSiteBuilder:
    """Interactive construction site builder with GUI."""
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.cell_size = 30
        self.grid = np.zeros((height, width), dtype=int)
        
        # Building elements
        self.materials = {}
        self.tasks = []
        self.spawn_points = []
        
        # UI state
        self.current_tool = "material"  # material, task, spawn, erase
        self.current_material = "brick"
        
        # Pygame setup
        pygame.init()
        pygame.display.init()
        
        # Create display
        display_width = width * self.cell_size + 300  # Extra space for toolbar
        display_height = height * self.cell_size + 100   # Extra space for controls
        
        self.screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("MARL Construction Site Builder")
        self.clock = pygame.time.Clock()
        
        # Initialize fonts
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Colors
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'material': (139, 69, 19),
            'task': (255, 0, 255),
            'spawn': (0, 255, 0),
            'wall': (128, 128, 128),
            'button': (220, 220, 220),
            'button_active': (150, 150, 255),
            'text': (0, 0, 0)
        }
        
        self.running = True
        
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_click(event.pos)
                    
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)
                
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse clicks."""
        x, y = pos
        
        # Check if click is on grid
        if x < self.width * self.cell_size and y < self.height * self.cell_size:
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            self.place_element(grid_x, grid_y)
            
        # Check toolbar buttons
        elif x > self.width * self.cell_size:
            self.handle_toolbar_click(x, y)
            
    def place_element(self, x: int, y: int):
        """Place element at grid position."""
        if self.current_tool == "material":
            self.add_material(x, y, self.current_material)
            
        elif self.current_tool == "task":
            self.add_task(x, y)
            
        elif self.current_tool == "spawn":
            self.add_spawn(x, y)
                
        elif self.current_tool == "erase":
            self.remove_element(x, y)
            
    def add_material(self, x: int, y: int, material_type: str = "brick"):
        """Add material at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.materials[(x, y)] = material_type
            self.grid[y, x] = 1
            
    def add_task(self, x: int, y: int):
        """Add construction task at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            task_id = len(self.tasks)
            self.tasks.append({
                "id": task_id,
                "type": "build_wall",
                "position": (x, y),
                "materials_needed": 2,
                "completed": False,
                "assigned_agents": []
            })
            
    def add_spawn(self, x: int, y: int):
        """Add agent spawn point at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if len(self.spawn_points) < 8:  # Max 8 agents
                self.spawn_points.append((x, y))
                
    def remove_element(self, x: int, y: int):
        """Remove element at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Remove material
            if (x, y) in self.materials:
                del self.materials[(x, y)]
            self.grid[y, x] = 0
            
            # Remove tasks at this position
            self.tasks = [t for t in self.tasks if t["position"] != (x, y)]
            
            # Remove spawn points
            if (x, y) in self.spawn_points:
                self.spawn_points.remove((x, y))
                
    def handle_toolbar_click(self, x: int, y: int):
        """Handle clicks on toolbar buttons."""
        toolbar_x = self.width * self.cell_size + 10
        
        # Tool buttons
        button_height = 40
        button_spacing = 50
        
        tools = ["material", "task", "spawn", "erase"]
        for i, tool in enumerate(tools):
            button_y = 20 + i * button_spacing
            if (toolbar_x <= x <= toolbar_x + 100 and 
                button_y <= y <= button_y + button_height):
                self.current_tool = tool
                
        # Material type buttons (when material tool is selected)
        if self.current_tool == "material":
            materials = ["brick", "steel", "concrete", "wood"]
            for i, material in enumerate(materials):
                button_y = 250 + i * 35
                if (toolbar_x <= x <= toolbar_x + 100 and 
                    button_y <= y <= button_y + 30):
                    self.current_material = material
                    
        # Action buttons
        action_y = self.height * self.cell_size + 10
        
        # Save button
        if (10 <= x <= 80 and action_y <= y <= action_y + 30):
            self.save_site()
            
        # Load button
        elif (90 <= x <= 160 and action_y <= y <= action_y + 30):
            self.load_site()
            
        # Clear button
        elif (170 <= x <= 240 and action_y <= y <= action_y + 30):
            self.clear_site()
            
        # Test button
        elif (250 <= x <= 320 and action_y <= y <= action_y + 30):
            self.test_site()
            
    def handle_keypress(self, key):
        """Handle keyboard input."""
        if key == pygame.K_1:
            self.current_tool = "material"
        elif key == pygame.K_2:
            self.current_tool = "task"
        elif key == pygame.K_3:
            self.current_tool = "spawn"
        elif key == pygame.K_4:
            self.current_tool = "erase"
        elif key == pygame.K_s:
            self.save_site()
        elif key == pygame.K_l:
            self.load_site()
        elif key == pygame.K_c:
            self.clear_site()
        elif key == pygame.K_t:
            self.test_site()
            
    def draw(self):
        """Draw the builder interface."""
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw elements
        self.draw_elements()
        
        # Draw toolbar
        self.draw_toolbar()
        
        # Draw controls
        self.draw_controls()
        
        # Draw info
        self.draw_info()
        
        pygame.display.flip()
        
    def draw_grid(self):
        """Draw the construction site grid."""
        for x in range(self.width + 1):
            pygame.draw.line(
                self.screen, 
                self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.height * self.cell_size)
            )
            
        for y in range(self.height + 1):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (0, y * self.cell_size),
                (self.width * self.cell_size, y * self.cell_size)
            )
            
    def draw_elements(self):
        """Draw placed elements."""
        # Draw materials
        for (x, y), material_type in self.materials.items():
            rect = pygame.Rect(
                x * self.cell_size + 2,
                y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            pygame.draw.rect(self.screen, self.colors['material'], rect)
            
            # Material type indicator
            text = self.small_font.render(material_type[0].upper(), True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            
        # Draw tasks
        for task in self.tasks:
            x, y = task["position"]
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.colors['task'], rect, 3)
            
            # Task ID
            text = self.small_font.render(str(task["id"]), True, self.colors['task'])
            self.screen.blit(text, (x * self.cell_size + 5, y * self.cell_size + 5))
            
        # Draw spawn points
        for i, (x, y) in enumerate(self.spawn_points):
            center = (
                x * self.cell_size + self.cell_size // 2,
                y * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.screen, self.colors['spawn'], center, self.cell_size // 3)
            
            # Agent number
            text = self.small_font.render(str(i), True, (0, 0, 0))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)
            
    def draw_toolbar(self):
        """Draw the toolbar."""
        toolbar_x = self.width * self.cell_size + 10
        
        # Title
        title = self.font.render("Tools", True, self.colors['text'])
        self.screen.blit(title, (toolbar_x, 0))
        
        # Tool buttons
        tools = [
            ("1. Material", "material"),
            ("2. Task", "task"), 
            ("3. Spawn", "spawn"),
            ("4. Erase", "erase")
        ]
        
        button_height = 40
        button_spacing = 50
        
        for i, (label, tool) in enumerate(tools):
            button_y = 20 + i * button_spacing
            color = self.colors['button_active'] if tool == self.current_tool else self.colors['button']
            
            button_rect = pygame.Rect(toolbar_x, button_y, 100, button_height)
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, self.colors['text'], button_rect, 2)
            
            text = self.small_font.render(label, True, self.colors['text'])
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
        # Material type selector (when material tool selected)
        if self.current_tool == "material":
            materials_title = self.font.render("Materials", True, self.colors['text'])
            self.screen.blit(materials_title, (toolbar_x, 220))
            
            materials = ["brick", "steel", "concrete", "wood"]
            for i, material in enumerate(materials):
                button_y = 250 + i * 35
                color = self.colors['button_active'] if material == self.current_material else self.colors['button']
                
                button_rect = pygame.Rect(toolbar_x, button_y, 100, 30)
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, self.colors['text'], button_rect, 2)
                
                text = self.small_font.render(material.title(), True, self.colors['text'])
                text_rect = text.get_rect(center=button_rect.center)
                self.screen.blit(text, text_rect)
                
    def draw_controls(self):
        """Draw control buttons."""
        action_y = self.height * self.cell_size + 10
        
        buttons = [
            ("Save (S)", 10),
            ("Load (L)", 90),
            ("Clear (C)", 170),
            ("Test (T)", 250)
        ]
        
        for label, x in buttons:
            button_rect = pygame.Rect(x, action_y, 70, 30)
            pygame.draw.rect(self.screen, self.colors['button'], button_rect)
            pygame.draw.rect(self.screen, self.colors['text'], button_rect, 2)
            
            text = self.small_font.render(label, True, self.colors['text'])
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
    def draw_info(self):
        """Draw information panel."""
        info_y = self.height * self.cell_size + 50
        
        info_lines = [
            f"Tool: {self.current_tool.title()}",
            f"Materials: {len(self.materials)}",
            f"Tasks: {len(self.tasks)}",
            f"Spawn Points: {len(self.spawn_points)}"
        ]
        
        if self.current_tool == "material":
            info_lines.append(f"Material: {self.current_material.title()}")
            
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, self.colors['text'])
            self.screen.blit(text, (10, info_y + i * 20))
            
    def save_site(self):
        """Save current site to JSON file."""
        site_data = {
            "width": self.width,
            "height": self.height,
            "materials": {f"{x},{y}": material for (x, y), material in self.materials.items()},
            "tasks": self.tasks,
            "spawn_points": self.spawn_points
        }
        
        filename = f"custom_site_{self.width}x{self.height}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(site_data, f, indent=2)
            print(f"Site saved to {filename}")
        except Exception as e:
            print(f"Error saving site: {e}")
            
    def load_site(self):
        """Load site from JSON file."""
        filename = f"custom_site_{self.width}x{self.height}.json"
        try:
            with open(filename, 'r') as f:
                site_data = json.load(f)
                
            self.materials = {
                tuple(map(int, pos.split(','))): material 
                for pos, material in site_data["materials"].items()
            }
            self.tasks = site_data["tasks"]
            self.spawn_points = [tuple(point) for point in site_data["spawn_points"]]
            
            # Update grid
            self.grid = np.zeros((self.height, self.width), dtype=int)
            for (x, y) in self.materials:
                self.grid[y, x] = 1
                
            print(f"Site loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved site found: {filename}")
        except Exception as e:
            print(f"Error loading site: {e}")
            
    def clear_site(self):
        """Clear all elements from the site."""
        self.materials.clear()
        self.tasks.clear()
        self.spawn_points.clear()
        self.grid = np.zeros((self.height, self.width), dtype=int)
        print("Site cleared")
        
    def test_site(self):
        """Test the current site with MARL agents."""
        if len(self.spawn_points) < 2:
            print("Need at least 2 spawn points to test!")
            return
            
        print("Testing site with MARL agents...")
        
        # Save current site first
        self.save_site()
        
        # Close builder
        self.running = False
        
        try:
            # Create custom environment
            env = CustomConstructionEnv(self.to_construction_site())
            
            # Run a quick test
            from ..agents import ConstructionAgent
            import torch
            import time
            
            obs, _ = env.reset()
            
            # Create agents
            obs_dim = len(obs[env.agents[0]])
            action_dim = env.action_spaces[env.agents[0]].n
            
            agents = {}
            roles = ["builder", "transporter", "crane_operator", "supervisor"]
            for i, agent_id in enumerate(env.agents):
                role = roles[i % len(roles)]
                agents[agent_id] = ConstructionAgent(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=32,
                    role=role
                )
                
            print("Running test simulation...")
            print("Close the window to stop the test.")
            
            # Run test episode
            for step in range(100):
                env.render()
                
                # Get actions
                actions = {}
                for agent_id, agent in agents.items():
                    obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                    action, _ = agent.get_action(obs_tensor)
                    actions[agent_id] = action
                    
                obs, rewards, terms, truncs, _ = env.step(actions)
                
                if step % 20 == 0:
                    completed = len([t for t in env.construction_site.tasks if t["completed"]])
                    total = len(env.construction_site.tasks)
                    print(f"Step {step}: {completed}/{total} tasks completed")
                
                if any(terms.values()) or any(truncs.values()):
                    break
                    
                time.sleep(0.1)
                
            env.close()
            print("Test completed!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("Make sure all dependencies are installed.")
        
        # Don't restart builder automatically
        print("Test finished. Run the builder again to continue designing.")
        
    def to_construction_site(self) -> ConstructionSite:
        """Convert builder state to ConstructionSite object."""
        site = ConstructionSite(self.width, self.height)
        site.materials = self.materials.copy()
        site.tasks = self.tasks.copy()
        site.grid = self.grid.copy()
        return site
        
    def run(self):
        """Main builder loop."""
        print("=== Construction Site Builder ===")
        print("Controls:")
        print("  1/2/3/4: Select tool (Material/Task/Spawn/Erase)")
        print("  Mouse: Click to place elements")
        print("  S: Save site")
        print("  L: Load site")  
        print("  C: Clear site")
        print("  T: Test site with agents")
        print("\nClick and drag to build your construction site!")
        
        try:
            while self.running:
                self.handle_events()
                
                # Check if display is still valid
                if pygame.get_init() and pygame.display.get_surface():
                    self.draw()
                    self.clock.tick(60)
                else:
                    print("Display surface lost, exiting...")
                    break
                    
        except pygame.error as e:
            print(f"Pygame error: {e}")
        except Exception as e:
            print(f"Error in builder: {e}")
        finally:
            if pygame.get_init():
                pygame.quit()
            print("Builder closed.")

class CustomConstructionEnv(ConstructionEnv):
    """Construction environment that uses a custom site."""
    
    def __init__(self, construction_site: ConstructionSite):
        self.custom_site = construction_site
        super().__init__(
            num_agents=min(len(construction_site.materials) + len(construction_site.tasks), 8),
            site_width=construction_site.width,
            site_height=construction_site.height,
            render_mode="human"
        )
        
    def _generate_construction_scenario(self):
        """Use the custom site instead of generating random scenario."""
        self.construction_site = self.custom_site