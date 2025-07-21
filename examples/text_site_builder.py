#!/usr/bin/env python3
"""
Text-based Construction Site Builder

Alternative to the GUI builder for systems that have display issues.
Create custom construction scenarios using text commands.
"""

import json
import argparse
from typing import Dict, List, Tuple

class TextSiteBuilder:
    """Text-based construction site builder."""
    
    def __init__(self, width: int = 15, height: int = 15):
        self.width = width
        self.height = height
        self.materials = {}
        self.tasks = []
        self.spawn_points = []
        
    def print_site(self):
        """Print current site layout."""
        print(f"\nCurrent Site ({self.width}x{self.height}):")
        print("Legend: M=Material, T=Task, S=Spawn, .=Empty")
        print("  " + "".join(f"{i%10}" for i in range(self.width)))
        
        for y in range(self.height):
            row = f"{y%10} "
            for x in range(self.width):
                if (x, y) in self.materials:
                    row += "M"
                elif any(t["position"] == [x, y] for t in self.tasks):
                    row += "T"
                elif (x, y) in self.spawn_points:
                    row += "S"
                else:
                    row += "."
            print(row)
            
        print(f"\nElements:")
        print(f"  Materials: {len(self.materials)}")
        print(f"  Tasks: {len(self.tasks)}")
        print(f"  Spawn Points: {len(self.spawn_points)}")
        
    def add_material(self, x: int, y: int, material_type: str = "brick"):
        """Add material at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.materials[(x, y)] = material_type
            print(f"Added {material_type} at ({x}, {y})")
        else:
            print(f"Position ({x}, {y}) is out of bounds!")
            
    def add_task(self, x: int, y: int):
        """Add construction task at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            task_id = len(self.tasks)
            self.tasks.append({
                "id": task_id,
                "type": "build_wall",
                "position": [x, y],
                "materials_needed": 2,
                "completed": False,
                "assigned_agents": []
            })
            print(f"Added task {task_id} at ({x}, {y})")
        else:
            print(f"Position ({x}, {y}) is out of bounds!")
            
    def add_spawn(self, x: int, y: int):
        """Add agent spawn point at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if len(self.spawn_points) < 8:
                self.spawn_points.append((x, y))
                agent_num = len(self.spawn_points) - 1
                print(f"Added spawn point {agent_num} at ({x}, {y})")
            else:
                print("Maximum 8 spawn points allowed!")
        else:
            print(f"Position ({x}, {y}) is out of bounds!")
            
    def remove_element(self, x: int, y: int):
        """Remove element at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            removed = False
            
            # Remove material
            if (x, y) in self.materials:
                del self.materials[(x, y)]
                print(f"Removed material at ({x}, {y})")
                removed = True
                
            # Remove task
            for i, task in enumerate(self.tasks):
                if task["position"] == [x, y]:
                    del self.tasks[i]
                    print(f"Removed task at ({x}, {y})")
                    removed = True
                    break
                    
            # Remove spawn point
            if (x, y) in self.spawn_points:
                self.spawn_points.remove((x, y))
                print(f"Removed spawn point at ({x}, {y})")
                removed = True
                
            if not removed:
                print(f"No element found at ({x}, {y})")
        else:
            print(f"Position ({x}, {y}) is out of bounds!")
            
    def save_site(self, filename: str = None):
        """Save site to JSON file."""
        if filename is None:
            filename = f"custom_site_{self.width}x{self.height}.json"
            
        site_data = {
            "width": self.width,
            "height": self.height,
            "materials": {f"{x},{y}": material for (x, y), material in self.materials.items()},
            "tasks": self.tasks,
            "spawn_points": self.spawn_points
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(site_data, f, indent=2)
            print(f"Site saved to {filename}")
        except Exception as e:
            print(f"Error saving site: {e}")
            
    def load_site(self, filename: str = None):
        """Load site from JSON file."""
        if filename is None:
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
            
            print(f"Site loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved site found: {filename}")
        except Exception as e:
            print(f"Error loading site: {e}")
            
    def clear_site(self):
        """Clear all elements."""
        self.materials.clear()
        self.tasks.clear()
        self.spawn_points.clear()
        print("Site cleared")
        
    def create_preset(self, preset_name: str):
        """Create a preset site layout."""
        self.clear_site()
        
        if preset_name == "simple":
            # Simple 2-agent scenario
            self.add_spawn(1, 1)
            self.add_spawn(self.width-2, self.height-2)
            self.add_material(5, 5, "brick")
            self.add_material(7, 7, "brick")
            self.add_task(10, 10)
            print("Created simple preset")
            
        elif preset_name == "coordination":
            # Requires coordination between multiple agents
            self.add_spawn(0, 0)
            self.add_spawn(self.width-1, 0)
            self.add_spawn(0, self.height-1)
            self.add_spawn(self.width-1, self.height-1)
            
            # Materials on edges
            for i in range(0, self.width, 3):
                self.add_material(i, 1, "brick")
                self.add_material(i, self.height-2, "steel")
                
            # Tasks in center
            center_x, center_y = self.width//2, self.height//2
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]:
                self.add_task(center_x + dx, center_y + dy)
                
            print("Created coordination preset")
            
        elif preset_name == "resource_scarcity":
            # Few materials, many tasks
            self.add_spawn(2, 2)
            self.add_spawn(self.width-3, self.height-3)
            
            # Only 3 materials
            self.add_material(5, 5, "brick")
            self.add_material(8, 8, "steel")
            self.add_material(12, 3, "concrete")
            
            # Many tasks
            for i in range(6):
                x = (i * 3 + 2) % (self.width - 2) + 1
                y = (i * 2 + 3) % (self.height - 2) + 1
                self.add_task(x, y)
                
            print("Created resource scarcity preset")
            
        else:
            print(f"Unknown preset: {preset_name}")
            print("Available presets: simple, coordination, resource_scarcity")
            
    def run_interactive(self):
        """Run interactive text-based builder."""
        print("=== Text-Based Construction Site Builder ===")
        print("Commands:")
        print("  show - Display current site")
        print("  material <x> <y> [type] - Add material (type: brick/steel/concrete/wood)")
        print("  task <x> <y> - Add construction task")
        print("  spawn <x> <y> - Add agent spawn point")
        print("  remove <x> <y> - Remove element")
        print("  save [filename] - Save site")
        print("  load [filename] - Load site")
        print("  clear - Clear all elements")
        print("  preset <name> - Load preset (simple/coordination/resource_scarcity)")
        print("  test - Test site with agents")
        print("  quit - Exit builder")
        print()
        
        self.print_site()
        
        while True:
            try:
                command = input("\nBuilder> ").strip().split()
                if not command:
                    continue
                    
                cmd = command[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "show":
                    self.print_site()
                elif cmd == "material" and len(command) >= 3:
                    x, y = int(command[1]), int(command[2])
                    material_type = command[3] if len(command) > 3 else "brick"
                    self.add_material(x, y, material_type)
                elif cmd == "task" and len(command) >= 3:
                    x, y = int(command[1]), int(command[2])
                    self.add_task(x, y)
                elif cmd == "spawn" and len(command) >= 3:
                    x, y = int(command[1]), int(command[2])
                    self.add_spawn(x, y)
                elif cmd == "remove" and len(command) >= 3:
                    x, y = int(command[1]), int(command[2])
                    self.remove_element(x, y)
                elif cmd == "save":
                    filename = command[1] if len(command) > 1 else None
                    self.save_site(filename)
                elif cmd == "load":
                    filename = command[1] if len(command) > 1 else None
                    self.load_site(filename)
                elif cmd == "clear":
                    self.clear_site()
                elif cmd == "preset" and len(command) >= 2:
                    self.create_preset(command[1])
                elif cmd == "test":
                    self.test_site()
                else:
                    print("Invalid command. Type 'quit' to exit.")
                    
            except (ValueError, IndexError):
                print("Invalid command format. Check your syntax.")
            except KeyboardInterrupt:
                break
                
        print("Builder closed.")
        
    def test_site(self):
        """Test the site with MARL agents."""
        if len(self.spawn_points) < 2:
            print("Need at least 2 spawn points to test!")
            return
            
        print("Testing site would launch MARL simulation...")
        print("(GUI version of site builder has full testing capability)")
        print("Save your site and use: python examples/train_custom_site.py --demo")

def main():
    parser = argparse.ArgumentParser(description="Text-based Construction Site Builder")
    parser.add_argument("--width", type=int, default=15, help="Site width")
    parser.add_argument("--height", type=int, default=15, help="Site height")
    parser.add_argument("--preset", type=str, help="Load preset (simple/coordination/resource_scarcity)")
    parser.add_argument("--file", type=str, help="Load from file")
    
    args = parser.parse_args()
    
    builder = TextSiteBuilder(args.width, args.height)
    
    if args.file:
        builder.load_site(args.file)
    elif args.preset:
        builder.create_preset(args.preset)
        
    builder.run_interactive()

if __name__ == "__main__":
    main()