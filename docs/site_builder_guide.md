# 🏗️ Construction Site Builder Guide

The Construction Site Builder allows you to create custom construction scenarios for MARL training. Design your own layouts with materials, tasks, and agent spawn points.

## 🚀 Quick Start

### Launch the Builder
```bash
python examples/site_builder.py
```

### Custom Size
```bash
python examples/site_builder.py --width 20 --height 20
```

## 🎮 Controls

### Mouse Controls
- **Left Click**: Place elements on the grid
- **Toolbar Clicks**: Select tools and options

### Keyboard Shortcuts
- **1**: Material tool
- **2**: Task tool  
- **3**: Spawn point tool
- **4**: Erase tool
- **S**: Save site
- **L**: Load site
- **C**: Clear site
- **T**: Test site with agents

## 🛠️ Tools

### 1. Material Tool
- Place construction materials on the grid
- **Material Types**:
  - **Brick**: Standard building material
  - **Steel**: Heavy construction material
  - **Concrete**: Foundation material
  - **Wood**: Light construction material
- **Visual**: Brown squares with material type letter

### 2. Task Tool
- Place construction tasks (what needs to be built)
- Each task requires materials to complete
- **Visual**: Magenta border squares with task ID
- **Properties**:
  - Unique task ID
  - Position on grid
  - Materials needed (default: 2)

### 3. Spawn Point Tool
- Set starting positions for agents
- Maximum 8 spawn points (8 agents max)
- **Visual**: Green circles with agent number
- **Agent Roles** (assigned automatically):
  - Agent 0: Builder
  - Agent 1: Transporter
  - Agent 2: Crane Operator
  - Agent 3: Supervisor
  - (Pattern repeats for more agents)

### 4. Erase Tool
- Remove any element from the grid
- Works on materials, tasks, and spawn points

## 💾 File Operations

### Save Site
- **Button**: Save or press **S**
- Saves to `custom_site_WIDTHxHEIGHT.json`
- **Format**: JSON with materials, tasks, and spawn points

### Load Site
- **Button**: Load or press **L**  
- Loads from `custom_site_WIDTHxHEIGHT.json`
- Restores all elements to the grid

### Clear Site
- **Button**: Clear or press **C**
- Removes all elements from the grid
- Fresh start for designing

## 🧪 Test Site

### Live Testing
- **Button**: Test or press **T**
- Closes builder and launches live simulation
- Agents will attempt to complete your construction tasks
- **Requirements**: At least 2 spawn points needed

### Watch Agent Behavior
- **Builders**: Seek construction sites and materials
- **Transporters**: Move materials around the site
- **Crane Operators**: Handle heavy lifting and placement
- **Supervisors**: Coordinate team activities

## 🎯 Training on Custom Sites

### Train Agents
```bash
python examples/train_custom_site.py --site custom_site_15x15.json --episodes 1000
```

### Demo Mode
```bash
python examples/train_custom_site.py --site custom_site_15x15.json --demo
```

### Auto-detect Site
```bash
# Uses default custom_site_15x15.json
python examples/train_custom_site.py --episodes 500
```

## 📐 Design Tips

### Effective Layouts
1. **Spread Materials**: Don't cluster all materials in one area
2. **Strategic Tasks**: Place tasks requiring coordination between agents
3. **Clear Paths**: Leave space for agents to move around
4. **Spawn Placement**: Position spawn points for good starting coverage

### Challenging Scenarios
- **Resource Scarcity**: Few materials, many tasks
- **Coordination Required**: Tasks requiring multiple agent types
- **Complex Layouts**: Materials and tasks spread across large areas
- **Bottlenecks**: Narrow passages requiring coordination

### Training Scenarios
- **Simple**: 2-3 tasks, abundant materials, small site
- **Medium**: 4-6 tasks, moderate materials, medium site  
- **Complex**: 8+ tasks, limited materials, large site
- **Expert**: Custom constraints and specialized layouts

## 📊 Site File Format

### JSON Structure
```json
{
  "width": 15,
  "height": 15,
  "materials": {
    "5,3": "brick",
    "7,8": "steel",
    "2,12": "concrete"
  },
  "tasks": [
    {
      "id": 0,
      "type": "build_wall", 
      "position": [10, 5],
      "materials_needed": 2,
      "completed": false,
      "assigned_agents": []
    }
  ],
  "spawn_points": [
    [1, 1],
    [14, 14],
    [1, 14], 
    [14, 1]
  ]
}
```

## 🔧 Advanced Usage

### Batch Site Creation
Create multiple sites programmatically:
```python
from marl_construction.envs import ConstructionSiteBuilder

# Create and save multiple sites
for size in [10, 15, 20]:
    builder = ConstructionSiteBuilder(size, size)
    # Add elements programmatically
    builder.save_site()
```

### Custom Training Loops
```python
from marl_construction.envs import CustomConstructionEnv
import json

# Load custom site
with open('my_site.json', 'r') as f:
    site_data = json.load(f)

# Create environment
env = CustomConstructionEnv(site_data)

# Train with custom parameters
trainer = MARLTrainer(env, agents, custom_config)
```

## 🎨 Visual Legend

| Element | Visual | Description |
|---------|--------|-------------|
| **Materials** | 🟤 Brown Square | Construction materials |
| **Tasks** | 🟪 Magenta Border | Construction objectives |
| **Spawn Points** | 🟢 Green Circle | Agent starting positions |
| **Grid** | ⬜ Gray Lines | Construction site layout |
| **Empty** | ⬜ White Space | Available building area |

## 🚀 Next Steps

1. **Create Your First Site**: Start with a simple 10x10 layout
2. **Test and Iterate**: Use the test function to see how agents behave
3. **Train Agents**: Run training on your custom scenarios
4. **Share Sites**: Save and share interesting site designs
5. **Advanced Scenarios**: Create complex multi-objective sites

The Site Builder gives you complete control over the construction environment, enabling custom training scenarios and research into multi-agent coordination!