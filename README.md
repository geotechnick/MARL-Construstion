# MARL Construction Simulation

A Multi-Agent Reinforcement Learning framework for construction site simulation and coordination. This project implements specialized agents (builders, transporters, crane operators, supervisors) that learn to collaborate on construction tasks through reinforcement learning.

## 🏗️ Features

- **Multi-Agent Environment**: Construction site simulation with materials, tasks, and agent coordination
- **Specialized Agent Roles**: Different agent types with unique capabilities and responsibilities
- **MARL Training**: PPO-based training with role-specific rewards and communication
- **Visualization**: Real-time rendering of construction progress with Pygame
- **Evaluation Tools**: Comprehensive analysis and visualization of training results

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/geotechnick/MARL-Construstion
cd MARL-Construction
```

2. **Run the installation script**:
```bash
python scripts/install.py
```

3. **Quick demo**:
```bash
python scripts/quick_start.py
```

### Manual Installation

If the installation script doesn't work:

```bash
# Create virtual environment (recommended)
python -m venv marl_env
source marl_env/bin/activate  # On Windows: marl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🎮 Usage

### Basic Training

```bash
python examples/basic_training.py
```

This will:
- Create a construction environment with 4 specialized agents
- Train agents for 500 episodes using PPO
- Save models and training plots
- Evaluate final performance

### Evaluation

```bash
python examples/evaluate_agents.py
```

Evaluates trained models with detailed analysis and visualization.

### Quick Start Options

```bash
python scripts/quick_start.py --mode demo        # Environment demo only
python scripts/quick_start.py --mode train       # Quick training only  
python scripts/quick_start.py --mode interactive # Interactive demo only
python scripts/quick_start.py --mode all         # All modes (default)
```

## 🏛️ Architecture

### Environment (`ConstructionEnv`)
- Grid-based construction site
- Materials and construction tasks
- Multi-agent coordination requirements
- Role-specific action spaces

### Agents (`ConstructionAgent`)
- **Builder**: Constructs structures from materials
- **Transporter**: Moves materials around the site
- **Crane Operator**: Handles heavy lifting and precise placement
- **Supervisor**: Coordinates team activities

### Training (`MARLTrainer`)
- PPO with centralized training, decentralized execution
- Role-specific reward functions
- Communication learning between agents
- Tensorboard logging and model checkpointing

## 📊 Agent Roles

| Role | Primary Actions | Reward Focus |
|------|----------------|--------------|
| **Builder** | Construct structures | Building completion, proximity to tasks |
| **Transporter** | Carry materials | Material delivery, efficient routing |
| **Crane Operator** | Heavy lifting | Precise placement, coordination |
| **Supervisor** | Team coordination | Overall project efficiency |

## 🔧 Configuration

Key parameters in training:

```python
env = ConstructionEnv(
    num_agents=4,           # Number of agents
    site_width=20,          # Construction site width
    site_height=20,         # Construction site height
    max_timesteps=500,      # Episode length
    render_mode="human"     # Visualization mode
)

trainer = MARLTrainer(
    env=env,
    agents=agents,
    lr=3e-4,                # Learning rate
    gamma=0.99,             # Discount factor
    communication_enabled=True  # Agent communication
)
```

## 📈 Results

After training, you can expect:
- Coordinated behavior between specialized agents
- Efficient task completion strategies
- Emergent communication patterns
- Success rates of 70-90% on construction tasks

## 🛠️ Development

### Project Structure
```
MARL-Construction/
├── marl_construction/          # Main package
│   ├── envs/                  # Environment implementations
│   ├── agents/                # Agent architectures
│   └── training/              # Training algorithms
├── examples/                  # Usage examples
├── scripts/                   # Setup and utility scripts
├── logs/                      # Training logs and models
└── requirements.txt           # Dependencies
```

### Adding New Agent Types

1. Extend `ConstructionAgent` with role-specific networks
2. Update `RoleSpecificReward` for new reward functions
3. Modify environment action/observation spaces if needed

### Custom Environments

Inherit from `ConstructionEnv` and override:
- `_generate_construction_scenario()` for custom tasks
- `_get_observation()` for different state representations
- `_process_agent_action()` for new action types

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- PettingZoo 1.24+
- Pygame 2.1+
- NumPy, Matplotlib, TensorBoard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [MARL Papers Collection](https://github.com/LantaoYu/MARL-Papers)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)

## 📧 Support

For questions and support:
- Open an issue on GitHub
- Check the examples/ directory for usage patterns
- Review logs in ./logs/ for debugging
