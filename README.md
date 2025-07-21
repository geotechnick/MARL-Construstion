# MARL Construction Cost Estimation

A Multi-Agent Reinforcement Learning system for intelligent construction cost estimation. This project implements specialized agents (Owner, Contractor, Supplier, Regulator, Estimator) that negotiate and collaborate to produce accurate, data-driven cost estimates for construction projects across multiple sectors.

## 🏗️ Features

- **Intelligent Cost Estimation**: Multi-agent negotiation produces consensus cost estimates
- **Project Type Support**: Residential, Commercial, Transportation, Water Infrastructure  
- **Market-Responsive Pricing**: Dynamic adjustment based on economic conditions
- **Risk Assessment**: Comprehensive analysis and mitigation strategies
- **User Customization**: Override default rates and project parameters
- **Historical Learning**: Database storage and trend analysis for improved accuracy

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/geotechnick/MARL-Construstion
cd MARL-Construction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch the system**:
```bash
# GUI Interface (Recommended)
python main.py --gui

# Command Line Demo
python main.py --demo

# System Tests
python main.py --test
```

### Requirements

```bash
python -m pip install -r requirements.txt
```

## 💻 Usage

### GUI Interface (Recommended)
```bash
python main.py --gui
```
Launch the complete graphical interface for project configuration, cost estimation, and results analysis.

### Command Line Estimation
```bash
python main.py --estimate project_config.json --output results.json
```
Estimate costs from a JSON configuration file and save results.

### System Demonstration
```bash
python main.py --demo
```
Run comprehensive demonstration showing all system capabilities.

### Testing & Validation
```bash
python main.py --test
```
Execute the complete test suite and generate performance reports.

## 🏛️ System Architecture

### Multi-Agent Environment
- PettingZoo-based negotiation environment
- Market condition modeling and dynamics
- Project-specific parameter spaces
- Agent communication and coordination

### Intelligent Agents
- **Owner Agent**: Minimizes cost, risk, and duration while maximizing quality
- **Contractor Agent**: Optimizes pricing strategies, resource allocation, and profit margins  
- **Supplier Agent**: Sets material prices and manages availability based on market conditions
- **Regulator Agent**: Enforces compliance requirements and assesses regulatory risks
- **Estimator Agent**: Provides final consensus estimates using historical project data

### Cost Estimation Engine
- Multi-agent negotiation orchestration
- Consensus building and convergence analysis
- Risk assessment and adjustment
- Historical learning and adaptation

### Data Management
- SQLite database for project storage
- Regional cost databases with user overrides
- Market trend tracking and analysis
- Historical project similarity matching

## 🏗️ Supported Project Types

| Project Type | Examples | Key Parameters |
|--------------|----------|----------------|
| **Residential** | Homes, apartments, condos | Area, stories, foundation, finish level |
| **Commercial** | Offices, retail, warehouses | Structural system, occupancy, MEP complexity |
| **Transportation** | Roads, bridges, railways | Length, lanes, terrain, bridge count |
| **Water Infrastructure** | Treatment plants, pipelines | Capacity, treatment level, pipe depth |

## ⚙️ Configuration Examples

### Project Configuration
```json
{
  "project": {
    "project_type": "residential",
    "location": "midwest", 
    "total_area": 2500,
    "duration_months": 12,
    "complexity_score": 0.6,
    "risk_factors": {
      "environmental_sensitivity": 0.3,
      "geotechnical_risk": 0.2,
      "weather_exposure": 0.4,
      "regulatory_complexity": 0.25
    },
    "custom_parameters": {
      "foundation_type": "basement",
      "stories": 2,
      "finish_level": "standard"
    }
  }
}
```

### Python API Usage
```python
from marl_construction.estimation.cost_engine import CostEstimationEngine
from marl_construction.agents.cost_estimation_agents import ProjectSpec, MarketConditions

# Create estimation engine
engine = CostEstimationEngine()

# Define project
project = ProjectSpec(
    project_type="commercial",
    location="west", 
    total_area=50000,
    duration_months=24,
    complexity_score=0.7,
    # ... additional parameters
)

# Run estimation
result = engine.estimate_project_cost(project, market_conditions)
print(f"Estimated cost: ${result.total_cost:,.2f}")
```

## 📊 Industry Pain Points Addressed

| Pain Point | MARL Solution |
|------------|---------------|
| **Scope Creep** | Real-time change impact analysis with agent re-negotiation |
| **Price Volatility** | Dynamic market condition modeling and adjustment |
| **Communication Delays** | Transparent agent negotiations simulate real-world dynamics |
| **Risk Underestimation** | Comprehensive risk assessment with uncertainty quantification |
| **Change Order Processing** | Instant scope change impact analysis and re-estimation |

## 🛠️ Development

### Project Structure
```
MARL-Construction/
├── marl_construction/          # Main package
│   ├── agents/                # Agent implementations (Owner, Contractor, etc.)
│   ├── envs/                  # Cost estimation environment
│   ├── estimation/            # Core estimation engine
│   ├── projects/              # Project type definitions
│   ├── data/                  # Database and default rates
│   ├── ui/                    # User interface components
│   └── evaluation/            # Metrics and testing
├── examples/                  # Usage examples and demos
├── logs/                      # System logs and model checkpoints
└── requirements.txt           # Dependencies
```

### Adding New Project Types

1. Create project type class in `marl_construction/projects/project_types.py`
2. Define work breakdown structure and cost drivers
3. Implement complexity multipliers and quantity estimation
4. Update UI forms and agent logic as needed

### Customizing Agents

1. Extend base agent classes in `marl_construction/agents/cost_estimation_agents.py`
2. Modify reward functions and negotiation strategies
3. Update observation and action spaces in environment
4. Retrain or fine-tune agent models

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- PettingZoo 1.24+
- NumPy, Pandas, Matplotlib
- tkinter, Seaborn (for GUI)
- SQLite3 (included with Python)

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
