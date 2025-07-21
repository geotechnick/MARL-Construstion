import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from pettingzoo import ParallelEnv
import random
from dataclasses import asdict

from ..agents.cost_estimation_agents import (
    OwnerAgent, ContractorAgent, SupplierAgent, RegulatorAgent, EstimatorAgent,
    ProjectSpec, MarketConditions, Bid, AgentType
)

class CostEstimationEnvironment(ParallelEnv):
    metadata = {"render_modes": ["human", "text"], "name": "cost_estimation_v1"}
    
    def __init__(self,
                 max_episodes: int = 100,
                 max_negotiation_rounds: int = 5,
                 render_mode: Optional[str] = None):
        
        self.max_episodes = max_episodes
        self.max_negotiation_rounds = max_negotiation_rounds
        self.render_mode = render_mode
        
        # Define agent types and their roles
        self.agent_types = [
            "owner_agent",
            "contractor_agent", 
            "supplier_agent",
            "regulator_agent",
            "estimator_agent"
        ]
        
        self.agents = self.agent_types[:]
        self.possible_agents = self.agents[:]
        
        # State tracking
        self.current_project: Optional[ProjectSpec] = None
        self.current_market: Optional[MarketConditions] = None
        self.negotiation_round = 0
        self.episode_count = 0
        self.agent_bids: Dict[str, Bid] = {}
        self.negotiation_history: List[Dict] = []
        
        # Observation space: project features + market conditions + negotiation state
        obs_dim = 25  # Project (10) + Market (6) + Negotiation state (9)
        
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Action spaces vary by agent type
        self.action_spaces = {
            "owner_agent": spaces.Discrete(10),      # Bid evaluation, negotiation terms
            "contractor_agent": spaces.Discrete(15), # Bid calculation, adjustments
            "supplier_agent": spaces.Discrete(12),   # Price setting, availability
            "regulator_agent": spaces.Discrete(8),   # Compliance assessment, timeline
            "estimator_agent": spaces.Discrete(20)   # Analysis, recommendations
        }
        
        # Initialize agent models
        self._initialize_agents()
        
        # Default data for projects and market
        self._initialize_default_data()
        
    def _initialize_agents(self):
        obs_dim = 25
        self.agent_models = {
            "owner_agent": OwnerAgent(obs_dim),
            "contractor_agent": ContractorAgent(obs_dim),
            "supplier_agent": SupplierAgent(obs_dim),
            "regulator_agent": RegulatorAgent(obs_dim),
            "estimator_agent": EstimatorAgent(obs_dim)
        }
        
    def _initialize_default_data(self):
        # Default regional rates and data
        self.default_data = {
            "labor_rates": {
                "northeast": {"carpenter": 35, "electrician": 45, "plumber": 40, "laborer": 25},
                "southeast": {"carpenter": 28, "electrician": 38, "plumber": 35, "laborer": 20},
                "midwest": {"carpenter": 30, "electrician": 40, "plumber": 37, "laborer": 22},
                "west": {"carpenter": 40, "electrician": 50, "plumber": 45, "laborer": 30}
            },
            "material_costs": {
                "concrete": 120,
                "steel": 0.85,
                "lumber": 450,
                "electrical": 2.5,
                "plumbing": 8.0
            },
            "equipment_rates": {
                "excavator": 350,      # per day
                "crane": 800,          # per day  
                "concrete_pump": 600,  # per day
                "bulldozer": 400       # per day
            }
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate new project and market conditions
        self.current_project = self._generate_random_project()
        self.current_market = self._generate_market_conditions()
        
        # Reset negotiation state
        self.negotiation_round = 0
        self.agent_bids = {}
        self.negotiation_history = []
        
        # Generate initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {"project": asdict(self.current_project), 
                        "market": asdict(self.current_market)} for agent in self.agents}
        
        return observations, infos
        
    def _generate_random_project(self) -> ProjectSpec:
        project_types = ["residential", "commercial", "transportation", "water_infrastructure"]
        locations = ["northeast", "southeast", "midwest", "west"]
        
        project_type = random.choice(project_types)
        location = random.choice(locations)
        
        # Generate project parameters based on type
        if project_type == "residential":
            area = random.uniform(1200, 8000)  # sq ft
            duration = random.randint(6, 18)   # months
            complexity = random.uniform(0.2, 0.7)
        elif project_type == "commercial":
            area = random.uniform(5000, 100000)
            duration = random.randint(12, 36)
            complexity = random.uniform(0.4, 0.8)
        elif project_type == "transportation":
            area = random.uniform(50000, 500000)  # road miles converted to sq ft equivalent
            duration = random.randint(18, 60)
            complexity = random.uniform(0.6, 0.9)
        else:  # water_infrastructure
            area = random.uniform(10000, 200000)
            duration = random.randint(15, 48)
            complexity = random.uniform(0.5, 0.9)
            
        risk_factors = {
            "environmental_sensitivity": random.uniform(0.1, 0.8),
            "geotechnical_risk": random.uniform(0.1, 0.6),
            "weather_exposure": random.uniform(0.2, 0.7),
            "regulatory_complexity": random.uniform(0.1, 0.5)
        }
        
        custom_parameters = {
            "energy_efficiency_target": random.choice(["standard", "high", "net_zero"]),
            "sustainability_certification": random.choice([None, "LEED", "BREEAM"]),
            "owner_experience": random.choice(["novice", "experienced", "expert"])
        }
        
        return ProjectSpec(
            project_type=project_type,
            location=location,
            total_area=area,
            duration_months=duration,
            complexity_score=complexity,
            risk_factors=risk_factors,
            custom_parameters=custom_parameters
        )
        
    def _generate_market_conditions(self) -> MarketConditions:
        return MarketConditions(
            labor_availability=random.uniform(0.4, 0.9),
            material_inflation=random.uniform(0.02, 0.15),
            supply_chain_stability=random.uniform(0.5, 0.95),
            economic_volatility=random.uniform(0.1, 0.4),
            fuel_surcharge=random.uniform(0.05, 0.25),
            weather_risk=random.uniform(0.1, 0.6)
        )
        
    def _get_observation(self, agent: str) -> np.ndarray:
        obs = []
        
        # Project features (10 dimensions)
        obs.extend([
            hash(self.current_project.project_type) % 100 / 100.0,  # normalized hash
            hash(self.current_project.location) % 100 / 100.0,
            min(self.current_project.total_area / 100000.0, 1.0),  # normalized area
            self.current_project.duration_months / 60.0,  # normalized duration
            self.current_project.complexity_score,
            self.current_project.risk_factors["environmental_sensitivity"],
            self.current_project.risk_factors["geotechnical_risk"], 
            self.current_project.risk_factors["weather_exposure"],
            self.current_project.risk_factors["regulatory_complexity"],
            1.0 if self.current_project.custom_parameters.get("sustainability_certification") else 0.0
        ])
        
        # Market conditions (6 dimensions)
        obs.extend([
            self.current_market.labor_availability,
            self.current_market.material_inflation,
            self.current_market.supply_chain_stability,
            self.current_market.economic_volatility,
            self.current_market.fuel_surcharge,
            self.current_market.weather_risk
        ])
        
        # Negotiation state (9 dimensions)
        obs.extend([
            self.negotiation_round / self.max_negotiation_rounds,
            len(self.agent_bids) / len(self.agents),  # bid completion ratio
            len(self.negotiation_history) / 20.0,    # normalized history length
        ])
        
        # Agent-specific bid information (6 dimensions)
        if agent in self.agent_bids:
            bid = self.agent_bids[agent]
            obs.extend([
                min(bid.cost_estimate / (self.current_project.total_area * 300), 2.0),  # normalized cost
                bid.timeline_estimate / 60.0,  # normalized timeline
                bid.confidence,
                bid.risk_premium,
                bid.profit_margin,
                1.0  # has bid indicator
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # no bid yet
            
        return np.array(obs, dtype=np.float32)
        
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # Process each agent's action
        self._process_agent_actions(actions, rewards, infos)
        
        # Check if negotiation round is complete
        if len(self.agent_bids) == len(self.agents):
            self.negotiation_round += 1
            
            # Calculate negotiation outcome
            self._calculate_negotiation_rewards(rewards)
            
            # Update negotiation history
            round_summary = {
                "round": self.negotiation_round,
                "bids": dict(self.agent_bids),
                "market_conditions": asdict(self.current_market)
            }
            self.negotiation_history.append(round_summary)
            
            # Reset for next round if not terminal
            if self.negotiation_round < self.max_negotiation_rounds:
                self.agent_bids = {}
            else:
                # Episode complete
                final_rewards = self._calculate_final_rewards()
                for agent in self.agents:
                    rewards[agent] += final_rewards[agent]
                    terminations[agent] = True
                    
        # Generate new observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        # Add detailed info for analysis
        for agent in self.agents:
            infos[agent].update({
                "negotiation_round": self.negotiation_round,
                "total_bids": len(self.agent_bids),
                "project_type": self.current_project.project_type,
                "market_volatility": self.current_market.economic_volatility
            })
            
        return observations, rewards, terminations, truncations, infos
        
    def _process_agent_actions(self, actions: Dict[str, int], rewards: Dict[str, float], infos: Dict):
        for agent, action in actions.items():
            if agent == "contractor_agent":
                # Contractor generates bid
                if agent not in self.agent_bids:
                    bid = self.agent_models[agent].calculate_bid(self.current_project, self.current_market)
                    self.agent_bids[agent] = bid
                    rewards[agent] += 1.0  # Reward for participating
                    infos[agent]["bid_submitted"] = True
                else:
                    # Contractor adjusts existing bid based on action
                    existing_bid = self.agent_bids[agent]
                    adjustment_factor = 1.0 + (action - 7) * 0.02  # Actions 0-14 map to adjustments
                    existing_bid.cost_estimate *= adjustment_factor
                    rewards[agent] += 0.5
                    
            elif agent == "supplier_agent":
                # Supplier sets prices and availability
                prices = self.agent_models[agent].set_material_prices(self.current_project, self.current_market)
                material_reqs = self._estimate_material_requirements()
                availability = self.agent_models[agent].check_availability(material_reqs, self.current_project.duration_months)
                
                # Generate supplier "bid" for material costs
                total_material_cost = sum(prices[mat] * qty for mat, qty in material_reqs.items())
                
                supplier_bid = Bid(
                    agent_id=agent,
                    cost_estimate=total_material_cost,
                    timeline_estimate=self.current_project.duration_months,
                    confidence=0.9 if all(availability.values()) else 0.6,
                    risk_premium=0.05,
                    profit_margin=0.1,
                    conditions=["material_availability_dependent"]
                )
                
                self.agent_bids[agent] = supplier_bid
                rewards[agent] += 1.0 if all(availability.values()) else 0.5
                infos[agent]["material_availability"] = availability
                
            elif agent == "owner_agent":
                # Owner evaluates bids and sets negotiation terms
                if len(self.agent_bids) > 1:
                    bids_list = list(self.agent_bids.values())
                    terms = self.agent_models[agent].negotiate_terms(bids_list)
                    infos[agent]["negotiation_terms"] = terms
                    rewards[agent] += 0.8
                    
            elif agent == "regulator_agent":
                # Regulator assesses compliance and timeline
                permit_time = self.agent_models[agent].estimate_permit_timeline(self.current_project)
                compliance_cost = self.agent_models[agent].calculate_compliance_costs(self.current_project)
                feasibility = self.agent_models[agent].assess_project_feasibility(self.current_project)
                
                # Create regulatory "bid" 
                regulatory_bid = Bid(
                    agent_id=agent,
                    cost_estimate=compliance_cost,
                    timeline_estimate=permit_time,
                    confidence=min(feasibility.values()),
                    risk_premium=0.1,
                    profit_margin=0.0,  # No profit for regulator
                    conditions=["regulatory_compliance"]
                )
                
                self.agent_bids[agent] = regulatory_bid
                rewards[agent] += 1.0
                infos[agent]["feasibility_assessment"] = feasibility
                
            elif agent == "estimator_agent":
                # Estimator analyzes all bids and provides final estimate
                if len(self.agent_bids) >= 2:
                    other_bids = [bid for aid, bid in self.agent_bids.items() if aid != agent]
                    estimate = self.agent_models[agent].generate_cost_estimate(
                        self.current_project, self.current_market, other_bids
                    )
                    
                    estimator_bid = Bid(
                        agent_id=agent,
                        cost_estimate=estimate["base_estimate"],
                        timeline_estimate=self.current_project.duration_months,
                        confidence=1.0 - estimate["confidence_interval"],
                        risk_premium=estimate["risk_factors"]["scope_creep_risk"],
                        profit_margin=0.05,  # Small estimator fee
                        conditions=["final_estimate"]
                    )
                    
                    self.agent_bids[agent] = estimator_bid
                    rewards[agent] += 2.0  # Higher reward for comprehensive analysis
                    infos[agent]["detailed_estimate"] = estimate
                    
    def _estimate_material_requirements(self) -> Dict[str, float]:
        # Simplified material requirement estimation based on project
        area_factor = self.current_project.total_area / 1000.0
        
        if self.current_project.project_type == "residential":
            return {
                "concrete": area_factor * 15,
                "steel": area_factor * 2000,
                "lumber": area_factor * 8,
                "electrical": area_factor * 200,
                "plumbing": area_factor * 150
            }
        elif self.current_project.project_type == "commercial":
            return {
                "concrete": area_factor * 25,
                "steel": area_factor * 3500,
                "lumber": area_factor * 3,
                "electrical": area_factor * 400,
                "plumbing": area_factor * 200
            }
        else:  # transportation or water
            return {
                "concrete": area_factor * 40,
                "steel": area_factor * 5000,
                "lumber": area_factor * 1,
                "electrical": area_factor * 150,
                "plumbing": area_factor * 300
            }
            
    def _calculate_negotiation_rewards(self, rewards: Dict[str, float]):
        # Reward agents based on negotiation effectiveness
        if len(self.agent_bids) >= 3:
            cost_estimates = [bid.cost_estimate for bid in self.agent_bids.values()]
            mean_cost = np.mean(cost_estimates)
            std_cost = np.std(cost_estimates)
            
            # Reward convergence (lower standard deviation)
            convergence_reward = max(0, 2.0 - std_cost / mean_cost)
            
            for agent in self.agents:
                if agent in self.agent_bids:
                    # Additional reward for being close to consensus
                    bid_deviation = abs(self.agent_bids[agent].cost_estimate - mean_cost) / mean_cost
                    consensus_reward = max(0, 1.0 - bid_deviation)
                    rewards[agent] += convergence_reward + consensus_reward
                    
    def _calculate_final_rewards(self) -> Dict[str, float]:
        final_rewards = {agent: 0.0 for agent in self.agents}
        
        if not self.agent_bids:
            return final_rewards
            
        # Get final cost estimates
        cost_estimates = [bid.cost_estimate for bid in self.agent_bids.values()]
        mean_cost = np.mean(cost_estimates)
        
        # Reward based on agent-specific objectives
        for agent, bid in self.agent_bids.items():
            if agent == "owner_agent":
                # Owner wants low cost and high confidence
                cost_score = max(0, 2.0 - bid.cost_estimate / mean_cost)
                confidence_score = bid.confidence * 2.0
                final_rewards[agent] = cost_score + confidence_score
                
            elif agent == "contractor_agent":
                # Contractor wants reasonable profit margin and successful bid
                profit_score = min(2.0, bid.profit_margin * 10)  # Cap at 2.0
                competitiveness = max(0, 2.0 - abs(bid.cost_estimate - mean_cost) / mean_cost)
                final_rewards[agent] = profit_score + competitiveness
                
            elif agent == "supplier_agent":
                # Supplier wants market-appropriate pricing
                market_alignment = max(0, 2.0 - abs(bid.cost_estimate - mean_cost * 0.4) / (mean_cost * 0.4))
                final_rewards[agent] = market_alignment + 1.0  # Base reward for participation
                
            elif agent == "regulator_agent":
                # Regulator wants realistic compliance assessment
                realism_score = bid.confidence * 2.0
                final_rewards[agent] = realism_score + 1.0
                
            elif agent == "estimator_agent":
                # Estimator wants accurate final prediction
                if len(cost_estimates) > 1:
                    other_estimates = [ce for i, ce in enumerate(cost_estimates) if list(self.agent_bids.keys())[i] != agent]
                    accuracy_score = max(0, 2.0 - abs(bid.cost_estimate - np.mean(other_estimates)) / np.mean(other_estimates))
                    final_rewards[agent] = accuracy_score + 2.0  # Base reward for comprehensive analysis
                else:
                    final_rewards[agent] = 2.0
                    
        return final_rewards
        
    def render(self):
        if self.render_mode == "human" or self.render_mode == "text":
            print(f"\n=== Cost Estimation Environment - Round {self.negotiation_round} ===")
            print(f"Project: {self.current_project.project_type} | {self.current_project.total_area:.0f} sq ft")
            print(f"Location: {self.current_project.location} | Duration: {self.current_project.duration_months} months")
            print(f"Market - Labor: {self.current_market.labor_availability:.2f} | Inflation: {self.current_market.material_inflation:.2f}")
            
            if self.agent_bids:
                print("\nCurrent Bids:")
                for agent, bid in self.agent_bids.items():
                    print(f"  {agent}: ${bid.cost_estimate:,.0f} | {bid.timeline_estimate} mo | Conf: {bid.confidence:.2f}")
            else:
                print("\nNo bids submitted yet.")
                
            print("=" * 60)
            
    def close(self):
        pass