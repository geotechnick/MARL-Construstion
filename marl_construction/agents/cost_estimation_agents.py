import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    OWNER = "owner"
    CONTRACTOR = "contractor"
    SUPPLIER = "supplier"
    REGULATOR = "regulator"
    ESTIMATOR = "estimator"

@dataclass
class ProjectSpec:
    project_type: str
    location: str
    total_area: float
    duration_months: int
    complexity_score: float
    risk_factors: Dict[str, float]
    custom_parameters: Dict[str, Any]

@dataclass
class MarketConditions:
    labor_availability: float
    material_inflation: float
    supply_chain_stability: float
    economic_volatility: float
    fuel_surcharge: float
    weather_risk: float

@dataclass
class Bid:
    agent_id: str
    cost_estimate: float
    timeline_estimate: int
    confidence: float
    risk_premium: float
    profit_margin: float
    conditions: List[str]

class BaseEstimationAgent(nn.Module):
    def __init__(self, 
                 agent_type: AgentType,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.agent_type = agent_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Agent-specific policy network
        self.policy_net = self._build_policy_network()
        
        # Value network for learning
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Historical memory for project similarity
        self.memory_size = 1000
        self.project_memory = []
        
    def _build_policy_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.action_dim)
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(obs)
        policy_output = self.policy_net(features)
        value = self.value_net(features)
        return policy_output, value
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_output, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(policy_output, dim=-1)
        else:
            action_probs = F.softmax(policy_output, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze()
            
        return action, value
        
    def add_to_memory(self, project_data: Dict):
        if len(self.project_memory) >= self.memory_size:
            self.project_memory.pop(0)
        self.project_memory.append(project_data)
        
    def find_similar_projects(self, current_project: ProjectSpec, top_k: int = 5) -> List[Dict]:
        if not self.project_memory:
            return []
            
        similarities = []
        for past_project in self.project_memory:
            similarity = self._calculate_project_similarity(current_project, past_project)
            similarities.append((similarity, past_project))
            
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [proj for _, proj in similarities[:top_k]]
        
    def _calculate_project_similarity(self, proj1: ProjectSpec, proj2: Dict) -> float:
        similarity = 0.0
        
        # Project type similarity
        if proj1.project_type == proj2.get('project_type', ''):
            similarity += 0.3
            
        # Size similarity
        size_diff = abs(proj1.total_area - proj2.get('total_area', 0)) / max(proj1.total_area, 1)
        similarity += 0.2 * (1 - min(size_diff, 1.0))
        
        # Location similarity (simplified)
        if proj1.location == proj2.get('location', ''):
            similarity += 0.2
            
        # Complexity similarity
        complexity_diff = abs(proj1.complexity_score - proj2.get('complexity_score', 0))
        similarity += 0.15 * (1 - min(complexity_diff, 1.0))
        
        # Duration similarity
        duration_diff = abs(proj1.duration_months - proj2.get('duration_months', 0)) / max(proj1.duration_months, 1)
        similarity += 0.15 * (1 - min(duration_diff, 1.0))
        
        return similarity

class OwnerAgent(BaseEstimationAgent):
    def __init__(self, obs_dim: int, action_dim: int = 10, hidden_dim: int = 256):
        super().__init__(AgentType.OWNER, obs_dim, action_dim, hidden_dim)
        
        # Owner-specific objectives: minimize cost, risk, and duration while maximizing quality
        self.cost_weight = 0.4
        self.risk_weight = 0.3
        self.duration_weight = 0.2
        self.quality_weight = 0.1
        
    def evaluate_bid(self, bid: Bid, project: ProjectSpec, market: MarketConditions) -> float:
        # Owner's utility function
        cost_utility = 1.0 - min(bid.cost_estimate / (project.total_area * 200), 1.0)  # Normalize by expected cost per sq ft
        risk_utility = 1.0 - bid.risk_premium
        duration_utility = 1.0 - min(bid.timeline_estimate / (project.duration_months * 1.2), 1.0)
        quality_utility = bid.confidence
        
        total_utility = (self.cost_weight * cost_utility +
                        self.risk_weight * risk_utility +
                        self.duration_weight * duration_utility +
                        self.quality_weight * quality_utility)
        
        return total_utility
        
    def negotiate_terms(self, bids: List[Bid]) -> Dict[str, float]:
        # Owner's negotiation strategy
        best_bid = max(bids, key=lambda b: self.evaluate_bid(b, None, None))
        
        negotiation_terms = {
            "target_cost_reduction": 0.05,  # Seek 5% cost reduction
            "acceptable_risk_increase": 0.02,  # Accept small risk increase
            "quality_requirements": 0.9,  # High quality requirement
            "timeline_flexibility": 0.1  # 10% timeline flexibility
        }
        
        return negotiation_terms

class ContractorAgent(BaseEstimationAgent):
    def __init__(self, obs_dim: int, action_dim: int = 15, hidden_dim: int = 256):
        super().__init__(AgentType.CONTRACTOR, obs_dim, action_dim, hidden_dim)
        
        # Contractor objectives: maximize profit while managing risk and resource allocation
        self.profit_weight = 0.5
        self.risk_weight = 0.2
        self.resource_efficiency_weight = 0.2
        self.market_position_weight = 0.1
        
        # Contractor-specific parameters
        self.base_profit_margin = 0.15
        self.risk_tolerance = 0.3
        self.resource_capacity = 1.0
        
    def calculate_bid(self, project: ProjectSpec, market: MarketConditions) -> Bid:
        # Base cost calculation using historical data
        similar_projects = self.find_similar_projects(project)
        base_cost = self._estimate_base_cost(project, similar_projects)
        
        # Market adjustments
        labor_multiplier = 1.0 + (1.0 - market.labor_availability) * 0.3
        material_multiplier = 1.0 + market.material_inflation
        supply_chain_multiplier = 1.0 + (1.0 - market.supply_chain_stability) * 0.2
        
        adjusted_cost = base_cost * labor_multiplier * material_multiplier * supply_chain_multiplier
        
        # Risk premium calculation
        project_risk = sum(project.risk_factors.values()) / len(project.risk_factors)
        market_risk = (market.economic_volatility + market.weather_risk) / 2
        total_risk = (project_risk + market_risk) / 2
        risk_premium = total_risk * 0.2
        
        # Profit margin adjustment based on competition and market position
        profit_margin = self.base_profit_margin + risk_premium
        
        # Final bid calculation
        final_cost = adjusted_cost * (1 + profit_margin)
        
        # Timeline estimation
        timeline = self._estimate_timeline(project, market)
        
        # Confidence based on historical accuracy and project similarity
        confidence = self._calculate_confidence(project, similar_projects)
        
        return Bid(
            agent_id=f"contractor_{id(self)}",
            cost_estimate=final_cost,
            timeline_estimate=timeline,
            confidence=confidence,
            risk_premium=risk_premium,
            profit_margin=profit_margin,
            conditions=["standard_terms", "weather_clauses"]
        )
        
    def _estimate_base_cost(self, project: ProjectSpec, similar_projects: List[Dict]) -> float:
        if similar_projects:
            # Use weighted average of similar projects
            weights = [proj.get('similarity', 0.5) for proj in similar_projects]
            costs = [proj.get('actual_cost', project.total_area * 150) for proj in similar_projects]
            weighted_avg = sum(w * c for w, c in zip(weights, costs)) / sum(weights)
            return weighted_avg
        else:
            # Fallback to area-based estimation
            cost_per_sqft = self._get_cost_per_sqft_by_type(project.project_type)
            return project.total_area * cost_per_sqft
            
    def _get_cost_per_sqft_by_type(self, project_type: str) -> float:
        base_costs = {
            "residential": 120,
            "commercial": 180,
            "transportation": 250,
            "water_infrastructure": 300
        }
        return base_costs.get(project_type, 150)
        
    def _estimate_timeline(self, project: ProjectSpec, market: MarketConditions) -> int:
        base_timeline = project.duration_months
        
        # Adjust for market conditions
        if market.labor_availability < 0.7:
            base_timeline *= 1.2
        if market.supply_chain_stability < 0.7:
            base_timeline *= 1.1
            
        return int(base_timeline)
        
    def _calculate_confidence(self, project: ProjectSpec, similar_projects: List[Dict]) -> float:
        base_confidence = 0.7
        
        # Increase confidence with more similar projects
        if len(similar_projects) > 3:
            base_confidence += 0.15
        elif len(similar_projects) > 1:
            base_confidence += 0.1
            
        # Adjust for project complexity
        complexity_penalty = project.complexity_score * 0.2
        base_confidence -= complexity_penalty
        
        return max(0.3, min(0.95, base_confidence))

class SupplierAgent(BaseEstimationAgent):
    def __init__(self, obs_dim: int, action_dim: int = 12, hidden_dim: int = 256):
        super().__init__(AgentType.SUPPLIER, obs_dim, action_dim, hidden_dim)
        
        self.inventory_levels = {
            "concrete": 1000,
            "steel": 500,
            "lumber": 2000,
            "electrical": 300,
            "plumbing": 200
        }
        
        self.base_prices = {
            "concrete": 120,  # per cubic yard
            "steel": 0.85,   # per pound
            "lumber": 450,   # per thousand board feet
            "electrical": 2.5,  # per linear foot
            "plumbing": 8.0  # per linear foot
        }
        
    def set_material_prices(self, project: ProjectSpec, market: MarketConditions) -> Dict[str, float]:
        prices = {}
        
        for material, base_price in self.base_prices.items():
            # Market adjustments
            inflation_adj = base_price * (1 + market.material_inflation)
            supply_adj = inflation_adj * (1 + (1 - market.supply_chain_stability) * 0.3)
            fuel_adj = supply_adj * (1 + market.fuel_surcharge)
            
            # Inventory-based pricing
            inventory_ratio = self.inventory_levels[material] / 1000
            if inventory_ratio < 0.3:
                inventory_multiplier = 1.4  # High demand premium
            elif inventory_ratio < 0.6:
                inventory_multiplier = 1.2
            else:
                inventory_multiplier = 1.0
                
            final_price = fuel_adj * inventory_multiplier
            prices[material] = final_price
            
        return prices
        
    def check_availability(self, material_requirements: Dict[str, float], timeline: int) -> Dict[str, bool]:
        availability = {}
        
        for material, quantity in material_requirements.items():
            current_stock = self.inventory_levels.get(material, 0)
            # Consider production capacity over timeline
            projected_supply = current_stock + (timeline * 50)  # 50 units per month production
            availability[material] = projected_supply >= quantity
            
        return availability

class RegulatorAgent(BaseEstimationAgent):
    def __init__(self, obs_dim: int, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__(AgentType.REGULATOR, obs_dim, action_dim, hidden_dim)
        
        self.permit_complexity = {
            "residential": 0.3,
            "commercial": 0.6,
            "transportation": 0.9,
            "water_infrastructure": 0.8
        }
        
        self.base_permit_times = {
            "residential": 2,  # months
            "commercial": 4,
            "transportation": 8,
            "water_infrastructure": 6
        }
        
    def estimate_permit_timeline(self, project: ProjectSpec) -> int:
        base_time = self.base_permit_times.get(project.project_type, 3)
        complexity_multiplier = 1 + self.permit_complexity.get(project.project_type, 0.5)
        
        # Environmental factors
        if 'environmental_sensitivity' in project.risk_factors:
            complexity_multiplier *= (1 + project.risk_factors['environmental_sensitivity'])
            
        return int(base_time * complexity_multiplier)
        
    def calculate_compliance_costs(self, project: ProjectSpec) -> float:
        base_cost = project.total_area * 2.5  # $2.5 per sq ft base compliance cost
        
        # Complexity adjustments
        complexity_factor = self.permit_complexity.get(project.project_type, 0.5)
        compliance_cost = base_cost * (1 + complexity_factor)
        
        return compliance_cost
        
    def assess_project_feasibility(self, project: ProjectSpec) -> Dict[str, float]:
        assessment = {
            "zoning_compliance": 0.9,  # 90% likely to pass
            "environmental_impact": 0.8,
            "safety_requirements": 0.95,
            "building_code_compliance": 0.9
        }
        
        # Adjust based on project risk factors
        for risk_factor, value in project.risk_factors.items():
            if risk_factor in assessment:
                assessment[risk_factor] *= (1 - value * 0.3)
                
        return assessment

class EstimatorAgent(BaseEstimationAgent):
    def __init__(self, obs_dim: int, action_dim: int = 20, hidden_dim: int = 256):
        super().__init__(AgentType.ESTIMATOR, obs_dim, action_dim, hidden_dim)
        
        # Historical cost database simulation
        self.cost_database = {
            "residential": [],
            "commercial": [],
            "transportation": [],
            "water_infrastructure": []
        }
        
    def generate_cost_estimate(self, project: ProjectSpec, market: MarketConditions, 
                             agent_bids: List[Bid]) -> Dict[str, Any]:
        
        # Aggregate analysis from all agent bids
        bid_costs = [bid.cost_estimate for bid in agent_bids if bid.cost_estimate > 0]
        
        if not bid_costs:
            # Fallback to historical data
            estimate = self._historical_estimate(project, market)
        else:
            # Statistical analysis of bids
            mean_bid = np.mean(bid_costs)
            std_bid = np.std(bid_costs)
            median_bid = np.median(bid_costs)
            
            # Weight bids by confidence
            confidence_weights = [bid.confidence for bid in agent_bids if bid.cost_estimate > 0]
            if confidence_weights:
                weighted_avg = np.average(bid_costs, weights=confidence_weights)
            else:
                weighted_avg = mean_bid
                
            estimate = {
                "base_estimate": weighted_avg,
                "low_estimate": median_bid * 0.85,
                "high_estimate": median_bid * 1.25,
                "confidence_interval": std_bid / mean_bid if mean_bid > 0 else 0.2
            }
            
        # Add market-based adjustments
        estimate["market_adjusted"] = self._apply_market_adjustments(estimate["base_estimate"], market)
        
        # Risk assessment
        estimate["risk_factors"] = self._assess_cost_risks(project, market)
        
        # Timeline correlation
        estimate["cost_timeline_correlation"] = self._estimate_cost_timeline_relationship(project)
        
        return estimate
        
    def _historical_estimate(self, project: ProjectSpec, market: MarketConditions) -> Dict[str, Any]:
        # Simplified historical cost lookup
        base_costs = {
            "residential": 150,
            "commercial": 200,
            "transportation": 350,
            "water_infrastructure": 400
        }
        
        base_cost_per_sqft = base_costs.get(project.project_type, 175)
        total_base = project.total_area * base_cost_per_sqft
        
        return {
            "base_estimate": total_base,
            "low_estimate": total_base * 0.8,
            "high_estimate": total_base * 1.3,
            "confidence_interval": 0.25
        }
        
    def _apply_market_adjustments(self, base_estimate: float, market: MarketConditions) -> float:
        adjustment_factor = 1.0
        adjustment_factor *= (1 + market.material_inflation)
        adjustment_factor *= (1 + (1 - market.labor_availability) * 0.2)
        adjustment_factor *= (1 + market.economic_volatility * 0.1)
        
        return base_estimate * adjustment_factor
        
    def _assess_cost_risks(self, project: ProjectSpec, market: MarketConditions) -> Dict[str, float]:
        risks = {
            "scope_creep_risk": 0.15,  # 15% likelihood of scope increase
            "material_escalation_risk": market.material_inflation + 0.05,
            "labor_shortage_risk": max(0, 1 - market.labor_availability) * 0.8,
            "weather_delay_risk": market.weather_risk * 0.3,
            "regulatory_risk": sum(project.risk_factors.values()) / len(project.risk_factors) * 0.2
        }
        
        return risks
        
    def _estimate_cost_timeline_relationship(self, project: ProjectSpec) -> Dict[str, float]:
        # Relationship between cost and timeline
        return {
            "acceleration_premium": 0.15,  # 15% cost increase for 20% time reduction
            "delay_cost_impact": 0.08,     # 8% cost increase per month delay
            "optimal_duration_months": project.duration_months,
            "crash_duration_months": int(project.duration_months * 0.8)
        }