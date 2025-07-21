import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import json

from ..agents.cost_estimation_agents import (
    OwnerAgent, ContractorAgent, SupplierAgent, RegulatorAgent, EstimatorAgent,
    ProjectSpec, MarketConditions, Bid, AgentType
)
from ..projects.project_types import get_project_type, get_available_project_types
from ..data.default_rates import get_default_rates_manager, get_labor_cost, get_material_total_cost, get_equipment_total_cost
from ..envs.cost_estimation_env import CostEstimationEnvironment

@dataclass
class EstimationResult:
    project_id: str
    total_cost: float
    cost_breakdown: Dict[str, float]
    timeline_months: int
    confidence_score: float
    risk_assessment: Dict[str, float]
    agent_bids: Dict[str, Bid]
    market_factors: Dict[str, float]
    cost_per_sqft: float
    negotiation_rounds: int
    convergence_score: float
    created_timestamp: str

@dataclass
class EstimationConfig:
    max_negotiation_rounds: int = 5
    convergence_threshold: float = 0.05  # 5% deviation for convergence
    enable_user_overrides: bool = True
    include_risk_premium: bool = True
    market_adjustment_factor: float = 1.0
    learning_enabled: bool = True

class CostEstimationEngine:
    def __init__(self, config: Optional[EstimationConfig] = None):
        self.config = config or EstimationConfig()
        self.rates_manager = get_default_rates_manager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize MARL environment
        self.env = CostEstimationEnvironment(
            max_negotiation_rounds=self.config.max_negotiation_rounds
        )
        
        # Agent models (initialized when needed)
        self.agent_models = {}
        self._initialize_agents()
        
        # Historical data for learning
        self.estimation_history: List[EstimationResult] = []
        
    def _initialize_agents(self):
        """Initialize all agent models"""
        obs_dim = 25  # Must match environment observation space
        
        self.agent_models = {
            "owner_agent": OwnerAgent(obs_dim),
            "contractor_agent": ContractorAgent(obs_dim),
            "supplier_agent": SupplierAgent(obs_dim),
            "regulator_agent": RegulatorAgent(obs_dim),
            "estimator_agent": EstimatorAgent(obs_dim)
        }
        
    def estimate_project_cost(self, 
                            project_spec: ProjectSpec,
                            market_conditions: Optional[MarketConditions] = None,
                            user_overrides: Optional[Dict[str, Any]] = None) -> EstimationResult:
        """
        Main entry point for cost estimation using MARL agents
        """
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate market conditions if not provided
        if market_conditions is None:
            market_conditions = self._generate_market_conditions(project_spec)
            
        # Apply user overrides if provided
        if user_overrides and self.config.enable_user_overrides:
            self._apply_user_overrides(project_spec, market_conditions, user_overrides)
            
        # Set up environment with project and market data
        self.env.current_project = project_spec
        self.env.current_market = market_conditions
        
        # Reset environment and get initial observations
        observations, _ = self.env.reset()
        
        # Run multi-agent negotiation
        negotiation_results = self._run_negotiation(observations)
        
        # Generate final cost estimate
        estimation_result = self._compile_estimation_result(
            project_id, project_spec, market_conditions, negotiation_results
        )
        
        # Store for learning
        if self.config.learning_enabled:
            self.estimation_history.append(estimation_result)
            self._update_agent_memory(estimation_result)
            
        return estimation_result
        
    def _generate_market_conditions(self, project_spec: ProjectSpec) -> MarketConditions:
        """Generate realistic market conditions based on project and current data"""
        # Base conditions - could be enhanced with real market data APIs
        base_conditions = MarketConditions(
            labor_availability=np.random.uniform(0.6, 0.9),
            material_inflation=np.random.uniform(0.02, 0.12),
            supply_chain_stability=np.random.uniform(0.7, 0.95),
            economic_volatility=np.random.uniform(0.1, 0.3),
            fuel_surcharge=np.random.uniform(0.05, 0.2),
            weather_risk=np.random.uniform(0.1, 0.5)
        )
        
        # Adjust based on project characteristics
        if project_spec.project_type == "transportation":
            base_conditions.weather_risk *= 1.3  # Higher weather exposure
            base_conditions.fuel_surcharge *= 1.2  # More fuel-dependent
            
        if project_spec.complexity_score > 0.7:
            base_conditions.labor_availability *= 0.9  # Complex projects need skilled labor
            
        return base_conditions
        
    def _apply_user_overrides(self, project_spec: ProjectSpec, 
                            market_conditions: MarketConditions,
                            user_overrides: Dict[str, Any]):
        """Apply user-specified rate overrides"""
        for override_key, override_value in user_overrides.items():
            if override_key.startswith("labor_"):
                trade = override_key.replace("labor_", "")
                self.rates_manager.set_user_override(
                    "labor_rates", 
                    f"{trade}_{project_spec.location}",
                    "base_rate",
                    override_value
                )
                
            elif override_key.startswith("material_"):
                material = override_key.replace("material_", "")
                self.rates_manager.set_user_override(
                    "material_costs",
                    f"{material}_{project_spec.location}",
                    "base_cost", 
                    override_value
                )
                
            elif override_key.startswith("market_"):
                market_field = override_key.replace("market_", "")
                if hasattr(market_conditions, market_field):
                    setattr(market_conditions, market_field, override_value)
                    
    def _run_negotiation(self, initial_observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run the multi-agent negotiation process"""
        observations = initial_observations
        total_rewards = {agent: 0.0 for agent in self.env.agents}
        negotiation_history = []
        
        for round_num in range(self.config.max_negotiation_rounds):
            # Get actions from each agent
            actions = {}
            agent_outputs = {}
            
            for agent_id, obs in observations.items():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, value = self.agent_models[agent_id].get_action(obs_tensor)
                actions[agent_id] = action.item() if torch.is_tensor(action) else action
                agent_outputs[agent_id] = {"action": actions[agent_id], "value": value}
                
            # Step environment
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Update total rewards
            for agent in self.env.agents:
                total_rewards[agent] += rewards[agent]
                
            # Store round results
            round_data = {
                "round": round_num + 1,
                "actions": actions.copy(),
                "rewards": rewards.copy(),
                "agent_bids": dict(self.env.agent_bids),
                "convergence": self._check_convergence()
            }
            negotiation_history.append(round_data)
            
            # Check for early termination
            if all(terminations.values()) or self._check_convergence():
                break
                
        return {
            "total_rewards": total_rewards,
            "negotiation_history": negotiation_history,
            "final_bids": dict(self.env.agent_bids),
            "rounds_completed": len(negotiation_history)
        }
        
    def _check_convergence(self) -> bool:
        """Check if agent bids have converged"""
        if len(self.env.agent_bids) < 3:
            return False
            
        cost_estimates = [bid.cost_estimate for bid in self.env.agent_bids.values()]
        if len(cost_estimates) < 2:
            return False
            
        mean_cost = np.mean(cost_estimates)
        std_cost = np.std(cost_estimates)
        
        # Convergence if standard deviation is less than threshold
        return (std_cost / mean_cost) < self.config.convergence_threshold
        
    def _compile_estimation_result(self, 
                                 project_id: str,
                                 project_spec: ProjectSpec,
                                 market_conditions: MarketConditions,
                                 negotiation_results: Dict[str, Any]) -> EstimationResult:
        """Compile final estimation result from negotiation outcomes"""
        
        agent_bids = negotiation_results["final_bids"]
        
        # Calculate consensus cost estimate
        if agent_bids:
            cost_estimates = [bid.cost_estimate for bid in agent_bids.values()]
            confidence_scores = [bid.confidence for bid in agent_bids.values()]
            
            # Weighted average by confidence
            if confidence_scores:
                total_cost = np.average(cost_estimates, weights=confidence_scores)
                confidence_score = np.mean(confidence_scores)
            else:
                total_cost = np.mean(cost_estimates)
                confidence_score = 0.7
        else:
            # Fallback to basic estimation
            total_cost = self._fallback_cost_estimate(project_spec)
            confidence_score = 0.5
            
        # Generate detailed cost breakdown
        cost_breakdown = self._generate_cost_breakdown(project_spec, total_cost)
        
        # Calculate timeline
        if agent_bids:
            timeline_estimates = [bid.timeline_estimate for bid in agent_bids.values()]
            timeline_months = int(np.median(timeline_estimates))
        else:
            timeline_months = project_spec.duration_months
            
        # Risk assessment
        risk_assessment = self._calculate_risk_assessment(project_spec, market_conditions, agent_bids)
        
        # Market factors impact
        market_factors = {\n            \"labor_impact\": 1.0 + (1.0 - market_conditions.labor_availability) * 0.2,\n            \"material_impact\": 1.0 + market_conditions.material_inflation,\n            \"supply_chain_impact\": 1.0 + (1.0 - market_conditions.supply_chain_stability) * 0.15,\n            \"economic_impact\": 1.0 + market_conditions.economic_volatility * 0.1\n        }\n        \n        # Calculate convergence score\n        convergence_score = self._calculate_convergence_score(negotiation_results)\n        \n        return EstimationResult(\n            project_id=project_id,\n            total_cost=total_cost,\n            cost_breakdown=cost_breakdown,\n            timeline_months=timeline_months,\n            confidence_score=confidence_score,\n            risk_assessment=risk_assessment,\n            agent_bids=agent_bids,\n            market_factors=market_factors,\n            cost_per_sqft=total_cost / project_spec.total_area if project_spec.total_area > 0 else 0,\n            negotiation_rounds=negotiation_results[\"rounds_completed\"],\n            convergence_score=convergence_score,\n            created_timestamp=datetime.now().isoformat()\n        )\n        \n    def _generate_cost_breakdown(self, project_spec: ProjectSpec, total_cost: float) -> Dict[str, float]:\n        \"\"\"Generate detailed cost breakdown by category\"\"\"\n        # Get project type handler for detailed breakdown\n        project_type_handler = get_project_type(project_spec.project_type)\n        \n        # Base breakdown percentages by project type\n        if project_spec.project_type == \"residential\":\n            breakdown_pcts = {\n                \"labor\": 0.35,\n                \"materials\": 0.30,\n                \"equipment\": 0.08,\n                \"permits_fees\": 0.05,\n                \"overhead\": 0.12,\n                \"profit\": 0.10\n            }\n        elif project_spec.project_type == \"commercial\":\n            breakdown_pcts = {\n                \"labor\": 0.40,\n                \"materials\": 0.28,\n                \"equipment\": 0.12,\n                \"permits_fees\": 0.06,\n                \"overhead\": 0.09,\n                \"profit\": 0.05\n            }\n        elif project_spec.project_type == \"transportation\":\n            breakdown_pcts = {\n                \"labor\": 0.25,\n                \"materials\": 0.45,\n                \"equipment\": 0.18,\n                \"permits_fees\": 0.04,\n                \"overhead\": 0.06,\n                \"profit\": 0.02\n            }\n        else:  # water_infrastructure\n            breakdown_pcts = {\n                \"labor\": 0.30,\n                \"materials\": 0.35,\n                \"equipment\": 0.20,\n                \"permits_fees\": 0.08,\n                \"overhead\": 0.05,\n                \"profit\": 0.02\n            }\n            \n        # Calculate actual costs\n        cost_breakdown = {}\n        for category, percentage in breakdown_pcts.items():\n            cost_breakdown[category] = total_cost * percentage\n            \n        return cost_breakdown\n        \n    def _calculate_risk_assessment(self, \n                                 project_spec: ProjectSpec,\n                                 market_conditions: MarketConditions,\n                                 agent_bids: Dict[str, Bid]) -> Dict[str, float]:\n        \"\"\"Calculate comprehensive risk assessment\"\"\"\n        \n        risk_assessment = {\n            \"scope_creep_risk\": 0.15,  # Base 15% risk\n            \"weather_risk\": market_conditions.weather_risk,\n            \"material_escalation_risk\": market_conditions.material_inflation + 0.05,\n            \"labor_shortage_risk\": max(0, 1.0 - market_conditions.labor_availability) * 0.8,\n            \"supply_chain_risk\": (1.0 - market_conditions.supply_chain_stability) * 0.6,\n            \"regulatory_risk\": sum(project_spec.risk_factors.values()) / len(project_spec.risk_factors) * 0.3\n        }\n        \n        # Adjust based on project complexity\n        complexity_multiplier = 1.0 + project_spec.complexity_score * 0.5\n        for risk_type in risk_assessment:\n            risk_assessment[risk_type] *= complexity_multiplier\n            risk_assessment[risk_type] = min(risk_assessment[risk_type], 0.8)  # Cap at 80%\n            \n        # Factor in agent confidence\n        if agent_bids:\n            avg_confidence = np.mean([bid.confidence for bid in agent_bids.values()])\n            confidence_adjustment = 1.0 - avg_confidence * 0.3\n            for risk_type in risk_assessment:\n                risk_assessment[risk_type] *= confidence_adjustment\n                \n        return risk_assessment\n        \n    def _calculate_convergence_score(self, negotiation_results: Dict[str, Any]) -> float:\n        \"\"\"Calculate how well agents converged during negotiation\"\"\"\n        history = negotiation_results[\"negotiation_history\"]\n        if not history:\n            return 0.0\n            \n        final_round = history[-1]\n        if \"agent_bids\" not in final_round or len(final_round[\"agent_bids\"]) < 2:\n            return 0.0\n            \n        costs = [bid.cost_estimate for bid in final_round[\"agent_bids\"].values()]\n        mean_cost = np.mean(costs)\n        std_cost = np.std(costs)\n        \n        # Higher convergence score for lower relative standard deviation\n        if mean_cost > 0:\n            relative_std = std_cost / mean_cost\n            convergence_score = max(0.0, 1.0 - relative_std * 5)  # Scale factor of 5\n        else:\n            convergence_score = 0.0\n            \n        return min(convergence_score, 1.0)\n        \n    def _fallback_cost_estimate(self, project_spec: ProjectSpec) -> float:\n        \"\"\"Fallback cost estimation when agents fail\"\"\"\n        project_type_handler = get_project_type(project_spec.project_type)\n        cost_per_sqft = project_type_handler.get_cost_per_sqft_estimate(asdict(project_spec))\n        return project_spec.total_area * cost_per_sqft\n        \n    def _update_agent_memory(self, estimation_result: EstimationResult):\n        \"\"\"Update agent memory with project results for learning\"\"\"\n        project_data = {\n            \"project_id\": estimation_result.project_id,\n            \"project_type\": self.env.current_project.project_type,\n            \"location\": self.env.current_project.location,\n            \"total_area\": self.env.current_project.total_area,\n            \"duration_months\": estimation_result.timeline_months,\n            \"complexity_score\": self.env.current_project.complexity_score,\n            \"actual_cost\": estimation_result.total_cost,\n            \"cost_per_sqft\": estimation_result.cost_per_sqft,\n            \"confidence_score\": estimation_result.confidence_score,\n            \"similarity\": 1.0  # Will be calculated during similarity search\n        }\n        \n        # Add to each agent's memory\n        for agent_model in self.agent_models.values():\n            agent_model.add_to_memory(project_data)\n            \n    def estimate_multiple_scenarios(self, \n                                  base_project: ProjectSpec,\n                                  scenario_variations: List[Dict[str, Any]]) -> List[EstimationResult]:\n        \"\"\"Run estimation for multiple scenario variations\"\"\"\n        results = []\n        \n        for i, variation in enumerate(scenario_variations):\n            # Create modified project spec\n            modified_project = ProjectSpec(**{**asdict(base_project), **variation})\n            \n            # Run estimation\n            result = self.estimate_project_cost(modified_project)\n            results.append(result)\n            \n        return results\n        \n    def get_estimation_insights(self, estimation_result: EstimationResult) -> Dict[str, Any]:\n        \"\"\"Generate insights and recommendations from estimation result\"\"\"\n        insights = {\n            \"cost_drivers\": [],\n            \"risk_mitigation\": [],\n            \"optimization_opportunities\": [],\n            \"confidence_factors\": []\n        }\n        \n        # Identify top cost drivers\n        sorted_breakdown = sorted(estimation_result.cost_breakdown.items(), \n                                key=lambda x: x[1], reverse=True)\n        insights[\"cost_drivers\"] = [\n            f\"{category}: ${amount:,.0f} ({amount/estimation_result.total_cost*100:.1f}%)\"\n            for category, amount in sorted_breakdown[:3]\n        ]\n        \n        # Risk mitigation suggestions\n        high_risks = {k: v for k, v in estimation_result.risk_assessment.items() if v > 0.3}\n        for risk_type, risk_level in high_risks.items():\n            if risk_type == \"scope_creep_risk\":\n                insights[\"risk_mitigation\"].append(\n                    \"Implement detailed change order management process\"\n                )\n            elif risk_type == \"weather_risk\":\n                insights[\"risk_mitigation\"].append(\n                    \"Consider weather protection measures and schedule buffers\"\n                )\n            elif risk_type == \"material_escalation_risk\":\n                insights[\"risk_mitigation\"].append(\n                    \"Lock in material prices early or include escalation clauses\"\n                )\n                \n        # Optimization opportunities\n        if estimation_result.cost_per_sqft > 200:  # High cost per sq ft\n            insights[\"optimization_opportunities\"].append(\n                \"Consider value engineering to reduce cost per square foot\"\n            )\n            \n        if estimation_result.timeline_months > 24:  # Long timeline\n            insights[\"optimization_opportunities\"].append(\n                \"Explore fast-track construction methods to reduce timeline\"\n            )\n            \n        # Confidence factors\n        if estimation_result.confidence_score > 0.8:\n            insights[\"confidence_factors\"].append(\"High agent consensus on estimate\")\n        elif estimation_result.confidence_score < 0.6:\n            insights[\"confidence_factors\"].append(\"Low confidence - consider additional analysis\")\n            \n        if estimation_result.convergence_score > 0.8:\n            insights[\"confidence_factors\"].append(\"Strong negotiation convergence\")\n            \n        return insights\n        \n    def export_estimation_report(self, estimation_result: EstimationResult, \n                               output_path: str, format: str = \"json\"):\n        \"\"\"Export detailed estimation report\"\"\"\n        report_data = {\n            \"estimation_result\": asdict(estimation_result),\n            \"project_details\": asdict(self.env.current_project) if self.env.current_project else {},\n            \"market_conditions\": asdict(self.env.current_market) if self.env.current_market else {},\n            \"insights\": self.get_estimation_insights(estimation_result),\n            \"generated_timestamp\": datetime.now().isoformat()\n        }\n        \n        if format.lower() == \"json\":\n            with open(output_path, 'w') as f:\n                json.dump(report_data, f, indent=2, default=str)\n        else:\n            raise ValueError(f\"Unsupported export format: {format}\")\n            \n    def get_historical_performance(self) -> Dict[str, Any]:\n        \"\"\"Get performance metrics from historical estimations\"\"\"\n        if not self.estimation_history:\n            return {\"message\": \"No historical data available\"}\n            \n        # Calculate performance metrics\n        confidence_scores = [r.confidence_score for r in self.estimation_history]\n        convergence_scores = [r.convergence_score for r in self.estimation_history]\n        negotiation_rounds = [r.negotiation_rounds for r in self.estimation_history]\n        \n        return {\n            \"total_estimations\": len(self.estimation_history),\n            \"average_confidence\": np.mean(confidence_scores),\n            \"average_convergence\": np.mean(convergence_scores),\n            \"average_negotiation_rounds\": np.mean(negotiation_rounds),\n            \"confidence_trend\": confidence_scores[-10:],  # Last 10 estimations\n            \"project_types_analyzed\": list(set([r.project_id.split('_')[0] for r in self.estimation_history[-20:]]))\n        }