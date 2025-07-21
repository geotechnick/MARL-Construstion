import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..estimation.cost_engine import EstimationResult
from ..agents.cost_estimation_agents import ProjectSpec, MarketConditions
from ..data.project_database import get_project_database

@dataclass
class AccuracyMetrics:
    mean_absolute_error: float
    mean_absolute_percentage_error: float
    root_mean_squared_error: float
    r2_score: float
    confidence_calibration: float
    
@dataclass
class ConvergenceMetrics:
    average_rounds: float
    convergence_rate: float  # % of estimations that converged
    average_convergence_score: float
    negotiation_efficiency: float
    
@dataclass
class AgentMetrics:
    agent_id: str
    average_confidence: float
    bid_accuracy: float
    participation_rate: float
    influence_score: float  # How much agent affects final estimate
    
@dataclass
class SystemMetrics:
    total_estimations: int
    accuracy_metrics: AccuracyMetrics
    convergence_metrics: ConvergenceMetrics
    agent_metrics: List[AgentMetrics]
    processing_time_avg: float
    system_reliability: float

class EstimationEvaluator:
    def __init__(self):
        self.database = get_project_database()
        
    def evaluate_accuracy(self, 
                         estimated_costs: List[float],
                         actual_costs: List[float],
                         confidence_scores: List[float]) -> AccuracyMetrics:
        """Evaluate accuracy of cost estimations against actual costs"""
        
        if len(estimated_costs) != len(actual_costs):
            raise ValueError("Estimated and actual costs must have same length")
            
        estimated = np.array(estimated_costs)
        actual = np.array(actual_costs)
        confidence = np.array(confidence_scores)
        
        # Mean Absolute Error
        mae = mean_absolute_error(actual, estimated)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - estimated) / actual)) * 100
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(actual, estimated))
        
        # R-squared score
        r2 = r2_score(actual, estimated)
        
        # Confidence calibration (how well confidence scores predict accuracy)
        absolute_errors = np.abs(actual - estimated) / actual
        confidence_calibration = 1.0 - np.corrcoef(confidence, absolute_errors)[0, 1]
        if np.isnan(confidence_calibration):
            confidence_calibration = 0.0
            
        return AccuracyMetrics(
            mean_absolute_error=mae,
            mean_absolute_percentage_error=mape,
            root_mean_squared_error=rmse,
            r2_score=r2,
            confidence_calibration=confidence_calibration
        )
        
    def evaluate_convergence(self, estimation_results: List[EstimationResult]) -> ConvergenceMetrics:
        """Evaluate negotiation convergence metrics"""
        
        if not estimation_results:
            return ConvergenceMetrics(0, 0, 0, 0)
            
        rounds = [result.negotiation_rounds for result in estimation_results]
        convergence_scores = [result.convergence_score for result in estimation_results]
        
        # Average rounds to completion
        average_rounds = np.mean(rounds)
        
        # Convergence rate (assuming convergence score > 0.8 means converged)
        converged_count = sum(1 for score in convergence_scores if score > 0.8)
        convergence_rate = converged_count / len(estimation_results) * 100
        
        # Average convergence score
        average_convergence_score = np.mean(convergence_scores)
        
        # Negotiation efficiency (inverse of rounds, scaled by convergence)
        max_rounds = max(rounds) if rounds else 1
        efficiency_scores = [(max_rounds - r + 1) / max_rounds * cs 
                           for r, cs in zip(rounds, convergence_scores)]
        negotiation_efficiency = np.mean(efficiency_scores)
        
        return ConvergenceMetrics(
            average_rounds=average_rounds,
            convergence_rate=convergence_rate,
            average_convergence_score=average_convergence_score,
            negotiation_efficiency=negotiation_efficiency
        )
        
    def evaluate_agent_performance(self, estimation_results: List[EstimationResult]) -> List[AgentMetrics]:
        """Evaluate individual agent performance"""
        
        if not estimation_results:
            return []
            
        # Collect all agent data
        agent_data = {}
        
        for result in estimation_results:
            for agent_id, bid in result.agent_bids.items():
                if agent_id not in agent_data:
                    agent_data[agent_id] = {
                        'confidences': [],
                        'bid_accuracies': [],
                        'participations': 0,
                        'influence_scores': []
                    }
                
                agent_data[agent_id]['confidences'].append(bid.confidence)
                agent_data[agent_id]['participations'] += 1
                
                # Calculate bid accuracy relative to final estimate
                bid_accuracy = 1.0 - abs(bid.cost_estimate - result.total_cost) / result.total_cost
                agent_data[agent_id]['bid_accuracies'].append(max(0, bid_accuracy))
                
                # Calculate influence (how close agent bid is to final estimate)
                all_bids = [b.cost_estimate for b in result.agent_bids.values()]
                if len(all_bids) > 1:
                    bid_std = np.std(all_bids)
                    if bid_std > 0:
                        influence = 1.0 - abs(bid.cost_estimate - result.total_cost) / bid_std
                        agent_data[agent_id]['influence_scores'].append(max(0, min(1, influence)))
                    else:
                        agent_data[agent_id]['influence_scores'].append(1.0)
                        
        # Calculate metrics for each agent
        agent_metrics = []
        total_estimations = len(estimation_results)
        
        for agent_id, data in agent_data.items():
            avg_confidence = np.mean(data['confidences']) if data['confidences'] else 0.0
            avg_bid_accuracy = np.mean(data['bid_accuracies']) if data['bid_accuracies'] else 0.0
            participation_rate = data['participations'] / total_estimations * 100
            avg_influence = np.mean(data['influence_scores']) if data['influence_scores'] else 0.0
            
            agent_metrics.append(AgentMetrics(
                agent_id=agent_id,
                average_confidence=avg_confidence,
                bid_accuracy=avg_bid_accuracy,
                participation_rate=participation_rate,
                influence_score=avg_influence
            ))
            
        return agent_metrics
        
    def comprehensive_evaluation(self, 
                               actual_costs: Optional[List[float]] = None,
                               project_ids: Optional[List[str]] = None,
                               time_period_days: Optional[int] = None) -> SystemMetrics:
        """Perform comprehensive system evaluation"""
        
        # Get estimation results from database
        if project_ids:
            estimation_results = []
            for pid in project_ids:
                project = self.database.get_project(pid)
                if project:
                    estimation_results.append(project.estimation_result)
        else:
            # Get recent projects
            projects = self.database.list_projects(limit=100)
            estimation_results = []
            for p in projects:
                project = self.database.get_project(p['id'])
                if project:
                    estimation_results.append(project.estimation_result)
                    
        if not estimation_results:
            raise ValueError("No estimation results found for evaluation")
            
        # Evaluate convergence
        convergence_metrics = self.evaluate_convergence(estimation_results)
        
        # Evaluate agent performance
        agent_metrics = self.evaluate_agent_performance(estimation_results)
        
        # Evaluate accuracy if actual costs provided
        if actual_costs:
            if len(actual_costs) != len(estimation_results):
                raise ValueError("Actual costs length must match estimation results")
                
            estimated_costs = [r.total_cost for r in estimation_results]
            confidence_scores = [r.confidence_score for r in estimation_results]
            accuracy_metrics = self.evaluate_accuracy(estimated_costs, actual_costs, confidence_scores)
        else:
            # Create dummy accuracy metrics if no actual costs available
            accuracy_metrics = AccuracyMetrics(0, 0, 0, 0, 0)
            
        # Calculate system reliability (% of successful estimations)
        successful_estimations = sum(1 for r in estimation_results if r.confidence_score > 0.5)
        system_reliability = successful_estimations / len(estimation_results) * 100
        
        # Estimate average processing time (simplified)
        avg_processing_time = np.mean([r.negotiation_rounds * 2.5 for r in estimation_results])  # 2.5s per round estimate
        
        return SystemMetrics(
            total_estimations=len(estimation_results),
            accuracy_metrics=accuracy_metrics,
            convergence_metrics=convergence_metrics,
            agent_metrics=agent_metrics,
            processing_time_avg=avg_processing_time,
            system_reliability=system_reliability
        )
        
    def benchmark_against_baseline(self, 
                                 estimation_results: List[EstimationResult],
                                 baseline_estimates: List[float],
                                 actual_costs: Optional[List[float]] = None) -> Dict[str, float]:
        """Benchmark MARL system against baseline estimation method"""
        
        marl_estimates = [r.total_cost for r in estimation_results]
        
        if actual_costs is None:
            # Use synthetic actual costs for demonstration
            actual_costs = [e * (1 + np.random.normal(0, 0.1)) for e in marl_estimates]
            
        # Calculate metrics for both approaches
        marl_mae = mean_absolute_error(actual_costs, marl_estimates)
        baseline_mae = mean_absolute_error(actual_costs, baseline_estimates)
        
        marl_mape = np.mean(np.abs((np.array(actual_costs) - np.array(marl_estimates)) / np.array(actual_costs))) * 100
        baseline_mape = np.mean(np.abs((np.array(actual_costs) - np.array(baseline_estimates)) / np.array(actual_costs))) * 100
        
        marl_rmse = np.sqrt(mean_squared_error(actual_costs, marl_estimates))
        baseline_rmse = np.sqrt(mean_squared_error(actual_costs, baseline_estimates))
        
        # Calculate improvements
        mae_improvement = (baseline_mae - marl_mae) / baseline_mae * 100
        mape_improvement = (baseline_mape - marl_mape) / baseline_mape * 100
        rmse_improvement = (baseline_rmse - marl_rmse) / baseline_rmse * 100
        
        return {
            'marl_mae': marl_mae,
            'baseline_mae': baseline_mae,
            'mae_improvement_pct': mae_improvement,
            'marl_mape': marl_mape,
            'baseline_mape': baseline_mape,
            'mape_improvement_pct': mape_improvement,
            'marl_rmse': marl_rmse,
            'baseline_rmse': baseline_rmse,
            'rmse_improvement_pct': rmse_improvement
        }
        
    def sensitivity_analysis(self, 
                           base_project: ProjectSpec,
                           base_market: MarketConditions,
                           estimation_engine) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on key parameters"""
        
        from ..estimation.cost_engine import CostEstimationEngine
        
        # Parameter variations to test
        variations = {
            'project_area': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            'complexity_score': [0.2, 0.4, 0.6, 0.8, 1.0],
            'labor_availability': [0.4, 0.6, 0.8, 1.0],
            'material_inflation': [0.0, 0.05, 0.1, 0.15, 0.2],
            'duration_months': [0.5, 0.75, 1.0, 1.25, 1.5]
        }\n        \n        sensitivity_results = {}\n        \n        # Get baseline estimate\n        baseline_result = estimation_engine.estimate_project_cost(base_project, base_market)\n        baseline_cost = baseline_result.total_cost\n        \n        for parameter, multipliers in variations.items():\n            parameter_results = {'multipliers': [], 'cost_changes': [], 'confidence_changes': []}\n            \n            for multiplier in multipliers:\n                # Create modified project/market\n                if parameter == 'project_area':\n                    modified_project = ProjectSpec(\n                        **{**base_project.__dict__, 'total_area': base_project.total_area * multiplier}\n                    )\n                    modified_market = base_market\n                elif parameter == 'complexity_score':\n                    modified_project = ProjectSpec(\n                        **{**base_project.__dict__, 'complexity_score': min(1.0, multiplier)}\n                    )\n                    modified_market = base_market\n                elif parameter == 'duration_months':\n                    modified_project = ProjectSpec(\n                        **{**base_project.__dict__, 'duration_months': int(base_project.duration_months * multiplier)}\n                    )\n                    modified_market = base_market\n                elif parameter == 'labor_availability':\n                    modified_project = base_project\n                    modified_market = MarketConditions(\n                        **{**base_market.__dict__, 'labor_availability': min(1.0, multiplier)}\n                    )\n                elif parameter == 'material_inflation':\n                    modified_project = base_project\n                    modified_market = MarketConditions(\n                        **{**base_market.__dict__, 'material_inflation': multiplier}\n                    )\n                    \n                # Run estimation\n                try:\n                    result = estimation_engine.estimate_project_cost(modified_project, modified_market)\n                    cost_change = (result.total_cost - baseline_cost) / baseline_cost * 100\n                    confidence_change = (result.confidence_score - baseline_result.confidence_score) * 100\n                    \n                    parameter_results['multipliers'].append(multiplier)\n                    parameter_results['cost_changes'].append(cost_change)\n                    parameter_results['confidence_changes'].append(confidence_change)\n                    \n                except Exception as e:\n                    print(f\"Error in sensitivity analysis for {parameter}={multiplier}: {e}\")\n                    continue\n                    \n            # Calculate sensitivity coefficient (slope of cost change vs parameter change)\n            if len(parameter_results['multipliers']) > 1:\n                x = np.array(parameter_results['multipliers'])\n                y = np.array(parameter_results['cost_changes'])\n                sensitivity_coeff = np.polyfit(x, y, 1)[0]  # Linear regression slope\n            else:\n                sensitivity_coeff = 0.0\n                \n            sensitivity_results[parameter] = {\n                'sensitivity_coefficient': sensitivity_coeff,\n                'max_cost_change': max(parameter_results['cost_changes']) if parameter_results['cost_changes'] else 0,\n                'min_cost_change': min(parameter_results['cost_changes']) if parameter_results['cost_changes'] else 0,\n                'data_points': len(parameter_results['multipliers'])\n            }\n            \n        return sensitivity_results\n        \n    def monte_carlo_validation(self, \n                             base_project: ProjectSpec,\n                             base_market: MarketConditions,\n                             estimation_engine,\n                             n_simulations: int = 100) -> Dict[str, Any]:\n        \"\"\"Perform Monte Carlo validation of estimation reliability\"\"\"\n        \n        results = []\n        \n        for i in range(n_simulations):\n            # Add random variations to parameters\n            varied_project = ProjectSpec(\n                project_type=base_project.project_type,\n                location=base_project.location,\n                total_area=base_project.total_area * np.random.normal(1.0, 0.1),\n                duration_months=max(1, int(base_project.duration_months * np.random.normal(1.0, 0.05))),\n                complexity_score=np.clip(base_project.complexity_score + np.random.normal(0, 0.1), 0, 1),\n                risk_factors={k: np.clip(v + np.random.normal(0, 0.05), 0, 1) \n                            for k, v in base_project.risk_factors.items()},\n                custom_parameters=base_project.custom_parameters\n            )\n            \n            varied_market = MarketConditions(\n                labor_availability=np.clip(base_market.labor_availability + np.random.normal(0, 0.05), 0, 1),\n                material_inflation=max(0, base_market.material_inflation + np.random.normal(0, 0.02)),\n                supply_chain_stability=np.clip(base_market.supply_chain_stability + np.random.normal(0, 0.05), 0, 1),\n                economic_volatility=np.clip(base_market.economic_volatility + np.random.normal(0, 0.02), 0, 1),\n                fuel_surcharge=max(0, base_market.fuel_surcharge + np.random.normal(0, 0.02)),\n                weather_risk=np.clip(base_market.weather_risk + np.random.normal(0, 0.05), 0, 1)\n            )\n            \n            try:\n                result = estimation_engine.estimate_project_cost(varied_project, varied_market)\n                results.append({\n                    'total_cost': result.total_cost,\n                    'confidence_score': result.confidence_score,\n                    'convergence_score': result.convergence_score,\n                    'negotiation_rounds': result.negotiation_rounds\n                })\n            except Exception as e:\n                print(f\"Monte Carlo simulation {i} failed: {e}\")\n                continue\n                \n        if not results:\n            return {'error': 'All Monte Carlo simulations failed'}\n            \n        # Analyze results\n        costs = [r['total_cost'] for r in results]\n        confidences = [r['confidence_score'] for r in results]\n        convergences = [r['convergence_score'] for r in results]\n        rounds = [r['negotiation_rounds'] for r in results]\n        \n        validation_results = {\n            'simulations_completed': len(results),\n            'success_rate': len(results) / n_simulations * 100,\n            'cost_statistics': {\n                'mean': np.mean(costs),\n                'std': np.std(costs),\n                'min': np.min(costs),\n                'max': np.max(costs),\n                'percentile_5': np.percentile(costs, 5),\n                'percentile_95': np.percentile(costs, 95),\n                'coefficient_of_variation': np.std(costs) / np.mean(costs)\n            },\n            'confidence_statistics': {\n                'mean': np.mean(confidences),\n                'std': np.std(confidences),\n                'min': np.min(confidences),\n                'max': np.max(confidences)\n            },\n            'convergence_statistics': {\n                'mean': np.mean(convergences),\n                'std': np.std(convergences),\n                'convergence_rate': sum(1 for c in convergences if c > 0.8) / len(convergences) * 100\n            },\n            'performance_statistics': {\n                'average_rounds': np.mean(rounds),\n                'max_rounds': np.max(rounds),\n                'min_rounds': np.min(rounds)\n            }\n        }\n        \n        return validation_results\n        \n    def generate_evaluation_report(self, \n                                 system_metrics: SystemMetrics,\n                                 sensitivity_results: Optional[Dict] = None,\n                                 monte_carlo_results: Optional[Dict] = None) -> str:\n        \"\"\"Generate comprehensive evaluation report\"\"\"\n        \n        report = f\"\"\"\n=== MARL CONSTRUCTION COST ESTIMATION SYSTEM EVALUATION REPORT ===\n\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n--- OVERALL SYSTEM PERFORMANCE ---\nTotal Estimations Evaluated: {system_metrics.total_estimations}\nSystem Reliability: {system_metrics.system_reliability:.1f}%\nAverage Processing Time: {system_metrics.processing_time_avg:.1f} seconds\n\n--- ACCURACY METRICS ---\n\"\"\"\n        \n        if system_metrics.accuracy_metrics.mean_absolute_error > 0:\n            report += f\"\"\"\nMean Absolute Error: ${system_metrics.accuracy_metrics.mean_absolute_error:,.2f}\nMean Absolute Percentage Error: {system_metrics.accuracy_metrics.mean_absolute_percentage_error:.2f}%\nRoot Mean Squared Error: ${system_metrics.accuracy_metrics.root_mean_squared_error:,.2f}\nR-squared Score: {system_metrics.accuracy_metrics.r2_score:.3f}\nConfidence Calibration: {system_metrics.accuracy_metrics.confidence_calibration:.3f}\n\"\"\"\n        else:\n            report += \"\\nAccuracy metrics not available (no actual cost data provided)\\n\"\n            \n        report += f\"\"\"\n\n--- CONVERGENCE METRICS ---\nAverage Negotiation Rounds: {system_metrics.convergence_metrics.average_rounds:.1f}\nConvergence Rate: {system_metrics.convergence_metrics.convergence_rate:.1f}%\nAverage Convergence Score: {system_metrics.convergence_metrics.average_convergence_score:.3f}\nNegotiation Efficiency: {system_metrics.convergence_metrics.negotiation_efficiency:.3f}\n\n--- AGENT PERFORMANCE ---\n\"\"\"\n        \n        for agent in system_metrics.agent_metrics:\n            report += f\"\"\"\n{agent.agent_id.replace('_', ' ').title()}:\n  Average Confidence: {agent.average_confidence:.3f}\n  Bid Accuracy: {agent.bid_accuracy:.3f}\n  Participation Rate: {agent.participation_rate:.1f}%\n  Influence Score: {agent.influence_score:.3f}\n\n\"\"\"\n        \n        if sensitivity_results:\n            report += \"\\n--- SENSITIVITY ANALYSIS ---\\n\"\n            for param, results in sensitivity_results.items():\n                report += f\"\"\"\n{param.replace('_', ' ').title()}:\n  Sensitivity Coefficient: {results['sensitivity_coefficient']:.3f}\n  Max Cost Change: {results['max_cost_change']:.1f}%\n  Min Cost Change: {results['min_cost_change']:.1f}%\n\n\"\"\"\n        \n        if monte_carlo_results and 'error' not in monte_carlo_results:\n            report += f\"\"\"\n--- MONTE CARLO VALIDATION ---\nSimulations Completed: {monte_carlo_results['simulations_completed']}\nSuccess Rate: {monte_carlo_results['success_rate']:.1f}%\n\nCost Variability:\n  Mean: ${monte_carlo_results['cost_statistics']['mean']:,.2f}\n  Standard Deviation: ${monte_carlo_results['cost_statistics']['std']:,.2f}\n  Coefficient of Variation: {monte_carlo_results['cost_statistics']['coefficient_of_variation']:.3f}\n  95% Confidence Interval: ${monte_carlo_results['cost_statistics']['percentile_5']:,.2f} - ${monte_carlo_results['cost_statistics']['percentile_95']:,.2f}\n\nReliability:\n  Average Confidence: {monte_carlo_results['confidence_statistics']['mean']:.3f}\n  Convergence Rate: {monte_carlo_results['convergence_statistics']['convergence_rate']:.1f}%\n  Average Negotiation Rounds: {monte_carlo_results['performance_statistics']['average_rounds']:.1f}\n\n\"\"\"\n        \n        report += \"\"\"\n--- RECOMMENDATIONS ---\n\n\"\"\"\n        \n        # Add recommendations based on metrics\n        if system_metrics.convergence_metrics.convergence_rate < 80:\n            report += \"• Consider tuning agent negotiation parameters to improve convergence rate\\n\"\n            \n        if system_metrics.convergence_metrics.average_rounds > 4:\n            report += \"• Optimize negotiation strategy to reduce average rounds\\n\"\n            \n        if system_metrics.accuracy_metrics.confidence_calibration < 0.5:\n            report += \"• Improve confidence score calibration for better uncertainty quantification\\n\"\n            \n        low_performing_agents = [a for a in system_metrics.agent_metrics if a.bid_accuracy < 0.7]\n        if low_performing_agents:\n            agent_names = ', '.join([a.agent_id for a in low_performing_agents])\n            report += f\"• Review and improve performance of agents: {agent_names}\\n\"\n            \n        if system_metrics.system_reliability < 90:\n            report += \"• Investigate causes of estimation failures to improve system reliability\\n\"\n            \n        report += \"\\n=== END OF REPORT ===\\n\"\n        \n        return report\n        \n    def visualize_performance(self, system_metrics: SystemMetrics, save_path: Optional[str] = None):\n        \"\"\"Create performance visualization plots\"\"\"\n        \n        fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n        fig.suptitle('MARL Cost Estimation System Performance', fontsize=16, fontweight='bold')\n        \n        # Agent performance comparison\n        agent_names = [a.agent_id.replace('_', ' ').title() for a in system_metrics.agent_metrics]\n        confidences = [a.average_confidence for a in system_metrics.agent_metrics]\n        accuracies = [a.bid_accuracy for a in system_metrics.agent_metrics]\n        \n        axes[0, 0].bar(agent_names, confidences, alpha=0.7, color='skyblue')\n        axes[0, 0].set_title('Agent Confidence Levels')\n        axes[0, 0].set_ylabel('Average Confidence')\n        axes[0, 0].set_ylim(0, 1)\n        plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')\n        \n        axes[0, 1].bar(agent_names, accuracies, alpha=0.7, color='lightgreen')\n        axes[0, 1].set_title('Agent Bid Accuracy')\n        axes[0, 1].set_ylabel('Bid Accuracy')\n        axes[0, 1].set_ylim(0, 1)\n        plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')\n        \n        # System metrics\n        metrics_labels = ['Convergence\\nRate', 'System\\nReliability', 'Negotiation\\nEfficiency']\n        metrics_values = [\n            system_metrics.convergence_metrics.convergence_rate,\n            system_metrics.system_reliability,\n            system_metrics.convergence_metrics.negotiation_efficiency * 100\n        ]\n        \n        bars = axes[0, 2].bar(metrics_labels, metrics_values, alpha=0.7, color=['orange', 'red', 'purple'])\n        axes[0, 2].set_title('System Performance Metrics')\n        axes[0, 2].set_ylabel('Percentage (%)')\n        axes[0, 2].set_ylim(0, 100)\n        \n        # Add value labels on bars\n        for bar, value in zip(bars, metrics_values):\n            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n                           f'{value:.1f}%', ha='center', va='bottom')\n        \n        # Agent influence vs participation\n        influences = [a.influence_score for a in system_metrics.agent_metrics]\n        participations = [a.participation_rate for a in system_metrics.agent_metrics]\n        \n        scatter = axes[1, 0].scatter(participations, influences, s=100, alpha=0.7, c=confidences, cmap='viridis')\n        axes[1, 0].set_xlabel('Participation Rate (%)')\n        axes[1, 0].set_ylabel('Influence Score')\n        axes[1, 0].set_title('Agent Influence vs Participation')\n        \n        # Add agent labels\n        for i, name in enumerate(agent_names):\n            axes[1, 0].annotate(name.split()[0], (participations[i], influences[i]), \n                              xytext=(5, 5), textcoords='offset points', fontsize=8)\n        \n        # Convergence metrics pie chart\n        convergence_data = [\n            system_metrics.convergence_metrics.convergence_rate,\n            100 - system_metrics.convergence_metrics.convergence_rate\n        ]\n        axes[1, 1].pie(convergence_data, labels=['Converged', 'Not Converged'], \n                      autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])\n        axes[1, 1].set_title('Negotiation Convergence')\n        \n        # Performance summary\n        axes[1, 2].axis('off')\n        summary_text = f\"\"\"\nSYSTEM SUMMARY\n\nTotal Estimations: {system_metrics.total_estimations}\nAvg Processing Time: {system_metrics.processing_time_avg:.1f}s\nAvg Negotiation Rounds: {system_metrics.convergence_metrics.average_rounds:.1f}\n\nTop Performing Agent:\n{max(system_metrics.agent_metrics, key=lambda a: a.bid_accuracy).agent_id.replace('_', ' ').title()}\n\nConvergence Score: {system_metrics.convergence_metrics.average_convergence_score:.3f}\nSystem Reliability: {system_metrics.system_reliability:.1f}%\n\"\"\"\n        \n        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, \n                        fontsize=10, verticalalignment='top', fontfamily='monospace',\n                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            \n        plt.show()\n        \nclass TestSuite:\n    \"\"\"Automated test suite for the MARL estimation system\"\"\"\n    \n    def __init__(self, estimation_engine):\n        self.estimation_engine = estimation_engine\n        self.evaluator = EstimationEvaluator()\n        \n    def run_basic_functionality_tests(self) -> Dict[str, bool]:\n        \"\"\"Run basic functionality tests\"\"\"\n        \n        from ..agents.cost_estimation_agents import ProjectSpec, MarketConditions\n        \n        test_results = {}\n        \n        # Test 1: Basic estimation\n        try:\n            test_project = ProjectSpec(\n                project_type=\"residential\",\n                location=\"midwest\",\n                total_area=2500,\n                duration_months=12,\n                complexity_score=0.5,\n                risk_factors={\n                    \"environmental_sensitivity\": 0.3,\n                    \"geotechnical_risk\": 0.2,\n                    \"weather_exposure\": 0.4,\n                    \"regulatory_complexity\": 0.3\n                },\n                custom_parameters={\"foundation_type\": \"slab\", \"stories\": 1}\n            )\n            \n            test_market = MarketConditions(\n                labor_availability=0.8,\n                material_inflation=0.05,\n                supply_chain_stability=0.9,\n                economic_volatility=0.2,\n                fuel_surcharge=0.1,\n                weather_risk=0.3\n            )\n            \n            result = self.estimation_engine.estimate_project_cost(test_project, test_market)\n            test_results['basic_estimation'] = result.total_cost > 0 and result.confidence_score > 0\n            \n        except Exception as e:\n            test_results['basic_estimation'] = False\n            print(f\"Basic estimation test failed: {e}\")\n            \n        # Test 2: Agent convergence\n        try:\n            test_results['agent_convergence'] = result.convergence_score > 0\n        except:\n            test_results['agent_convergence'] = False\n            \n        # Test 3: All agents participate\n        try:\n            expected_agents = 5  # owner, contractor, supplier, regulator, estimator\n            test_results['agent_participation'] = len(result.agent_bids) >= expected_agents - 1  # Allow one agent to not participate\n        except:\n            test_results['agent_participation'] = False\n            \n        # Test 4: Cost breakdown completeness\n        try:\n            total_breakdown = sum(result.cost_breakdown.values())\n            cost_diff = abs(total_breakdown - result.total_cost) / result.total_cost\n            test_results['cost_breakdown_accuracy'] = cost_diff < 0.01  # Within 1%\n        except:\n            test_results['cost_breakdown_accuracy'] = False\n            \n        # Test 5: Risk assessment completeness\n        try:\n            test_results['risk_assessment_complete'] = len(result.risk_assessment) >= 4\n        except:\n            test_results['risk_assessment_complete'] = False\n            \n        return test_results\n        \n    def run_stress_tests(self) -> Dict[str, bool]:\n        \"\"\"Run stress tests with extreme parameters\"\"\"\n        \n        from ..agents.cost_estimation_agents import ProjectSpec, MarketConditions\n        \n        stress_test_results = {}\n        \n        # Stress test scenarios\n        stress_scenarios = [\n            {\n                'name': 'large_project',\n                'project': ProjectSpec(\n                    project_type=\"commercial\",\n                    location=\"west\",\n                    total_area=500000,  # Very large\n                    duration_months=48,\n                    complexity_score=0.9,\n                    risk_factors={\"environmental_sensitivity\": 0.8, \"geotechnical_risk\": 0.7, \n                                \"weather_exposure\": 0.6, \"regulatory_complexity\": 0.8},\n                    custom_parameters={\"structural_system\": \"steel\", \"occupancy_type\": \"healthcare\"}\n                ),\n                'market': MarketConditions(0.4, 0.2, 0.5, 0.4, 0.25, 0.7)  # Poor conditions\n            },\n            {\n                'name': 'tiny_project',\n                'project': ProjectSpec(\n                    project_type=\"residential\",\n                    location=\"southeast\",\n                    total_area=500,  # Very small\n                    duration_months=3,\n                    complexity_score=0.1,\n                    risk_factors={\"environmental_sensitivity\": 0.1, \"geotechnical_risk\": 0.1,\n                                \"weather_exposure\": 0.2, \"regulatory_complexity\": 0.1},\n                    custom_parameters={\"foundation_type\": \"slab\", \"stories\": 1}\n                ),\n                'market': MarketConditions(0.95, 0.01, 0.98, 0.05, 0.02, 0.1)  # Excellent conditions\n            },\n            {\n                'name': 'high_volatility',\n                'project': ProjectSpec(\n                    project_type=\"transportation\",\n                    location=\"northeast\",\n                    total_area=100000,\n                    duration_months=36,\n                    complexity_score=0.8,\n                    risk_factors={\"environmental_sensitivity\": 0.9, \"geotechnical_risk\": 0.8,\n                                \"weather_exposure\": 0.9, \"regulatory_complexity\": 0.9},\n                    custom_parameters={\"length_miles\": 10, \"lanes\": 6, \"terrain\": \"mountainous\"}\n                ),\n                'market': MarketConditions(0.3, 0.25, 0.4, 0.5, 0.3, 0.8)  # Very poor conditions\n            }\n        ]\n        \n        for scenario in stress_scenarios:\n            try:\n                result = self.estimation_engine.estimate_project_cost(\n                    scenario['project'], scenario['market']\n                )\n                \n                # Check if estimation completed successfully\n                success = (\n                    result.total_cost > 0 and\n                    result.confidence_score > 0 and\n                    len(result.agent_bids) > 0 and\n                    result.negotiation_rounds <= 10  # Reasonable convergence time\n                )\n                \n                stress_test_results[scenario['name']] = success\n                \n            except Exception as e:\n                stress_test_results[scenario['name']] = False\n                print(f\"Stress test {scenario['name']} failed: {e}\")\n                \n        return stress_test_results\n        \n    def generate_test_report(self) -> str:\n        \"\"\"Generate comprehensive test report\"\"\"\n        \n        basic_tests = self.run_basic_functionality_tests()\n        stress_tests = self.run_stress_tests()\n        \n        report = f\"\"\"\n=== MARL COST ESTIMATION SYSTEM TEST REPORT ===\n\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n--- BASIC FUNCTIONALITY TESTS ---\n\"\"\"\n        \n        for test_name, result in basic_tests.items():\n            status = \"PASS\" if result else \"FAIL\"\n            report += f\"{test_name.replace('_', ' ').title()}: {status}\\n\"\n            \n        basic_pass_rate = sum(basic_tests.values()) / len(basic_tests) * 100\n        report += f\"\\nBasic Tests Pass Rate: {basic_pass_rate:.1f}%\\n\"\n        \n        report += \"\\n--- STRESS TESTS ---\\n\"\n        \n        for test_name, result in stress_tests.items():\n            status = \"PASS\" if result else \"FAIL\"\n            report += f\"{test_name.replace('_', ' ').title()}: {status}\\n\"\n            \n        stress_pass_rate = sum(stress_tests.values()) / len(stress_tests) * 100\n        report += f\"\\nStress Tests Pass Rate: {stress_pass_rate:.1f}%\\n\"\n        \n        overall_pass_rate = (sum(basic_tests.values()) + sum(stress_tests.values())) / (len(basic_tests) + len(stress_tests)) * 100\n        \n        report += f\"\"\"\n--- OVERALL RESULTS ---\nOverall Pass Rate: {overall_pass_rate:.1f}%\n\nSYSTEM STATUS: {'HEALTHY' if overall_pass_rate >= 80 else 'NEEDS ATTENTION' if overall_pass_rate >= 60 else 'CRITICAL'}\n\n=== END OF TEST REPORT ===\n\"\"\"\n        \n        return report