#!/usr/bin/env python3
"""
Complete MARL Construction Cost Estimation Demo

This script demonstrates the full capabilities of the MARL Construction Cost Estimation system,
including project setup, agent negotiation, cost estimation, and result analysis.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marl_construction.agents.cost_estimation_agents import ProjectSpec, MarketConditions
from marl_construction.estimation.cost_engine import CostEstimationEngine, EstimationConfig
from marl_construction.projects.project_types import get_project_type, get_available_project_types
from marl_construction.data.project_database import get_project_database
from marl_construction.evaluation.metrics import EstimationEvaluator, TestSuite
import json
from datetime import datetime

def demo_residential_project():
    """Demonstrate residential project cost estimation"""
    print("=== RESIDENTIAL PROJECT ESTIMATION DEMO ===\n")
    
    # Create a residential project specification
    residential_project = ProjectSpec(
        project_type="residential",
        location="midwest",
        total_area=2800.0,  # sq ft
        duration_months=14,
        complexity_score=0.6,
        risk_factors={
            "environmental_sensitivity": 0.3,
            "geotechnical_risk": 0.2,
            "weather_exposure": 0.4,
            "regulatory_complexity": 0.25
        },
        custom_parameters={
            "foundation_type": "basement",
            "stories": 2,
            "finish_level": "standard",
            "energy_efficiency_target": "high"
        }
    )
    
    # Define market conditions
    market_conditions = MarketConditions(
        labor_availability=0.75,
        material_inflation=0.08,
        supply_chain_stability=0.85,
        economic_volatility=0.15,
        fuel_surcharge=0.12,
        weather_risk=0.3
    )
    
    # Create estimation engine
    config = EstimationConfig(
        max_negotiation_rounds=5,
        convergence_threshold=0.05,
        enable_user_overrides=True,
        include_risk_premium=True
    )
    
    engine = CostEstimationEngine(config)
    
    # Run estimation
    print("Running multi-agent cost estimation...")
    result = engine.estimate_project_cost(residential_project, market_conditions)
    
    # Display results
    print(f"\n--- ESTIMATION RESULTS ---")
    print(f"Project ID: {result.project_id}")
    print(f"Total Cost: ${result.total_cost:,.2f}")
    print(f"Cost per sq ft: ${result.cost_per_sqft:.2f}")
    print(f"Timeline: {result.timeline_months} months")
    print(f"Confidence Score: {result.confidence_score:.1%}")
    print(f"Negotiation Rounds: {result.negotiation_rounds}")
    print(f"Convergence Score: {result.convergence_score:.1%}")
    
    print(f"\n--- COST BREAKDOWN ---")
    for category, amount in result.cost_breakdown.items():
        percentage = (amount / result.total_cost) * 100
        print(f"{category.replace('_', ' ').title()}: ${amount:,.2f} ({percentage:.1f}%)")
    
    print(f"\n--- RISK ASSESSMENT ---")
    for risk_type, risk_level in result.risk_assessment.items():
        print(f"{risk_type.replace('_', ' ').title()}: {risk_level:.1%}")
    
    print(f"\n--- AGENT BIDS ---")
    for agent_id, bid in result.agent_bids.items():
        print(f"{agent_id.replace('_', ' ').title()}:")
        print(f"  Cost: ${bid.cost_estimate:,.2f}")
        print(f"  Timeline: {bid.timeline_estimate} months")
        print(f"  Confidence: {bid.confidence:.1%}")
        print(f"  Risk Premium: {bid.risk_premium:.1%}")
    
    # Get insights
    insights = engine.get_estimation_insights(result)
    print(f"\n--- INSIGHTS ---")
    print("Top Cost Drivers:")
    for driver in insights["cost_drivers"]:
        print(f"  • {driver}")
    
    if insights["risk_mitigation"]:
        print("\nRisk Mitigation Recommendations:")
        for rec in insights["risk_mitigation"]:
            print(f"  • {rec}")
    
    return result

def demo_commercial_project():
    """Demonstrate commercial project cost estimation"""
    print("\n=== COMMERCIAL PROJECT ESTIMATION DEMO ===\n")
    
    # Create a commercial project specification
    commercial_project = ProjectSpec(
        project_type="commercial",
        location="west",
        total_area=45000.0,  # sq ft
        duration_months=28,
        complexity_score=0.7,
        risk_factors={
            "environmental_sensitivity": 0.4,
            "geotechnical_risk": 0.3,
            "weather_exposure": 0.2,
            "regulatory_complexity": 0.5
        },
        custom_parameters={
            "structural_system": "steel",
            "occupancy_type": "office",
            "mep_complexity": "high"
        }
    )
    
    # Define different market conditions
    market_conditions = MarketConditions(
        labor_availability=0.65,
        material_inflation=0.12,
        supply_chain_stability=0.75,
        economic_volatility=0.25,
        fuel_surcharge=0.18,
        weather_risk=0.15
    )
    
    # Create estimation engine
    engine = CostEstimationEngine()
    
    # Run estimation
    print("Running multi-agent cost estimation...")
    result = engine.estimate_project_cost(commercial_project, market_conditions)
    
    # Display summary results
    print(f"\n--- ESTIMATION SUMMARY ---")
    print(f"Total Cost: ${result.total_cost:,.2f}")
    print(f"Cost per sq ft: ${result.cost_per_sqft:.2f}")
    print(f"Timeline: {result.timeline_months} months")
    print(f"Confidence Score: {result.confidence_score:.1%}")
    
    # Show market impact
    print(f"\n--- MARKET IMPACT FACTORS ---")
    for factor, impact in result.market_factors.items():
        print(f"{factor.replace('_', ' ').title()}: {impact:.1%} impact")
    
    return result

def demo_scenario_comparison():
    """Demonstrate scenario comparison functionality"""
    print("\n=== SCENARIO COMPARISON DEMO ===\n")
    
    # Base project
    base_project = ProjectSpec(
        project_type="transportation",
        location="northeast",
        total_area=200000.0,  # equivalent area for 5 mile road
        duration_months=36,
        complexity_score=0.5,
        risk_factors={
            "environmental_sensitivity": 0.4,
            "geotechnical_risk": 0.3,
            "weather_exposure": 0.6,
            "regulatory_complexity": 0.4
        },
        custom_parameters={
            "length_miles": 5,
            "lanes": 4,
            "terrain": "rolling",
            "bridge_count": 2
        }
    )
    
    # Create different scenarios
    scenarios = [
        {
            "name": "Baseline",
            "project_changes": {},
            "market": MarketConditions(0.8, 0.05, 0.9, 0.15, 0.1, 0.3)
        },
        {
            "name": "High Material Inflation",
            "project_changes": {},
            "market": MarketConditions(0.8, 0.15, 0.9, 0.15, 0.1, 0.3)
        },
        {
            "name": "Labor Shortage",
            "project_changes": {},
            "market": MarketConditions(0.5, 0.05, 0.9, 0.15, 0.1, 0.3)
        },
        {
            "name": "Complex Terrain",
            "project_changes": {"custom_parameters": {**base_project.custom_parameters, "terrain": "mountainous"}},
            "market": MarketConditions(0.8, 0.05, 0.9, 0.15, 0.1, 0.3)
        }
    ]
    
    engine = CostEstimationEngine()
    scenario_results = []
    
    print("Running scenario analysis...")
    
    for scenario in scenarios:
        # Create modified project
        project_data = {**base_project.__dict__, **scenario["project_changes"]}
        modified_project = ProjectSpec(**project_data)
        
        # Run estimation
        result = engine.estimate_project_cost(modified_project, scenario["market"])
        scenario_results.append((scenario["name"], result))
        
        print(f"\nScenario: {scenario['name']}")
        print(f"  Total Cost: ${result.total_cost:,.2f}")
        print(f"  Timeline: {result.timeline_months} months")
        print(f"  Confidence: {result.confidence_score:.1%}")
    
    # Compare scenarios
    print(f"\n--- SCENARIO COMPARISON ---")
    baseline_cost = scenario_results[0][1].total_cost
    
    for name, result in scenario_results:
        cost_diff = result.total_cost - baseline_cost
        cost_diff_pct = (cost_diff / baseline_cost) * 100
        print(f"{name}:")
        print(f"  Cost: ${result.total_cost:,.2f} ({cost_diff_pct:+.1f}%)")
        print(f"  Timeline: {result.timeline_months} months")
    
    return scenario_results

def demo_database_integration():
    """Demonstrate database storage and retrieval"""
    print("\n=== DATABASE INTEGRATION DEMO ===\n")
    
    # Get database instance
    database = get_project_database()
    
    # Create a sample project
    project = ProjectSpec(
        project_type="water_infrastructure",
        location="southeast", 
        total_area=50000.0,
        duration_months=24,
        complexity_score=0.8,
        risk_factors={
            "environmental_sensitivity": 0.6,
            "geotechnical_risk": 0.4,
            "weather_exposure": 0.3,
            "regulatory_complexity": 0.7
        },
        custom_parameters={
            "capacity_mgd": 15,
            "treatment_level": "tertiary",
            "pipe_miles": 8,
            "soil_conditions": "poor"
        }
    )
    
    market = MarketConditions(0.7, 0.1, 0.8, 0.2, 0.15, 0.4)
    
    # Run estimation
    engine = CostEstimationEngine()
    result = engine.estimate_project_cost(project, market)
    
    # Save to database
    project_id = database.save_project(
        project, market, result,
        name="Demo Water Treatment Plant",
        tags=["demo", "water_infrastructure", "treatment"],
        notes="Demonstration project for MARL cost estimation system"
    )
    
    print(f"Project saved to database with ID: {project_id}")
    
    # Retrieve and display
    retrieved_project = database.get_project(project_id)
    if retrieved_project:
        print(f"\nRetrieved project: {retrieved_project.name}")
        print(f"Total cost: ${retrieved_project.estimation_result.total_cost:,.2f}")
        print(f"Created: {retrieved_project.created_date}")
        print(f"Tags: {', '.join(retrieved_project.tags)}")
    
    # List recent projects
    print(f"\n--- RECENT PROJECTS ---")
    recent_projects = database.list_projects(limit=5)
    for project_info in recent_projects:
        print(f"{project_info['name']} ({project_info['project_type']}): ${project_info.get('total_cost', 'N/A')}")
    
    # Get database statistics
    stats = database.get_database_stats()
    print(f"\n--- DATABASE STATISTICS ---")
    print(f"Total projects: {stats.get('projects_count', 0)}")
    print(f"Project types: {stats.get('project_type_distribution', {})}")
    print(f"Locations: {stats.get('location_distribution', {})}")
    
    return project_id

def demo_evaluation_metrics():
    """Demonstrate evaluation and testing capabilities"""
    print("\n=== EVALUATION METRICS DEMO ===\n")
    
    # Create estimation engine
    engine = CostEstimationEngine()
    
    # Run test suite
    test_suite = TestSuite(engine)
    
    print("Running basic functionality tests...")
    basic_tests = test_suite.run_basic_functionality_tests()
    
    print("Basic Test Results:")
    for test_name, result in basic_tests.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print("\nRunning stress tests...")
    stress_tests = test_suite.run_stress_tests()
    
    print("Stress Test Results:")
    for test_name, result in stress_tests.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    # Generate full test report
    test_report = test_suite.generate_test_report()
    
    # Save test report
    with open("test_report.txt", "w") as f:
        f.write(test_report)
    
    print(f"\nFull test report saved to test_report.txt")
    
    # Demonstrate evaluation metrics
    evaluator = EstimationEvaluator()
    
    # Create some sample data for evaluation
    estimation_results = []
    for i in range(5):
        # Create a sample project
        sample_project = ProjectSpec(
            project_type="residential",
            location="midwest",
            total_area=2000 + i * 500,
            duration_months=10 + i * 2,
            complexity_score=0.3 + i * 0.1,
            risk_factors={
                "environmental_sensitivity": 0.2 + i * 0.1,
                "geotechnical_risk": 0.2,
                "weather_exposure": 0.3,
                "regulatory_complexity": 0.2
            },
            custom_parameters={"foundation_type": "slab", "stories": 1}
        )
        
        sample_market = MarketConditions(0.8, 0.05, 0.9, 0.1, 0.08, 0.2)
        result = engine.estimate_project_cost(sample_project, sample_market)
        estimation_results.append(result)
    
    # Evaluate system performance
    try:
        system_metrics = evaluator.comprehensive_evaluation()
        print(f"\n--- SYSTEM PERFORMANCE METRICS ---")
        print(f"Total estimations evaluated: {system_metrics.total_estimations}")
        print(f"System reliability: {system_metrics.system_reliability:.1f}%")
        print(f"Average convergence score: {system_metrics.convergence_metrics.average_convergence_score:.3f}")
        print(f"Average negotiation rounds: {system_metrics.convergence_metrics.average_rounds:.1f}")
        
        # Generate evaluation report
        eval_report = evaluator.generate_evaluation_report(system_metrics)
        
        # Save evaluation report
        with open("evaluation_report.txt", "w") as f:
            f.write(eval_report)
            
        print(f"\nEvaluation report saved to evaluation_report.txt")
        
    except Exception as e:
        print(f"Could not generate comprehensive evaluation: {e}")

def demo_user_overrides():
    """Demonstrate user override functionality"""
    print("\n=== USER OVERRIDES DEMO ===\n")
    
    # Base project
    project = ProjectSpec(
        project_type="residential",
        location="west",
        total_area=3000.0,
        duration_months=15,
        complexity_score=0.5,
        risk_factors={
            "environmental_sensitivity": 0.3,
            "geotechnical_risk": 0.25,
            "weather_exposure": 0.4,
            "regulatory_complexity": 0.3
        },
        custom_parameters={
            "foundation_type": "slab",
            "stories": 1,
            "finish_level": "luxury"
        }
    )
    
    market = MarketConditions(0.75, 0.08, 0.85, 0.18, 0.12, 0.25)
    
    engine = CostEstimationEngine()
    
    # Run baseline estimation
    print("Running baseline estimation...")
    baseline_result = engine.estimate_project_cost(project, market)
    
    print(f"Baseline cost: ${baseline_result.total_cost:,.2f}")
    
    # Run with user overrides
    print("\nApplying user overrides...")
    user_overrides = {
        "labor_carpenter": 50.0,  # Override carpenter rate to $50/hr
        "material_lumber": 600.0,  # Override lumber cost to $600/mbf
        "market_labor_availability": 0.9  # Override market conditions
    }
    
    override_result = engine.estimate_project_cost(project, market, user_overrides)
    
    print(f"Cost with overrides: ${override_result.total_cost:,.2f}")
    
    cost_difference = override_result.total_cost - baseline_result.total_cost
    cost_diff_pct = (cost_difference / baseline_result.total_cost) * 100
    
    print(f"Cost difference: ${cost_difference:,.2f} ({cost_diff_pct:+.1f}%)")
    
    return baseline_result, override_result

def main():
    """Run complete demonstration of MARL construction cost estimation system"""
    print("🏗️  MARL CONSTRUCTION COST ESTIMATION SYSTEM DEMO 🏗️")
    print("=" * 60)
    
    try:
        # Demo 1: Residential project
        residential_result = demo_residential_project()
        
        # Demo 2: Commercial project
        commercial_result = demo_commercial_project()
        
        # Demo 3: Scenario comparison
        scenario_results = demo_scenario_comparison()
        
        # Demo 4: Database integration
        project_id = demo_database_integration()
        
        # Demo 5: User overrides
        baseline_result, override_result = demo_user_overrides()
        
        # Demo 6: Evaluation metrics
        demo_evaluation_metrics()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
        # Summary
        print(f"\n--- DEMO SUMMARY ---")
        print(f"Residential project cost: ${residential_result.total_cost:,.2f}")
        print(f"Commercial project cost: ${commercial_result.total_cost:,.2f}")
        print(f"Database project ID: {project_id}")
        print(f"User override impact: {((override_result.total_cost - baseline_result.total_cost) / baseline_result.total_cost * 100):+.1f}%")
        
        print(f"\nFiles generated:")
        print(f"  - test_report.txt")
        print(f"  - evaluation_report.txt")
        
        print(f"\nSystem demonstrates:")
        print(f"  ✓ Multi-agent reinforcement learning")
        print(f"  ✓ Project type specialization")
        print(f"  ✓ Market condition modeling")
        print(f"  ✓ User customization")
        print(f"  ✓ Database persistence")
        print(f"  ✓ Comprehensive evaluation")
        print(f"  ✓ Risk assessment")
        print(f"  ✓ Scenario comparison")
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)