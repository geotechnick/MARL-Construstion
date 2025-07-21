#!/usr/bin/env python3
"""
MARL Construction Cost Estimation System - Main Entry Point

This is the main entry point for the MARL Construction Cost Estimation system.
It provides a command-line interface and GUI launcher for the complete system.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def launch_gui():
    """Launch the graphical user interface"""
    try:
        from marl_construction.ui.project_builder import main as gui_main
        print("🏗️  Launching MARL Construction Cost Estimator GUI...")
        gui_main()
    except ImportError as e:
        print(f"❌ GUI dependencies not available: {e}")
        print("Please install required GUI packages: tkinter, matplotlib")
        return 1
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1
    return 0

def run_demo():
    """Run the complete system demonstration"""
    try:
        from examples.complete_estimation_demo import main as demo_main
        print("🏗️  Running MARL Construction Cost Estimation Demo...")
        return demo_main()
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")
        return 1

def run_tests():
    """Run the system test suite"""
    try:
        from marl_construction.estimation.cost_engine import CostEstimationEngine
        from marl_construction.evaluation.metrics import TestSuite
        
        print("🏗️  Running MARL Construction Cost Estimation Tests...")
        
        engine = CostEstimationEngine()
        test_suite = TestSuite(engine)
        
        # Run tests
        basic_tests = test_suite.run_basic_functionality_tests()
        stress_tests = test_suite.run_stress_tests()
        
        # Generate report
        report = test_suite.generate_test_report()
        
        # Save report
        with open("test_results.txt", "w") as f:
            f.write(report)
        
        print("✅ Tests completed. Results saved to test_results.txt")
        
        # Calculate pass rate
        all_tests = {**basic_tests, **stress_tests}
        pass_rate = sum(all_tests.values()) / len(all_tests) * 100
        
        print(f"Overall pass rate: {pass_rate:.1f}%")
        
        return 0 if pass_rate >= 80 else 1
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return 1

def estimate_project(project_file: str, output_file: str = None):
    """Estimate project cost from configuration file"""
    try:
        import json
        from marl_construction.agents.cost_estimation_agents import ProjectSpec, MarketConditions
        from marl_construction.estimation.cost_engine import CostEstimationEngine
        
        print(f"🏗️  Loading project from {project_file}...")
        
        # Load project configuration
        with open(project_file, 'r') as f:
            config = json.load(f)
        
        # Create project spec
        project = ProjectSpec(**config.get('project', {}))
        market = MarketConditions(**config.get('market', {}))
        user_overrides = config.get('overrides', {})
        
        # Run estimation
        engine = CostEstimationEngine()
        result = engine.estimate_project_cost(project, market, user_overrides)
        
        # Display results
        print(f"\n--- ESTIMATION RESULTS ---")
        print(f"Total Cost: ${result.total_cost:,.2f}")
        print(f"Cost per sq ft: ${result.cost_per_sqft:.2f}")
        print(f"Timeline: {result.timeline_months} months")
        print(f"Confidence: {result.confidence_score:.1%}")
        
        # Save results if output file specified
        if output_file:
            engine.export_estimation_report(result, output_file)
            print(f"✅ Results saved to {output_file}")
        
        return 0
        
    except FileNotFoundError:
        print(f"❌ Project file not found: {project_file}")
        return 1
    except Exception as e:
        print(f"❌ Failed to estimate project: {e}")
        return 1

def show_system_info():
    """Display system information and capabilities"""
    print("""
🏗️  MARL CONSTRUCTION COST ESTIMATION SYSTEM

📋 SYSTEM CAPABILITIES:
• Multi-Agent Reinforcement Learning cost estimation
• Support for 4 project types: Residential, Commercial, Transportation, Water Infrastructure
• Regional cost databases with user overrides
• Real-time market condition modeling
• Risk assessment and mitigation recommendations
• Scenario comparison and sensitivity analysis
• Historical project tracking and analysis
• Comprehensive evaluation metrics
• Interactive GUI and command-line interfaces

🤖 INTELLIGENT AGENTS:
• Owner Agent: Minimizes cost, risk, and duration while maximizing quality
• Contractor Agent: Optimizes pricing, resource allocation, and profit margins
• Supplier Agent: Manages material pricing and availability
• Regulator Agent: Enforces compliance and assesses regulatory risks
• Estimator Agent: Provides final consensus estimates based on historical data

📊 KEY FEATURES:
• Dynamic agent negotiation with convergence analysis
• Customizable default rates by region and trade
• Work breakdown structure (WBS) generation
• Cost breakdown by category (labor, materials, equipment, etc.)
• Risk factor analysis (scope creep, weather, regulatory, etc.)
• Market impact modeling (inflation, labor availability, supply chain)
• Database storage for historical analysis
• Comprehensive testing and evaluation framework

🎯 ADDRESSING INDUSTRY PAIN POINTS:
• Scope creep through real-time change impact analysis
• Labor/material price volatility via market condition modeling
• Communication delays through transparent agent negotiations
• Risk underestimation via comprehensive risk assessment
• Change order processing through instant scope impact analysis

💻 USAGE MODES:
• GUI: Interactive graphical interface
• CLI: Command-line estimation from project files
• Demo: Complete system demonstration
• API: Integration into existing construction software

🔧 TECHNICAL STACK:
• Multi-Agent RL: PettingZoo, PyTorch
• Data: SQLite, pandas
• UI: tkinter, matplotlib, seaborn
• Testing: Custom evaluation framework
""")

def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="MARL Construction Cost Estimation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --gui                    # Launch GUI
  python main.py --demo                   # Run demonstration  
  python main.py --test                   # Run test suite
  python main.py --estimate project.json # Estimate from file
  python main.py --info                   # Show system info
        """
    )
    
    parser.add_argument("--gui", action="store_true", 
                       help="Launch graphical user interface")
    parser.add_argument("--demo", action="store_true",
                       help="Run complete system demonstration")
    parser.add_argument("--test", action="store_true",
                       help="Run system test suite")
    parser.add_argument("--estimate", type=str, metavar="FILE",
                       help="Estimate project cost from JSON configuration file")
    parser.add_argument("--output", type=str, metavar="FILE",
                       help="Output file for estimation results (JSON)")
    parser.add_argument("--info", action="store_true",
                       help="Show system information and capabilities")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help and system info
    if len(sys.argv) == 1:
        parser.print_help()
        print()
        show_system_info()
        return 0
    
    # Handle different modes
    if args.info:
        show_system_info()
        return 0
    elif args.gui:
        return launch_gui()
    elif args.demo:
        return run_demo()
    elif args.test:
        return run_tests()
    elif args.estimate:
        return estimate_project(args.estimate, args.output)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)