#!/usr/bin/env python3
"""
Claude Projects Integration Demo

This script demonstrates how to integrate data from Claude Projects or other external
project management systems into the MARL Construction Cost Estimation system.
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marl_construction.integrations.claude_projects_connector import ClaudeProjectsConnector, create_sample_import_template
from marl_construction.data.project_database import get_project_database
from marl_construction.estimation.cost_engine import CostEstimationEngine

def demo_json_import():
    """Demonstrate importing from JSON export"""
    print("=== JSON IMPORT DEMO ===\n")
    
    # Create sample data file
    create_sample_import_template()
    print("✅ Created sample import template: claude_projects_import_template.json")
    
    # Initialize connector
    connector = ClaudeProjectsConnector()
    
    # Import projects
    try:
        imported_project_ids = connector.import_from_json_export("claude_projects_import_template.json")
        
        print(f"\n✅ Successfully imported {len(imported_project_ids)} projects:")
        for project_id in imported_project_ids:
            print(f"  - {project_id}")
            
        # Verify imports by retrieving them
        database = get_project_database()
        print(f"\n--- IMPORTED PROJECT DETAILS ---")
        
        for project_id in imported_project_ids:
            project = database.get_project(project_id)
            if project:
                print(f"\nProject: {project.name}")
                print(f"  Type: {project.project_spec.project_type}")
                print(f"  Location: {project.project_spec.location}")
                print(f"  Area: {project.project_spec.total_area:,.0f} sq ft")
                print(f"  Estimated Cost: ${project.estimation_result.total_cost:,.2f}")
                print(f"  Cost/sq ft: ${project.estimation_result.cost_per_sqft:.2f}")
                print(f"  Tags: {', '.join(project.tags)}")
                
    except Exception as e:
        print(f"❌ Import failed: {e}")

def demo_csv_import():
    """Demonstrate importing from CSV export"""
    print("\n=== CSV IMPORT DEMO ===\n")
    
    # Create sample CSV file
    csv_content = """id,name,type,location,area,duration,actual_cost,completion_date
cp_003,Warehouse Project,commercial,west,75000,18,6500000,2023-11-20
cp_004,Highway Bridge,infrastructure,northeast,25000,30,12000000,2023-09-15
cp_005,Water Treatment Plant,water,southeast,15000,36,8500000,2024-01-10"""
    
    with open("sample_projects.csv", "w") as f:
        f.write(csv_content)
    
    print("✅ Created sample CSV file: sample_projects.csv")
    
    # Define field mapping
    field_mapping = {
        'id': 'id',
        'name': 'name', 
        'type': 'type',
        'location': 'location',
        'area': 'area',
        'duration': 'duration',
        'actual_cost': 'actual_cost',
        'completion_date': 'completion_date'
    }
    
    # Initialize connector and import
    connector = ClaudeProjectsConnector()
    
    try:
        imported_project_ids = connector.import_from_csv_export("sample_projects.csv", field_mapping)
        
        print(f"\n✅ Successfully imported {len(imported_project_ids)} projects from CSV:")
        for project_id in imported_project_ids:
            print(f"  - {project_id}")
            
        # Show imported data
        database = get_project_database()
        print(f"\n--- CSV IMPORTED PROJECT DETAILS ---")
        
        for project_id in imported_project_ids:
            project = database.get_project(project_id)
            if project:
                print(f"\nProject: {project.name}")
                print(f"  Type: {project.project_spec.project_type}")
                print(f"  Area: {project.project_spec.total_area:,.0f} sq ft") 
                print(f"  Estimated Cost: ${project.estimation_result.total_cost:,.2f}")
                
    except Exception as e:
        print(f"❌ CSV import failed: {e}")

def demo_data_export():
    """Demonstrate exporting data for Claude Projects"""
    print("\n=== DATA EXPORT DEMO ===\n")
    
    connector = ClaudeProjectsConnector()
    database = get_project_database()
    
    # Get some projects to export
    recent_projects = database.list_projects(limit=5)
    project_ids = [p['id'] for p in recent_projects]
    
    if project_ids:
        try:
            exported_count = connector.export_for_claude_projects("exported_projects.json", project_ids)
            
            print(f"✅ Exported {exported_count} projects to exported_projects.json")
            
            # Show what was exported
            with open("exported_projects.json", "r") as f:
                export_data = json.load(f)
                
            print(f"\n--- EXPORT PREVIEW ---")
            for project in export_data["projects"][:2]:  # Show first 2
                print(f"\nProject: {project['name']}")
                print(f"  Type: {project['type']}")
                print(f"  Area: {project['area']:,.0f} sq ft")
                print(f"  Estimated Cost: ${project['estimated_cost']:,.2f}")
                print(f"  Confidence: {project['confidence']:.1%}")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    else:
        print("⚠️  No projects found to export")

def demo_historical_learning():
    """Demonstrate how imported data improves estimation accuracy"""
    print("\n=== HISTORICAL LEARNING DEMO ===\n")
    
    # Create estimation engine
    engine = CostEstimationEngine()
    database = get_project_database()
    
    # Show database statistics before and after import
    stats = database.get_database_stats()
    print(f"Database Statistics:")
    print(f"  Total projects: {stats.get('projects_count', 0)}")
    print(f"  Project types: {stats.get('project_type_distribution', {})}")
    
    # Demo estimation on a new project similar to imported ones
    from marl_construction.agents.cost_estimation_agents import ProjectSpec, MarketConditions
    
    test_project = ProjectSpec(
        project_type="commercial",
        location="west",
        total_area=65000,  # Similar to imported warehouse
        duration_months=20,
        complexity_score=0.6,
        risk_factors={
            "environmental_sensitivity": 0.3,
            "geotechnical_risk": 0.3,
            "weather_exposure": 0.2,
            "regulatory_complexity": 0.4
        },
        custom_parameters={
            "structural_system": "steel",
            "occupancy_type": "warehouse",
            "mep_complexity": "standard"
        }
    )
    
    test_market = MarketConditions(0.75, 0.08, 0.85, 0.15, 0.12, 0.25)
    
    print(f"\n--- ESTIMATING NEW PROJECT ---")
    print(f"Project: 65,000 sq ft warehouse in west region")
    
    try:
        result = engine.estimate_project_cost(test_project, test_market)
        
        print(f"\nEstimation Results:")
        print(f"  Total Cost: ${result.total_cost:,.2f}")
        print(f"  Cost/sq ft: ${result.cost_per_sqft:.2f}")
        print(f"  Confidence: {result.confidence_score:.1%}")
        print(f"  Negotiation Rounds: {result.negotiation_rounds}")
        
        # Show if any historical projects were used for similarity
        similar_projects = []
        for agent_model in engine.agent_models.values():
            if hasattr(agent_model, 'project_memory') and agent_model.project_memory:
                similar = agent_model.find_similar_projects(test_project, top_k=3)
                if similar:
                    similar_projects.extend(similar)
        
        if similar_projects:
            print(f"\n--- SIMILAR HISTORICAL PROJECTS FOUND ---")
            for i, proj in enumerate(similar_projects[:3]):
                print(f"  {i+1}. Similarity: {proj.get('similarity', 0):.1%}")
                print(f"     Type: {proj.get('project_type', 'Unknown')}")
                print(f"     Area: {proj.get('total_area', 0):,.0f} sq ft")
        else:
            print(f"\n⚠️  No similar historical projects found - estimation based on defaults")
            
    except Exception as e:
        print(f"❌ Estimation failed: {e}")

def demo_api_integration_template():
    """Show template for API integration"""
    print("\n=== API INTEGRATION TEMPLATE ===\n")
    
    print("""
🔗 API Integration Example:

# If Claude Projects has an API, you can integrate like this:

connector = ClaudeProjectsConnector(
    api_key="your_api_key_here",
    base_url="https://api.claude-projects.com/v1"
)

# Sync projects from API
imported_ids = connector.sync_with_api(
    endpoint="projects",
    headers={
        'Authorization': 'Bearer your_token',
        'Content-Type': 'application/json'
    }
)

# The system will automatically:
# 1. Fetch projects from the API
# 2. Transform them to our ProjectSpec format
# 3. Store them in our database
# 4. Make them available for historical learning

📋 Required API Response Format:
{
  "projects": [
    {
      "id": "project_123",
      "name": "Project Name",
      "type": "commercial",
      "location": "west",
      "area": 50000,
      "duration": 24,
      "actual_cost": 8500000,
      "completion_date": "2023-12-15",
      // ... additional fields
    }
  ]
}
""")

def cleanup_demo_files():
    """Clean up demo files"""
    files_to_remove = [
        "claude_projects_import_template.json",
        "sample_projects.csv", 
        "exported_projects.json"
    ]
    
    for file_name in files_to_remove:
        try:
            os.remove(file_name)
            print(f"🗑️  Removed {file_name}")
        except FileNotFoundError:
            pass

def main():
    """Run Claude Projects integration demonstration"""
    print("🔗 CLAUDE PROJECTS INTEGRATION DEMO")
    print("=" * 50)
    
    try:
        # Demo JSON import
        demo_json_import()
        
        # Demo CSV import  
        demo_csv_import()
        
        # Demo data export
        demo_data_export()
        
        # Demo historical learning
        demo_historical_learning()
        
        # Show API integration template
        demo_api_integration_template()
        
        print("\n" + "=" * 50)
        print("🎉 INTEGRATION DEMO COMPLETED!")
        print("=" * 50)
        
        print(f"\n--- INTEGRATION SUMMARY ---")
        print(f"✅ JSON Import: Demonstrated with sample data")
        print(f"✅ CSV Import: Demonstrated with custom field mapping")
        print(f"✅ Data Export: Created export file for Claude Projects") 
        print(f"✅ Historical Learning: Showed how imported data improves accuracy")
        print(f"✅ API Template: Provided integration example")
        
        print(f"\n--- FILES CREATED ---")
        print(f"📄 claude_projects_import_template.json - Sample import format")
        print(f"📄 sample_projects.csv - Sample CSV import")
        print(f"📄 exported_projects.json - Export format example")
        
        # Ask if user wants to clean up
        cleanup = input(f"\nDelete demo files? (y/n): ").lower().strip()
        if cleanup == 'y':
            cleanup_demo_files()
            
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)