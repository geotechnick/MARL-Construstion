"""
Claude Projects Integration Module

This module provides integration capabilities with Claude Projects or other external
project management systems to import historical project data for improved estimation accuracy.
"""

import json
import csv
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..agents.cost_estimation_agents import ProjectSpec, MarketConditions
from ..data.project_database import get_project_database
from ..estimation.cost_engine import CostEstimationEngine

@dataclass
class ExternalProjectData:
    """Data structure for external project information"""
    external_id: str
    name: str
    type: str
    location: str
    area: float
    duration: int
    actual_cost: Optional[float] = None
    completion_date: Optional[str] = None
    custom_fields: Dict[str, Any] = None

class ClaudeProjectsConnector:
    """Connector for Claude Projects data integration"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.database = get_project_database()
        
    def import_from_json_export(self, file_path: str) -> List[str]:
        """Import projects from Claude Projects JSON export"""
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            imported_projects = []
            
            for project_data in data.get('projects', []):
                # Transform external data to our format
                project_spec = self._transform_external_project(project_data)
                
                if project_spec:
                    # Generate market conditions (or use historical if available)
                    market_conditions = self._estimate_historical_market_conditions(project_data)
                    
                    # Create estimation result from actual data
                    estimation_result = self._create_estimation_from_actual(project_data, project_spec)
                    
                    # Save to database
                    project_id = self.database.save_project(
                        project_spec,
                        market_conditions, 
                        estimation_result,
                        name=project_data.get('name', 'Imported Project'),
                        tags=['imported', 'claude_projects'],
                        notes=f"Imported from Claude Projects on {datetime.now().isoformat()}"
                    )
                    
                    imported_projects.append(project_id)
                    
            return imported_projects
            
        except Exception as e:
            raise Exception(f"Failed to import from JSON: {str(e)}")
    
    def import_from_csv_export(self, file_path: str, field_mapping: Dict[str, str]) -> List[str]:
        """Import projects from CSV export with custom field mapping"""
        
        try:
            imported_projects = []
            
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Map CSV fields to our structure
                    project_data = {}
                    for our_field, csv_field in field_mapping.items():
                        if csv_field in row:
                            project_data[our_field] = row[csv_field]
                    
                    # Transform to our format
                    project_spec = self._transform_external_project(project_data)
                    
                    if project_spec:
                        market_conditions = self._estimate_historical_market_conditions(project_data)
                        estimation_result = self._create_estimation_from_actual(project_data, project_spec)
                        
                        project_id = self.database.save_project(
                            project_spec,
                            market_conditions,
                            estimation_result,
                            name=project_data.get('name', 'CSV Import'),
                            tags=['imported', 'csv'],
                            notes=f"Imported from CSV on {datetime.now().isoformat()}"
                        )
                        
                        imported_projects.append(project_id)
            
            return imported_projects
            
        except Exception as e:
            raise Exception(f"Failed to import from CSV: {str(e)}")
    
    def sync_with_api(self, endpoint: str, headers: Optional[Dict] = None) -> List[str]:
        """Sync projects via API connection"""
        
        if not self.api_key or not self.base_url:
            raise ValueError("API key and base URL required for API sync")
        
        try:
            # Set up headers
            if not headers:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            
            # Fetch projects from API
            response = requests.get(f"{self.base_url}/{endpoint}", headers=headers)
            response.raise_for_status()
            
            projects_data = response.json()
            imported_projects = []
            
            for project_data in projects_data.get('projects', []):
                # Check if project already exists (by external_id)
                external_id = project_data.get('id', '')
                
                if not self._project_exists(external_id):
                    project_spec = self._transform_external_project(project_data)
                    
                    if project_spec:
                        market_conditions = self._estimate_historical_market_conditions(project_data)
                        estimation_result = self._create_estimation_from_actual(project_data, project_spec)
                        
                        project_id = self.database.save_project(
                            project_spec,
                            market_conditions,
                            estimation_result,
                            name=project_data.get('name', 'API Import'),
                            tags=['imported', 'api_sync'],
                            notes=f"Synced from API on {datetime.now().isoformat()}. External ID: {external_id}"
                        )
                        
                        imported_projects.append(project_id)
            
            return imported_projects
            
        except Exception as e:
            raise Exception(f"Failed to sync with API: {str(e)}")
    
    def _transform_external_project(self, external_data: Dict[str, Any]) -> Optional[ProjectSpec]:
        """Transform external project data to ProjectSpec"""
        
        try:
            # Map project types
            project_type_mapping = {
                'residential': 'residential',
                'commercial': 'commercial', 
                'office': 'commercial',
                'retail': 'commercial',
                'warehouse': 'commercial',
                'infrastructure': 'transportation',
                'road': 'transportation',
                'bridge': 'transportation',
                'water': 'water_infrastructure',
                'wastewater': 'water_infrastructure',
                'treatment': 'water_infrastructure'
            }
            
            external_type = external_data.get('type', '').lower()
            project_type = project_type_mapping.get(external_type, 'commercial')
            
            # Map locations to regions
            location_mapping = {
                'northeast': 'northeast',
                'southeast': 'southeast', 
                'midwest': 'midwest',
                'west': 'west',
                'california': 'west',
                'texas': 'southeast',
                'florida': 'southeast',
                'new york': 'northeast',
                # Add more location mappings as needed
            }
            
            external_location = external_data.get('location', '').lower()
            location = location_mapping.get(external_location, 'midwest')  # Default to midwest
            
            # Extract other parameters
            total_area = float(external_data.get('area', 0))
            duration_months = int(external_data.get('duration', 12))
            
            # Estimate complexity from available data
            complexity_score = self._estimate_complexity(external_data)
            
            # Create risk factors (default if not available)
            risk_factors = {
                "environmental_sensitivity": external_data.get('environmental_risk', 0.3),
                "geotechnical_risk": external_data.get('geotechnical_risk', 0.3),
                "weather_exposure": external_data.get('weather_risk', 0.3),
                "regulatory_complexity": external_data.get('regulatory_risk', 0.3)
            }
            
            # Extract custom parameters based on project type
            custom_parameters = self._extract_custom_parameters(external_data, project_type)
            
            return ProjectSpec(
                project_type=project_type,
                location=location,
                total_area=total_area,
                duration_months=duration_months,
                complexity_score=complexity_score,
                risk_factors=risk_factors,
                custom_parameters=custom_parameters
            )
            
        except Exception as e:
            print(f"Failed to transform project {external_data.get('id', 'unknown')}: {e}")
            return None
    
    def _estimate_historical_market_conditions(self, project_data: Dict[str, Any]) -> MarketConditions:
        """Estimate market conditions from historical project data"""
        
        # Default market conditions (could be enhanced with historical economic data)
        return MarketConditions(
            labor_availability=project_data.get('labor_availability', 0.75),
            material_inflation=project_data.get('material_inflation', 0.05),
            supply_chain_stability=project_data.get('supply_chain_stability', 0.85),
            economic_volatility=project_data.get('economic_volatility', 0.15),
            fuel_surcharge=project_data.get('fuel_surcharge', 0.1),
            weather_risk=project_data.get('weather_risk', 0.3)
        )
    
    def _create_estimation_from_actual(self, project_data: Dict[str, Any], project_spec: ProjectSpec):
        """Create estimation result from actual project data"""
        
        from ..estimation.cost_engine import EstimationResult
        
        actual_cost = project_data.get('actual_cost', 0)
        
        # If no actual cost, create a dummy estimation
        if actual_cost == 0:
            engine = CostEstimationEngine()
            market = self._estimate_historical_market_conditions(project_data)
            return engine.estimate_project_cost(project_spec, market)
        
        # Create estimation result from actual data
        cost_per_sqft = actual_cost / project_spec.total_area if project_spec.total_area > 0 else 0
        
        # Estimate cost breakdown (simplified)
        cost_breakdown = {
            "labor": actual_cost * 0.35,
            "materials": actual_cost * 0.30,
            "equipment": actual_cost * 0.10,
            "permits_fees": actual_cost * 0.05,
            "overhead": actual_cost * 0.12,
            "profit": actual_cost * 0.08
        }
        
        # Default risk assessment
        risk_assessment = {
            "scope_creep_risk": 0.15,
            "material_escalation_risk": 0.1,
            "labor_shortage_risk": 0.2,
            "weather_delay_risk": 0.25,
            "regulatory_risk": 0.1
        }
        
        # Default market factors
        market_factors = {
            "labor_impact": 1.0,
            "material_impact": 1.05,
            "supply_chain_impact": 1.0,
            "economic_impact": 1.0
        }
        
        return EstimationResult(
            project_id=f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_cost=actual_cost,
            cost_breakdown=cost_breakdown,
            timeline_months=project_spec.duration_months,
            confidence_score=0.9,  # High confidence for actual data
            risk_assessment=risk_assessment,
            agent_bids={},  # No agent bids for imported data
            market_factors=market_factors,
            cost_per_sqft=cost_per_sqft,
            negotiation_rounds=1,
            convergence_score=1.0,  # Perfect convergence for actual data
            created_timestamp=datetime.now().isoformat()
        )
    
    def _estimate_complexity(self, project_data: Dict[str, Any]) -> float:
        """Estimate project complexity from available data"""
        
        complexity = 0.5  # Default medium complexity
        
        # Adjust based on available indicators
        if 'complexity' in project_data:
            return float(project_data['complexity'])
        
        # Infer from project characteristics
        area = project_data.get('area', 0)
        if area > 100000:  # Large project
            complexity += 0.2
        elif area < 1000:  # Small project
            complexity -= 0.2
        
        # Duration-based complexity
        duration = project_data.get('duration', 12)
        if duration > 36:  # Long duration suggests complexity
            complexity += 0.1
        
        # Ensure within bounds
        return max(0.1, min(0.9, complexity))
    
    def _extract_custom_parameters(self, project_data: Dict[str, Any], project_type: str) -> Dict[str, Any]:
        """Extract project-type specific custom parameters"""
        
        custom_params = {}
        
        if project_type == "residential":
            custom_params = {
                "foundation_type": project_data.get('foundation_type', 'slab'),
                "stories": project_data.get('stories', 1),
                "finish_level": project_data.get('finish_level', 'standard'),
                "energy_efficiency_target": project_data.get('energy_target', 'standard')
            }
        elif project_type == "commercial":
            custom_params = {
                "structural_system": project_data.get('structural_system', 'steel'),
                "occupancy_type": project_data.get('occupancy_type', 'office'),
                "mep_complexity": project_data.get('mep_complexity', 'standard')
            }
        elif project_type == "transportation":
            custom_params = {
                "length_miles": project_data.get('length_miles', 5),
                "lanes": project_data.get('lanes', 4),
                "terrain": project_data.get('terrain', 'flat'),
                "bridge_count": project_data.get('bridge_count', 0)
            }
        elif project_type == "water_infrastructure":
            custom_params = {
                "capacity_mgd": project_data.get('capacity_mgd', 10),
                "treatment_level": project_data.get('treatment_level', 'secondary'),
                "pipe_miles": project_data.get('pipe_miles', 5),
                "soil_conditions": project_data.get('soil_conditions', 'normal')
            }
        
        return custom_params
    
    def _project_exists(self, external_id: str) -> bool:
        """Check if project with external ID already exists"""
        
        # Query database for projects with this external ID in notes
        projects = self.database.list_projects()
        
        for project in projects:
            project_record = self.database.get_project(project['id'])
            if project_record and external_id in project_record.notes:
                return True
        
        return False
    
    def export_for_claude_projects(self, output_file: str, project_ids: Optional[List[str]] = None):
        """Export project data in format suitable for Claude Projects"""
        
        if project_ids is None:
            # Export all projects
            projects = self.database.list_projects()
            project_ids = [p['id'] for p in projects]
        
        export_data = {
            "exported_date": datetime.now().isoformat(),
            "source": "MARL Construction Cost Estimation",
            "projects": []
        }
        
        for project_id in project_ids:
            project = self.database.get_project(project_id)
            if project:
                export_project = {
                    "id": project.id,
                    "name": project.name,
                    "type": project.project_spec.project_type,
                    "location": project.project_spec.location,
                    "area": project.project_spec.total_area,
                    "duration": project.project_spec.duration_months,
                    "complexity": project.project_spec.complexity_score,
                    "estimated_cost": project.estimation_result.total_cost,
                    "cost_per_sqft": project.estimation_result.cost_per_sqft,
                    "confidence": project.estimation_result.confidence_score,
                    "risk_factors": project.project_spec.risk_factors,
                    "custom_parameters": project.project_spec.custom_parameters,
                    "created_date": project.created_date
                }
                export_data["projects"].append(export_project)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return len(export_data["projects"])

def create_sample_import_template():
    """Create a sample import template for Claude Projects data"""
    
    sample_data = {
        "projects": [
            {
                "id": "cp_001",
                "name": "Downtown Office Building",
                "type": "commercial",
                "location": "northeast", 
                "area": 50000,
                "duration": 24,
                "actual_cost": 8500000,
                "completion_date": "2023-12-15",
                "complexity": 0.7,
                "structural_system": "steel",
                "occupancy_type": "office",
                "mep_complexity": "high",
                "environmental_risk": 0.3,
                "geotechnical_risk": 0.4,
                "weather_risk": 0.2,
                "regulatory_risk": 0.5
            },
            {
                "id": "cp_002", 
                "name": "Suburban Residence",
                "type": "residential",
                "location": "midwest",
                "area": 2800,
                "duration": 8,
                "actual_cost": 420000,
                "completion_date": "2023-08-30",
                "complexity": 0.4,
                "foundation_type": "basement",
                "stories": 2,
                "finish_level": "standard",
                "energy_target": "high"
            }
        ]
    }
    
    with open("claude_projects_import_template.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample import template created: claude_projects_import_template.json")