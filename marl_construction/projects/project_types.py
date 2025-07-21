from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

@dataclass
class WorkBreakdownStructure:
    task_id: str
    task_name: str
    description: str
    duration_days: int
    labor_hours: Dict[str, float]  # trade -> hours
    materials: Dict[str, float]    # material -> quantity
    equipment: Dict[str, float]    # equipment -> days
    dependencies: List[str]        # task_ids this depends on
    cost_per_unit: Optional[float] = None
    quantity: Optional[float] = None

@dataclass
class ProjectTypeConfig:
    name: str
    description: str
    typical_size_range: tuple  # (min, max) area
    typical_duration_range: tuple  # (min, max) months
    complexity_factors: Dict[str, float]
    default_wbs: List[WorkBreakdownStructure]
    cost_drivers: List[str]
    risk_categories: List[str]
    
class BaseProjectType(ABC):
    def __init__(self, config: ProjectTypeConfig):
        self.config = config
        
    @abstractmethod
    def generate_wbs(self, project_parameters: Dict[str, Any]) -> List[WorkBreakdownStructure]:
        pass
        
    @abstractmethod
    def estimate_quantities(self, project_parameters: Dict[str, Any]) -> Dict[str, float]:
        pass
        
    @abstractmethod
    def calculate_complexity_multiplier(self, project_parameters: Dict[str, Any]) -> float:
        pass
        
    @abstractmethod  
    def get_cost_per_sqft_estimate(self, project_parameters: Dict[str, Any]) -> float:
        pass

class ResidentialProject(BaseProjectType):
    def __init__(self):
        wbs_templates = [
            WorkBreakdownStructure(
                task_id="RES_001",
                task_name="Site Preparation",
                description="Excavation, grading, and site utilities",
                duration_days=5,
                labor_hours={"laborer": 32, "equipment_operator": 16},
                materials={"gravel": 15, "sand": 10},
                equipment={"excavator": 3, "bulldozer": 2},
                dependencies=[]
            ),
            WorkBreakdownStructure(
                task_id="RES_002", 
                task_name="Foundation",
                description="Concrete foundation and basement/crawlspace",
                duration_days=10,
                labor_hours={"concrete_finisher": 48, "laborer": 64, "carpenter": 24},
                materials={"concrete": 25, "rebar": 2000, "lumber": 1.5},
                equipment={"concrete_pump": 2},
                dependencies=["RES_001"]
            ),
            WorkBreakdownStructure(
                task_id="RES_003",
                task_name="Framing",
                description="Wall, floor, and roof framing",
                duration_days=15,
                labor_hours={"carpenter": 120, "laborer": 80},
                materials={"lumber": 15, "fasteners": 50},
                equipment={},
                dependencies=["RES_002"]
            ),
            WorkBreakdownStructure(
                task_id="RES_004",
                task_name="Roofing",
                description="Roof sheathing, underlayment, and shingles",
                duration_days=8,
                labor_hours={"roofer": 64, "laborer": 32},
                materials={"roofing_material": 30, "underlayment": 35},
                equipment={},
                dependencies=["RES_003"]
            ),
            WorkBreakdownStructure(
                task_id="RES_005",
                task_name="Electrical Rough-In",
                description="Electrical wiring and panel installation",
                duration_days=6,
                labor_hours={"electrician": 48},
                materials={"electrical": 200},
                equipment={},
                dependencies=["RES_003"]
            ),
            WorkBreakdownStructure(
                task_id="RES_006",
                task_name="Plumbing Rough-In", 
                description="Plumbing lines and fixture preparation",
                duration_days=5,
                labor_hours={"plumber": 40},
                materials={"plumbing": 150},
                equipment={},
                dependencies=["RES_003"]
            ),
            WorkBreakdownStructure(
                task_id="RES_007",
                task_name="HVAC Installation",
                description="Heating, ventilation, and AC systems",
                duration_days=4,
                labor_hours={"hvac_tech": 32},
                materials={"hvac_equipment": 1},
                equipment={},
                dependencies=["RES_003"]
            ),
            WorkBreakdownStructure(
                task_id="RES_008",
                task_name="Insulation",
                description="Wall and attic insulation",
                duration_days=3,
                labor_hours={"insulation_installer": 24},
                materials={"insulation": 1200},
                equipment={},
                dependencies=["RES_005", "RES_006", "RES_007"]
            ),
            WorkBreakdownStructure(
                task_id="RES_009",
                task_name="Drywall",
                description="Drywall installation and finishing", 
                duration_days=12,
                labor_hours={"drywaller": 96, "painter": 32},
                materials={"drywall": 50, "joint_compound": 25},
                equipment={},
                dependencies=["RES_008"]
            ),
            WorkBreakdownStructure(
                task_id="RES_010",
                task_name="Interior Finishes",
                description="Flooring, trim, cabinets, and final fixtures",
                duration_days=20,
                labor_hours={"carpenter": 80, "flooring_installer": 40, "painter": 60},
                materials={"flooring": 35, "cabinets": 12, "trim": 8},
                equipment={},
                dependencies=["RES_009"]
            )
        ]
        
        config = ProjectTypeConfig(
            name="Residential Construction",
            description="Single-family homes, duplexes, apartments, condominiums",
            typical_size_range=(800, 8000),  # sq ft
            typical_duration_range=(4, 18),  # months
            complexity_factors={
                "foundation_type": 0.15,
                "stories": 0.1,
                "finish_level": 0.2,
                "energy_efficiency": 0.1,
                "custom_features": 0.15
            },
            default_wbs=wbs_templates,
            cost_drivers=["area", "finish_level", "foundation_type", "stories"],
            risk_categories=["weather", "permit_delays", "material_availability", "skilled_labor"]
        )
        super().__init__(config)
        
    def generate_wbs(self, project_parameters: Dict[str, Any]) -> List[WorkBreakdownStructure]:
        wbs = self.config.default_wbs.copy()
        
        # Adjust quantities based on project size
        area_multiplier = project_parameters.get("total_area", 2000) / 2000.0
        stories = project_parameters.get("stories", 1)
        
        for task in wbs:
            # Scale labor hours and materials by area
            for trade in task.labor_hours:
                task.labor_hours[trade] *= area_multiplier
            for material in task.materials:
                task.materials[material] *= area_multiplier
                
            # Adjust for multi-story
            if task.task_id in ["RES_003", "RES_005", "RES_006", "RES_008", "RES_009"]:
                story_multiplier = 1.0 + (stories - 1) * 0.8
                for trade in task.labor_hours:
                    task.labor_hours[trade] *= story_multiplier
                    
        # Add custom features if specified
        if project_parameters.get("custom_features", []):
            for feature in project_parameters["custom_features"]:
                if feature == "basement":
                    basement_task = WorkBreakdownStructure(
                        task_id="RES_011",
                        task_name="Basement Construction",
                        description="Basement excavation and finishing",
                        duration_days=15,
                        labor_hours={"concrete_finisher": 60, "carpenter": 40},
                        materials={"concrete": 40, "lumber": 5},
                        equipment={"excavator": 5},
                        dependencies=["RES_001"]
                    )
                    wbs.append(basement_task)
                    
        return wbs
        
    def estimate_quantities(self, project_parameters: Dict[str, Any]) -> Dict[str, float]:
        area = project_parameters.get("total_area", 2000)
        stories = project_parameters.get("stories", 1)
        
        # Base quantities per sq ft
        quantities = {
            "concrete": area * 0.012,  # cubic yards
            "lumber": area * 0.008,    # thousand board feet  
            "roofing_material": (area / stories) * 1.1,  # sq ft (roof area)
            "drywall": area * 2.2,     # sq ft (walls + ceiling)
            "electrical": area * 0.1,  # linear feet per sq ft
            "plumbing": area * 0.075,  # linear feet per sq ft
            "insulation": area * 1.1,  # sq ft
            "flooring": area * 0.9     # sq ft (excluding bathrooms)
        }
        
        return quantities
        
    def calculate_complexity_multiplier(self, project_parameters: Dict[str, Any]) -> float:
        multiplier = 1.0
        
        # Foundation complexity
        foundation_type = project_parameters.get("foundation_type", "slab")
        if foundation_type == "basement":
            multiplier *= 1.3
        elif foundation_type == "crawlspace":
            multiplier *= 1.1
            
        # Story complexity
        stories = project_parameters.get("stories", 1)
        if stories > 1:
            multiplier *= (1.0 + (stories - 1) * 0.15)
            
        # Finish level
        finish_level = project_parameters.get("finish_level", "standard")
        finish_multipliers = {"economy": 0.85, "standard": 1.0, "luxury": 1.4}
        multiplier *= finish_multipliers.get(finish_level, 1.0)
        
        # Energy efficiency
        energy_target = project_parameters.get("energy_efficiency_target", "standard")
        if energy_target == "high":
            multiplier *= 1.1
        elif energy_target == "net_zero":
            multiplier *= 1.25
            
        return multiplier
        
    def get_cost_per_sqft_estimate(self, project_parameters: Dict[str, Any]) -> float:
        base_cost = 120  # Base cost per sq ft
        complexity_mult = self.calculate_complexity_multiplier(project_parameters)
        return base_cost * complexity_mult

class CommercialProject(BaseProjectType):
    def __init__(self):
        wbs_templates = [
            WorkBreakdownStructure(
                task_id="COM_001",
                task_name="Site Work",
                description="Excavation, utilities, and site preparation",
                duration_days=15,
                labor_hours={"equipment_operator": 80, "laborer": 120},
                materials={"gravel": 100, "sand": 50},
                equipment={"excavator": 10, "bulldozer": 8},
                dependencies=[]
            ),
            WorkBreakdownStructure(
                task_id="COM_002",
                task_name="Foundation & Structure",
                description="Concrete foundation and structural steel/concrete",
                duration_days=30,
                labor_hours={"ironworker": 200, "concrete_finisher": 150, "crane_operator": 100},
                materials={"concrete": 200, "steel": 50000, "rebar": 10000},
                equipment={"crane": 15, "concrete_pump": 8},
                dependencies=["COM_001"]
            ),
            WorkBreakdownStructure(
                task_id="COM_003", 
                task_name="Building Envelope",
                description="Exterior walls, roofing, and weatherproofing",
                duration_days=25,
                labor_hours={"roofer": 120, "glazier": 80, "mason": 160},
                materials={"roofing_membrane": 50, "curtain_wall": 200, "masonry": 500},
                equipment={"scissor_lift": 10},
                dependencies=["COM_002"]
            ),
            WorkBreakdownStructure(
                task_id="COM_004",
                task_name="MEP Rough-In",
                description="Mechanical, electrical, plumbing rough installation",
                duration_days=35,
                labor_hours={"electrician": 300, "plumber": 200, "hvac_tech": 250},
                materials={"electrical": 1000, "plumbing": 500, "hvac_equipment": 8},
                equipment={},
                dependencies=["COM_002"]
            ),
            WorkBreakdownStructure(
                task_id="COM_005",
                task_name="Interior Construction",
                description="Drywall, ceilings, flooring, and interior finishes",
                duration_days=40,
                labor_hours={"drywaller": 200, "flooring_installer": 120, "painter": 180},
                materials={"drywall": 300, "ceiling_tile": 100, "flooring": 150},
                equipment={"scissor_lift": 15},
                dependencies=["COM_004"]
            ),
            WorkBreakdownStructure(
                task_id="COM_006",
                task_name="MEP Finish",
                description="Final electrical, plumbing, and HVAC installation",
                duration_days=20,
                labor_hours={"electrician": 150, "plumber": 100, "hvac_tech": 120},
                materials={"fixtures": 50, "controls": 25},
                equipment={},
                dependencies=["COM_005"]
            )
        ]
        
        config = ProjectTypeConfig(
            name="Commercial Construction",
            description="Office buildings, retail, warehouses, manufacturing",
            typical_size_range=(5000, 200000),  # sq ft
            typical_duration_range=(8, 48),     # months
            complexity_factors={
                "structural_system": 0.2,
                "mep_complexity": 0.25,
                "finish_level": 0.15,
                "occupancy_type": 0.15,
                "height": 0.1
            },
            default_wbs=wbs_templates,
            cost_drivers=["area", "structural_system", "mep_complexity", "finish_level"],
            risk_categories=["design_changes", "permit_delays", "material_escalation", "coordination"]
        )
        super().__init__(config)
        
    def generate_wbs(self, project_parameters: Dict[str, Any]) -> List[WorkBreakdownStructure]:
        wbs = self.config.default_wbs.copy()
        
        area_multiplier = project_parameters.get("total_area", 20000) / 20000.0
        height_factor = project_parameters.get("stories", 1)
        
        for task in wbs:
            # Scale by area
            for trade in task.labor_hours:
                task.labor_hours[trade] *= area_multiplier
            for material in task.materials:
                task.materials[material] *= area_multiplier
                
            # Height adjustments
            if task.task_id in ["COM_002", "COM_003"]:  # Structure and envelope
                task.duration_days = int(task.duration_days * (1 + (height_factor - 1) * 0.3))
                
        return wbs
        
    def estimate_quantities(self, project_parameters: Dict[str, Any]) -> Dict[str, float]:
        area = project_parameters.get("total_area", 20000)
        
        quantities = {
            "concrete": area * 0.02,   # cubic yards
            "steel": area * 2.5,       # pounds
            "roofing_membrane": area * 0.05,  # sq ft of roof per floor area
            "drywall": area * 1.8,     # sq ft
            "electrical": area * 0.15, # linear feet per sq ft
            "plumbing": area * 0.05,   # linear feet per sq ft  
            "hvac_ductwork": area * 0.1  # linear feet per sq ft
        }
        
        return quantities
        
    def calculate_complexity_multiplier(self, project_parameters: Dict[str, Any]) -> float:
        multiplier = 1.0
        
        # Structural system
        structural = project_parameters.get("structural_system", "steel")
        if structural == "concrete":
            multiplier *= 1.15
        elif structural == "mass_timber":
            multiplier *= 1.25
            
        # MEP complexity
        mep_level = project_parameters.get("mep_complexity", "standard")
        if mep_level == "high":
            multiplier *= 1.3
        elif mep_level == "specialized":
            multiplier *= 1.6
            
        # Occupancy type
        occupancy = project_parameters.get("occupancy_type", "office")
        occupancy_mults = {
            "office": 1.0, "retail": 1.1, "warehouse": 0.8, 
            "manufacturing": 1.4, "healthcare": 1.8, "laboratory": 2.2
        }
        multiplier *= occupancy_mults.get(occupancy, 1.0)
        
        return multiplier
        
    def get_cost_per_sqft_estimate(self, project_parameters: Dict[str, Any]) -> float:
        base_cost = 180
        complexity_mult = self.calculate_complexity_multiplier(project_parameters)
        return base_cost * complexity_mult

class TransportationProject(BaseProjectType):
    def __init__(self):
        wbs_templates = [
            WorkBreakdownStructure(
                task_id="TRANS_001",
                task_name="Survey & Design",
                description="Surveying, design verification, and stakeout",
                duration_days=20,
                labor_hours={"surveyor": 120, "engineer": 80},
                materials={},
                equipment={"survey_equipment": 15},
                dependencies=[]
            ),
            WorkBreakdownStructure(
                task_id="TRANS_002",
                task_name="Right-of-Way Preparation",
                description="Clearing, grubbing, and utility relocation",
                duration_days=30,
                labor_hours={"equipment_operator": 200, "laborer": 150},
                materials={},
                equipment={"bulldozer": 20, "excavator": 15},
                dependencies=["TRANS_001"]
            ),
            WorkBreakdownStructure(
                task_id="TRANS_003",
                task_name="Earthwork",
                description="Excavation, embankment, and grading",
                duration_days=45,
                labor_hours={"equipment_operator": 300, "grade_checker": 100},
                materials={"aggregate_base": 500},
                equipment={"excavator": 30, "haul_truck": 25},
                dependencies=["TRANS_002"]
            ),
            WorkBreakdownStructure(
                task_id="TRANS_004",
                task_name="Drainage Systems",
                description="Storm drainage, culverts, and detention",
                duration_days=25,
                labor_hours={"pipe_layer": 150, "equipment_operator": 100},
                materials={"storm_pipe": 1000, "catch_basins": 20},
                equipment={"excavator": 15, "crane": 5},
                dependencies=["TRANS_003"]
            ),
            WorkBreakdownStructure(
                task_id="TRANS_005",
                task_name="Pavement Structure",
                description="Base course and pavement installation",
                duration_days=35,
                labor_hours={"paving_crew": 250, "equipment_operator": 200},
                materials={"asphalt": 800, "base_course": 1200},
                equipment={"paver": 20, "roller": 20},
                dependencies=["TRANS_004"]
            ),
            WorkBreakdownStructure(
                task_id="TRANS_006",
                task_name="Traffic Control & Signage",
                description="Signals, signs, and pavement markings",
                duration_days=15,
                labor_hours={"electrician": 80, "sign_installer": 60},
                materials={"traffic_signals": 8, "signage": 30},
                equipment={"bucket_truck": 8},
                dependencies=["TRANS_005"]
            )
        ]
        
        config = ProjectTypeConfig(
            name="Transportation Infrastructure",
            description="Roads, bridges, tunnels, railways, transit systems",
            typical_size_range=(1, 50),  # miles
            typical_duration_range=(12, 72),  # months
            complexity_factors={
                "terrain": 0.25,
                "traffic_control": 0.15,
                "environmental": 0.2,
                "utility_conflicts": 0.15,
                "bridge_structures": 0.25
            },
            default_wbs=wbs_templates,
            cost_drivers=["length", "width", "pavement_type", "bridge_count", "terrain"],
            risk_categories=["weather", "environmental", "geotechnical", "utility_conflicts", "traffic_maintenance"]
        )
        super().__init__(config)
        
    def generate_wbs(self, project_parameters: Dict[str, Any]) -> List[WorkBreakdownStructure]:
        wbs = self.config.default_wbs.copy()
        
        length_miles = project_parameters.get("length_miles", 5)
        lanes = project_parameters.get("lanes", 4)
        
        for task in wbs:
            # Scale by project length
            length_multiplier = length_miles / 5.0
            for trade in task.labor_hours:
                task.labor_hours[trade] *= length_multiplier
            for material in task.materials:
                task.materials[material] *= length_multiplier
                
            # Lane width factor
            lane_multiplier = lanes / 4.0
            for material in task.materials:
                if material in ["asphalt", "base_course", "aggregate_base"]:
                    task.materials[material] *= lane_multiplier
                    
        # Add bridge tasks if needed
        if project_parameters.get("bridge_count", 0) > 0:
            bridge_task = WorkBreakdownStructure(
                task_id="TRANS_007",
                task_name="Bridge Construction",
                description="Bridge foundations, structures, and deck",
                duration_days=60,
                labor_hours={"ironworker": 300, "concrete_finisher": 200, "crane_operator": 150},
                materials={"concrete": 500, "steel": 80000, "bridge_deck": 100},
                equipment={"crane": 40, "concrete_pump": 15},
                dependencies=["TRANS_003"]
            )
            wbs.append(bridge_task)
            
        return wbs
        
    def estimate_quantities(self, project_parameters: Dict[str, Any]) -> Dict[str, float]:
        length_miles = project_parameters.get("length_miles", 5)
        lanes = project_parameters.get("lanes", 4) 
        
        # Convert to linear feet
        length_ft = length_miles * 5280
        lane_width_ft = 12  # Standard lane width
        
        quantities = {
            "earthwork": length_ft * lanes * lane_width_ft * 2,  # cubic yards (assume 2 ft depth)
            "aggregate_base": length_ft * lanes * lane_width_ft * 0.5,  # tons
            "asphalt": length_ft * lanes * lane_width_ft * 0.15,  # tons
            "storm_pipe": length_ft * 0.1,  # linear feet of pipe per project foot
            "signage": length_miles * 8,    # signs per mile
        }
        
        return quantities
        
    def calculate_complexity_multiplier(self, project_parameters: Dict[str, Any]) -> float:
        multiplier = 1.0
        
        # Terrain complexity
        terrain = project_parameters.get("terrain", "flat")
        terrain_mults = {"flat": 1.0, "rolling": 1.2, "mountainous": 1.6}
        multiplier *= terrain_mults.get(terrain, 1.0)
        
        # Environmental sensitivity
        if project_parameters.get("environmental_sensitivity", 0) > 0.5:
            multiplier *= 1.3
            
        # Bridge complexity
        bridge_count = project_parameters.get("bridge_count", 0)
        multiplier *= (1.0 + bridge_count * 0.4)
        
        # Urban vs rural
        setting = project_parameters.get("setting", "rural")
        if setting == "urban":
            multiplier *= 1.4  # Traffic control, utility conflicts
            
        return multiplier
        
    def get_cost_per_sqft_estimate(self, project_parameters: Dict[str, Any]) -> float:
        # Transportation uses cost per mile, convert to equivalent
        base_cost_per_mile = 2000000  # $2M per mile base
        complexity_mult = self.calculate_complexity_multiplier(project_parameters)
        cost_per_mile = base_cost_per_mile * complexity_mult
        
        # Convert to "cost per sq ft" equivalent for consistency
        # (This is somewhat artificial but maintains interface)
        length_miles = project_parameters.get("length_miles", 5)
        lanes = project_parameters.get("lanes", 4)
        equivalent_sqft = length_miles * lanes * 12 * 5280  # miles * lanes * width * ft_per_mile
        
        return (cost_per_mile * length_miles) / equivalent_sqft

class WaterInfrastructureProject(BaseProjectType):
    def __init__(self):
        wbs_templates = [
            WorkBreakdownStructure(
                task_id="WATER_001",
                task_name="Site Investigation",
                description="Geotechnical, environmental, and utility investigation",
                duration_days=25,
                labor_hours={"engineer": 120, "technician": 200},
                materials={},
                equipment={"drill_rig": 10, "survey_equipment": 15},
                dependencies=[]
            ),
            WorkBreakdownStructure(
                task_id="WATER_002", 
                task_name="Excavation & Shoring",
                description="Trench excavation and safety shoring systems",
                duration_days=40,
                labor_hours={"equipment_operator": 250, "laborer": 200},
                materials={"shoring_materials": 100},
                equipment={"excavator": 30, "trench_box": 25},
                dependencies=["WATER_001"]
            ),
            WorkBreakdownStructure(
                task_id="WATER_003",
                task_name="Pipeline Installation",
                description="Pipe laying, joints, and connections",
                duration_days=50,
                labor_hours={"pipe_layer": 300, "welder": 150},
                materials={"water_pipe": 5000, "fittings": 200},
                equipment={"crane": 20, "pipe_layer": 30},
                dependencies=["WATER_002"]
            ),
            WorkBreakdownStructure(
                task_id="WATER_004",
                task_name="Pumping Stations",
                description="Pump station construction and equipment",
                duration_days=35,
                labor_hours={"electrician": 200, "mechanical_tech": 180, "concrete_finisher": 150},
                materials={"concrete": 150, "pumps": 4, "electrical_controls": 1},
                equipment={"crane": 15},
                dependencies=["WATER_003"]
            ),
            WorkBreakdownStructure(
                task_id="WATER_005",
                task_name="Treatment Systems",
                description="Water treatment equipment and processes",
                duration_days=45,
                labor_hours={"process_tech": 250, "electrician": 180, "instrument_tech": 200},
                materials={"treatment_equipment": 8, "instrumentation": 50},
                equipment={"crane": 10},
                dependencies=["WATER_004"]
            ),
            WorkBreakdownStructure(
                task_id="WATER_006",
                task_name="Testing & Commissioning",
                description="System testing, disinfection, and startup",
                duration_days=20,
                labor_hours={"engineer": 80, "technician": 160},
                materials={"testing_materials": 10},
                equipment={},
                dependencies=["WATER_005"]
            )
        ]
        
        config = ProjectTypeConfig(
            name="Water Infrastructure", 
            description="Water supply, wastewater treatment, stormwater systems",
            typical_size_range=(1, 100),  # MGD capacity or pipeline miles
            typical_duration_range=(18, 60),  # months
            complexity_factors={
                "treatment_level": 0.3,
                "pipe_depth": 0.2,
                "soil_conditions": 0.2, 
                "live_connections": 0.15,
                "regulatory_requirements": 0.15
            },
            default_wbs=wbs_templates,
            cost_drivers=["capacity", "treatment_complexity", "pipe_diameter", "depth", "soil_conditions"],
            risk_categories=["geotechnical", "environmental", "regulatory", "live_tie_ins", "contamination"]
        )
        super().__init__(config)
        
    def generate_wbs(self, project_parameters: Dict[str, Any]) -> List[WorkBreakdownStructure]:
        wbs = self.config.default_wbs.copy()
        
        capacity = project_parameters.get("capacity_mgd", 10)  # Million gallons per day
        pipe_miles = project_parameters.get("pipe_miles", 5)
        
        # Scale by capacity/size
        capacity_multiplier = capacity / 10.0
        length_multiplier = pipe_miles / 5.0
        
        for task in wbs:
            if task.task_id in ["WATER_004", "WATER_005"]:  # Treatment related
                for trade in task.labor_hours:
                    task.labor_hours[trade] *= capacity_multiplier
                for material in task.materials:
                    task.materials[material] *= capacity_multiplier
            elif task.task_id in ["WATER_002", "WATER_003"]:  # Pipeline related  
                for trade in task.labor_hours:
                    task.labor_hours[trade] *= length_multiplier
                for material in task.materials:
                    task.materials[material] *= length_multiplier
                    
        return wbs
        
    def estimate_quantities(self, project_parameters: Dict[str, Any]) -> Dict[str, float]:
        capacity = project_parameters.get("capacity_mgd", 10)
        pipe_miles = project_parameters.get("pipe_miles", 5)
        
        quantities = {
            "excavation": pipe_miles * 5280 * 8 * 8,  # cubic yards (8ft wide x 8ft deep)
            "water_pipe": pipe_miles * 5280,  # linear feet
            "concrete": capacity * 15,  # cubic yards for structures
            "treatment_equipment": capacity / 5,  # major equipment units
            "electrical": capacity * 1000,  # linear feet of electrical
            "instrumentation": capacity * 50  # control points
        }
        
        return quantities
        
    def calculate_complexity_multiplier(self, project_parameters: Dict[str, Any]) -> float:
        multiplier = 1.0
        
        # Treatment complexity
        treatment_level = project_parameters.get("treatment_level", "secondary")
        treatment_mults = {"primary": 0.8, "secondary": 1.0, "tertiary": 1.4, "advanced": 1.8}
        multiplier *= treatment_mults.get(treatment_level, 1.0)
        
        # Pipe depth complexity
        depth = project_parameters.get("average_depth_ft", 8)
        if depth > 15:
            multiplier *= 1.4
        elif depth > 10:
            multiplier *= 1.2
            
        # Soil conditions
        soil = project_parameters.get("soil_conditions", "normal")
        soil_mults = {"good": 0.9, "normal": 1.0, "poor": 1.3, "rock": 1.6}
        multiplier *= soil_mults.get(soil, 1.0)
        
        # Live connections complexity
        if project_parameters.get("live_connections", False):
            multiplier *= 1.3
            
        return multiplier
        
    def get_cost_per_sqft_estimate(self, project_parameters: Dict[str, Any]) -> float:
        # Water projects use cost per MGD or per mile
        base_cost_per_mgd = 5000000  # $5M per MGD
        complexity_mult = self.calculate_complexity_multiplier(project_parameters) 
        cost_per_mgd = base_cost_per_mgd * complexity_mult
        
        # Convert to equivalent area cost for interface consistency
        capacity = project_parameters.get("capacity_mgd", 10)
        equivalent_sqft = capacity * 10000  # Artificial conversion factor
        
        return (cost_per_mgd * capacity) / equivalent_sqft

# Project type registry
PROJECT_TYPES = {
    "residential": ResidentialProject,
    "commercial": CommercialProject,
    "transportation": TransportationProject,
    "water_infrastructure": WaterInfrastructureProject
}

def get_project_type(project_type_name: str) -> BaseProjectType:
    """Factory function to create project type instances"""
    if project_type_name not in PROJECT_TYPES:
        raise ValueError(f"Unknown project type: {project_type_name}")
    
    return PROJECT_TYPES[project_type_name]()

def get_available_project_types() -> List[str]:
    """Get list of available project type names"""
    return list(PROJECT_TYPES.keys())