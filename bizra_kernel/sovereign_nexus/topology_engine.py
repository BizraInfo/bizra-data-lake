"""
47-Discipline Topology Engine for BIZRA Sovereign Nexus

Implements the 7-layer, 47-discipline mapping for cross-domain synthesis.
Maps various academic and practical disciplines to BIZRA subsystems.
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from bizra_kernel.abstraction_elevator import DomainType


class Layer(Enum):
    """The seven foundational layers of knowledge in the BIZRA system."""
    L1_FOUNDATION = "L1_FOUNDATION"
    L2_PHYSICALITY = "L2_PHYSICALITY"
    L3_SOCIETAL = "L3_SOCIETAL"
    L4_CREATIVE = "L4_CREATIVE"
    L5_TRANSCENDENT = "L5_TRANSCENDENT"
    L6_APPLIED = "L6_APPLIED"
    L7_SYNTHESIS = "L7_SYNTHESIS"


@dataclass
class DisciplineMapping:
    """Represents a mapping between a discipline and BIZRA subsystems."""
    name: str
    layer: Layer
    bizra_subsystem: str
    description: str
    related_disciplines: List[str]


class DisciplineTopologyEngine:
    """Manages the 47-discipline topology and cross-layer bridge detection."""
    
    def __init__(self):
        self.layers = {
            Layer.L1_FOUNDATION: [
                "formal_logic", "number_theory", "set_theory",
                "graph_theory", "information_theory", "cybernetics", "systems_theory"
            ],
            Layer.L2_PHYSICALITY: [
                "thermodynamics", "quantum_mechanics", "neuroscience",
                "evolutionary_biology", "ecological_science", "materials_science"
            ],
            Layer.L3_SOCIETAL: [
                "economics", "game_theory", "sociology", "anthropology",
                "political_science", "law", "linguistics", "psychology"
            ],
            Layer.L4_CREATIVE: [
                "architecture", "design_thinking", "music_theory",
                "narratology", "visual_arts", "poetics"
            ],
            Layer.L5_TRANSCENDENT: [
                "ethics", "theology", "epistemology",
                "metaphysics", "aesthetics", "phenomenology"
            ],
            Layer.L6_APPLIED: [
                "computer_science", "cryptography", "data_science",
                "network_engineering", "robotics", "energy_engineering"
            ],
            Layer.L7_SYNTHESIS: [
                "chaos_theory", "complexity_science", "semiotics", "decision_theory",
                "pedagogy", "history", "futures_studies", "ihsan_studies"
            ]
        }
        
        # Map disciplines to BIZRA subsystems
        self.discipline_to_subsystem = self._create_discipline_mappings()
        
        # Cross-layer bridges
        self.bridges = self._identify_bridges()
    
    def _create_discipline_mappings(self) -> Dict[str, DisciplineMapping]:
        """Creates mappings from disciplines to BIZRA subsystems."""
        mappings = {}
        
        # L1 Foundation disciplines -> Core Logic/Reasoning
        for disc in self.layers[Layer.L1_FOUNDATION]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L1_FOUNDATION,
                bizra_subsystem="reasoning_engine",
                description=f"Foundation discipline for logical and mathematical reasoning: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L1_FOUNDATION] if d != disc]
            )
        
        # L2 Physicality disciplines -> Reality Modeling
        for disc in self.layers[Layer.L2_PHYSICALITY]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L2_PHYSICALITY,
                bizra_subsystem="reality_modeling",
                description=f"Physical reality modeling discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L2_PHYSICALITY] if d != disc]
            )
        
        # L3 Societal disciplines -> Social Dynamics
        for disc in self.layers[Layer.L3_SOCIETAL]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L3_SOCIETAL,
                bizra_subsystem="social_dynamics",
                description=f"Societal dynamics and governance discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L3_SOCIETAL] if d != disc]
            )
        
        # L4 Creative disciplines -> Creative Synthesis
        for disc in self.layers[Layer.L4_CREATIVE]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L4_CREATIVE,
                bizra_subsystem="creative_synthesis",
                description=f"Creative synthesis and design discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L4_CREATIVE] if d != disc]
            )
        
        # L5 Transcendent disciplines -> Ethical/Governance
        for disc in self.layers[Layer.L5_TRANSCENDENT]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L5_TRANSCENDENT,
                bizra_subsystem="ethical_governance",
                description=f"Ethical and transcendent discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L5_TRANSCENDENT] if d != disc]
            )
        
        # L6 Applied disciplines -> Applied Systems
        for disc in self.layers[Layer.L6_APPLIED]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L6_APPLIED,
                bizra_subsystem="applied_systems",
                description=f"Applied systems and engineering discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L6_APPLIED] if d != disc]
            )
        
        # L7 Synthesis disciplines -> Meta Integration
        for disc in self.layers[Layer.L7_SYNTHESIS]:
            mappings[disc] = DisciplineMapping(
                name=disc,
                layer=Layer.L7_SYNTHESIS,
                bizra_subsystem="meta_integration",
                description=f"Meta-integration and synthesis discipline: {disc}",
                related_disciplines=[d for d in self.layers[Layer.L7_SYNTHESIS] if d != disc]
            )
        
        return mappings
    
    def _identify_bridges(self) -> Dict[str, List[str]]:
        """Identifies bridges between layers and disciplines."""
        bridges = {}
        
        # Example bridges between adjacent layers
        bridges["L1-L2"] = [
            ("formal_logic", "thermodynamics"),  # Logical foundations of physical laws
            ("information_theory", "neuroscience")  # Information processing in brains
        ]
        
        bridges["L2-L3"] = [
            ("neuroscience", "psychology"),  # Neural basis of social behavior
            ("evolutionary_biology", "sociology")  # Evolutionary basis of society
        ]
        
        bridges["L3-L4"] = [
            ("linguistics", "narratology"),  # Language and storytelling
            ("psychology", "visual_arts")  # Psychological aspects of art
        ]
        
        bridges["L4-L5"] = [
            ("architecture", "aesthetics"),  # Architectural aesthetics
            ("poetics", "ethics")  # Ethics of narrative
        ]
        
        bridges["L5-L6"] = [
            ("ethics", "cryptography"),  # Ethical implications of privacy tech
            ("epistemology", "computer_science")  # Foundations of computation
        ]
        
        bridges["L6-L7"] = [
            ("complexity_science", "systems_theory"),  # Complex systems approaches
            ("decision_theory", "futures_studies")  # Decision making for futures
        ]
        
        return bridges
    
    def get_discipline_mapping(self, discipline: str) -> Optional[DisciplineMapping]:
        """Returns the mapping for a given discipline."""
        return self.discipline_to_subsystem.get(discipline)
    
    def get_layer_disciplines(self, layer: Layer) -> List[str]:
        """Returns all disciplines in a given layer."""
        return self.layers.get(layer, [])
    
    def get_cross_layer_bridges(self, layer1: Layer, layer2: Layer) -> List[tuple]:
        """Returns bridges between two layers."""
        key = f"{layer1.value[:2]}-{layer2.value[:2]}"
        return self.bridges.get(key, [])
    
    def map_to_abstraction_domain(self, discipline: str) -> Optional[DomainType]:
        """Maps a discipline to an AbstractionElevator DomainType."""
        # This integrates with the existing AbstractionElevator
        discipline_to_domain_map = {
            # L1 Foundation
            "formal_logic": DomainType.LOGIC,
            "number_theory": DomainType.TECHNICAL,
            "set_theory": DomainType.TECHNICAL,
            "graph_theory": DomainType.TECHNICAL,
            "information_theory": DomainType.TECHNICAL,
            "cybernetics": DomainType.TECHNICAL,
            "systems_theory": DomainType.TECHNICAL,
            
            # L2 Physicality
            "thermodynamics": DomainType.TECHNICAL,
            "quantum_mechanics": DomainType.TECHNICAL,
            "neuroscience": DomainType.TECHNICAL,
            "evolutionary_biology": DomainType.TECHNICAL,
            "ecological_science": DomainType.TECHNICAL,
            "materials_science": DomainType.TECHNICAL,
            
            # L3 Societal
            "economics": DomainType.ECONOMIC,
            "game_theory": DomainType.ECONOMIC,
            "sociology": DomainType.SOCIAL,
            "anthropology": DomainType.SOCIAL,
            "political_science": DomainType.SOCIAL,
            "law": DomainType.ETHICAL,
            "linguistics": DomainType.SOCIAL,
            "psychology": DomainType.SOCIAL,
            
            # L4 Creative
            "architecture": DomainType.TECHNICAL,
            "design_thinking": DomainType.TECHNICAL,
            "music_theory": DomainType.TECHNICAL,
            "narratology": DomainType.SOCIAL,
            "visual_arts": DomainType.SOCIAL,
            "poetics": DomainType.SOCIAL,
            
            # L5 Transcendent
            "ethics": DomainType.ETHICAL,
            "theology": DomainType.ETHICAL,
            "epistemology": DomainType.ETHICAL,
            "metaphysics": DomainType.ETHICAL,
            "aesthetics": DomainType.ETHICAL,
            "phenomenology": DomainType.ETHICAL,
            
            # L6 Applied
            "computer_science": DomainType.TECHNICAL,
            "cryptography": DomainType.TECHNICAL,
            "data_science": DomainType.TECHNICAL,
            "network_engineering": DomainType.TECHNICAL,
            "robotics": DomainType.TECHNICAL,
            "energy_engineering": DomainType.TECHNICAL,
            
            # L7 Synthesis
            "chaos_theory": DomainType.TECHNICAL,
            "complexity_science": DomainType.TECHNICAL,
            "semiotics": DomainType.SOCIAL,
            "decision_theory": DomainType.TECHNICAL,
            "pedagogy": DomainType.SOCIAL,
            "history": DomainType.SOCIAL,
            "futures_studies": DomainType.TEMPORAL,
            "ihsan_studies": DomainType.ETHICAL
        }
        
        return discipline_to_domain_map.get(discipline)
    
    def find_synergies(self, disciplines: List[str]) -> List[tuple]:
        """Finds synergies between provided disciplines."""
        synergies = []
        
        for i, disc1 in enumerate(disciplines):
            for j, disc2 in enumerate(disciplines[i+1:], i+1):
                # Look for cross-layer bridges
                for bridge in self.bridges.values():
                    for b_disc1, b_disc2 in bridge:
                        if (disc1 == b_disc1 and disc2 == b_disc2) or (disc1 == b_disc2 and disc2 == b_disc1):
                            synergies.append((disc1, disc2, "Cross-layer bridge"))
        
        return synergies