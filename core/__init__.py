"""
Multi-Agent PAS Framework
=========================

A flexible framework for multi-agent LLM experiments with:
- Configurable agent definitions
- Flexible interaction topologies
- Linear probe monitoring
- Activation injection
- Comprehensive visualization
- Multi-category probe training
- Multi-model architecture support
- LLM-DNA phylogenetic integration (built-in)
"""

from .agent_registry import AgentRegistry, AgentConfig, AgentBehavior, create_court_agents
from .interaction_graph import InteractionGraph, InteractionEdge, ConversationState, EdgeType
from .model_compatibility import (
    ModelCompatibility,
    ModelFamily,
    ArchitectureConfig,
    load_model_and_tokenizer,
    get_device,
)
from .orchestrator import (
    MultiAgentOrchestrator,
    ProbeConfig,
    InjectionConfig,
    AgentMetrics,
    ExperimentResult,
    quick_setup,
    create_orchestrator_from_config
)
from .probe_trainer import (
    MultiProbeTrainer,
    TrainedProbe,
    CategoryDatasets,
)
try:
    from .visualization import (
        ExperimentVisualizer,
        save_token_heatmap,
        visualize_from_json,
        calculate_ghost_divergence,
        calculate_repetition_rate,
        calculate_jaccard_distance,
    )
except ImportError:
    pass

# LLM-DNA utilities (built-in)
from .llm_dna import (
    DNASignature,
    PhylogeneticTree,
    DimensionalityReducer,
    DistanceMetric,
    compute_distance,
    compute_distance_matrix,
    build_tree_from_signatures,
    load_signatures_from_directory,
    generate_itol_colorstrip,
    generate_itol_labels,
)

# PAS-specific DNA integration
try:
    from .dna_integration import (
        PASSignature,
        SignatureExtractor,
        DNAExporter,
        PhylogeneticTreeBuilder,
        build_phylogenetic_tree,
        export_experiment_to_dna,
    )
except ImportError:
    pass

from .steered_agent import SteeredAgent

# Tools (RAG, fact-checking)
from .tools import (
    AgentTool,
    RAGTool,
    LegalSearchTool,
    ToolRegistry,
    ToolResult,
    create_court_tools,
)

# Court Orchestrator
from .court_orchestrator import (
    CourtOrchestrator,
    CourtTranscript,
    JudgeRuling,
    create_court_from_model,
    create_court_from_api,
)

__version__ = "2.0.0"
__all__ = [
    # Registry
    "AgentRegistry",
    "AgentConfig", 
    "AgentBehavior",
    # Graph
    "InteractionGraph",
    "InteractionEdge",
    "ConversationState",
    "EdgeType",
    # Model Compatibility
    "ModelCompatibility",
    "ModelFamily",
    "ArchitectureConfig",
    "load_model_and_tokenizer",
    "get_device",
    # Orchestrator
    "MultiAgentOrchestrator",
    "ProbeConfig",
    "InjectionConfig",
    "AgentMetrics",
    "ExperimentResult",
    # Probe Training
    "MultiProbeTrainer",
    "TrainedProbe",
    "CategoryDatasets",
    # Visualization
    "ExperimentVisualizer",
    "save_token_heatmap",
    "visualize_from_json",
    "calculate_ghost_divergence",
    "calculate_repetition_rate",
    "calculate_jaccard_distance",
    # LLM-DNA (built-in)
    "DNASignature",
    "PhylogeneticTree",
    "DimensionalityReducer",
    "DistanceMetric",
    "compute_distance",
    "compute_distance_matrix",
    "build_tree_from_signatures",
    "load_signatures_from_directory",
    "generate_itol_colorstrip",
    "generate_itol_labels",
    # PAS DNA Integration
    "PASSignature",
    "SignatureExtractor",
    "DNAExporter",
    "PhylogeneticTreeBuilder",
    "build_phylogenetic_tree",
    "export_experiment_to_dna",
    # Factory functions
    "quick_setup",
    "create_orchestrator_from_config",
    # Tools
    "AgentTool",
    "RAGTool",
    "LegalSearchTool",
    "ToolRegistry",
    "ToolResult",
    "create_court_tools",
    # Court Orchestrator
    "CourtOrchestrator",
    "CourtTranscript",
    "JudgeRuling",
    "create_court_agents",
    "create_court_from_model",
    "create_court_from_api",
]