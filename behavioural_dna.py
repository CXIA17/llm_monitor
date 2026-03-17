"""
Agent Behavioral DNA - Fine-Grained Feature Extraction
=======================================================

Extends the DNASignature paradigm to capture fine-grained behavioral signals
from multi-agent interactions (e.g., Federal Court simulations).

Captures:
- Token-level dynamics (entropy, gradients, peaks)
- Temporal patterns (drift, oscillation, momentum)
- Cross-agent interactions (reactivity, mirroring, dominance)
- Phase-specific signatures
- Linguistic markers
- Probe interactions
- Injection response characteristics
- Role compliance metrics
- Composite behavioral indices
- Anomaly detection

Compatible with existing DNASignature for phylogenetic analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

# Try to import scipy for advanced analysis
try:
    from scipy import stats
    from scipy.signal import find_peaks, correlate
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import existing DNASignature for compatibility
try:
    from dna_signature import DNASignature, DNAMetadata, DNACollection
    HAS_DNA_SIGNATURE = True
except ImportError:
    HAS_DNA_SIGNATURE = False


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AgentRole(str, Enum):
    """Agent roles in court simulation."""
    JUDGE = "judge"
    PLAINTIFF_COUNSEL = "plaintiff_counsel"
    DEFENSE_COUNSEL = "defense_counsel"
    JURY_FOREPERSON = "jury_foreperson"
    CLERK = "clerk"
    WITNESS = "witness"
    UNKNOWN = "unknown"


class CourtPhase(str, Enum):
    """Phases of court proceedings."""
    MOTIONS = "motions"
    OPENING = "opening"
    EXAMINATION = "examination"
    CLOSING = "closing"
    DELIBERATION = "deliberation"
    VERDICT = "verdict"


class TrajectoryArchetype(str, Enum):
    """Behavioral trajectory patterns."""
    STEADY_STATE = "steady_state"
    ESCALATING = "escalating"
    DE_ESCALATING = "de_escalating"
    VOLATILE = "volatile"
    PHASE_SHIFT = "phase_shift"
    DAMPENED_OSCILLATION = "dampened_oscillation"
    RUNAWAY = "runaway"
    UNKNOWN = "unknown"


# Feature dimension constants
TOKEN_FEATURES_DIM = 12
TEMPORAL_FEATURES_DIM = 10
CROSS_AGENT_FEATURES_DIM = 8
PHASE_FEATURES_DIM = 6  # Per phase
LINGUISTIC_FEATURES_DIM = 10
PROBE_INTERACTION_DIM = 6
INJECTION_FEATURES_DIM = 8
ROLE_COMPLIANCE_DIM = 6
COMPOSITE_INDICES_DIM = 8

TOTAL_DNA_DIM = 128  # Fixed dimension for ML compatibility

# SAE fingerprint constants (used when SAE enrichment is enabled)
SAE_SUMMARY_DIM = 16  # Compact SAE summary features appended to DNA vector
SAE_ENRICHED_DNA_DIM = TOTAL_DNA_DIM + SAE_SUMMARY_DIM  # 144D with SAE


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenLevelFeatures:
    """Features extracted at the token level within a single response."""
    
    # Basic statistics
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_range: float = 0.0
    
    # Distribution characteristics
    score_entropy: float = 0.0
    score_skewness: float = 0.0
    score_kurtosis: float = 0.0
    
    # Gradient analysis
    mean_gradient: float = 0.0
    max_gradient: float = 0.0
    gradient_sign_changes: int = 0
    
    # Peak analysis
    num_peaks: int = 0
    num_valleys: int = 0
    peak_prominence_mean: float = 0.0
    argmax_position: float = 0.0  # Normalized 0-1
    argmin_position: float = 0.0
    
    # Positional analysis
    first_token_bias: float = 0.0  # First 10% vs rest
    tail_bias: float = 0.0  # Last 10% vs rest
    middle_concentration: float = 0.0  # Middle 50% vs edges
    
    # Clustering
    score_bimodality: float = 0.0  # Dip test or similar
    dominant_mode: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.score_mean, self.score_std, self.score_range,
            self.score_entropy, self.score_skewness, self.score_kurtosis,
            self.mean_gradient, self.max_gradient, self.gradient_sign_changes / 100,
            self.argmax_position, self.argmin_position,
            self.first_token_bias,
        ], dtype=np.float32)


@dataclass
class TemporalFeatures:
    """Features tracking behavioral changes across rounds."""
    
    # Drift analysis
    drift_velocity: float = 0.0  # Rate of change
    drift_direction: float = 0.0  # +1 increasing, -1 decreasing
    total_drift: float = 0.0  # End - Start
    
    # Oscillation
    oscillation_frequency: float = 0.0  # Sign changes per round
    oscillation_amplitude: float = 0.0  # Mean absolute deviation
    
    # Momentum
    momentum: float = 0.0  # Autocorrelation lag-1
    acceleration: float = 0.0  # Second derivative
    
    # Convergence
    convergence_rate: float = 0.0  # Toward mean or extreme
    final_trajectory_slope: float = 0.0  # Last 3 points
    
    # Stability
    variance_trend: float = 0.0  # Is variance increasing or decreasing?
    recovery_rate: float = 0.0  # Return to baseline after perturbation
    
    # Pattern classification
    trajectory_archetype: TrajectoryArchetype = TrajectoryArchetype.UNKNOWN
    archetype_confidence: float = 0.0
    
    # Phase transitions
    num_phase_transitions: int = 0
    transition_locations: List[int] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.drift_velocity, self.drift_direction, self.total_drift,
            self.oscillation_frequency, self.oscillation_amplitude,
            self.momentum, self.acceleration,
            self.convergence_rate, self.final_trajectory_slope,
            self.variance_trend,
        ], dtype=np.float32)


@dataclass
class CrossAgentFeatures:
    """Features capturing interactions between agents."""
    
    # Per-opponent reactivity (how much this agent's score changes after opponent speaks)
    reactivity_scores: Dict[str, float] = field(default_factory=dict)
    mean_reactivity: float = 0.0
    max_reactivity: float = 0.0
    
    # Mirroring (correlation with other agents)
    mirroring_coefficients: Dict[str, float] = field(default_factory=dict)
    mean_mirroring: float = 0.0
    
    # Dominance (Granger causality - does this agent drive others?)
    dominance_index: float = 0.0
    influenced_by: List[str] = field(default_factory=list)
    influences: List[str] = field(default_factory=list)
    
    # Opposition
    opposition_strength: float = 0.0  # Mean |self - opponent|
    polarization_contribution: float = 0.0
    
    # Authority alignment
    judge_alignment: float = 0.0  # Correlation with judge scores
    deference_index: float = 0.0  # Score change toward judge's position
    
    # Contagion
    injection_contagion_rate: float = 0.0  # If another agent injected, how affected?
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.mean_reactivity, self.max_reactivity,
            self.mean_mirroring, self.dominance_index,
            self.opposition_strength, self.polarization_contribution,
            self.judge_alignment, self.injection_contagion_rate,
        ], dtype=np.float32)


@dataclass
class PhaseSpecificFeatures:
    """Features specific to each court phase."""
    
    phase: CourtPhase = CourtPhase.MOTIONS
    
    # Compliance with phase expectations
    phase_compliance_score: float = 0.0
    
    # Phase-specific metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Examples of phase-specific metrics:
    # MOTIONS: legal_citation_density, question_responsiveness, procedural_adherence
    # OPENING: narrative_vs_argument_ratio, chronological_coherence, promise_density
    # CLOSING: evidence_reference_density, instruction_application, emotional_intensity
    # DELIBERATION: evidence_recall, instruction_adherence, doubt_expression
    
    # Score statistics within this phase
    phase_mean_score: float = 0.0
    phase_score_variance: float = 0.0
    phase_score_trend: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        base = [
            self.phase_compliance_score,
            self.phase_mean_score,
            self.phase_score_variance,
            self.phase_score_trend,
        ]
        # Add top metrics
        metric_values = list(self.metrics.values())[:2] if self.metrics else [0.0, 0.0]
        while len(metric_values) < 2:
            metric_values.append(0.0)
        return np.array(base + metric_values, dtype=np.float32)


@dataclass
class LinguisticFeatures:
    """Features extracted from response text."""
    
    # Hedging and confidence
    hedging_index: float = 0.0  # "may", "might", "possibly", "arguably"
    assertiveness_score: float = 0.0  # "clearly", "certainly", "must", "is"
    certainty_ratio: float = 0.0  # assertive / (assertive + hedging)
    
    # Legal language (for court context)
    citation_density: float = 0.0  # Case citations per 100 tokens
    statute_references: float = 0.0
    precedent_invocations: float = 0.0
    
    # Emotional content
    emotional_valence: float = 0.0  # -1 to +1
    emotional_intensity: float = 0.0  # 0 to 1
    
    # Structural
    sentence_complexity: float = 0.0  # Mean words per sentence
    question_ratio: float = 0.0  # Questions / total sentences
    
    # Pronouns
    first_person_ratio: float = 0.0  # "I", "we"
    third_person_ratio: float = 0.0  # "the defendant", "the evidence"
    
    # Logical structure
    conditional_density: float = 0.0  # "if", "then", "would"
    negation_frequency: float = 0.0  # "not", "never", "no"
    causal_connector_density: float = 0.0  # "because", "therefore", "thus"
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.hedging_index, self.assertiveness_score, self.certainty_ratio,
            self.citation_density, self.emotional_valence, self.emotional_intensity,
            self.sentence_complexity, self.question_ratio,
            self.first_person_ratio, self.conditional_density,
        ], dtype=np.float32)


@dataclass 
class ProbeInteractionFeatures:
    """Features capturing how different probes interact."""
    
    # Multi-probe correlation matrix (upper triangle)
    probe_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Key composite signals
    confidence_persuasiveness_alignment: float = 0.0  # Positive = confident AND persuasive
    logic_emotion_balance: float = 0.0  # Legal precision vs emotional appeal
    calibration_score: float = 0.0  # Confidence vs factual accuracy
    
    # Probe dominance
    dominant_probe: str = ""
    dominant_probe_score: float = 0.0
    
    # Probe volatility (which probes vary most?)
    most_volatile_probe: str = ""
    probe_volatility_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.confidence_persuasiveness_alignment,
            self.logic_emotion_balance,
            self.calibration_score,
            self.dominant_probe_score,
            max(self.probe_volatility_scores.values()) if self.probe_volatility_scores else 0.0,
            len(self.probe_correlations) / 10,  # Normalized count
        ], dtype=np.float32)


@dataclass
class InjectionResponseFeatures:
    """Features characterizing response to behavioral injection."""
    
    # Was this agent injected?
    is_injected: bool = False
    injection_strength: float = 0.0
    injection_probe: str = ""
    
    # Response characteristics (if injected)
    absorption_rate: float = 0.0  # How quickly score changed
    peak_effect: float = 0.0  # Maximum score deviation from baseline
    decay_half_life: float = 0.0  # Rounds until effect halves
    residual_effect: float = 0.0  # Persistent change after decay
    
    # Amplification
    amplification_factor: float = 0.0  # Did agent amplify beyond injection?
    runaway_detected: bool = False
    
    # Response characteristics (if NOT injected but others were)
    cross_contamination: float = 0.0  # Score change due to others' injection
    resistance_score: float = 0.0  # Stability despite injection in system
    
    # Recovery
    recovery_time: int = 0  # Rounds to return to baseline
    full_recovery: bool = False
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            float(self.is_injected), self.injection_strength,
            self.absorption_rate, self.peak_effect,
            self.decay_half_life, self.amplification_factor,
            self.cross_contamination, self.resistance_score,
        ], dtype=np.float32)


@dataclass
class RoleComplianceFeatures:
    """Features measuring adherence to role expectations."""
    
    role: AgentRole = AgentRole.UNKNOWN
    
    # Overall compliance
    role_compliance_score: float = 0.0
    
    # Role-specific metrics
    role_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Examples per role:
    # JUDGE: neutrality_maintenance, procedural_intervention_appropriateness, 
    #        question_balance, instruction_clarity
    # PLAINTIFF: burden_awareness, narrative_construction, evidence_integration,
    #            rebuttal_effectiveness
    # DEFENSE: doubt_creation, counter_narrative_strength, prosecution_challenge,
    #          rights_invocation
    # JURY: instruction_following, evidence_focus, deliberation_facilitation
    
    # Deviation from role
    out_of_role_instances: int = 0
    role_confusion_score: float = 0.0  # Did agent act like another role?
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        base = [
            self.role_compliance_score,
            self.out_of_role_instances / 10,
            self.role_confusion_score,
        ]
        # Add top role metrics
        metric_values = list(self.role_metrics.values())[:3] if self.role_metrics else [0.0, 0.0, 0.0]
        while len(metric_values) < 3:
            metric_values.append(0.0)
        return np.array(base + metric_values, dtype=np.float32)


@dataclass
class CompositeIndices:
    """High-level behavioral indices computed from other features."""
    
    # Attorney effectiveness (for counsel roles)
    advocacy_effectiveness_index: float = 0.0
    
    # Judicial quality (for judge role)
    judicial_quality_index: float = 0.0
    
    # Deliberation quality (for jury role)
    deliberation_quality_index: float = 0.0
    
    # Universal indices
    calibration_index: float = 0.0  # Confidence vs accuracy alignment
    manipulation_susceptibility_index: float = 0.0
    behavioral_stability_index: float = 0.0
    engagement_index: float = 0.0  # Responsiveness and participation
    
    # System-level (computed across all agents)
    polarization_index: float = 0.0  # How adversarial is the debate?
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector."""
        return np.array([
            self.advocacy_effectiveness_index,
            self.judicial_quality_index,
            self.deliberation_quality_index,
            self.calibration_index,
            self.manipulation_susceptibility_index,
            self.behavioral_stability_index,
            self.engagement_index,
            self.polarization_index,
        ], dtype=np.float32)


@dataclass
class SAEFeatures:
    """
    Features derived from Sparse Autoencoder (SAE) analysis.

    When a reader LLM + pretrained SAE is available, these features capture
    high-level semantic properties of the agent's responses that go beyond
    hand-crafted linguistic markers. Each feature represents the activation
    frequency of a learned SAE latent concept across the agent's responses.

    This integrates with the SAE fingerprinting pipeline (core.sae_fingerprint)
    to provide per-agent SAE-based behavioral signatures.
    """

    # Whether SAE features have been computed
    is_populated: bool = False

    # Summary statistics from SAE activation frequencies
    n_active_features: int = 0            # How many SAE latents fired at least once
    mean_activation_frequency: float = 0.0  # Mean frequency across active latents
    activation_sparsity: float = 0.0       # Fraction of latents that never fired

    # Top SAE feature frequencies (sorted descending)
    # These are the most characteristic SAE latent activations for this agent
    top_feature_frequencies: List[float] = field(default_factory=list)
    top_feature_indices: List[int] = field(default_factory=list)
    top_feature_labels: List[str] = field(default_factory=list)

    # Distributional properties of the SAE activation pattern
    frequency_entropy: float = 0.0         # Entropy of the frequency distribution
    frequency_std: float = 0.0             # Std of frequencies across active latents
    frequency_skewness: float = 0.0        # Skewness of frequency distribution
    frequency_kurtosis: float = 0.0        # Kurtosis (peakedness)
    frequency_gini: float = 0.0            # Gini coefficient (inequality of activations)

    # Cross-feature statistics
    n_unique_features: int = 0             # Features unique to this agent vs population
    feature_overlap_with_population: float = 0.0  # Jaccard with population average

    # Full frequency vector (for advanced analysis, not included in to_vector)
    _full_frequencies: Optional[np.ndarray] = field(default=None, repr=False)

    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector (SAE_SUMMARY_DIM dimensions)."""
        # Use top-8 feature frequencies + 8 distributional stats = 16D
        top_freqs = list(self.top_feature_frequencies[:8])
        while len(top_freqs) < 8:
            top_freqs.append(0.0)

        stats = [
            self.mean_activation_frequency,
            self.activation_sparsity,
            self.frequency_entropy / 16.0,  # Normalize (max ~16 bits)
            self.frequency_std,
            self.frequency_skewness / 10.0,  # Normalize
            self.frequency_kurtosis / 100.0,  # Normalize
            self.frequency_gini,
            self.feature_overlap_with_population,
        ]

        return np.array(top_freqs + stats, dtype=np.float32)

    @classmethod
    def from_fingerprint(
        cls,
        frequencies: np.ndarray,
        feature_labels: Optional[Dict[int, str]] = None,
        population_frequencies: Optional[np.ndarray] = None,
        top_n: int = 8,
    ) -> 'SAEFeatures':
        """
        Create SAEFeatures from an SAE fingerprint frequency vector.

        Args:
            frequencies: Array of shape (d_sae,) with activation frequencies
            feature_labels: Optional mapping of latent index to label
            population_frequencies: Optional population-average frequencies
                for computing uniqueness metrics
            top_n: Number of top features to store

        Returns:
            Populated SAEFeatures instance
        """
        feature_labels = feature_labels or {}

        # Active features
        active_mask = frequencies > 0
        n_active = int(np.sum(active_mask))
        active_freqs = frequencies[active_mask]

        # Top features
        sorted_indices = np.argsort(frequencies)[::-1][:top_n]
        top_freqs = [float(frequencies[i]) for i in sorted_indices]
        top_indices = [int(i) for i in sorted_indices]
        top_labels = [feature_labels.get(int(i), f"SAE Latent #{i}") for i in sorted_indices]

        # Distributional stats
        mean_freq = float(np.mean(active_freqs)) if n_active > 0 else 0.0
        sparsity = 1.0 - (n_active / len(frequencies))

        freq_entropy = 0.0
        freq_std = 0.0
        freq_skew = 0.0
        freq_kurt = 0.0
        freq_gini = 0.0

        if n_active > 1:
            freq_std = float(np.std(active_freqs))
            # Entropy
            p = active_freqs / np.sum(active_freqs)
            freq_entropy = float(-np.sum(p * np.log2(p + 1e-10)))
            # Gini coefficient
            sorted_f = np.sort(active_freqs)
            n = len(sorted_f)
            index = np.arange(1, n + 1)
            freq_gini = float(
                (2 * np.sum(index * sorted_f) - (n + 1) * np.sum(sorted_f))
                / (n * np.sum(sorted_f) + 1e-10)
            )

            try:
                from scipy import stats as sp_stats
                freq_skew = float(sp_stats.skew(active_freqs))
                freq_kurt = float(sp_stats.kurtosis(active_freqs))
            except ImportError:
                pass

        # Population comparison
        n_unique = 0
        overlap = 0.0
        if population_frequencies is not None:
            pop_active = population_frequencies > 0.01
            agent_active = frequencies > 0
            n_unique = int(np.sum(agent_active & ~pop_active))
            intersection = np.sum(agent_active & pop_active)
            union = np.sum(agent_active | pop_active)
            overlap = float(intersection / union) if union > 0 else 0.0

        return cls(
            is_populated=True,
            n_active_features=n_active,
            mean_activation_frequency=mean_freq,
            activation_sparsity=sparsity,
            top_feature_frequencies=top_freqs,
            top_feature_indices=top_indices,
            top_feature_labels=top_labels,
            frequency_entropy=freq_entropy,
            frequency_std=freq_std,
            frequency_skewness=freq_skew,
            frequency_kurtosis=freq_kurt,
            frequency_gini=freq_gini,
            n_unique_features=n_unique,
            feature_overlap_with_population=overlap,
            _full_frequencies=frequencies.copy(),
        )


@dataclass
class AnomalyReport:
    """Detected anomalies in agent behavior."""
    
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Anomaly types detected
    out_of_role_detected: bool = False
    phase_violation_detected: bool = False
    sudden_shift_detected: bool = False
    correlation_break_detected: bool = False
    convergence_anomaly_detected: bool = False
    dead_channel_detected: bool = False
    
    # Severity
    total_anomaly_count: int = 0
    max_severity: float = 0.0
    
    def add_anomaly(
        self,
        anomaly_type: str,
        round_num: int,
        severity: float,
        description: str,
        details: Dict[str, Any] = None
    ):
        """Add an anomaly to the report."""
        self.anomalies.append({
            "type": anomaly_type,
            "round": round_num,
            "severity": severity,
            "description": description,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
        self.total_anomaly_count += 1
        self.max_severity = max(self.max_severity, severity)
        
        # Set type flags
        if anomaly_type == "out_of_role":
            self.out_of_role_detected = True
        elif anomaly_type == "phase_violation":
            self.phase_violation_detected = True
        elif anomaly_type == "sudden_shift":
            self.sudden_shift_detected = True


@dataclass
class AgentBehavioralDNAMetadata:
    """Metadata for Agent Behavioral DNA."""
    
    # Agent identification
    agent_id: str
    agent_role: AgentRole
    agent_name: str
    
    # Session context
    session_id: str
    case_id: str
    trial_type: str
    
    # Model info
    model_name: str
    
    # Extraction info
    num_rounds: int
    num_statements: int
    phases_included: List[str]
    probes_used: List[str]
    
    # Injection info
    was_injected: bool
    injection_probe: str
    injection_strength: float
    
    # Timing
    extraction_time: str
    computation_time_seconds: float
    
    # DNA dimensions
    total_dimension: int
    feature_breakdown: Dict[str, int]


# =============================================================================
# MAIN CLASS: Agent Behavioral DNA
# =============================================================================

class AgentBehavioralDNA:
    """
    Comprehensive behavioral DNA signature for an agent in multi-agent simulation.
    
    Captures fine-grained behavioral features across multiple dimensions:
    - Token-level dynamics
    - Temporal patterns
    - Cross-agent interactions
    - Phase-specific signatures
    - Linguistic markers
    - Probe interactions
    - Injection response
    - Role compliance
    - Composite indices
    - Anomaly detection
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_role: AgentRole,
        metadata: AgentBehavioralDNAMetadata,
    ):
        """
        Initialize Agent Behavioral DNA.
        
        Args:
            agent_id: Unique agent identifier
            agent_role: Role of the agent
            metadata: Extraction metadata
        """
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)
        
        # Feature components
        self.token_features = TokenLevelFeatures()
        self.temporal_features = TemporalFeatures()
        self.cross_agent_features = CrossAgentFeatures()
        self.phase_features: Dict[str, PhaseSpecificFeatures] = {}
        self.linguistic_features = LinguisticFeatures()
        self.probe_interaction_features = ProbeInteractionFeatures()
        self.injection_features = InjectionResponseFeatures()
        self.role_compliance_features = RoleComplianceFeatures(role=agent_role)
        self.composite_indices = CompositeIndices()
        self.sae_features = SAEFeatures()
        self.anomaly_report = AnomalyReport()

        # Raw data storage
        self._raw_scores: List[List[float]] = []  # Per-statement token scores
        self._round_scores: List[float] = []  # Per-round mean scores
        self._statements: List[str] = []  # Raw text
        self._timestamps: List[str] = []
        
        # Cached full vector
        self._full_vector: Optional[np.ndarray] = None
        self._vector_dirty = True
    
    # =========================================================================
    # DATA INGESTION
    # =========================================================================
    
    def add_statement(
        self,
        text: str,
        token_scores: List[float],
        round_num: int,
        phase: CourtPhase,
        probe_scores: Dict[str, float] = None,
        timestamp: str = None,
    ):
        """
        Add a statement from this agent.
        
        Args:
            text: Raw response text
            token_scores: Per-token probe scores
            round_num: Round number
            phase: Court phase
            probe_scores: Scores from multiple probes
            timestamp: When statement was made
        """
        self._statements.append(text)
        self._raw_scores.append(token_scores)
        self._round_scores.append(float(np.mean(token_scores)) if token_scores else 0.0)
        self._timestamps.append(timestamp or datetime.now().isoformat())
        
        # Initialize phase features if needed
        if phase.value not in self.phase_features:
            self.phase_features[phase.value] = PhaseSpecificFeatures(phase=phase)
        
        self._vector_dirty = True
    
    def set_injection_info(
        self,
        is_injected: bool,
        strength: float = 0.0,
        probe: str = "",
    ):
        """Set injection information for this agent."""
        self.injection_features.is_injected = is_injected
        self.injection_features.injection_strength = strength
        self.injection_features.injection_probe = probe
        self._vector_dirty = True
    
    def set_other_agents_scores(self, other_scores: Dict[str, List[float]]):
        """
        Set scores from other agents for cross-agent analysis.
        
        Args:
            other_scores: Dict of agent_id -> list of round scores
        """
        self._other_agent_scores = other_scores
        self._vector_dirty = True
    
    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    
    def extract_all_features(self):
        """Extract all features from ingested data."""
        if not self._round_scores:
            self.logger.warning("No data to extract features from")
            return

        self._extract_token_features()
        self._extract_temporal_features()
        self._extract_cross_agent_features()
        self._extract_phase_features()
        self._extract_linguistic_features()
        self._extract_probe_interaction_features()
        self._extract_injection_features()
        self._extract_role_compliance_features()
        self._compute_composite_indices()
        self._detect_anomalies()
        # SAE features are populated externally via enrich_with_sae()

        self._vector_dirty = True

    def enrich_with_sae(
        self,
        sae_frequencies: np.ndarray,
        feature_labels: Optional[Dict[int, str]] = None,
        population_frequencies: Optional[np.ndarray] = None,
    ):
        """
        Enrich this agent's DNA with SAE-based features.

        Call this after running the SAE fingerprinting pipeline
        (core.sae_fingerprint) on this agent's responses.

        Args:
            sae_frequencies: SAE latent frequency vector for this agent's
                responses, shape (d_sae,), values in [0, 1]
            feature_labels: Optional mapping of SAE latent index → label
            population_frequencies: Optional population-average frequencies
                for computing uniqueness metrics
        """
        self.sae_features = SAEFeatures.from_fingerprint(
            frequencies=sae_frequencies,
            feature_labels=feature_labels,
            population_frequencies=population_frequencies,
        )
        self._vector_dirty = True
    
    def _extract_token_features(self):
        """Extract token-level features."""
        if not self._raw_scores:
            return
        
        # Flatten all token scores
        all_tokens = []
        for scores in self._raw_scores:
            all_tokens.extend(scores)
        
        if not all_tokens:
            return
        
        all_tokens = np.array(all_tokens)
        
        tf = self.token_features
        
        # Basic statistics
        tf.score_mean = float(np.mean(all_tokens))
        tf.score_std = float(np.std(all_tokens))
        tf.score_min = float(np.min(all_tokens))
        tf.score_max = float(np.max(all_tokens))
        tf.score_range = tf.score_max - tf.score_min
        
        # Distribution characteristics
        tf.score_entropy = self._compute_entropy(all_tokens)
        
        if HAS_SCIPY and len(all_tokens) > 2:
            tf.score_skewness = float(stats.skew(all_tokens))
            tf.score_kurtosis = float(stats.kurtosis(all_tokens))
        
        # Gradient analysis
        if len(all_tokens) > 1:
            gradients = np.diff(all_tokens)
            tf.mean_gradient = float(np.mean(gradients))
            tf.max_gradient = float(np.max(np.abs(gradients)))
            tf.gradient_sign_changes = int(np.sum(np.diff(np.sign(gradients)) != 0))
        
        # Peak analysis
        if HAS_SCIPY and len(all_tokens) > 3:
            peaks, properties = find_peaks(all_tokens, prominence=0.1)
            valleys, _ = find_peaks(-all_tokens, prominence=0.1)
            tf.num_peaks = len(peaks)
            tf.num_valleys = len(valleys)
            if len(peaks) > 0 and 'prominences' in properties:
                tf.peak_prominence_mean = float(np.mean(properties['prominences']))
        
        # Position analysis
        tf.argmax_position = float(np.argmax(all_tokens) / len(all_tokens))
        tf.argmin_position = float(np.argmin(all_tokens) / len(all_tokens))
        
        # Positional bias
        n = len(all_tokens)
        first_10 = all_tokens[:max(1, n // 10)]
        last_10 = all_tokens[-max(1, n // 10):]
        middle = all_tokens[n // 4: 3 * n // 4]
        
        tf.first_token_bias = float(np.mean(first_10) - np.mean(all_tokens))
        tf.tail_bias = float(np.mean(last_10) - np.mean(all_tokens))
        if len(middle) > 0:
            edges = np.concatenate([all_tokens[:n // 4], all_tokens[3 * n // 4:]])
            tf.middle_concentration = float(np.mean(middle) - np.mean(edges))
    
    def _extract_temporal_features(self):
        """Extract temporal features across rounds."""
        scores = np.array(self._round_scores)
        
        if len(scores) < 2:
            return
        
        tf = self.temporal_features
        
        # Drift
        tf.total_drift = float(scores[-1] - scores[0])
        tf.drift_velocity = tf.total_drift / len(scores)
        tf.drift_direction = float(np.sign(tf.total_drift))
        
        # Oscillation
        sign_changes = np.sum(np.diff(np.sign(scores - np.mean(scores))) != 0)
        tf.oscillation_frequency = float(sign_changes / len(scores))
        tf.oscillation_amplitude = float(np.mean(np.abs(scores - np.mean(scores))))
        
        # Momentum (autocorrelation)
        if len(scores) > 2:
            diffs = np.diff(scores)
            if len(diffs) > 1:
                tf.momentum = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1]) if np.std(diffs) > 0 else 0.0
            
            # Acceleration
            if len(diffs) > 1:
                tf.acceleration = float(np.mean(np.diff(diffs)))
        
        # Convergence
        mean_score = np.mean(scores)
        distances_to_mean = np.abs(scores - mean_score)
        if len(distances_to_mean) > 1:
            tf.convergence_rate = float(np.polyfit(range(len(distances_to_mean)), distances_to_mean, 1)[0])
        
        # Final trajectory
        if len(scores) >= 3:
            tf.final_trajectory_slope = float(np.polyfit(range(3), scores[-3:], 1)[0])
        
        # Variance trend
        if len(scores) >= 4:
            half = len(scores) // 2
            first_half_var = np.var(scores[:half])
            second_half_var = np.var(scores[half:])
            tf.variance_trend = float(second_half_var - first_half_var)
        
        # Classify trajectory archetype
        tf.trajectory_archetype, tf.archetype_confidence = self._classify_trajectory(scores)
        
        # Phase transitions (change points)
        if HAS_SCIPY and len(scores) > 4:
            tf.transition_locations = self._detect_change_points(scores)
            tf.num_phase_transitions = len(tf.transition_locations)
    
    def _extract_cross_agent_features(self):
        """Extract cross-agent interaction features."""
        if not hasattr(self, '_other_agent_scores'):
            return
        
        cf = self.cross_agent_features
        my_scores = np.array(self._round_scores)
        
        reactivities = []
        mirroring = []
        
        for other_id, other_scores in self._other_agent_scores.items():
            other_arr = np.array(other_scores)
            
            if len(other_arr) != len(my_scores):
                continue
            
            # Reactivity: how much do I change after they speak?
            if len(my_scores) > 1:
                # Assuming turn-taking, reactivity is my change following their statement
                my_changes = np.diff(my_scores)
                reactivity = float(np.mean(np.abs(my_changes)))
                cf.reactivity_scores[other_id] = reactivity
                reactivities.append(reactivity)
            
            # Mirroring: correlation with other agent
            if np.std(my_scores) > 0 and np.std(other_arr) > 0:
                corr = float(np.corrcoef(my_scores, other_arr)[0, 1])
                cf.mirroring_coefficients[other_id] = corr
                mirroring.append(corr)
            
            # Opposition strength
            cf.opposition_strength = float(np.mean(np.abs(my_scores - other_arr)))
            
            # Judge alignment (if other is judge)
            if 'judge' in other_id.lower():
                if np.std(my_scores) > 0 and np.std(other_arr) > 0:
                    cf.judge_alignment = float(np.corrcoef(my_scores, other_arr)[0, 1])
        
        if reactivities:
            cf.mean_reactivity = float(np.mean(reactivities))
            cf.max_reactivity = float(np.max(reactivities))
        
        if mirroring:
            cf.mean_mirroring = float(np.mean(mirroring))
    
    def _extract_phase_features(self):
        """Extract phase-specific features."""
        # Aggregate scores by phase would require phase tracking per statement
        # For now, compute basic stats for phases we have
        for phase_name, pf in self.phase_features.items():
            pf.phase_mean_score = float(np.mean(self._round_scores)) if self._round_scores else 0.0
            pf.phase_score_variance = float(np.var(self._round_scores)) if len(self._round_scores) > 1 else 0.0
    
    def _extract_linguistic_features(self):
        """Extract linguistic features from text."""
        if not self._statements:
            return
        
        lf = self.linguistic_features
        
        all_text = " ".join(self._statements).lower()
        words = all_text.split()
        word_count = len(words)
        
        if word_count == 0:
            return
        
        # Hedging markers
        hedge_words = ['may', 'might', 'possibly', 'perhaps', 'arguably', 'could', 'potentially']
        hedge_count = sum(all_text.count(w) for w in hedge_words)
        lf.hedging_index = hedge_count / word_count * 100
        
        # Assertiveness markers
        assert_words = ['clearly', 'certainly', 'must', 'definitely', 'obviously', 'undoubtedly']
        assert_count = sum(all_text.count(w) for w in assert_words)
        lf.assertiveness_score = assert_count / word_count * 100
        
        # Certainty ratio
        total_markers = hedge_count + assert_count
        lf.certainty_ratio = assert_count / total_markers if total_markers > 0 else 0.5
        
        # Legal citations (simplified pattern)
        import re
        citation_pattern = r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+'
        citations = re.findall(citation_pattern, " ".join(self._statements))
        lf.citation_density = len(citations) / word_count * 100
        
        # Question ratio
        sentences = [s.strip() for s in re.split(r'[.!?]', all_text) if s.strip()]
        questions = [s for s in sentences if s.endswith('?') or s.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'do', 'does', 'can', 'could', 'would', 'should'))]
        lf.question_ratio = len(questions) / len(sentences) if sentences else 0.0
        
        # Sentence complexity
        if sentences:
            lf.sentence_complexity = word_count / len(sentences)
        
        # Pronouns
        first_person = ['i', 'we', 'my', 'our', 'me', 'us']
        first_count = sum(words.count(w) for w in first_person)
        lf.first_person_ratio = first_count / word_count * 100
        
        # Conditionals
        conditionals = ['if', 'then', 'would', 'could', 'should', 'unless', 'whether']
        cond_count = sum(all_text.count(w) for w in conditionals)
        lf.conditional_density = cond_count / word_count * 100
        
        # Negations
        negations = ['not', 'never', 'no', "n't", 'neither', 'nor', 'cannot']
        neg_count = sum(all_text.count(w) for w in negations)
        lf.negation_frequency = neg_count / word_count * 100
    
    def _extract_probe_interaction_features(self):
        """Extract probe interaction features."""
        # This would require multi-probe scores per statement
        # Placeholder for now
        pass
    
    def _extract_injection_features(self):
        """Extract injection response features."""
        if not self._round_scores:
            return
        
        scores = np.array(self._round_scores)
        inf = self.injection_features
        
        if inf.is_injected:
            # Absorption: how quickly did score change?
            if len(scores) > 1:
                initial_change = abs(scores[1] - scores[0]) if len(scores) > 1 else 0
                inf.absorption_rate = float(initial_change)
            
            # Peak effect
            baseline = scores[0] if len(scores) > 0 else 0
            inf.peak_effect = float(np.max(np.abs(scores - baseline)))
            
            # Amplification
            if inf.injection_strength != 0:
                observed_effect = inf.peak_effect
                expected_effect = abs(inf.injection_strength) * 0.15  # Rough expectation
                inf.amplification_factor = observed_effect / expected_effect if expected_effect > 0 else 1.0
                inf.runaway_detected = inf.amplification_factor > 2.0
        else:
            # Resistance for non-injected agents
            inf.resistance_score = 1.0 - float(np.std(scores)) if np.std(scores) < 1 else 0.0
    
    def _extract_role_compliance_features(self):
        """Extract role compliance features."""
        rcf = self.role_compliance_features
        
        # Basic compliance based on score stability for judge (should be neutral)
        if self.agent_role == AgentRole.JUDGE:
            scores = np.array(self._round_scores)
            if len(scores) > 0:
                # Judge should have scores near 0 (neutral)
                neutrality = 1.0 - min(1.0, np.mean(np.abs(scores)))
                rcf.role_metrics['neutrality'] = neutrality
                rcf.role_compliance_score = neutrality
        
        elif self.agent_role in [AgentRole.PLAINTIFF_COUNSEL, AgentRole.DEFENSE_COUNSEL]:
            # Attorneys should be consistently advocating
            scores = np.array(self._round_scores)
            if len(scores) > 0:
                consistency = 1.0 - min(1.0, np.std(scores))
                rcf.role_metrics['consistency'] = consistency
                rcf.role_compliance_score = consistency
    
    def _compute_composite_indices(self):
        """Compute composite behavioral indices."""
        ci = self.composite_indices
        
        # Calibration: confidence vs accuracy (using assertiveness vs hedging as proxy)
        ci.calibration_index = self.linguistic_features.certainty_ratio
        
        # Behavioral stability
        if self._round_scores:
            scores = np.array(self._round_scores)
            ci.behavioral_stability_index = 1.0 - min(1.0, float(np.std(scores)))
        
        # Manipulation susceptibility
        if self.injection_features.is_injected:
            ci.manipulation_susceptibility_index = min(1.0, self.injection_features.absorption_rate)
        else:
            ci.manipulation_susceptibility_index = 1.0 - self.injection_features.resistance_score
        
        # Role-specific indices
        if self.agent_role == AgentRole.JUDGE:
            ci.judicial_quality_index = self.role_compliance_features.role_compliance_score
        elif self.agent_role in [AgentRole.PLAINTIFF_COUNSEL, AgentRole.DEFENSE_COUNSEL]:
            ci.advocacy_effectiveness_index = (
                0.5 * self.linguistic_features.assertiveness_score / 10 +
                0.5 * self.role_compliance_features.role_compliance_score
            )
        elif self.agent_role == AgentRole.JURY_FOREPERSON:
            ci.deliberation_quality_index = self.role_compliance_features.role_compliance_score
    
    def _detect_anomalies(self):
        """Detect behavioral anomalies."""
        scores = np.array(self._round_scores)
        
        if len(scores) < 2:
            return
        
        # Sudden shift detection
        if len(scores) > 2:
            diffs = np.abs(np.diff(scores))
            threshold = np.mean(diffs) + 2 * np.std(diffs)
            for i, diff in enumerate(diffs):
                if diff > threshold:
                    self.anomaly_report.add_anomaly(
                        "sudden_shift",
                        round_num=i + 1,
                        severity=float(diff / threshold),
                        description=f"Score jumped by {diff:.3f} (threshold: {threshold:.3f})"
                    )
        
        # Dead channel detection
        if np.std(scores) < 0.01:
            self.anomaly_report.add_anomaly(
                "dead_channel",
                round_num=0,
                severity=0.5,
                description="Near-zero variance in scores"
            )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _compute_entropy(self, values: np.ndarray, bins: int = 20) -> float:
        """Compute entropy of value distribution."""
        hist, _ = np.histogram(values, bins=bins)
        hist = hist + 1e-10
        probs = hist / np.sum(hist)
        return float(-np.sum(probs * np.log2(probs)))
    
    def _classify_trajectory(self, scores: np.ndarray) -> Tuple[TrajectoryArchetype, float]:
        """Classify trajectory pattern."""
        if len(scores) < 3:
            return TrajectoryArchetype.UNKNOWN, 0.0
        
        # Fit linear trend
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        predicted = slope * x + intercept
        r_squared = 1 - np.sum((scores - predicted) ** 2) / np.sum((scores - np.mean(scores)) ** 2)
        
        std = np.std(scores)
        sign_changes = np.sum(np.diff(np.sign(scores - np.mean(scores))) != 0)
        
        # Classification logic
        if std < 0.05:
            return TrajectoryArchetype.STEADY_STATE, 0.9
        elif r_squared > 0.7 and slope > 0.05:
            return TrajectoryArchetype.ESCALATING, float(r_squared)
        elif r_squared > 0.7 and slope < -0.05:
            return TrajectoryArchetype.DE_ESCALATING, float(r_squared)
        elif sign_changes > len(scores) * 0.5:
            return TrajectoryArchetype.VOLATILE, float(sign_changes / len(scores))
        else:
            return TrajectoryArchetype.UNKNOWN, 0.5
    
    def _detect_change_points(self, scores: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Detect change points in score sequence."""
        if len(scores) < 4:
            return []
        
        # Simple method: look for points where local mean changes significantly
        window = max(2, len(scores) // 4)
        change_points = []
        
        for i in range(window, len(scores) - window):
            before = np.mean(scores[i - window:i])
            after = np.mean(scores[i:i + window])
            if abs(after - before) > threshold:
                change_points.append(i)
        
        return change_points
    
    # =========================================================================
    # VECTOR REPRESENTATION
    # =========================================================================
    
    def to_vector(self, normalize: bool = True, include_sae: bool = False) -> np.ndarray:
        """
        Convert to fixed-size vector for ML.

        Args:
            normalize: Whether to L2-normalize the vector
            include_sae: Whether to append SAE summary features.
                If True and SAE features are populated, returns a
                SAE_ENRICHED_DNA_DIM (144D) vector. Otherwise returns
                the standard TOTAL_DNA_DIM (128D) vector.

        Returns:
            numpy array of shape (TOTAL_DNA_DIM,) or (SAE_ENRICHED_DNA_DIM,)
        """
        if not self._vector_dirty and self._full_vector is not None:
            if not include_sae:
                return self._full_vector[:TOTAL_DNA_DIM]
            elif self.sae_features.is_populated:
                return self._full_vector

        # Concatenate all feature vectors
        components = [
            self.token_features.to_vector(),
            self.temporal_features.to_vector(),
            self.cross_agent_features.to_vector(),
            self.linguistic_features.to_vector(),
            self.probe_interaction_features.to_vector(),
            self.injection_features.to_vector(),
            self.role_compliance_features.to_vector(),
            self.composite_indices.to_vector(),
        ]

        # Add phase features (up to 4 phases)
        phase_vecs = []
        for phase in [CourtPhase.MOTIONS, CourtPhase.OPENING, CourtPhase.CLOSING, CourtPhase.DELIBERATION]:
            if phase.value in self.phase_features:
                phase_vecs.append(self.phase_features[phase.value].to_vector())
            else:
                phase_vecs.append(np.zeros(PHASE_FEATURES_DIM, dtype=np.float32))

        components.extend(phase_vecs)

        # Concatenate base features
        full_vec = np.concatenate(components)

        # Pad or truncate to fixed size
        if len(full_vec) < TOTAL_DNA_DIM:
            full_vec = np.pad(full_vec, (0, TOTAL_DNA_DIM - len(full_vec)))
        else:
            full_vec = full_vec[:TOTAL_DNA_DIM]

        # Append SAE summary features if available and requested
        if include_sae and self.sae_features.is_populated:
            sae_vec = self.sae_features.to_vector()
            full_vec = np.concatenate([full_vec, sae_vec])

        # Handle NaN/Inf
        full_vec = np.nan_to_num(full_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        if normalize:
            norm = np.linalg.norm(full_vec)
            if norm > 0:
                full_vec = full_vec / norm

        self._full_vector = full_vec.astype(np.float32)
        self._vector_dirty = False

        return self._full_vector
    
    def to_dna_signature(self) -> 'DNASignature':
        """
        Convert to standard DNASignature for compatibility with phylogenetic analysis.
        
        Returns:
            DNASignature object
        """
        if not HAS_DNA_SIGNATURE:
            raise ImportError("DNASignature class not available")
        
        vec = self.to_vector()
        
        metadata = DNAMetadata(
            model_name=self.metadata.model_name,
            extraction_method=f"behavioral_dna_{self.agent_role.value}",
            probe_set_id=",".join(self.metadata.probes_used),
            probe_count=len(self.metadata.probes_used),
            dna_dimension=len(vec),
            embedding_dimension=TOTAL_DNA_DIM,
            reduction_method="feature_extraction",
            extraction_time=self.metadata.extraction_time,
            computation_time_seconds=self.metadata.computation_time_seconds,
            model_metadata={"agent_id": self.agent_id, "role": self.agent_role.value},
            extractor_config={"version": "1.0", "features": "full"},
        )
        
        return DNASignature(vec, metadata)
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def get_behavioral_fingerprint(self) -> str:
        """Generate human-readable behavioral summary."""
        lines = [
            f"=== Behavioral DNA: {self.agent_id} ({self.agent_role.value}) ===",
            "",
            f"Trajectory: {self.temporal_features.trajectory_archetype.value} "
            f"(confidence: {self.temporal_features.archetype_confidence:.2f})",
            f"Drift: {self.temporal_features.total_drift:+.3f} "
            f"(velocity: {self.temporal_features.drift_velocity:+.4f}/round)",
            "",
            f"Token-level:",
            f"  Mean score: {self.token_features.score_mean:.3f} ± {self.token_features.score_std:.3f}",
            f"  Entropy: {self.token_features.score_entropy:.2f}",
            f"  First-token bias: {self.token_features.first_token_bias:+.3f}",
            "",
            f"Linguistic markers:",
            f"  Hedging: {self.linguistic_features.hedging_index:.1f}%",
            f"  Assertiveness: {self.linguistic_features.assertiveness_score:.1f}%",
            f"  Certainty ratio: {self.linguistic_features.certainty_ratio:.2f}",
            "",
            f"Role compliance: {self.role_compliance_features.role_compliance_score:.2f}",
            f"Calibration index: {self.composite_indices.calibration_index:.2f}",
            f"Stability index: {self.composite_indices.behavioral_stability_index:.2f}",
        ]
        
        if self.sae_features.is_populated:
            lines.extend([
                "",
                f"SAE Fingerprint:",
                f"  Active features: {self.sae_features.n_active_features}",
                f"  Sparsity: {self.sae_features.activation_sparsity:.3f}",
                f"  Mean frequency: {self.sae_features.mean_activation_frequency:.4f}",
                f"  Gini coefficient: {self.sae_features.frequency_gini:.3f}",
            ])
            if self.sae_features.top_feature_labels:
                lines.append(f"  Top features:")
                for label, freq in zip(
                    self.sae_features.top_feature_labels[:3],
                    self.sae_features.top_feature_frequencies[:3],
                ):
                    lines.append(f"    - {label}: {freq * 100:.1f}%")

        if self.injection_features.is_injected:
            lines.extend([
                "",
                f"INJECTION DETECTED:",
                f"  Strength: {self.injection_features.injection_strength:+.2f}",
                f"  Absorption rate: {self.injection_features.absorption_rate:.3f}",
                f"  Peak effect: {self.injection_features.peak_effect:.3f}",
                f"  Amplification: {self.injection_features.amplification_factor:.2f}x",
            ])

        if self.anomaly_report.total_anomaly_count > 0:
            lines.extend([
                "",
                f"ANOMALIES: {self.anomaly_report.total_anomaly_count} detected",
            ])
            for anomaly in self.anomaly_report.anomalies[:3]:
                lines.append(f"  - {anomaly['type']}: {anomaly['description']}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "num_statements": len(self._statements),
            "num_rounds": len(self._round_scores),
            "token_features": asdict(self.token_features),
            "temporal_features": {
                "drift_velocity": self.temporal_features.drift_velocity,
                "total_drift": self.temporal_features.total_drift,
                "trajectory_archetype": self.temporal_features.trajectory_archetype.value,
                "oscillation_frequency": self.temporal_features.oscillation_frequency,
            },
            "linguistic_features": {
                "hedging_index": self.linguistic_features.hedging_index,
                "assertiveness_score": self.linguistic_features.assertiveness_score,
                "certainty_ratio": self.linguistic_features.certainty_ratio,
            },
            "composite_indices": asdict(self.composite_indices),
            "injection": {
                "is_injected": self.injection_features.is_injected,
                "strength": self.injection_features.injection_strength,
                "absorption_rate": self.injection_features.absorption_rate,
            },
            "anomalies": self.anomaly_report.total_anomaly_count,
            "sae_features": {
                "is_populated": self.sae_features.is_populated,
                "n_active_features": self.sae_features.n_active_features,
                "activation_sparsity": self.sae_features.activation_sparsity,
                "mean_activation_frequency": self.sae_features.mean_activation_frequency,
                "frequency_gini": self.sae_features.frequency_gini,
                "top_features": list(zip(
                    self.sae_features.top_feature_labels[:5],
                    self.sae_features.top_feature_frequencies[:5],
                )) if self.sae_features.is_populated else [],
            },
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def save(self, filepath: Union[str, Path], format: str = "json"):
        """Save behavioral DNA to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "metadata": asdict(self.metadata),
            "vector": self.to_vector().tolist(),
            "statistics": self.get_statistics(),
            "fingerprint": self.get_behavioral_fingerprint(),
            "raw_round_scores": self._round_scores,
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = "auto") -> 'AgentBehavioralDNA':
        """Load behavioral DNA from file."""
        filepath = Path(filepath)
        
        if format == "auto":
            format = filepath.suffix.lstrip('.')
        
        if format == "pickle" or format == "pkl":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct from saved data
            metadata = AgentBehavioralDNAMetadata(**data["metadata"])
            dna = cls(
                agent_id=data["agent_id"],
                agent_role=AgentRole(data["agent_role"]),
                metadata=metadata,
            )
            dna._round_scores = data.get("raw_round_scores", [])
            dna._full_vector = np.array(data["vector"], dtype=np.float32)
            dna._vector_dirty = False
            
            return dna
        else:
            raise ValueError(f"Unknown format: {format}")


# =============================================================================
# COLLECTION CLASS
# =============================================================================

class AgentBehavioralDNACollection:
    """Collection of Agent Behavioral DNA signatures."""
    
    def __init__(self, signatures: Optional[List[AgentBehavioralDNA]] = None):
        self.signatures: Dict[str, AgentBehavioralDNA] = {}
        if signatures:
            for sig in signatures:
                self.signatures[sig.agent_id] = sig
    
    def add(self, signature: AgentBehavioralDNA):
        """Add signature to collection."""
        self.signatures[signature.agent_id] = signature
    
    def __len__(self) -> int:
        return len(self.signatures)
    
    def __iter__(self):
        return iter(self.signatures.values())
    
    def __getitem__(self, agent_id: str) -> AgentBehavioralDNA:
        return self.signatures[agent_id]
    
    def get_distance_matrix(self, metric: str = "euclidean") -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise distance matrix."""
        agents = list(self.signatures.keys())
        n = len(agents)
        distances = np.zeros((n, n))
        
        vectors = {aid: self.signatures[aid].to_vector() for aid in agents}
        
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = vectors[agents[i]], vectors[agents[j]]
                
                if metric == "euclidean":
                    dist = np.linalg.norm(v1 - v2)
                elif metric == "cosine":
                    dot = np.dot(v1, v2)
                    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                    dist = 1 - dot / norms if norms > 0 else 1.0
                else:
                    dist = np.linalg.norm(v1 - v2)
                
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances, agents
    
    def compute_polarization_index(self) -> float:
        """Compute system-wide polarization."""
        if len(self.signatures) < 2:
            return 0.0
        
        # Get counsel agents
        plaintiff = None
        defense = None
        
        for sig in self.signatures.values():
            if sig.agent_role == AgentRole.PLAINTIFF_COUNSEL:
                plaintiff = sig
            elif sig.agent_role == AgentRole.DEFENSE_COUNSEL:
                defense = sig
        
        if plaintiff and defense:
            p_scores = np.array(plaintiff._round_scores)
            d_scores = np.array(defense._round_scores)
            
            if len(p_scores) > 0 and len(d_scores) > 0:
                min_len = min(len(p_scores), len(d_scores))
                return float(np.mean(np.abs(p_scores[:min_len] - d_scores[:min_len])))
        
        return 0.0
    
    def to_dna_collection(self) -> 'DNACollection':
        """Convert to standard DNACollection."""
        if not HAS_DNA_SIGNATURE:
            raise ImportError("DNASignature not available")
        
        return DNACollection([sig.to_dna_signature() for sig in self.signatures.values()])
    
    def save(self, filepath: Union[str, Path]):
        """Save collection to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "agents": {aid: sig.get_statistics() for aid, sig in self.signatures.items()},
            "vectors": {aid: sig.to_vector().tolist() for aid, sig in self.signatures.items()},
            "polarization_index": self.compute_polarization_index(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def extract_behavioral_dna_from_session(
    session_data: Dict[str, Any],
    model_name: str = "unknown",
) -> AgentBehavioralDNACollection:
    """
    Extract behavioral DNA from a court session.
    
    Args:
        session_data: Dictionary containing:
            - round_results: List of round data
            - agents: Dict of agent configs
            - injection_target: Which agent was injected
            - injection_strength: Injection strength
        model_name: Model name for metadata
        
    Returns:
        Collection of Agent Behavioral DNA signatures
    """
    collection = AgentBehavioralDNACollection()
    
    round_results = session_data.get("round_results", [])
    agents_config = session_data.get("agents", {})
    injection_target = session_data.get("injection_target", "")
    injection_strength = session_data.get("injection_strength", 0.0)
    
    # Collect data per agent
    agent_data = defaultdict(lambda: {
        "scores": [],
        "texts": [],
        "phases": [],
        "token_scores": [],
    })
    
    for round_data in round_results:
        round_num = round_data.get("round", 0)
        phase = round_data.get("phase", "unknown")
        
        for agent_id, data in round_data.get("agents", {}).items():
            agent_data[agent_id]["scores"].append(data.get("mean_score", 0.0))
            agent_data[agent_id]["texts"].append(data.get("text", ""))
            agent_data[agent_id]["phases"].append(phase)
            agent_data[agent_id]["token_scores"].append(data.get("scores", []))
    
    # Build DNA for each agent
    for agent_id, data in agent_data.items():
        if not data["scores"]:
            continue
        
        # Determine role
        role = AgentRole.UNKNOWN
        for r in AgentRole:
            if r.value in agent_id.lower():
                role = r
                break
        
        # Create metadata
        metadata = AgentBehavioralDNAMetadata(
            agent_id=agent_id,
            agent_role=role,
            agent_name=agents_config.get(agent_id, {}).get("name", agent_id),
            session_id=session_data.get("session_id", "unknown"),
            case_id=session_data.get("case_id", "unknown"),
            trial_type=session_data.get("trial_type", "unknown"),
            model_name=model_name,
            num_rounds=len(data["scores"]),
            num_statements=len(data["texts"]),
            phases_included=list(set(data["phases"])),
            probes_used=session_data.get("probes", ["overconfidence"]),
            was_injected=(agent_id == injection_target),
            injection_probe=session_data.get("probe", "overconfidence") if agent_id == injection_target else "",
            injection_strength=injection_strength if agent_id == injection_target else 0.0,
            extraction_time=datetime.now().isoformat(),
            computation_time_seconds=0.0,
            total_dimension=TOTAL_DNA_DIM,
            feature_breakdown={
                "token": TOKEN_FEATURES_DIM,
                "temporal": TEMPORAL_FEATURES_DIM,
                "cross_agent": CROSS_AGENT_FEATURES_DIM,
                "linguistic": LINGUISTIC_FEATURES_DIM,
            },
        )
        
        # Create DNA
        dna = AgentBehavioralDNA(agent_id, role, metadata)
        
        # Add statements
        for i, (text, tokens, phase) in enumerate(zip(data["texts"], data["token_scores"], data["phases"])):
            try:
                phase_enum = CourtPhase(phase) if phase in [p.value for p in CourtPhase] else CourtPhase.MOTIONS
            except:
                phase_enum = CourtPhase.MOTIONS
            
            dna.add_statement(
                text=text,
                token_scores=tokens if tokens else [data["scores"][i]],
                round_num=i + 1,
                phase=phase_enum,
            )
        
        # Set injection info
        dna.set_injection_info(
            is_injected=(agent_id == injection_target),
            strength=injection_strength if agent_id == injection_target else 0.0,
        )
        
        # Set other agents' scores for cross-agent analysis
        other_scores = {
            aid: adata["scores"]
            for aid, adata in agent_data.items()
            if aid != agent_id
        }
        dna.set_other_agents_scores(other_scores)
        
        # Extract features
        dna.extract_all_features()
        
        collection.add(dna)
    
    return collection


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'AgentBehavioralDNA',
    'AgentBehavioralDNACollection',
    'AgentBehavioralDNAMetadata',
    
    # Feature classes
    'TokenLevelFeatures',
    'TemporalFeatures',
    'CrossAgentFeatures',
    'PhaseSpecificFeatures',
    'LinguisticFeatures',
    'ProbeInteractionFeatures',
    'InjectionResponseFeatures',
    'RoleComplianceFeatures',
    'CompositeIndices',
    'SAEFeatures',
    'AnomalyReport',
    
    # Enums
    'AgentRole',
    'CourtPhase',
    'TrajectoryArchetype',
    
    # Factory
    'extract_behavioral_dna_from_session',
    
    # Constants
    'TOTAL_DNA_DIM',
    'SAE_SUMMARY_DIM',
    'SAE_ENRICHED_DNA_DIM',
]