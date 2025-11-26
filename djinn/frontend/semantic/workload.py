from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class WorkloadType(str, Enum):
	LLM = "llm"
	VISION = "vision"
	MULTIMODAL = "multimodal"
	RECSYS = "recsys"
	UNKNOWN = "unknown"


@dataclass
class MatchedPattern:
	pattern_name: str
	confidence: float
	matched_nodes: List[str] = field(default_factory=list)  # Node IDs that matched this pattern
	subgraph: Any | None = None
	optimization_hints: Dict[str, Any] | None = None
	metadata: Dict[str, Any] | None = None


@dataclass
class WorkloadProfile:
	workload_type: WorkloadType
	patterns: List[MatchedPattern]
	metadata: Dict[str, Any] | None = None
	structure: "StructuralInfo" | Dict[str, Any] | None = None
	context: Dict[str, Any] | None = None

	# Optional fields to align with interface-contracts for future phases
	confidence: float | None = None
	phases: Dict[str, List[str]] | None = None
	modalities: Dict[str, Any] | None = None
	dependencies: List[Any] | None = None
	compute_intensity: float | None = None
	memory_bandwidth: float | None = None
	latency_sensitivity: str | None = None
	optimization_hints: Dict[str, Any] | None = None


# Structural information extracted from FX or other static analyses
@dataclass
class StructuralInfo:
	modules: Dict[str, Any]
	architecture: Optional[str]
	depth: int
	width: int
	parameters: int


class WorkloadClassifier:
	"""Rule-based classifier mapping matched patterns to workload type.

	Simple heuristic version aligned with the specification. Can be replaced or
	enhanced later without changing callers.
	"""

	def classify(self, patterns: List[MatchedPattern]) -> WorkloadType:
		scores = {p.pattern_name: p.confidence for p in patterns}
		# Multimodal: strong signals from both LLM and Vision patterns
		if (scores.get("llm", 0.0) >= 0.85) and (
			scores.get("vision", 0.0) >= 0.85
			or scores.get("conv_pattern", 0.0) >= 0.85
			or scores.get("conv_activation", 0.0) >= 0.85
		):
			return WorkloadType.MULTIMODAL
		# Prefer explicit workload-named patterns if available
		if scores.get("llm", 0.0) > 0.8:
			return WorkloadType.LLM
		if scores.get("vision", 0.0) > 0.8 or scores.get("conv_pattern", 0.0) > 0.85:
			return WorkloadType.VISION
		if scores.get("multimodal", 0.0) > 0.7:
			return WorkloadType.MULTIMODAL
		if scores.get("recsys", 0.0) > 0.7:
			return WorkloadType.RECSYS
		return WorkloadType.UNKNOWN


# Planner surface (interface-only for now)
@dataclass
class PlanFragment:
	fragment_id: str
	subgraph: Any
	inputs: List[Any]
	outputs: List[Any]


@dataclass
class ExecutionPlan:
	plan_id: str
	fragments: List[PlanFragment]
	placement: Dict[str, Any]  # fragment_id -> device/node
	transfers: List[Dict[str, Any]]
	feature_flags: Dict[str, bool]