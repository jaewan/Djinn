from __future__ import annotations

import time
from typing import Dict, Any, Union

from djinn.core.graph_interface import Graph as DjinnGraph
from .pattern_registry import PatternRegistry
from .workload import WorkloadProfile, WorkloadType, WorkloadClassifier
from .hooks import HookManager
from .graph_utils import analyze_operations_advanced, track_performance
import logging
import os


class SemanticAnalyzer:
	"""Three-tier semantic analyzer (dynamic, patterns, hooks).

	Updated for Refactoring #3: Now supports unified graph interface.
	Updated for Refactoring #5: Now supports dependency injection for pattern matchers.
	"""

	def __init__(
		self, 
		pattern_registry: PatternRegistry | None = None,
		pattern_matcher=None
	) -> None:
		"""Initialize semantic analyzer.
		
		Args:
			pattern_registry: (Deprecated) Legacy pattern registry. Use pattern_matcher instead.
			pattern_matcher: IPatternMatcher instance for pattern matching.
						   If None, uses default NetworkXPatternMatcher.
		"""
		import warnings
		
		# Support both old and new initialization
		if pattern_matcher is not None:
			# New approach: Use injected pattern matcher
			self.pattern_matcher = pattern_matcher
			self.pattern_registry = None  # Deprecated
		elif pattern_registry is not None:
			# Old approach: Wrap pattern_registry in NetworkXPatternMatcher
			warnings.warn(
				"pattern_registry parameter is deprecated. Use pattern_matcher instead. "
				"See docs/REFACTORING_MIGRATION_GUIDE.md for migration instructions.",
				DeprecationWarning,
				stacklevel=2
			)
			from .pattern_matcher import NetworkXPatternMatcher
			self.pattern_matcher = NetworkXPatternMatcher(pattern_registry)
			self.pattern_registry = pattern_registry  # Keep for backward compat
		else:
			# Default: Use NetworkX matcher with default patterns
			from .pattern_matcher import get_default_pattern_matcher
			self.pattern_matcher = get_default_pattern_matcher()
			self.pattern_registry = None
		
		self.hook_manager = HookManager()
		self._analysis_stats: Dict[str, float] = {}

	@track_performance
	def analyze_graph(self, graph: DjinnGraph) -> WorkloadProfile:
		"""Analyze graph with performance tracking and advanced algorithms.
		
		Args:
			graph: Unified Graph interface (DjinnGraph)
		"""
		start_time = time.perf_counter()
		logger = logging.getLogger(__name__)
		
		# Check if profiler is available
		try:
			from djinn.server.profiling_context import ProfilingContext
			profiler = ProfilingContext.get_current()
			profiler_active = profiler is not None
		except (ImportError, AttributeError):
			profiler_active = False
			profiler = None
		
		# ✅ PHASE 3: Use unified Graph interface with stable_id()
		# Stable graph-id caching (best-effort in-process)
		cache_enabled = os.getenv("GENIE_ANALYZER_CACHE", "1") == "1"
		graph_id: str | None = None
		if profiler_active:
			graph_id_context = profiler.profile_component("semantic_graph_id_computation")
			graph_id_context.__enter__()
		else:
			graph_id_context = None
		
		try:
			if cache_enabled:
				try:
					# Use unified stable_id() method from Graph interface
					graph_id = graph.stable_id()
				except Exception:
					graph_id = None
					logger.debug("Graph ID computation failed", exc_info=True)
			if cache_enabled and graph_id:
				cached = getattr(self, "_cache", {}).get(graph_id)
				if cached is not None:
					return cached
		finally:
			if graph_id_context is not None:
				graph_id_context.__exit__(None, None, None)
		
		# Use advanced operation analysis
		if profiler_active:
			op_analysis_context = profiler.profile_component("semantic_operation_analysis")
			op_analysis_context.__enter__()
		else:
			op_analysis_context = None
		
		try:
			ops_metadata = analyze_operations_advanced(graph)
		finally:
			if op_analysis_context is not None:
				op_analysis_context.__exit__(None, None, None)
		
		# Structural analysis removed (FX dependency eliminated)
		
		# Hook context
		if profiler_active:
			hook_context_profiler = profiler.profile_component("semantic_hook_context")
			hook_context_profiler.__enter__()
		else:
			hook_context_profiler = None
		
		try:
			semantic_context = self.hook_manager.get_context(graph)
		finally:
			if hook_context_profiler is not None:
				hook_context_profiler.__exit__(None, None, None)
		
		# Match patterns using injected pattern matcher (Refactoring #5)
		if profiler_active:
			pattern_match_context = profiler.profile_component("semantic_pattern_matching")
			pattern_match_context.__enter__()
		else:
			pattern_match_context = None
		
		try:
			pattern_result = self.pattern_matcher.match_patterns(graph)
			if pattern_result.is_ok:
				patterns = pattern_result.unwrap()
			else:
				# Log error but continue with empty patterns
				logger.warning(f"Pattern matching had errors: {pattern_result.error}")
				patterns = []
		finally:
			if pattern_match_context is not None:
				pattern_match_context.__exit__(None, None, None)
		
		# Workload classification
		if profiler_active:
			classify_context = profiler.profile_component("semantic_workload_classification")
			classify_context.__enter__()
		else:
			classify_context = None
		
		try:
			workload_type = WorkloadClassifier().classify(patterns)
		finally:
			if classify_context is not None:
				classify_context.__exit__(None, None, None)
		
		total_time = time.perf_counter() - start_time
		self._analysis_stats['last_analysis_time'] = total_time
		
		# Feature-flagged logging
		slow_ms_env = os.getenv("GENIE_ANALYZER_SLOW_MS", "100")
		try:
			slow_threshold = float(slow_ms_env) / 1000.0
		except Exception:
			slow_threshold = 0.1
		debug_enabled = os.getenv("GENIE_ANALYZER_DEBUG", "0") == "1"
		if total_time > slow_threshold:
			logger.warning("Graph analysis took %.1fms (>%.1fms target)", total_time * 1000.0, slow_threshold * 1000.0)
		elif debug_enabled:
			logger.debug("Graph analysis took %.3fms", total_time * 1000.0)
		
		profile = WorkloadProfile(
			workload_type=workload_type,
			patterns=patterns,
			metadata=ops_metadata,
			context=semantic_context,
		)
		if cache_enabled and graph_id:
			cache = getattr(self, "_cache", None)
			if cache is None:
				self._cache = {}
				cache = self._cache
			cache[graph_id] = profile
		return profile

	def get_performance_report(self) -> Dict[str, Any]:
		"""Get comprehensive performance report."""
		return {
			"analyzer_stats": self._analysis_stats,
			"pattern_stats": self.pattern_matcher.get_performance_report()
		}

	def generate_stub_plan(self, graph, profile: "WorkloadProfile") -> "ExecutionPlan":
		"""Emit a minimal ExecutionPlan with placement hints.

		This is a Phase-1 stub that wraps the entire graph into a single fragment
		with conservative placement based on workload type.
		"""
		from .workload import ExecutionPlan, PlanFragment, WorkloadType
		fragment_id = "fragment_0"
		fragment = PlanFragment(fragment_id=fragment_id, subgraph=graph, inputs=[], outputs=[])
		# Simple placement heuristic
		placement_device = {
			WorkloadType.VISION: "remote_accelerator:0",
			WorkloadType.LLM: "remote_accelerator:0",
			WorkloadType.MULTIMODAL: "remote_accelerator:0",
			WorkloadType.RECSYS: "cpu",
			WorkloadType.UNKNOWN: "cpu",
		}.get(profile.workload_type, "cpu")
		plan = ExecutionPlan(
			plan_id="plan_stub_0",
			fragments=[fragment],
			placement={fragment_id: placement_device},
			transfers=[],
			feature_flags={"overlap_io": False, "micro_batching": False},
		)
		return plan