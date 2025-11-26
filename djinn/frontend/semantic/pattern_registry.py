from __future__ import annotations

import time
import threading
from typing import Dict, List, Optional, Set
from enum import Enum
import networkx as nx

from djinn.core.exceptions import Result, PatternMatchError
from djinn.core.types import MatchingMode  # Import from shared types
from djinn.core.graph_interface import Graph as DjinnGraph
from .graph_utils import graph_to_networkx
import logging
import os
import importlib
from typing import Type
try:  # Python 3.8+
	from importlib import metadata as importlib_metadata  # type: ignore
except Exception:  # pragma: no cover
	import importlib_metadata  # type: ignore
from .workload import MatchedPattern
from ..patterns.base import PatternPlugin
from .pattern_index import HierarchicalPatternIndex


class MatchingMetrics:
    """Metrics for pattern matching performance."""
    def __init__(
        self,
        total_time_ms: float,
        patterns_tried: int,
        patterns_matched: int,
        cache_hits: int,
        cache_misses: int,
        index_hits: int,
        index_misses: int
    ):
        self.total_time_ms = total_time_ms
        self.patterns_tried = patterns_tried
        self.patterns_matched = patterns_matched
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.index_hits = index_hits
        self.index_misses = index_misses

    def to_dict(self):
        total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total if total > 0 else 0

        total_index = self.index_hits + self.index_misses
        index_hit_rate = self.index_hits / total_index if total_index > 0 else 0

        return {
            'total_time_ms': self.total_time_ms,
            'patterns_tried': self.patterns_tried,
            'patterns_matched': self.patterns_matched,
            'cache_hit_rate': cache_hit_rate,
            'index_hit_rate': index_hit_rate,
        }


class PatternRegistry:
	"""
	Registry and orchestrator for pattern plugins with thread safety and observability.
	
	NOTE: This is the NEW PatternRegistry (production). There is also an old PatternRegistry
	in djinn/frontend/semantic/patterns/base_matcher.py that is deprecated and only used
	for backward compatibility with PatternMatcher interface. This class uses PatternPlugin
	interface and returns MatchedPattern results.
	"""

	def __init__(self) -> None:
		# Thread safety
		self._lock = threading.RLock()

		# Core pattern registry
		self._patterns: Dict[str, PatternPlugin] = {}
		self._performance_stats: Dict[str, List[float]] = {}

		# Hierarchical pattern index for fast candidate selection
		self._index = HierarchicalPatternIndex()

		# Comprehensive metrics collection
		self._matching_metrics: List['MatchingMetrics'] = []
		self._cache_hits = 0
		self._cache_misses = 0
		self._index_hits = 0
		self._index_misses = 0

		# Initialize patterns
		self.register_builtin_patterns()
		self.register_lazy_dag_patterns()  # ✅ NEW: Register LazyDAG patterns
		# Attempt to load external plugins via entry points and env var
		self._load_external_plugins()

	def register_pattern(self, pattern: PatternPlugin) -> None:
		"""Thread-safe pattern registration with hierarchical indexing."""
		with self._lock:
			self._patterns[pattern.name] = pattern
			self._performance_stats[pattern.name] = []
			# Register with hierarchical index for fast candidate selection
			self._index.register_pattern(pattern)

	def register_builtin_patterns(self) -> None:
		"""Register built-in patterns with hierarchical indexing."""
		# Import here to avoid circular dependency
		from ..patterns.advanced_patterns import (
			AdvancedLLMPattern, AdvancedVisionPattern, RecSysPattern, MultiModalPattern,
			ResidualBlockPattern
		)
		# Register advanced patterns (index registration happens in register_pattern)
		self.register_pattern(AdvancedLLMPattern())
		self.register_pattern(AdvancedVisionPattern())
		self.register_pattern(RecSysPattern())
		self.register_pattern(MultiModalPattern())
		self.register_pattern(ResidualBlockPattern())

	def register_lazy_dag_patterns(self) -> None:
		"""Register LazyDAG-specific pattern matchers.
		
		NOTE: LazyDAGAttentionMatcher and LazyDAGConvolutionMatcher have been
		consolidated into AdvancedLLMPattern and AdvancedVisionPattern respectively.
		Only specialized LazyDAG patterns remain (KV cache, linear, activation).
		"""
		# Import here to avoid circular dependency
		from .patterns.lazy_dag_patterns import (
			LazyDAGKVCacheMatcher,
			LazyDAGLinearMatcher,
			LazyDAGActivationMatcher
		)
		# Register only specialized LazyDAG patterns
		# Attention and convolution are now handled by AdvancedLLMPattern/AdvancedVisionPattern
		self.register_pattern(LazyDAGKVCacheMatcher())
		self.register_pattern(LazyDAGLinearMatcher())
		self.register_pattern(LazyDAGActivationMatcher())

	def _load_external_plugins(self) -> None:
		"""Load pattern plugins from entry points and environment variable.

		Entry point groups checked:
		- 'genie.patterns'
		- 'genie.pattern_plugins'

		Env var:
		- GENIE_PATTERN_PLUGINS=module1:Factory,module2:ClassName,module3
		  Where Factory() -> PatternPlugin | list[PatternPlugin], ClassName is a PatternPlugin subclass,
		  or module has 'get_patterns()' returning list[PatternPlugin].
		"""
		logger = logging.getLogger(__name__)
		groups = ["genie.patterns", "genie.pattern_plugins"]
		for group in groups:
			try:
				for ep in importlib_metadata.entry_points().select(group=group):  # type: ignore[attr-defined]
					try:
						obj = ep.load()
						self._register_from_object(obj, source=f"entry_point:{ep.name}")
					except Exception as e:  # pragma: no cover
						logger.warning("Failed loading pattern entry point %s: %s", ep.name, e)
			except Exception as e:  # pragma: no cover
				logger.debug("Entry point loading not available or failed for group %s: %s", group, e)

		# Env var plugins
		env = os.getenv("GENIE_PATTERN_PLUGINS")
		if not env:
			return
		for spec in env.split(","):
			spec = spec.strip()
			if not spec:
				continue
			module_name, _, symbol = spec.partition(":")
			try:
				module = importlib.import_module(module_name)
				obj = getattr(module, symbol) if symbol else module
				self._register_from_object(obj, source=f"env:{spec}")
			except Exception as e:  # pragma: no cover
				logger.warning("Failed loading env pattern %s: %s", spec, e)

	def _register_from_object(self, obj, source: str = "unknown") -> None:  # noqa: ANN001
		"""Register patterns from a variety of provider shapes."""
		def is_plugin_class(cls: Type) -> bool:  # noqa: ANN202
			try:
				from ..frontend.patterns.base import PatternPlugin as _PP
				return isinstance(cls, type) and issubclass(cls, _PP)
			except Exception:
				return False

		if obj is None:
			return
		# If it's a class, instantiate
		if is_plugin_class(obj):
			instance = obj()
			self.register_pattern(instance)
			return
		# If it's a function, call it
		if callable(obj):
			result = obj()
			# Accept single or list
			if isinstance(result, list):
				for pat in result:
					if pat is not None:
						self.register_pattern(pat)
				return
			if result is not None:
				self.register_pattern(result)
			return
		# If it's a module, try get_patterns()
		get_patterns = getattr(obj, "get_patterns", None)
		if callable(get_patterns):
			try:
				patterns = get_patterns()
				for pat in patterns:
					if pat is not None:
						self.register_pattern(pat)
			except Exception:
				pass

	def match_patterns(
		self,
		graph: DjinnGraph,
		mode: MatchingMode = MatchingMode.EXHAUSTIVE,
		required_patterns: Optional[Set[str]] = None,
		confidence_threshold: float = 0.90,
		min_patterns_to_try: int = 3
	) -> Result[List[MatchedPattern]]:
		"""
		Match patterns with configurable strategy and comprehensive observability.

		Args:
			graph: Unified Graph interface (DjinnGraph) to analyze
			mode: Matching mode (exhaustive, fast, required_only)
			required_patterns: Set of pattern names that MUST be tried
			confidence_threshold: Threshold for early termination
			min_patterns_to_try: Minimum patterns to try (safety)

		Returns:
			Result[List[MatchedPattern]]: Matched patterns or error
		"""
		start_time = time.perf_counter()
		logger = logging.getLogger(__name__)
		debug_enabled = os.getenv("GENIE_ANALYZER_DEBUG", "0") == "1"

		matches: List[MatchedPattern] = []
		errors: List[Exception] = []
		patterns_tried = 0

		# Extract graph properties for hierarchical indexing
		# ✅ FIX: Convert iterator to list to avoid "iterator has no len()" error
		nodes_list = list(graph.nodes())
		graph_ops = frozenset(
			node.operation.replace('aten::', '')
			for node in nodes_list
		)
		graph_size = len(nodes_list)
		# Skip cycle detection and max fanout for now (complex to compute with unified API)
		has_cycles = False  # Conservative assumption
		max_fanout = 10    # Conservative assumption
		graph_hash = str(hash(frozenset((node.id for node in nodes_list))))  # Stable hash

		# Get candidates from hierarchical index (reduces from O(n) to O(log n))
		with self._lock:  # Read lock for thread safety
			candidates = self._index.get_candidate_patterns(
				graph_hash, graph_ops, graph_size, has_cycles, max_fanout
			)

		logger.debug(f"Index: {len(self._patterns)} total, "
					f"{len(candidates)} candidates")

		# If no candidates from index, fall back to all patterns (safety net)
		if not candidates:
			logger.debug("Index returned no candidates - using all patterns")
			with self._lock:
				candidates = list(self._patterns.values())

		# Sort by priority (smarter heuristic)
		candidates = self._sort_by_priority(candidates, mode)

		# Ensure required patterns are tried first
		if required_patterns and mode == MatchingMode.REQUIRED_ONLY:
			required = [p for p in candidates if p.name in required_patterns]
			optional = [p for p in candidates if p.name not in required_patterns]
			candidates = required + optional

		for pattern in candidates:
			# Always try minimum number of patterns (safety)
			if patterns_tried >= min_patterns_to_try:
				# Check early termination conditions
				if mode == MatchingMode.FAST:
					if self._should_terminate_early(matches, confidence_threshold):
						logger.info(f"Early termination after {patterns_tried} patterns")
						break

			pattern_start = time.perf_counter()

			try:
				match = pattern.match(graph)  # Pass unified graph directly
				latency = time.perf_counter() - pattern_start
				patterns_tried += 1

				# Track performance
				with self._lock:
					self._performance_stats.setdefault(pattern.name, []).append(latency)

				# Feature-flagged warnings
				slow_ms_env = os.getenv("GENIE_ANALYZER_SLOW_MS", "50")
				try:
					slow_threshold = float(slow_ms_env) / 1000.0
				except Exception:
					slow_threshold = 0.05
				if latency > slow_threshold:
					logger.debug("Pattern %s took %.1fms (>%.1fms)", pattern.name, latency * 1000.0, slow_threshold * 1000.0)
				elif debug_enabled:
					logger.debug("Pattern %s took %.3fms", pattern.name, latency * 1000.0)

				if match is None:
					# Pattern didn't match - not an error, just no match
					continue

				matches.append(
					MatchedPattern(
						pattern_name=pattern.name,
						confidence=match.confidence,
						matched_nodes=getattr(match, "matched_nodes", []),
						subgraph=None,
						optimization_hints=getattr(pattern, "get_hints", lambda: {})(),
						metadata=getattr(match, "metadata", None),
					)
				)
			except Exception as e:
				# Pattern matching raised an exception
				error = PatternMatchError(
					f"Pattern {pattern.name} failed to match",
					context={'pattern': pattern.name, 'error': str(e)}
				)
				errors.append(error)
				logger.debug(f"Pattern {pattern.name} raised exception: {e}")
				patterns_tried += 1

		# Sort by confidence (deterministic)
		matches.sort(key=lambda m: (m.confidence, m.pattern_name), reverse=True)

		# Calculate metrics
		total_time_ms = (time.perf_counter() - start_time) * 1000

		metrics = MatchingMetrics(
			total_time_ms=total_time_ms,
			patterns_tried=patterns_tried,
			patterns_matched=len(matches),
			cache_hits=self._cache_hits,
			cache_misses=self._cache_misses,
			index_hits=self._index_hits,
			index_misses=self._index_misses
		)

		# Store metrics for observability
		with self._lock:
			self._matching_metrics.append(metrics)

		logger.info(f"Pattern matching: {metrics.to_dict()}")

		# Return Result based on what we found
		if matches:
			# We have some matches - success (even if some patterns failed)
			return Result.ok(matches)
		elif errors:
			# No matches and we have errors - return aggregated error
			return Result.err(PatternMatchError(
				f"All {len(errors)} patterns failed to match",
				context={'error_count': len(errors), 'errors': [str(e) for e in errors]}
			))
		else:
			# No matches but no errors either - return empty list as success
			return Result.ok([])

	def _sort_by_priority(self, patterns: List[PatternPlugin], mode: MatchingMode) -> List[PatternPlugin]:
		"""
		Sort patterns by priority using multiple signals.

		Uses: historical performance, pattern specificity, mode-specific hints.
		"""
		def priority_score(pattern: PatternPlugin) -> float:
			score = 0.0

			# Historical performance (exponential moving average)
			with self._lock:
				if pattern.name in self._performance_stats:
					times = self._performance_stats[pattern.name]
					if times:
						# Recent performance weighted more
						ema = self._compute_ema(times, alpha=0.3)
						# Lower time = higher priority
						score += (100 - min(ema * 1000, 100))  # Cap at 100

			# Pattern specificity (from pattern metadata)
			specificity = self._pattern_specificity(pattern)
			score += specificity * 0.5

			# Mode-specific boost
			if mode == MatchingMode.FAST:
				# In fast mode, prefer high-confidence patterns
				if pattern.name in ['llm', 'vision', 'recsys']:
					score += 100

			return score

		return sorted(patterns, key=priority_score, reverse=True)

	def _should_terminate_early(self, matches: List[MatchedPattern], threshold: float) -> bool:
		"""
		Decide if we can terminate early.

		Conditions:
		1. Have at least one high-confidence match (>threshold)
		2. Clear winner (significant gap to second-best)
		3. No ambiguity (not close to multiple patterns)
		"""
		if not matches:
			return False

		# Sort by confidence
		sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
		best = sorted_matches[0]

		# Check 1: High confidence
		if best.confidence < threshold:
			return False

		# Check 2: Clear winner (20% gap)
		if len(sorted_matches) > 1:
			second = sorted_matches[1]
			gap = best.confidence - second.confidence

			if gap < 0.20:
				# Too close - might have ambiguity
				logger.debug(f"Not terminating: ambiguous ({best.pattern_name}={best.confidence:.2f}, "
						   f"{second.pattern_name}={second.confidence:.2f})")
				return False

		# Check 3: No other high-confidence matches
		other_high_confidence = [m for m in sorted_matches[1:]
								if m.confidence > threshold - 0.1]
		if other_high_confidence:
			logger.debug(f"Not terminating: multiple high-confidence matches")
			return False

		# Safe to terminate
		return True

	@staticmethod
	def _compute_ema(values: List[float], alpha: float = 0.3) -> float:
		"""Compute exponential moving average (recent values weighted more)."""
		if not values:
			return 0.0

		ema = values[0]
		for value in values[1:]:
			ema = alpha * value + (1 - alpha) * ema

		return ema

	@staticmethod
	def _pattern_specificity(pattern: PatternPlugin) -> int:
		"""
		Compute pattern specificity score (higher = more specific).

		More specific patterns should be tried first.
		"""
		score = 0

		# More required operations = more specific
		if hasattr(pattern, 'expected_operations'):
			score += len(pattern.expected_operations) * 10

		# Narrower size range = more specific
		if hasattr(pattern, 'min_nodes') and hasattr(pattern, 'max_nodes'):
			size_range = pattern.max_nodes - pattern.min_nodes
			if size_range < 100:
				score += 50
			elif size_range < 1000:
				score += 20

		# Has sequence requirement = more specific
		if hasattr(pattern, 'operation_sequence'):
			score += 30

		return score


	def get_performance_report(self) -> Dict[str, Dict[str, float]]:
		"""Get comprehensive performance statistics including index metrics."""
		report = {}

		# Pattern-level statistics
		for pattern_name, times in self._performance_stats.items():
			if times:
				report[pattern_name] = {
					"avg_latency_ms": (sum(times) / len(times)) * 1000,
					"max_latency_ms": max(times) * 1000,
					"min_latency_ms": min(times) * 1000,
					"call_count": len(times),
					"total_time_ms": sum(times) * 1000
				}
			else:
				report[pattern_name] = {
					"avg_latency_ms": 0.0,
					"max_latency_ms": 0.0,
					"min_latency_ms": 0.0,
					"call_count": 0,
					"total_time_ms": 0.0
				}

		# Add index statistics
		report["_index"] = self._index.get_stats()

		# Add overall matching statistics
		if self._matching_metrics:
			recent_metrics = self._matching_metrics[-10:]  # Last 10 matches
			report["_matching"] = {
				"recent_avg_time_ms": sum(m.total_time_ms for m in recent_metrics) / len(recent_metrics),
				"recent_avg_patterns_tried": sum(m.patterns_tried for m in recent_metrics) / len(recent_metrics),
				"recent_avg_patterns_matched": sum(m.patterns_matched for m in recent_metrics) / len(recent_metrics),
				"total_matches": len(self._matching_metrics)
			}

		return report
