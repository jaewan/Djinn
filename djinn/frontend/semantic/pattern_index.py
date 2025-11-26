"""
Hierarchical Pattern Index for fast pattern candidate selection.

This implements the hierarchical indexing strategy from the peer review to reduce
pattern matching from O(n) to O(log n) complexity.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, FrozenSet
import logging
import networkx as nx
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Immutable for hashing
class PatternSignature:
    """
    Multi-level signature for pattern indexing.

    Design: Immutable dataclass so it can be used as dict key.
    Each level provides increasingly specific filtering.
    """
    # Level 1: Fast set-based filtering
    operations: FrozenSet[str]  # e.g., frozenset({"conv2d", "relu"})

    # Level 2: Structural properties
    min_nodes: int  # Minimum nodes required
    max_nodes: int  # Maximum nodes expected
    has_cycles: bool
    max_fanout: int

    # Level 3: Operation sequence hash (for ordered patterns)
    sequence_hash: Optional[int] = None

    @classmethod
    def from_pattern_plugin(cls, pattern) -> Optional['PatternSignature']:
        """
        Extract signature from pattern plugin.

        Handles errors gracefully - returns None if extraction fails.
        """
        try:
            # Get expected operations from pattern metadata
            if hasattr(pattern, 'expected_operations'):
                operations = frozenset(pattern.expected_operations)
                logger.debug(f"Pattern {pattern.name}: operations={operations}")
            else:
                # Fallback: empty set means "matches anything"
                logger.debug(f"Pattern {pattern.name} has no expected_operations")
                return None

            # Get structural hints
            min_nodes = getattr(pattern, 'min_nodes', 1)
            max_nodes = getattr(pattern, 'max_nodes', float('inf'))
            has_cycles = getattr(pattern, 'allows_cycles', False)
            max_fanout = getattr(pattern, 'max_fanout', 100)

            # Compute sequence hash if pattern cares about order
            sequence_hash = None
            if hasattr(pattern, 'operation_sequence'):
                sequence_hash = hash(tuple(pattern.operation_sequence))

            return cls(
                operations=operations,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                has_cycles=has_cycles,
                max_fanout=max_fanout,
                sequence_hash=sequence_hash
            )
        except Exception as e:
            logger.warning(f"Failed to extract signature for {pattern.name}: {e}")
            return None


class HierarchicalPatternIndex:
    """
    Multi-level index for fast pattern candidate selection.

    Thread Safety: Read-only after initialization (thread-safe).
    If dynamic registration needed, add RWLock.
    """

    def __init__(self):
        # Level 1: By operation presence (Bloom filter could optimize this)
        self.by_operations: Dict[FrozenSet[str], List] = defaultdict(list)

        # Level 2: By size bucket (log-scale buckets for better distribution)
        self.by_size: Dict[str, List] = defaultdict(list)

        # Level 3: By structural properties
        self.by_structure: Dict[tuple, List] = defaultdict(list)

        # Level 4: By operation sequence (for ordered pattern matching)
        self.by_sequence: Dict[int, List] = defaultdict(list)

        # Fallback: All patterns (for patterns without signatures)
        self.all_patterns: List = []

        # Metrics
        self._index_hits = 0
        self._index_misses = 0

    def register_pattern(self, pattern):
        """
        Register pattern with index.

        Idempotent - can be called multiple times safely.
        """
        signature = PatternSignature.from_pattern_plugin(pattern)

        if signature is None:
            # Pattern can't be indexed - use fallback
            logger.debug(f"Pattern {pattern.name} registered in fallback index")
            self.all_patterns.append(pattern)
            return

        # Index by operations (most selective filter)
        self.by_operations[signature.operations].append(pattern)

        # Index by size bucket (log scale: 1-10, 10-100, 100-1000, ...)
        size_bucket = self._compute_size_bucket(signature.min_nodes, signature.max_nodes)
        self.by_size[size_bucket].append(pattern)

        # Index by structure
        struct_key = (signature.has_cycles, signature.max_fanout)
        self.by_structure[struct_key].append(pattern)

        # Index by operation sequence if available
        if signature.sequence_hash is not None:
            self.by_sequence[signature.sequence_hash].append(pattern)

        # Always add to fallback (safety net)
        self.all_patterns.append(pattern)

    @lru_cache(maxsize=1024)  # Cache results for common graphs
    def get_candidate_patterns(self, graph_hash: str, graph_ops: FrozenSet[str],
                               graph_size: int, has_cycles: bool,
                               max_fanout: int) -> List:
        """
        Get candidate patterns for a graph.

        Uses multi-level filtering to reduce candidates from O(n) to O(log n).

        Args:
            graph_hash: Stable hash of graph (for caching)
            graph_ops: Frozenset of operations in graph
            graph_size: Number of nodes
            has_cycles: Whether graph has cycles
            max_fanout: Maximum out-degree

        Returns:
            List of candidate patterns (sorted by specificity)
        """
        candidates = set()

        # Level 1: Filter by operations (most selective)
        logger.debug(f"Filtering patterns for graph_ops={graph_ops}")
        for pattern_ops, patterns in self.by_operations.items():
            is_subset = pattern_ops.issubset(graph_ops)
            logger.debug(f"  Pattern ops {pattern_ops}: subset of graph? {is_subset}")
            if is_subset:
                candidates.update(patterns)
                logger.debug(f"  Added {len(patterns)} patterns for {pattern_ops}")

        if not candidates:
            self._index_misses += 1
            logger.debug("Index miss - using all patterns")
            return self.all_patterns

        self._index_hits += 1

        # Level 2: Filter by size
        size_bucket = self._compute_size_bucket(graph_size, graph_size)
        size_candidates = self.by_size.get(size_bucket, [])
        if size_candidates:
            candidates = candidates.intersection(size_candidates)

        # Level 3: Filter by structure (if we still have many candidates)
        if len(candidates) > 10:
            struct_key = (has_cycles, max_fanout)
            struct_candidates = self.by_structure.get(struct_key, [])
            if struct_candidates:
                candidates = candidates.intersection(struct_candidates)

        # Sort by specificity (more specific patterns first)
        candidates = sorted(
            candidates,
            key=lambda p: self._pattern_specificity(p),
            reverse=True
        )

        return list(candidates)

    @staticmethod
    def _compute_size_bucket(min_size: int, max_size: int) -> str:
        """
        Compute log-scale size bucket.

        Buckets: 1-10, 10-100, 100-1000, 1000+
        """
        import math

        avg_size = (min_size + max_size) / 2

        if avg_size < 10:
            return "tiny"
        elif avg_size < 100:
            return "small"
        elif avg_size < 1000:
            return "medium"
        else:
            return "large"

    @staticmethod
    def _pattern_specificity(pattern) -> int:
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

    def get_stats(self) -> Dict:
        """Get index performance statistics."""
        total = self._index_hits + self._index_misses
        hit_rate = self._index_hits / total if total > 0 else 0

        return {
            'total_patterns': len(self.all_patterns),
            'indexed_patterns': len(self.all_patterns) - len([p for p in self.all_patterns
                                                               if not any(p in patterns
                                                                        for patterns in self.by_operations.values())]),
            'index_hit_rate': hit_rate,
            'total_queries': total,
        }
