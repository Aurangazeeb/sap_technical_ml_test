"""In-process LRU cache for search results.

Provides a bounded cache keyed by ``(product_id, k)`` that evicts the
least-recently-used entry when ``max_size`` is exceeded.  Built on
``collections.OrderedDict`` for O(1) get/put/evict.
"""
from __future__ import annotations

from collections import OrderedDict

__all__ = ["SearchCache"]


class SearchCache:
    """Bounded LRU cache for ``find_similar_products`` results.

    Parameters
    ----------
    max_size : int
        Maximum number of entries before the least-recently-used is evicted.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self._max_size = max_size
        self._store: OrderedDict[tuple[str, int], list[str]] = OrderedDict()

    def get(self, product_id: str, k: int) -> list[str] | None:
        """Return cached result or ``None`` on miss.  Refreshes LRU order."""
        key = (product_id, k)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, product_id: str, k: int, result: list[str]) -> None:
        """Insert or update a cache entry, evicting LRU if full."""
        key = (product_id, k)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = result
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
