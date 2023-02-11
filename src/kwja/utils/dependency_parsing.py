from collections import defaultdict
from typing import Dict, Set


class DependencyManager:
    def __init__(self) -> None:
        self.directed_graph: Dict[int, Set[int]] = defaultdict(set)
        self.root = False

    def add_edge(self, source: int, target: int) -> None:
        self.directed_graph[source].add(target)

    def remove_edge(self, source: int, target: int) -> None:
        self.directed_graph[source].remove(target)

    def is_cyclic(self, source: int, visited: Set[int], cache: Dict[int, bool]) -> bool:
        if source in cache:
            return cache[source]

        if source in visited:
            return True
        else:
            visited.add(source)

        ret = False
        for target in self.directed_graph[source]:
            if self.is_cyclic(target, visited, cache):
                ret = True
                break

        cache[source] = ret
        return ret

    def has_cycle(self) -> bool:
        visited: Set[int] = set()
        cache: Dict[int, bool] = {}
        # cast keys to list to avoid RuntimeError: dictionary changed size during iteration
        for source in list(self.directed_graph.keys()):
            if self.is_cyclic(source, visited, cache):
                return True
        return False
