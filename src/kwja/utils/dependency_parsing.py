from collections import defaultdict
from typing import Dict, Set


class DependencyManager:
    def __init__(self) -> None:
        self.directed_graph: Dict[int, set] = defaultdict(set)
        self.root = False

    def add_edge(self, src: int, dst: int) -> None:
        self.directed_graph[src].add(dst)

    def remove_edge(self, src: int, dst: int) -> None:
        self.directed_graph[src].remove(dst)

    def is_cyclic(self, src: int, visited: Set[int], cache: Dict[int, bool]) -> bool:
        if src in cache:
            return cache[src]

        if src in visited:
            return True
        else:
            visited.add(src)

        ret = False
        for dst in self.directed_graph[src]:
            if self.is_cyclic(dst, visited, cache):
                ret = True
                break

        cache[src] = ret
        return ret

    def has_cycle(self) -> bool:
        visited: Set[int] = set()
        cache: Dict[int, bool] = {}
        for src in list(self.directed_graph.keys()):
            if self.is_cyclic(src, visited, cache):
                return True
        return False
