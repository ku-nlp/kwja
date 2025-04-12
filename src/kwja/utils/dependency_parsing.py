from collections import defaultdict


class DependencyManager:
    def __init__(self) -> None:
        self.directed_graph: dict[int, set[int]] = defaultdict(set)
        self.root = False

    def add_edge(self, source: int, target: int) -> None:
        self.directed_graph[source].add(target)

    def remove_edge(self, source: int, target: int) -> None:
        self.directed_graph[source].remove(target)

    def is_cyclic(self, source: int, visited: set[int], cache: dict[int, bool]) -> bool:
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
        visited: set[int] = set()
        cache: dict[int, bool] = {}
        # cast keys to list to avoid RuntimeError: dictionary changed size during iteration
        for source in list(self.directed_graph.keys()):
            if self.is_cyclic(source, visited, cache):
                return True
        return False
