"""tskit table accumulator for the hull simulator.

Records nodes and edges as the simulation runs. Edges are accumulated
in arbitrary order; tskit's ``sort()`` is called at finalization.
"""

import tskit


class TableBuilder:
    """Wraps a tskit ``TableCollection`` for incremental edge recording."""

    def __init__(self, sequence_length: float, num_populations: int = 1):
        self.tc = tskit.TableCollection(sequence_length=sequence_length)
        for _ in range(num_populations):
            self.tc.populations.add_row()

    def add_sample(
        self, time: float = 0.0, population: int = 0, metadata: bytes = b""
    ) -> int:
        return self.tc.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE,
            time=time,
            population=population,
            metadata=metadata,
        )

    def add_internal(
        self, time: float, population: int = 0, metadata: bytes = b""
    ) -> int:
        return self.tc.nodes.add_row(
            flags=0, time=time, population=population, metadata=metadata
        )

    def add_edge(self, left: float, right: float, parent: int, child: int):
        self.tc.edges.add_row(left=left, right=right, parent=parent, child=child)

    def finalize(self) -> tskit.TreeSequence:
        self.tc.sort()
        ts = self.tc.tree_sequence()
        return ts.simplify()
