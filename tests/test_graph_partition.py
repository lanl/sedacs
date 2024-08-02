import unittest

import numpy as np
from sedacs.graph import get_a_small_graph
from sedacs.graph_partition import (
    do_flips,
    do_flips_precomp,
    get_balance_from_indices,
    get_balancing,
    get_cut,
    get_parts_from_indices,
    get_parts_indices,
    metis_partition,
)


class TestGraphPartition(unittest.TestCase):
    def test_get_cut(self):
        passed = True
        nnodes = 9
        whichPart = np.zeros((nnodes), dtype=int)
        whichPart[0:4] = 0
        whichPart[4:7] = 1
        whichPart[7:9] = 2
        graph = np.zeros((nnodes, nnodes), dtype=int)
        graph[:, 0] = 1
        # A cyclic graph
        for i in range(nnodes - 1):
            graph[i, 1] = i + 1
        graph[8, 1] = 0
        # 3 segments will cut the graph in 3 points
        result = 3
        try:
            cut = get_cut(whichPart, graph)
            if result == cut:
                passed = True
            else:
                passed = False
        except Exception:
            passed = False
        self.assertTrue(passed)
        return passed

    def test_get_parts_indices(self):
        passed = True
        parts = [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
        nnodes = 9
        result = np.zeros((nnodes), dtype=int)
        result[0:4] = 0
        result[4:7] = 1
        result[7:9] = 2
        try:
            whichPart = get_parts_indices(parts, nnodes)
            if np.linalg.norm(result - whichPart) == 0:
                passed = True
            else:
                passed = False
        except Exception:
            passed = False
        self.assertTrue(passed)

    def test_get_parts_from_indices(self):
        passed = True
        nnodes = 9
        whichPart = np.zeros((nnodes), dtype=int)
        whichPart[0:4] = 0
        whichPart[4:7] = 1
        whichPart[7:9] = 2
        partsRef = [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
        parts = get_parts_from_indices(whichPart, 3)
        for element in parts:
            if element in partsRef:
                pass
            else:
                passed = False
        self.assertTrue(passed)

    def test_get_balancing(self):
        parts = [[0, 1, 2, 3], [0, 1]]
        result = 2
        try:
            bal = get_balancing(parts)
            if bal == result:
                passed = True
            else:
                passed = False
        except Exception:
            passed = False
        self.assertTrue(passed)

    def test_get_balance_from_indices(self):
        nnodes = 9
        whichPart = np.zeros(nnodes, dtype=int)
        whichPart[0:4] = 0
        whichPart[4:7] = 1
        whichPart[7:9] = 2
        nparts = 3
        try:
            bal = get_balance_from_indices(whichPart, nparts)
            if bal == 2:
                passed = True
            else:
                passed = False
        except Exception:
            passed = False
        self.assertTrue(passed)

    def test_do_flips_precomp(self):
        nnodes = 6
        graph = get_a_small_graph()
        whichPart = np.zeros((nnodes), dtype=int)
        result = np.zeros((nnodes), dtype=int)
        result[0:3] = 1
        whichPart[0] = 1
        whichPart[3] = 1
        whichPart[2] = 1
        nparts = 2
        for _ in range(10):
            whichPartNew = do_flips_precomp(whichPart, graph, nnodes, nparts)
            whichPart = whichPartNew
            cut = get_cut(whichPart, graph)
        if np.linalg.norm(whichPartNew - result) == 0:
            passed = True
        else:
            passed = False
        self.assertTrue(passed)

    def test_do_flips(self):
        nnodes = 6
        graph = get_a_small_graph()
        whichPart = np.zeros((nnodes), dtype=int)
        result = np.zeros((nnodes), dtype=int)
        result[0:3] = 1
        whichPart[0] = 1
        whichPart[3] = 1
        whichPart[2] = 1
        nparts = 2
        for _ in range(10):
            whichPartNew = do_flips(whichPart, graph)
            whichPart = whichPartNew
            cut = get_cut(whichPart, graph)
        if np.linalg.norm(whichPartNew - result) == 0:
            passed = True
        else:
            passed = False
        self.assertTrue(passed)

    # This test is disabled for now
    # def test_no_metis_partition(self):
    #     nnodes = 6
    #     graph = get_a_small_graph()
    #     whichPart = np.zeros((nnodes), dtype=int)
    #     result = np.zeros((nnodes), dtype=int)
    #     nparts = 2
    #     try:
    #         parts = metis_partition(graph, nparts)
    #         whichPart = get_parts_indices(parts, nnodes)
    #         cut = get_cut(whichPart, graph)
    #         if cut == 1:
    #             passed = True
    #         else:
    #             passed = False
    #     except Exception:
    #         passed = False
    #     self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
