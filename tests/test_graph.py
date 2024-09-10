import unittest

import numpy as np
from sedacs.graph import get_a_small_graph, get_random_adjacency_matrix
from sedacs.graph_partition import (
    get_cut,
    get_balance,
    get_balance_from_partitions,
    get_balance_from_partition_sizes,
    get_parts_from_indices,
    get_parts_indices
)


class TestGraph(unittest.TestCase):
    def test_get_parts_indices(self):
        passed = True
        small_graph = [[1, 1, 0, 0, 0, 0],
                       [1, 0, 2, 4, 0, 0],
                       [1, 1, 0, 0, 0, 0],
                       [1, 4, 0, 0, 0, 0],
                       [3, 1, 3, 5, 0, 0],
                       [1, 4, 0, 0, 0, 0]]
        small_graph = np.array(small_graph)
        test_small_graph = get_a_small_graph()
        passed = np.all(test_small_graph == small_graph)

        self.assertTrue(passed, msg="Small graph not constructed as expected")

    def test_randomadj(self):
        random_adj = get_random_adjacency_matrix(20, density=0.3)
        check1 = np.all(random_adj == random_adj.T)
        check2 = np.all(random_adj < 1.0001)
        random_adj_diag = get_random_adjacency_matrix(20,
                                                      density=0.5,
                                                      degreeOnDiagonal=True)
        ref_diag = random_adj_diag.diagonal()
        test_diag = np.sum(random_adj_diag, axis=0)/2
        print(ref_diag, test_diag)
        check3 = np.all(test_diag == ref_diag)

        passed = check1 and check2 and check3

        failMsg = f"Check1 {check1}, Check2 {check2}, Check3 {check3}"
        self.assertTrue(passed, msg=failMsg)

if __name__ == "__main__":
    unittest.main()
