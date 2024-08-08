import unittest
import numpy as np
import torch
from sedacs.neighbor_list import generate_neighbor_list, NeighborState

def generate_system(N, density, dtype=np.float32):
    vol_to_n = 1.0/density
    box_dim = int((N * vol_to_n)**(1/3))
    box = np.eye(3, dtype=dtype) * box_dim
    R = np.random.rand(N,3).astype(dtype) * box_dim
    return R, box

def generate_nbr_list(coords, box, cutoff, is_periodic, is_dense):
    from matscipy.neighbours import neighbour_list
    if is_periodic:
        pbc = np.array([True, True, True])
    else:
        pbc = np.array([False, False, False])
        box = np.eye(3) * 50.0
    id1, id2 = neighbour_list(quantities="ij",
                                          pbc=pbc,
                                          cell=box,
                                          positions=coords,
                                          cutoff=cutoff)
    if is_dense:
        # convert the neighbors to ELLPACK
        nbr_list = [[] for i in range(len(coords))]
        for i, j in zip(id1, id2):
            nbr_list[i].append(j)


        nbr_counts = [len(l) for l in nbr_list]
        max_c = max(nbr_counts)
        dummy_ind = -1
        for ids in nbr_list:
            diff = max_c - len(ids)
            ids.extend([dummy_ind] * diff)

        nbr_list = np.array(nbr_list)
        return nbr_list
    else:
        return id1, id2
    
def compare_dense(nbr_list1, nbr_list2, err_msg=""):
    nbr_list1 = np.sort(nbr_list1, axis=1)
    nbr_list2 = np.sort(nbr_list2, axis=1)
    np.testing.assert_equal(nbr_list1, nbr_list2, 
                            err_msg=err_msg)
    
def compare_sparse(source1, target1, source2, target2, err_msg=""):
    # sorting
    inds = np.lexsort((source1, target1))
    source1 = source1[inds]
    target1 = target1[inds]

    inds = np.lexsort((source2, target2))
    source2 = source2[inds]
    target2 = target2[inds]

    np.testing.assert_equal(source1, source1, 
                            err_msg=err_msg)
    np.testing.assert_equal(target2, target2, 
                            err_msg=err_msg)   

class TestNeighborList(unittest.TestCase):
    
    def test_periodic(self):
        for N in [10000, 20000]:
            coords, box = generate_system(N, density=0.1, dtype=np.float64)
            coords_t = torch.from_numpy(coords).T.contiguous() # torch expects 3xK
            box_t = torch.from_numpy(box)
            for cutoff in [5.0, 10.0]:
                # dense part
                target_nbr_list_2d = generate_nbr_list(coords, box, cutoff, True, True)
                new_nbr_2d = generate_neighbor_list(coords_t, box_t, cutoff, is_dense=True)
                new_nbr_2d = new_nbr_2d.cpu().numpy()
                compare_dense(new_nbr_2d, target_nbr_list_2d, err_msg=f"Periodic system, N:{N}, cutoff:{cutoff}, dense")
                # sparse part
                target_nbr_i, target_nbr_j = generate_nbr_list(coords, box, cutoff, True, False)
                nbr_i, nbr_j = generate_neighbor_list(coords_t, box_t, cutoff, is_dense=False)
                nbr_i = nbr_i.cpu().numpy()
                nbr_j = nbr_j.cpu().numpy()
                
                compare_sparse(nbr_i, nbr_j, target_nbr_i, target_nbr_j, err_msg=f"Periodic system, N:{N}, cutoff:{cutoff}, sparse")
                
    
    def test_nonperiodic(self):
        for N in [10000, 20000]:
            coords, _ = generate_system(N, density=0.1, dtype=np.float64)
            coords_t = torch.from_numpy(coords).T.contiguous() # torch expects 3xK
            for cutoff in [5.0, 10.0]:
                target_nbr_list_2d = generate_nbr_list(coords, None, cutoff, False, True)
                new_nbr_2d = generate_neighbor_list(coords_t, None, cutoff, is_dense=True)
                new_nbr_2d = new_nbr_2d.cpu().numpy()
                compare_dense(new_nbr_2d, target_nbr_list_2d, err_msg=f"Noneriodic system, N:{N}, cutoff:{cutoff}, dense")
                # sparse part
                target_nbr_i, target_nbr_j = generate_nbr_list(coords, None, cutoff, False, False)
                nbr_i, nbr_j = generate_neighbor_list(coords_t, None, cutoff, is_dense=False)
                nbr_i = nbr_i.cpu().numpy()
                nbr_j = nbr_j.cpu().numpy()
                
                compare_sparse(nbr_i, nbr_j, target_nbr_i, target_nbr_j, err_msg=f"Periodic system, N:{N}, cutoff:{cutoff}, sparse")

    def test_dataclass(self):
        N = 1000
        cutoff = 5.0
        coords, box = generate_system(N, density=0.1, dtype=np.float64)
        coords = torch.from_numpy(coords).T.contiguous() # torch expects 3xK
        box = torch.from_numpy(box)
        state = NeighborState(coords, box, None, cutoff)
        new_nbr_2d = generate_neighbor_list(coords, box, cutoff)
        compare_dense(new_nbr_2d, state.nbr_inds, err_msg=f"Test dataclass, N:{N}, cutoff:{cutoff}, dense")




