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

def generate_system_from_box(density, box_dims, dtype=np.float32):
    total_vol = np.prod(box_dims)
    N = int(total_vol * density)
    box = np.eye(3, dtype=dtype)
    for i in range(3):
        box[i,i] = box_dims[i]

    R = np.random.rand(N,3).astype(dtype)
    for i in range(3):
        R[:,i] = R[:,i] * box_dims[i]

    return R, box

def generate_sedacs_neighbor_list(coords, box, cutoff, is_dense):
    '''
    Generate and sort the neighbor list to make the comparison easier
    '''
    state = NeighborState(coords, box, None, cutoff, is_dense)
    nbr_inds = state.nbr_inds.cpu().numpy()
    dists = state.calculate_distance(coords).cpu().numpy()
    dists = np.where(dists > cutoff, 0, dists)
    disps = state.calculate_displacement(coords).cpu().numpy()
    disps = np.where(dists[None, ...] == 0, 0, disps)
    shifts = state.calculate_shift(coords).cpu().numpy()
    shifts = np.where(dists[None, ...] == 0, 0, shifts)

    if is_dense:
        disps = disps.transpose(1,2,0)
        shifts = shifts.transpose(1,2,0)
        atom_ids = np.arange(len(nbr_inds)).reshape(-1,1)
        inds = np.argsort(nbr_inds, axis=1)
        nbr_inds = nbr_inds[atom_ids,inds]
        dists = dists[atom_ids,inds] * (nbr_inds != -1)
        disps = disps[atom_ids,inds] * (nbr_inds != -1)[..., None]
        shifts = shifts[atom_ids,inds] * (nbr_inds != -1)[..., None]
    else:
        disps = disps.transpose(1,0)
        shifts = shifts.transpose(1,0)
        inds = np.lexsort((nbr_inds[0], nbr_inds[1]))
        nbr_inds[0] = nbr_inds[0][inds]
        nbr_inds[1] = nbr_inds[1][inds]
        dists = dists[inds]
        disps = disps[inds]
        shifts = shifts[inds]
    return (nbr_inds, dists, 
            disps, shifts)


def generate_target_neigbor_list(coords, box, cutoff, is_periodic, is_dense):
    '''
    Generate and sort the neighbor list to make the comparison easier
    '''
    from matscipy.neighbours import neighbour_list
    if is_periodic:
        pbc = np.array([True, True, True])
    else:
        pbc = np.array([False, False, False])
        box = np.eye(3) * 50.0
    [id1, id2, 
     dists, disps, shifts] = neighbour_list(quantities="ijdDS",
                                          pbc=pbc,
                                          cell=box,
                                          positions=coords,
                                          cutoff=cutoff)
    dists = np.where(dists > cutoff, 0, dists)
    dist_cond = dists.reshape(-1,1) == 0.0
    disps = np.where(dist_cond, 0, disps)
    shifts = np.where(dist_cond, 0, shifts)
    if is_dense:
        # convert the neighbors to ELLPACK
        nbr_list = [[] for i in range(len(coords))]
        nbr_dists = [[] for i in range(len(coords))]
        nbr_disps = [[] for i in range(len(coords))]
        nbr_shifts = [[] for i in range(len(coords))]
        for i, j, d, disp, shift in zip(id1, id2, dists, disps, shifts):
            nbr_list[i].append(j)
            nbr_dists[i].append(d)
            nbr_disps[i].append(list(disp))
            nbr_shifts[i].append(list(shift))


        nbr_counts = [len(l) for l in nbr_list]
        max_c = max(nbr_counts)
        dummy_ind = -1
        for i in range(len(nbr_list)):
            diff = max_c - len(nbr_list[i])
            nbr_list[i].extend([dummy_ind] * diff)
            nbr_dists[i].extend([0.0] * diff)
            nbr_disps[i].extend([[0.0,0.0,0.0]] * diff)
            nbr_shifts[i].extend([[0.0,0.0,0.0]] * diff)


        nbr_list = np.array(nbr_list)
        nbr_dists = np.array(nbr_dists)
        nbr_disps = np.array(nbr_disps)
        nbr_shifts = np.array(nbr_shifts)

        atom_ids = np.arange(len(nbr_list)).reshape(-1,1)
        inds = np.argsort(nbr_list, axis=1)
        nbr_list = nbr_list[atom_ids,inds]
        nbr_dists = nbr_dists[atom_ids,inds]
        nbr_disps = nbr_disps[atom_ids,inds]
        nbr_shifts = nbr_shifts[atom_ids,inds]

        return nbr_list, nbr_dists, nbr_disps, nbr_shifts 
    else:
        inds = np.lexsort((id1, id2))
        id1 = id1[inds]
        id2 = id2[inds]
        dists = dists[inds]
        disps = disps[inds]
        shifts = shifts[inds]

        return np.stack((id1, id2)), dists, disps, shifts
    
def compare(inds, dists, disps, shifts, t_inds, t_dists, t_disps, t_shifts, base_msg=""):
    np.testing.assert_equal(inds, t_inds, 
            err_msg=base_msg + ", nbr indices")
    np.testing.assert_almost_equal(dists, t_dists, 
            err_msg=base_msg + ", nbr distances")
    np.testing.assert_almost_equal(disps, t_disps, 
            err_msg=base_msg + ", nbr displacements")
    np.testing.assert_equal(shifts, t_shifts, 
            err_msg=base_msg + ", nbr shifts")  

class TestNeighborList(unittest.TestCase):
    
    def test_periodic(self):
        for device in ['cpu', 'cuda']:
            if device == "cuda" and torch.cuda.is_available() == False:
                print("[test_periodic]: CUDA is not available, skipping the GPU test")
                continue
            for box_dims in [[30.0, 30.0, 30.0], [45.0, 30.0, 30.0], [30.0, 30.0, 60.0]]:
                coords, box = generate_system_from_box(density=0.1, box_dims=box_dims, dtype=np.float64)
                N = len(coords)
                coords_t = torch.from_numpy(coords).T.contiguous() # torch expects 3xK
                box_t = torch.from_numpy(box)
                for cutoff in [5.0, 10.0]:
                    # dense part
                    t_nbr_inds, t_dists, t_disps, t_shifts = generate_target_neigbor_list(coords, box, cutoff, True, True)
                    nbr_inds, dists, disps, shifts = generate_sedacs_neighbor_list(coords_t, box_t, cutoff, is_dense=True)

                    base_msg = f"Periodic system, N:{N}, cutoff:{cutoff}, dense"
                    compare(nbr_inds, dists, disps, shifts, t_nbr_inds, t_dists, t_disps, t_shifts, base_msg=base_msg)
                    
                    # sparse part
                    t_nbr_inds, t_dists, t_disps, t_shifts = generate_target_neigbor_list(coords, box, cutoff, True, False)
                    nbr_inds, dists, disps, shifts = generate_sedacs_neighbor_list(coords_t, box_t, cutoff, is_dense=False)
                    base_msg = f"Periodic system, N:{N}, cutoff:{cutoff}, sparse"
                    compare(nbr_inds, dists, disps, shifts, t_nbr_inds, t_dists, t_disps, t_shifts, base_msg=base_msg)
                    
    
    def test_nonperiodic(self):
        for device in ['cpu', 'cuda']:
            if device == "cuda" and torch.cuda.is_available() == False:
                print("[test_nonperiodic]: CUDA is not available, skipping the GPU test")
                continue
            for N in [1000, 2000]:
                coords, _ = generate_system(N, density=0.1, dtype=np.float64)
                coords_t = torch.from_numpy(coords).T.contiguous() # torch expects 3xK
                coords_t = coords_t.to(device)
                for cutoff in [3.0, 5.0]:
                    t_nbr_inds, t_dists, t_disps, t_shifts = generate_target_neigbor_list(coords, None, cutoff, False, True)
                    nbr_inds, dists, disps, shifts = generate_sedacs_neighbor_list(coords_t, None, cutoff, is_dense=True)
                    base_msg = f"Nonperiodic system, N:{N}, cutoff:{cutoff}, dense"
                    compare(nbr_inds, dists, disps, shifts, t_nbr_inds, t_dists, t_disps, t_shifts, base_msg=base_msg)
                    # sparse part
                    t_nbr_inds, t_dists, t_disps, t_shifts = generate_target_neigbor_list(coords, None, cutoff, False, False)
                    nbr_inds, dists, disps, shifts = generate_sedacs_neighbor_list(coords_t, None, cutoff, is_dense=False)
                    base_msg = f"Nonperiodic system, N:{N}, cutoff:{cutoff}, sparse"
                    compare(nbr_inds, dists, disps, shifts, t_nbr_inds, t_dists, t_disps, t_shifts, base_msg=base_msg)



