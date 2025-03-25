import numpy as np
def generate_system(N, density, dtype=np.float32):
    vol_to_n = 1.0/density
    box_dim = int((N * vol_to_n)**(1/3))
    box = np.eye(3, dtype=dtype) * box_dim
    R = np.random.rand(N,3).astype(dtype) * box_dim
    return R, box

def generate_nbr_list(coords, box, cutoff, is_periodic, is_dense, with_extra_info=False):
    from matscipy.neighbours import neighbour_list
    if is_periodic:
        pbc = np.array([True, True, True])
    else:
        pbc = np.array([False, False, False])
        box = np.eye(3) * 50.0
    id1, id2, dists, disps, shifts = neighbour_list(quantities="ijdDS",
                                          pbc=pbc,
                                          cell=box,
                                          positions=coords,
                                          cutoff=cutoff)
    if is_dense:
        # convert the neighbors to ELLPACK
        nbr_list = [[] for i in range(len(coords))]
        nbr_list_dist = [[] for i in range(len(coords))]
        nbr_list_disp = [[] for i in range(len(coords))]
        nbr_list_shift = [[] for i in range(len(coords))]
        for i, j, d, disp, s in zip(id1, id2, dists, disps, shifts):
            nbr_list[i].append(j)
            nbr_list_dist[i].append(d)
            nbr_list_disp[i].append(disp)
            nbr_list_shift[i].append(s)


        nbr_counts = [len(l) for l in nbr_list]
        max_c = max(nbr_counts)
        dummy_ind = -1
        dummy_dist = 1.0
        dummy_disp = np.array([1.0,1.0,1.0])
        dummy_shift = [-1.0,-1.0,-1.0]
        for ids, ds, disps, shifts in zip(nbr_list, nbr_list_dist, nbr_list_disp, nbr_list_shift):
            diff = max_c - len(ids)
            ids.extend([dummy_ind] * diff)
            ds.extend([dummy_dist] * diff)
            disps.extend([dummy_disp] * diff)
            shifts.extend([dummy_shift] * diff)

        nbr_list = np.array(nbr_list)
        nbr_list_dist = np.array(nbr_list_dist)
        nbr_list_disp = np.array(nbr_list_disp)
        nbr_list_shift = np.array(nbr_list_shift)
        # move [x,y,z] dim to the beginning
        nbr_list_disp = nbr_list_disp.transpose(2,0,1)
        if with_extra_info:
            return nbr_list, nbr_list_dist, nbr_list_disp
        else:
            return nbr_list
    else:
        if with_extra_info:
            return id1, id2, dists, disps
        else:
            return id1, id2
