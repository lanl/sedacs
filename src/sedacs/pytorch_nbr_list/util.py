#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:53:14 2024

@author: mkaymak
"""
import numpy as np
import torch

def generate_system(N, density, dtype=np.float32):
    vol_to_n = 1.0/density
    box_dim = int((N * vol_to_n)**(1/3))
    box = np.eye(3, dtype=dtype) * box_dim
    R = np.random.rand(N,3).astype(dtype) * box_dim
    return R, box

def generate_nbr_list(coords, box, cutoff, device='cpu'):
    from matscipy.neighbours import neighbour_list
    pbc = np.array([True, True, True])
    id1, id2 = neighbour_list(quantities="ij",
                                          pbc=pbc,
                                          cell=box,
                                          positions=coords,
                                          cutoff=cutoff)
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


    nbr_list = torch.from_numpy(nbr_list).type(torch.int32).to(device)

    return nbr_list, id1, id2
