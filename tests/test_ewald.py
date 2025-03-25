from sedacs.ewald import ewald_real, ewald_kspace_part1, ewald_kspace_part2
from sedacs.ewald import construct_kspace
from sedacs.ewald import calculate_num_kvecs_dynamic, determine_alpha, CONV_FACTOR
from sedacs.ewald import calculate_PME_energy, init_PME_data, calculate_alpha_and_num_grids
from sedacs.neighbor_list import generate_neighbor_list, calculate_displacement
import pickle
import numpy as np
from ase.io import read
import torch
import os
from absl.testing import parameterized
import math

#TODO: for now, all the tests are on CPU. This means triton code is not tested.
device = "cpu"
dtype = torch.float32
ATOL = 1e-3
RTOL = 1e-4

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

def assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
    kw = {}
    if atol: kw['atol'] = atol
    if rtol: kw['rtol'] = rtol
    with np.errstate(invalid='ignore'):
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)

def calc_num_grids(cell, alpha, t_err):
    denom = 3.0 * math.pow(t_err, 0.2)
    nmesh1 = math.ceil(2.0 * alpha * cell[0][0] / denom)
    nmesh2 = math.ceil(2.0 * alpha * cell[1][1] / denom)
    nmesh3 = math.ceil(2.0 * alpha * cell[2][2] / denom)

    return [nmesh1, nmesh2, nmesh3]

def process_geo(geo_path, cutoff, acc):
    atoms = read(geo_path)
    types = np.array(atoms.get_chemical_symbols())
    charge_O = 0.9
    charge_H = -0.45

    charges = np.zeros(len(types))
    charges[types=="H"] = charge_H
    charges[types=="O"] = charge_O
    
    cell = np.array(atoms.cell)
    positions = atoms.get_positions()
    nbr_list, nbr_list_dist, nbr_list_disp = generate_nbr_list(positions, cell, cutoff, is_periodic=True, 
                                                                is_dense=True, with_extra_info=True)
    nbr_list = torch.from_numpy(nbr_list)
    nbr_list_dist = torch.from_numpy(nbr_list_dist)
    nbr_list_disp = torch.from_numpy(nbr_list_disp)
    alpha = determine_alpha(charges, acc, cutoff, cell)
    alpha = float(alpha)
    cutoff_kspace, kcounts = calculate_num_kvecs_dynamic(charges, cell, acc, alpha)

    grid_dimensions = calc_num_grids(cell, alpha, acc)
    positions = torch.from_numpy(positions).type(dtype)
    positions = positions.T.contiguous()
    cell = torch.from_numpy(cell).type(dtype)
    charges = torch.from_numpy(charges).type(dtype)
    
    I, kvecs = construct_kspace(cell, kcounts, cutoff_kspace, alpha)
    alpha = torch.tensor(alpha) 

    new_nbr_2d = generate_neighbor_list(positions, cell, cutoff, is_dense=True)
    lattice_lengths = torch.norm(cell, dim=1)
    new_nbr_list_disp = calculate_displacement(positions, new_nbr_2d, lattice_lengths)
    new_nbr_list_dist = torch.norm(new_nbr_list_disp, dim=0)
    new_nbr_list_dist = torch.where(new_nbr_2d == -1, 1.0, new_nbr_list_dist)

    PME_data = init_PME_data(grid_dimensions, cell, alpha, order=6)
    
    return {"sys_data":[positions, cell, charges],
            "nbr_data_matscipy":[nbr_list, nbr_list_dist, nbr_list_disp],
            "nbr_data_sedacs":[new_nbr_2d, new_nbr_list_dist, new_nbr_list_disp],
            "ewald_data":[alpha, I, kvecs],
            "PME_data":list(PME_data)}
    

TEST_DATA = []
def read_test_data(test_folder):
    cutoff = 10.0
    acc = 1e-6
    for root, sub_dirs, _ in os.walk(test_folder):
        for dir in sub_dirs:
            test_name = dir
            geo_path = f"{root}/{dir}/geo.pdb"
            results_path = f"{root}/{dir}/ewald_data.pkl"
            geo_data = process_geo(geo_path, cutoff, acc)
            for k in geo_data.keys():
                if type(geo_data[k]) == list:
                    for j in range(len(geo_data[k])):
                        if torch.is_tensor(geo_data[k][j]):
                            geo_data[k][j] = geo_data[k][j].to(device)
                    

                    
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                
            item = {"name":test_name,
                    "geo_path":geo_path,
                    "geo_data":geo_data,
                    "results":results,
                    "cutoff":cutoff,
                    "acc":acc}
            TEST_DATA.append(item)

read_test_data("tests/data")

class EwaldTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
    
    @parameterized.parameters(
     [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
    def test_real_part(self, i, name):
        # move data to device
        geo_data = TEST_DATA[i]['geo_data']  
        results = TEST_DATA[i]['results']           
        alpha, I, kvecs = geo_data["ewald_data"]
        alpha = float(alpha)
        positions, cell, charges = geo_data["sys_data"]
        nbr_list, nbr_list_dist, nbr_list_disp = geo_data["nbr_data_sedacs"]
        N = len(charges)
        en, forces, dq = ewald_real(nbr_list,
                                    nbr_list_disp, nbr_list_dist, charges,  
                                    alpha=alpha, cutoff=10.0, calculate_forces=1,
                                    calculate_dq=1)
        en = en * CONV_FACTOR
        forces = forces * CONV_FACTOR   
        
        assert_numpy_allclose(float(en), results['real_energy'], 
                              atol=ATOL, rtol=RTOL, err_msg="Ewald real - energy")
        assert_numpy_allclose(forces.detach().cpu().numpy(), results['real_forces'],
                               atol=ATOL, rtol=RTOL, err_msg="Ewald real - forces")

    @parameterized.parameters(
     [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
    def test_kspace_part(self, i, name):
        # move data to device
        geo_data = TEST_DATA[i]['geo_data']  
        results = TEST_DATA[i]['results']           
        alpha, I, kvecs = geo_data["ewald_data"]
        alpha = float(alpha)
        positions, cell, charges = geo_data["sys_data"]
        nbr_list, nbr_list_dist, nbr_list_disp = geo_data["nbr_data_sedacs"]
        
        my_r_vals, my_i_vals = ewald_kspace_part1(positions, charges, kvecs) 
        r_sum = torch.sum(my_r_vals, axis=1)
        i_sum = torch.sum(my_i_vals, axis=1)
        vol = torch.det(cell)
        en, forces, dq = ewald_kspace_part2(r_sum, i_sum, 
                                            my_r_vals, my_i_vals, 
                                            vol, kvecs, I,
                                            charges,
                                            positions,
                                            calculate_forces=1,
                                            calculate_dq=1)
        
        en = en * CONV_FACTOR
        forces = forces * CONV_FACTOR 
        
        assert_numpy_allclose(float(en), results['kspace_energy'], 
                              atol=ATOL, rtol=RTOL, err_msg="Ewald kspace - energy")
        assert_numpy_allclose(forces.detach().cpu().numpy(), results['kspace_forces'],
                               atol=ATOL, rtol=RTOL, err_msg="Ewald kspace - forces")
        

    @parameterized.parameters(
     [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
    def test_PME(self, i, name):
        geo_data = TEST_DATA[i]['geo_data']  
        results = TEST_DATA[i]['results']           
        positions, cell, charges = geo_data["sys_data"]
        nbr_list, nbr_list_dist, nbr_list_disp = geo_data["nbr_data_sedacs"]

        PME_data = geo_data['PME_data'] 
        alpha = geo_data['ewald_data'][0]

        charges.grad = None
        charges.requires_grad = True
        
        positions.grad = None
        positions.requires_grad = True
        pme_e = calculate_PME_energy(positions, charges, cell, alpha, PME_data) * CONV_FACTOR
        pme_e.backward()
        forces = -1 * positions.grad
        dq = charges.grad
    
        charges.requires_grad = False
        positions.requires_grad = False
        positions.grad = None
        charges.grad = None

        
        assert_numpy_allclose(float(pme_e), results['kspace_energy'], 
                              atol=ATOL, rtol=RTOL, err_msg="Ewald kspace - energy")
        assert_numpy_allclose(forces.detach().cpu().numpy(), results['kspace_forces'],
                               atol=ATOL, rtol=RTOL, err_msg="Ewald kspace - forces")