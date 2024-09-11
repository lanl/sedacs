"""Hamiltonian
Routines to build a Hamiltonian matrix. Typically
this will be done interfacing with an eng.

"""

import sys
import time
from sedacs.interface_files import get_hamiltonian_files
from sedacs.interface_modules import get_hamiltonian_module
from sedacs.interface_pyseqm import get_fock_pyseqm, ParamContainer, get_hcore_pyseqm, get_molecule_pyseqm
from sedacs.energy import get_eElec
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI
from seqm.seqm_functions.hcore import hcore
import torch
import numpy as np

__all__ = ["get_hamiltonian"]


## Build the non-scf Hamiltonian matrix.
# @brief This will build a Hamiltonian matrix. Typically this will be done interfacing with an eng.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param coords Positions for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @verbose Verbosity
#
def get_hamiltonian(eng, coords, types, symbols,
                    partsIndex, partsCoreHaloIndex, molSysData, P, doForces = False,
                    verbose=False):
    # Call the proper interface
    # If there is no interface, one should write its own Hamiltonian
    if eng.interface == "None":
        raise ValueError("ERROR!!! - Write your own Hamiltonian.")
    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        return get_hamiltonian_module(eng, coords, types, symbols, verb=verbose)
    # Using any available library. We will use MDI here.
    elif eng.interface == "MDI":
        raise NotImplemented("MDI interface not implemented yet")
    # Using unix sockets to interface the codes
    elif eng.interface == "Socket":
        raise NotImplemented("Sockets not implemented yet")
    # Using files as a form of communication and transfering data.
    elif eng.interface == "File":
        return get_hamiltonian_files(eng, coords, types, symbols, verb=verbose)
    elif eng.interface == "PySEQM":
        tic = time.time()
        block_indices = torch.tensor(partsCoreHaloIndex)
        sub_inds = torch.isin(molSysData.molecule_whole.idxi, block_indices) * torch.isin(molSysData.molecule_whole.idxj, block_indices)
        # Define the length of block indices
        block_size = len(block_indices)
        # Vectorize diagonal indices
        maskd_sub = torch.arange(0, block_size * block_size, block_size + 1)  # Diagonal indices
        # Vectorize upper triangle indices
        mask_sub = torch.cat([torch.arange(i * block_size + i + 1, (i + 1) * block_size) for i in range(block_size)])

        molSub = get_molecule_pyseqm(coords, symbols, types)[0]

        M_sub, _, _, _ = hcore(molSub, doTETCI=False)
        #print('  Time to compute hcore', time.time() - tic)
        print("t: h1elNonDi {:>7.3f} |".format(time.time() - tic), end=" ")

        tic = time.time()
        #rij_sub = molSysData.molecule_whole.rij[sub_inds]
        subIndsUnion = torch.isin(molSysData.molecule_whole.idxi, block_indices) + torch.isin(molSysData.molecule_whole.idxj, block_indices)

        coulInts_test, e1b, e2a, _, _ = TETCI(molSysData.molecule_whole.const, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion],
                                                        molSysData.molecule_whole.ni[subIndsUnion], molSysData.molecule_whole.nj[subIndsUnion], molSysData.molecule_whole.xij[subIndsUnion], molSysData.molecule_whole.rij[subIndsUnion], molSysData.molecule_whole.Z,\
                                    molSysData.molecule_whole.parameters['zeta_s'], molSysData.molecule_whole.parameters['zeta_p'], molSysData.molecule_whole.parameters['zeta_d'],\
                                    molSysData.molecule_whole.parameters['s_orb_exp_tail'], molSysData.molecule_whole.parameters['p_orb_exp_tail'], molSysData.molecule_whole.parameters['d_orb_exp_tail'],\
                                    molSysData.molecule_whole.parameters['g_ss'], molSysData.molecule_whole.parameters['g_pp'], molSysData.molecule_whole.parameters['g_p2'], molSysData.molecule_whole.parameters['h_sp'],\
                                    molSysData.molecule_whole.parameters['F0SD'], molSysData.molecule_whole.parameters['G2SD'], molSysData.molecule_whole.parameters['rho_core'],\
                                    molSysData.molecule_whole.alp, molSysData.molecule_whole.chi, molSysData.molecule_whole.method)
        #print('  Time to compute TETCI', time.time() - tic)
        print("TETCI&DiI {:>7.3f} |".format(time.time() - tic), end=" ")

        # for i in range(len(block_indices)):
        #     M_sub[molSub.maskd[i]] += torch.sum(e2a[mol_obj.idxj[subIndsUnion]==block_indices[i]], dim=0) + torch.sum(e1b[mol_obj.idxi[subIndsUnion]==block_indices[i]], dim=0)

        # tic = time.time()
        # idx_to_idx_mapping = {value: idx for idx, value in enumerate(partsCoreHaloIndex)}
        # new_idxi = torch.tensor([idx_to_idx_mapping[value.item()] for value in molSysData.molecule_whole.idxi[torch.isin(molSysData.molecule_whole.idxi, block_indices)]])
        # new_idxj = torch.tensor([idx_to_idx_mapping[value.item()] for value in molSysData.molecule_whole.idxj[torch.isin(molSysData.molecule_whole.idxj, block_indices)]])
        # print('Time to get indices', time.time() - tic)

        # tic = time.time()
        # idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
        # max_key = max(idx_to_idx_mapping.keys())
        # lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long)
        # # Populate the lookup tensor
        # for key, value in idx_to_idx_mapping.items():
        #     lookup_tensor[key] = value
        # new_idxi = lookup_tensor[molSysData.molecule_whole.idxi[torch.isin(molSysData.molecule_whole.idxi, block_indices)]]
        # new_idxj = lookup_tensor[molSysData.molecule_whole.idxj[torch.isin(molSysData.molecule_whole.idxj, block_indices)]]
        # #print('  t indices', time.time() - tic)
        # print("diIndsExp {:>7.3f} |".format(time.time() - tic), end=" ")

        tic = time.time()
        idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
        max_key = max(idx_to_idx_mapping.keys())
        lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long)
        # Populate the lookup tensor
        for key, value in idx_to_idx_mapping.items():
            lookup_tensor[key] = value
        max_i = molSysData.molecule_whole.idxi.max()
        max_j = molSysData.molecule_whole.idxj.max()
        atom_max = max(max_i,max_j)
        in_block_mask = torch.zeros(atom_max+1,dtype=torch.bool)
        in_block_mask[block_indices]=True
        #isini = in_block_mask[idxi].to(torch.bool)
        #isinj = in_block_mask[idxj].to(torch.bool)
        #loc_i = molSysData.molecule_whole.idxi[in_block_mask[idxi].to(torch.bool)]
        #loc_j = molSysData.molecule_whole.idxj[in_block_mask[idxj].to(torch.bool)]
        new_idxi = lookup_tensor[molSysData.molecule_whole.idxi[in_block_mask[molSysData.molecule_whole.idxi].to(torch.bool)]]
        new_idxj = lookup_tensor[molSysData.molecule_whole.idxj[in_block_mask[molSysData.molecule_whole.idxj].to(torch.bool)]]
        print("diIndsExp {:>7.3f} |".format(time.time() - tic), end=" ")
        #print((new_idxi==new_idxi_).all(), (new_idxj==new_idxj_).all())

        # torch.save(block_indices, 'block_indices.pt')
        # torch.save(molSysData.molecule_whole.idxi, 'idxi.pt')
        # torch.save(molSysData.molecule_whole.idxj, 'idxj.pt')

        # torch.save(new_idxi, 'new_idxi.pt')
        # torch.save(new_idxj, 'new_idxj.pt')
        # exit(0)
        # exit(0)

        
        tic = time.time()
        ### $$$ index_add_ is very slow!
        M_sub.index_add_(0,molSub.maskd[new_idxi], e1b[torch.isin(molSysData.molecule_whole.idxi[subIndsUnion], block_indices)])
        # M_sub = torch.index_add(M_sub, 0,molSub.maskd[new_idxi], e1b[torch.isin(molSysData.molecule_whole.idxi[subIndsUnion], block_indices)])

        M_sub.index_add_(0,molSub.maskd[new_idxj], e2a[torch.isin(molSysData.molecule_whole.idxj[subIndsUnion], block_indices)])
        del e1b, e2a, idx_to_idx_mapping, _

        #torch.cuda.synchronize()
        #print('  Time to update hcore diag', time.time() - tic)
        print("h1elDiUpd {:>7.3f} |".format(time.time() - tic), end=" ")

        #print(sub_inds.shape, subIndsUnion.shape, new_idxi.shape, new_idxj.shape)
        # M_sub_from_whole = torch.zeros(len(block_indices)*len(block_indices),4,4)
        # M_sub_from_whole[maskd_sub] = molSysData.M_whole[molSysData.molecule_whole.maskd[block_indices]]
        # M_sub_from_whole[mask_sub] = molSysData.M_whole[molSysData.molecule_whole.mask[sub_inds]]
        # print(torch.sum(abs(M_sub - M_sub_from_whole)))

        # ham_from_whole =      get_fock_pyseqm(P_diag, P_sub, M_sub, molSysData.w_whole[subIndsUnion], block_indices,
        #         molSysData.molecule_whole.nmol, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion], rij_sub,
        #         molSysData.molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
        
        tic = time.time()
        P_sub = torch.zeros(len(block_indices)*len(block_indices),4,4)
        P_sub[maskd_sub] = P[molSysData.molecule_whole.maskd[block_indices]]
        P_sub[mask_sub] = P[molSysData.molecule_whole.mask[sub_inds]]
        P_diag = P[molSysData.molecule_whole.maskd]

        ham = get_fock_pyseqm(P_diag, P_sub, M_sub, coulInts_test, block_indices,
                molSysData.molecule_whole.nmol, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion], molSub.rij,
                molSysData.molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
        del coulInts_test, sub_inds, subIndsUnion, new_idxi, new_idxj
        #print('  Time to compute Fock', time.time() - tic)
        print("FulSubFock {:>7.3f} |".format(time.time() - tic), end=" ")

        if doForces:
            h1elec_sub = M_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                     .transpose(2,3) \
                     .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            dm = P_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                     .transpose(2,3) \
                     .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            eElec = get_eElec(eng, dm, ham, h1elec_sub)

            L = eElec.sum()
            torch.save(P_sub, 'ham_n.pt')
            print(ham)
            print(eElec)
            #print(torch.sum(abs(coulInts_test - molSysData.w_whole[subIndsUnion])))
            L.backward(
                retain_graph=True
                )
            force = -molSysData.molecule_whole.coordinates.grad.detach()[0][partsIndex]
            molSysData.molecule_whole.coordinates.grad.zero_()
            print(force)

            force2 = -molSub.coordinates.grad.detach()[0][np.isin(partsCoreHaloIndex, partsIndex)]
            molSub.coordinates.grad.zero_()
            print(force2)
            print(force + force2)
            exit()
            #molSysData.molecule_whole.coordinates.grad.zero_()
            #force[0][np.isin(partsCoreHaloIndex, partsIndex)]

            return force



        return ham
        


    raise ValueError(f"ERROR!!!: Interface type not recognized: '{eng.interface}'. " +
                     f"Use any of the following: Module,File,Socket,MDI")

