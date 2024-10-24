"""Hamiltonian
Routines to build a Hamiltonian matrix. Typically
this will be done interfacing with an eng.

"""

import sys
import time
from sedacs.interface_files import get_hamiltonian_files
from sedacs.interface_modules import get_hamiltonian_module
from sedacs.interface_pyseqm import get_fock_pyseqm_2, get_molecule_pyseqm
from sedacs.energy import get_eElec
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI
from seqm.seqm_functions.hcore import hcore
from seqm.seqm_functions.pack import pack
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
                    partsIndex, partsCoreHaloIndex, molSysData, P, P_contr, graph_for_pairs, graph_maskd, core_indices_in_sub_expanded, doForces = False,
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
        block_indices = torch.tensor(partsCoreHaloIndex, dtype=torch.int64, device=P_contr.device)
        # Define the length of block indices
        block_size = torch.tensor(len(block_indices), device = P_contr.device)
        # Vectorize diagonal indices
        maskd_sub = torch.arange(0, block_size * block_size, block_size + 1, device = P_contr.device)  # Diagonal indices
        # Vectorize upper triangle indices
        mask_sub = torch.cat([torch.arange(i * block_size + i + 1, (i + 1) * block_size, device = P_contr.device) for i in range(block_size)])
        #mask_sub_lower_TEST = torch.cat([torch.arange(i * block_size + i-1, i * block_size-1, -1) for i in range(block_size-1, 0,-1)])


        print(P_contr.device, )
        molSub = get_molecule_pyseqm(molSysData.molecule_whole.coordinates[:,partsCoreHaloIndex], symbols, types, device=P_contr.device)[0]#.to(P_contr.device)
        M_sub, _, _, _ = hcore(molSub, doTETCI=False) # non-diagonal h1elec

        #print('  Time to compute hcore', time.time() - tic)
        print("t: h1elNonDi {:>7.3f} |".format(time.time() - tic), end=" ")

        # tic = time.time()
        # ###rij_sub = molSysData.molecule_whole.rij[sub_inds]
        # subIndsUnion_i = torch.isin(molSysData.molecule_whole.idxi, block_indices)
        # subIndsUnion_j = torch.isin(molSysData.molecule_whole.idxj, block_indices)
        # subIndsUnion = subIndsUnion_i + subIndsUnion_j
        # del subIndsUnion_i, subIndsUnion_j
        # #subIndsUnion = torch.isin(molSysData.molecule_whole.idxi, block_indices) + torch.isin(molSysData.molecule_whole.idxj, block_indices)
        # print("subIndsUnion {:>7.3f} |".format(time.time() - tic), end=" ")

        tic = time.time()
        ### first doing idxi because its sorted
        # Searchsorted gives you the indices where the elements should be placed to maintain order. Works with idxi (sorted) but not with idxj (not sorted)
        pos = torch.searchsorted(block_indices, molSysData.molecule_whole.idxi)
        # Ensure the indices are within bounds
        pos = torch.clamp(pos, max=len(block_indices) - 1)
        # Check if the positions are valid and match
        subIndsUnion_i = (pos < len(block_indices)) & (block_indices[pos] == molSysData.molecule_whole.idxi)

        ### second, doing indx i because its a sequence of sorted maxtrix triangle rows
        start_ind = 0
        end_ind = molSysData.molecule_whole.molsize - 1
        subIndsUnion_j = torch.zeros(int((molSysData.molecule_whole.molsize*(molSysData.molecule_whole.molsize-1)/2)), dtype=torch.bool, device=P_contr.device)
        tmp_j = molSysData.molecule_whole.idxj[start_ind:end_ind]
        pos = torch.searchsorted(block_indices, tmp_j)
        pos = torch.clamp(pos, max=len(block_indices) - 1)
        valid_top_row = (pos < len(block_indices)) & (block_indices[pos] == tmp_j)
        del tmp_j, pos
        for i in range(0,molSysData.molecule_whole.molsize): ### $$$ needs vecorization
            subIndsUnion_j[start_ind:end_ind] = valid_top_row[i:]
            start_ind = end_ind
            end_ind = end_ind + molSysData.molecule_whole.molsize - i - 2

        subIndsUnion = subIndsUnion_i + subIndsUnion_j
        del subIndsUnion_i, subIndsUnion_j, valid_top_row
        #subIndsUnion = torch.isin(molSysData.molecule_whole.idxi, block_indices) + torch.isin(molSysData.molecule_whole.idxj, block_indices)
        print("subIndsUnion {:>7.3f} |".format(time.time() - tic), end=" ")
        
        tic = time.time()
        coulInts_test, e1b, e2a, _, _ = TETCI(molSysData.molecule_whole.const, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion],
                molSysData.molecule_whole.ni[subIndsUnion], molSysData.molecule_whole.nj[subIndsUnion], molSysData.molecule_whole.xij[subIndsUnion], molSysData.molecule_whole.rij[subIndsUnion], molSysData.molecule_whole.Z,\
                molSysData.molecule_whole.parameters['zeta_s'], molSysData.molecule_whole.parameters['zeta_p'], molSysData.molecule_whole.parameters['zeta_d'],\
                molSysData.molecule_whole.parameters['s_orb_exp_tail'], molSysData.molecule_whole.parameters['p_orb_exp_tail'], molSysData.molecule_whole.parameters['d_orb_exp_tail'],\
                molSysData.molecule_whole.parameters['g_ss'], molSysData.molecule_whole.parameters['g_pp'], molSysData.molecule_whole.parameters['g_p2'], molSysData.molecule_whole.parameters['h_sp'],\
                molSysData.molecule_whole.parameters['F0SD'], molSysData.molecule_whole.parameters['G2SD'], molSysData.molecule_whole.parameters['rho_core'],\
                molSysData.molecule_whole.alp, molSysData.molecule_whole.chi, molSysData.molecule_whole.method)
        #print('  Time to compute TETCI', time.time() - tic)
        print("TETCI&DiI {:>7.3f} |".format(time.time() - tic), end=" ")

        tic = time.time()
        idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
        max_key = max(idx_to_idx_mapping.keys())
        lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long, device = P_contr.device)
        # Populate the lookup tensor
        for key, value in idx_to_idx_mapping.items():
            lookup_tensor[key] = value
        max_i = molSysData.molecule_whole.idxi.max()
        max_j = molSysData.molecule_whole.idxj.max()
        atom_max = max(max_i,max_j)
        in_block_mask = torch.zeros(atom_max+1,dtype=torch.bool, device = P_contr.device)
        in_block_mask[block_indices]=True
        new_idxi = lookup_tensor[molSysData.molecule_whole.idxi[in_block_mask[molSysData.molecule_whole.idxi].to(torch.bool)]]
        new_idxj = lookup_tensor[molSysData.molecule_whole.idxj[in_block_mask[molSysData.molecule_whole.idxj].to(torch.bool)]]
        print("diIndsExp {:>7.3f} |".format(time.time() - tic), end=" ")
        
        # add diagonal to h1elec
        tic = time.time()
        ### $$$ index_add_ is very slow!
        M_sub.index_add_(0,molSub.maskd[new_idxi], e1b[torch.isin(molSysData.molecule_whole.idxi[subIndsUnion], block_indices)])
        M_sub.index_add_(0,molSub.maskd[new_idxj], e2a[torch.isin(molSysData.molecule_whole.idxj[subIndsUnion], block_indices)])
        del e1b, e2a, _

        #torch.cuda.synchronize()
        #print('  Time to update hcore diag', time.time() - tic)
        print("h1elDiUpd {:>7.3f} |".format(time.time() - tic), end=" ")
        
        tic = time.time()
        
        P_sub_from_contr = torch.zeros(len(block_indices)*len(block_indices),4,4, device = P_contr.device)
        P_sub_from_contr = P_sub_from_contr.reshape(len(block_indices), len(block_indices), 4,4)

        graph_for_pairs = torch.from_numpy(graph_for_pairs).to(P_contr.device)

        for i in range(len(block_indices)): ### $$$ needs vecorization
            if block_indices[i] in partsIndex:
                
                #P_sub_from_contr[i] = P_contr[block_indices[i]][:graph_for_pairs[block_indices[i]][0]]
                #P_sub_from_contr[:,i] = P_contr[block_indices[i]][:graph_for_pairs[block_indices[i]][0]].transpose(1,2)
                #P_sub_from_contr[:,i] = P_contr[block_indices[i]][:graph_for_pairs[block_indices[i]][0]]

                P_sub_from_contr[:,i] = P_contr[:,block_indices[i]][:graph_for_pairs[block_indices[i]][0]]
                # if block_indices[i] == 36:
                #     print(block_indices[i], i)
                #     print(P_contr[:,block_indices[i]])
                # P_sub_from_contr[i] = P_contr[:,block_indices[i]][:graph_for_pairs[block_indices[i]][0]].transpose(1,2)
            else:
                P_sub_from_contr[:,i][lookup_tensor[graph_for_pairs[block_indices[i]][1:graph_for_pairs[block_indices[i]][0]+1][torch.isin( graph_for_pairs[block_indices[i]][1:graph_for_pairs[block_indices[i]][0]+1] , block_indices)]]] = \
                    P_contr[:,block_indices[i]][:graph_for_pairs[block_indices[i]][0]][[torch.isin( graph_for_pairs[block_indices[i]][1:graph_for_pairs[block_indices[i]][0]+1], block_indices)]]


        if eng.reconstruct_dm:
            sub_inds = torch.isin(molSysData.molecule_whole.idxi, block_indices) * torch.isin(molSysData.molecule_whole.idxj, block_indices)
            # Vectorize lower triangle indices
            mask_sub_lower = torch.cat([torch.arange(i * block_size, i * block_size + i) for i in range(1, block_size)])

            P_sub = torch.zeros(len(block_indices)*len(block_indices),4,4) # coreHalo DM
            P_sub[maskd_sub] = P[molSysData.molecule_whole.maskd[block_indices]]
            P_sub[mask_sub] = P[molSysData.molecule_whole.mask[sub_inds]]
            ### $$$ should be a better way
            mask_lower_triag_in_whole = torch.sort(molSysData.molecule_whole.mask[sub_inds]//(molSysData.molecule_whole.molsize) + molSysData.molecule_whole.mask[sub_inds]%(molSysData.molecule_whole.molsize)*molSysData.molecule_whole.molsize)[0]
            P_sub[mask_sub_lower] = P[mask_lower_triag_in_whole]#.transpose(1,2)
            P_diag = P[molSysData.molecule_whole.maskd] # diagonal DM of the whole
            del sub_inds, mask_sub_lower

        P_sub_from_contr = P_sub_from_contr.reshape(len(block_indices)*len(block_indices), 4,4)
        P_diag_contr = P_contr.transpose(0,1).reshape(molSysData.molecule_whole.molsize*(len(graph_for_pairs[0])-1), 4,4)[graph_maskd]#.transpose(0,1)

        #print('ERR',torch.sum(abs(P_sub_from_contr- P_sub[:])))
        # print(P_sub.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #              .transpose(2,3) \
        #              .reshape(1, 4*molSub.molsize, 4*molSub.molsize),'\n')
        
        # print(P_sub_from_contr.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #              .transpose(2,3) \
        #              .reshape(1, 4*molSub.molsize, 4*molSub.molsize),'\n')
        
            
        # print((P_sub.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #             .transpose(2,3) \
        #             .reshape(1, 4*molSub.molsize, 4*molSub.molsize) - P_sub_from_contr.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #             .transpose(2,3) \
        #             .reshape(1, 4*molSub.molsize, 4*molSub.molsize)))

        # print((P_sub.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #             .transpose(2,3) \
        #             .reshape(1, 4*molSub.molsize, 4*molSub.molsize) - P_sub_from_contr.reshape(1, molSub.molsize, molSub.molsize,4,4) \
        #             .transpose(2,3) \
        #             .reshape(1, 4*molSub.molsize, 4*molSub.molsize))[:,:,31*4:32*4])
    
        # ERR2 = abs((P_sub_from_contr.reshape(molSub.molsize, molSub.molsize,4,4)- P_sub.reshape(molSub.molsize, molSub.molsize,4,4)))
        # for i in range(len(block_indices)):
        #     for j in range(len(block_indices)):
        #         if torch.sum(ERR2[i:i+1][:,j:j+1]) > 0.0001:
        #             print(i,j)
        #             print(block_indices[i],block_indices[j])
        #             print('ERR2',torch.sum(ERR2[i:i+1][:,j:j+1]))

        ham_contr = get_fock_pyseqm_2(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, block_indices,
                molSysData.molecule_whole.nmol, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion], molSub.rij,
                molSysData.molecule_whole.parameters, maskd_sub, mask_sub) # slowest part

        print("FulSubFock {:>7.3f} |".format(time.time() - tic), end=" ")


        if eng.reconstruct_dm:
            h1elec_sub = M_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                    .transpose(2,3) \
                    .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)            
            h1elec_sub = h1elec_sub.triu()+h1elec_sub.triu(1).transpose(1,2)

            dm_contr = P_sub_from_contr.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                    .transpose(2,3) \
                    .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            
            #eElec_contr  = 0.5*torch.sum(dm_contr[:,core_indices_in_sub_expanded,:]*(h1elec_sub[:,core_indices_in_sub_expanded,:]+ham_contr[:,core_indices_in_sub_expanded,:]),dim=(1,2))
            eElec_contr  = 0.5*torch.sum(dm_contr[:,:,core_indices_in_sub_expanded]*(h1elec_sub[:,:,core_indices_in_sub_expanded]+ham_contr[:,:,core_indices_in_sub_expanded]),dim=(1,2))
        
            ham = get_fock_pyseqm_2(P_diag, P_sub, M_sub, coulInts_test, block_indices,
                    molSysData.molecule_whole.nmol, molSysData.molecule_whole.idxi[subIndsUnion], molSysData.molecule_whole.idxj[subIndsUnion], molSub.rij,
                    molSysData.molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
            
            dm = P_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                    .transpose(2,3) \
                    .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            eElec = 0.5*torch.sum(dm[:,:,core_indices_in_sub_expanded]*(h1elec_sub[:,:,core_indices_in_sub_expanded]+ham[:,:,core_indices_in_sub_expanded]),dim=(1,2))

            if abs(eElec_contr- eElec).sum() > 0.0000001:
                print()
                print('ERR EN',abs(eElec_contr- eElec).sum())
                print('ERR H',torch.sum(abs(ham_contr- ham)))
                print('ERR DM',torch.sum(abs(dm_contr- dm)))

            diag_err = torch.sum(abs(P_diag_contr-P_diag))
            print('diag_err', diag_err)

        del coulInts_test, subIndsUnion, new_idxi, new_idxj, P_diag_contr, maskd_sub, mask_sub, idx_to_idx_mapping, lookup_tensor

        if doForces:
            
            tic = time.time()
            h1elec_sub = M_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                    .transpose(2,3) \
                    .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize) 
            h1elec_sub = h1elec_sub.triu()+h1elec_sub.triu(1).transpose(1,2)

            dm_contr = P_sub_from_contr.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                    .transpose(2,3) \
                    .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            
            core_indices_in_sub_expanded#.to(dm_contr.device)

            eElec_contr  = 0.5*(dm_contr[:,:,core_indices_in_sub_expanded]*(h1elec_sub[:,:,core_indices_in_sub_expanded]+ham_contr[:,:,core_indices_in_sub_expanded])).sum()
            del dm_contr, h1elec_sub, ham_contr, P_sub_from_contr, M_sub, molSub
            L = eElec_contr.sum()
            print("En {:>7.3f} |".format(time.time() - tic), end=" ")

            tic = time.time()
            #del dm_contr, P_sub_from_contr, h1elec_sub, ham_contr, molSub
            L.backward(retain_graph=True)


            force = -molSysData.molecule_whole.coordinates.grad.detach()[0].cpu().numpy()
            molSysData.molecule_whole.coordinates.grad.zero_()
            
            if eng.reconstruct_dm:
                dm = P_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                     .transpose(2,3) \
                     .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
                eElec = 0.5*(dm[:,:,core_indices_in_sub_expanded]*(h1elec_sub[:,:,core_indices_in_sub_expanded]+ham[:,:,core_indices_in_sub_expanded])).sum()
                L = eElec.sum()
                L.backward(retain_graph=True)
                force_FULLDM = -molSysData.molecule_whole.coordinates.grad.detach()[0].numpy()
                molSysData.molecule_whole.coordinates.grad.zero_()
                del dm, ham, P_sub
                print('Force ERR',np.sum(abs(force_FULLDM - force)))
                
            del  L
            print("Force {:>7.3f} |".format(time.time() - tic), end=" ")
            torch.cuda.empty_cache()
            return force, eElec_contr.detach().cpu()
        return ham_contr
        


    raise ValueError(f"ERROR!!!: Interface type not recognized: '{eng.interface}'. " +
                     f"Use any of the following: Module,File,Socket,MDI")

