"""Hamiltonian
Routines to build a Hamiltonian matrix. Typically
this will be done interfacing with an eng.

"""

import sys
import time
from sedacs.interface_files import get_hamiltonian_files
from sedacs.interface_modules import get_hamiltonian_module
from sedacs.interface.pyseqm import get_fock_pyseqm, get_fock_pyseqm_u, get_molecule_pyseqm
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI
from seqm.seqm_functions.hcore import hcore
from seqm.seqm_functions.pack import pack
import torch
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import numpy as np
from mpi4py import MPI
import psutil
import gc

__all__ = ["get_hamiltonian"]

def get_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                yield obj
        except Exception as e:
            pass
def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)

## Build the non-scf Hamiltonian matrix.
# @brief This will build a Hamiltonian matrix. Typically this will be done interfacing with an eng.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param coords Positions for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @verbose Verbosity
#
def print_memory_usage(rank, node_rank, message):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message} | Rank: {rank}, Node Rank: {node_rank}, Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def get_hamiltonian(sdc, eng, coords, types, symbols,
                    partsIndex, partsCoreHaloIndex, molecule_whole, P_contr, graph_for_pairs, graph_maskd, core_indices_in_sub_expanded, ham_timing, doForces = False, verbose=False):
    '''
    Function constructs Fock matrix. Calculates electronic energy and electronic forces if doForces. If doForces==False, just electronic energy.
    sdc:
    eng:
    coords: coordanates of CH, list(n_atoms, 3)
    types: atoms types
    symbols: symbols corresponding to atoms types
    partsIndex: core
    partsCoreHaloIndex: core+halo
    partsCoreHalo: list of core+halo
    molecule_whole: pyseqm molecule object for the whole system
    P_contr: contracted dm. (sy.nats, sdc.maxDeg, 4,4)
    graph_for_pairs: graph of communities. E.g. graph_for_pairs[i] is a whole CH community in which atom i is, including itself. graph_for_pairs[i][0] is a community size
    graph_maskd: diagonal mask for P_contr
    core_indices_in_sub_expanded_list: indices of core columns in CH. E.g., CH[i] contains atoms [0,1,2,3], core atoms are [1,3], 4 AOs per atom. Then, core_indices_in_sub_expanded_list[i] is [4,5,6,7, 12,13,14,15].
    doForces: flag for forces computation. backprop.
    '''
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
        block_indices = torch.tensor(partsCoreHaloIndex, dtype=eng.torch_int_dt, device=P_contr.device) # core+halo as tensor
        # Define the length of block indices
        block_size = torch.tensor(len(block_indices), device = P_contr.device, dtype=eng.torch_int_dt) # size of core+halo
        # Vectorize diagonal indices
        maskd_sub = torch.arange(0, block_size * block_size, block_size + 1, device = P_contr.device)  # Diagonal indices of core+halo hamiltonian
        # Vectorize upper triangle indices
        mask_sub = torch.cat([torch.arange(i * block_size + i + 1, (i + 1) * block_size, device = P_contr.device) for i in range(block_size)])
        #mask_sub_lower_TEST = torch.cat([torch.arange(i * block_size + i-1, i * block_size-1, -1) for i in range(block_size-1, 0,-1)])

        if doForces: # pyseqm molecule object for core+halo
            molSub = get_molecule_pyseqm(sdc, molecule_whole.coordinates[:,partsCoreHaloIndex], symbols, types, device=P_contr.device)[0]
        else:
            with torch.no_grad():
                molSub = get_molecule_pyseqm(sdc, molecule_whole.coordinates[:,partsCoreHaloIndex], symbols, types, device=P_contr.device)[0]
        M_sub, _, __, ___ = hcore(molSub, doTETCI=False) # off-diagonal h1elec
        del _, __, ___
        ham_timing['h1elNonDi'] = time.time() - tic
        #print("t: h1elNonDi {:>7.3f} |".format(time.time() - tic), end=" ")

        if eng.use_pyseqm_lt: # if pyseqm "large tensors" are pre-computed (e.g. idxi, idxj, masks), these will be sliced and used. Not recommended for large systems
            # subIndsUnion_i = torch.isin(molecule_whole.idxi, block_indices)
            # subIndsUnion_j = torch.isin(molecule_whole.idxj, block_indices)
            # subIndsUnion = subIndsUnion_i + subIndsUnion_j

            tic = time.time()
            ### first doing idxi because its sorted
            # Searchsorted gives you the indices where the elements should be placed to maintain order. Works with idxi (sorted) but not with idxj (not sorted)
            pos = torch.searchsorted(block_indices, molecule_whole.idxi)
            # Ensure the indices are within bounds
            pos = torch.clamp(pos, max=len(block_indices) - 1)
            # Check if the positions are valid and match
            subIndsUnion_i = (pos < len(block_indices)) & (block_indices[pos] == molecule_whole.idxi)

            ### second, doing indx i because its a sequence of sorted maxtrix triangle rows
            start_ind = 0
            end_ind = molecule_whole.molsize - 1
            subIndsUnion_j = torch.zeros(int((molecule_whole.molsize*(molecule_whole.molsize-1)/2)), dtype=torch.bool, device=P_contr.device)
            tmp_j = molecule_whole.idxj[start_ind:end_ind]
            pos = torch.searchsorted(block_indices, tmp_j)
            pos = torch.clamp(pos, max=len(block_indices) - 1)
            valid_top_row = (pos < len(block_indices)) & (block_indices[pos] == tmp_j)
            del tmp_j, pos
            for i in range(0,molecule_whole.molsize): ### $$$ needs vecorization
                subIndsUnion_j[start_ind:end_ind] = valid_top_row[i:]
                start_ind = end_ind
                end_ind = end_ind + molecule_whole.molsize - i - 2

            subIndsUnion = subIndsUnion_i + subIndsUnion_j
            del subIndsUnion_i, subIndsUnion_j, valid_top_row
            #subIndsUnion = torch.isin(molecule_whole.idxi, block_indices) + torch.isin(molecule_whole.idxj, block_indices)
            ham_timing['subIndsUnion'] = time.time() - tic
            #print("subIndsUnion {:>7.3f} |".format(time.time() - tic), end=" ")

            tic = time.time() # compute 2c2e and diagonal h1elec
            coulInts_test, e1b, e2a, _, _ = TETCI(molecule_whole.const, molecule_whole.idxi[subIndsUnion], molecule_whole.idxj[subIndsUnion],
                molecule_whole.ni[subIndsUnion], molecule_whole.nj[subIndsUnion], molecule_whole.xij[subIndsUnion], molecule_whole.rij[subIndsUnion], molecule_whole.Z,\
                molecule_whole.parameters['zeta_s'], molecule_whole.parameters['zeta_p'], molecule_whole.parameters['zeta_d'],\
                molecule_whole.parameters['s_orb_exp_tail'], molecule_whole.parameters['p_orb_exp_tail'], molecule_whole.parameters['d_orb_exp_tail'],\
                molecule_whole.parameters['g_ss'], molecule_whole.parameters['g_pp'], molecule_whole.parameters['g_p2'], molecule_whole.parameters['h_sp'],\
                molecule_whole.parameters['F0SD'], molecule_whole.parameters['G2SD'], molecule_whole.parameters['rho_core'],\
                molecule_whole.alp, molecule_whole.chi, molecule_whole.method)
            ham_timing['TETCI&DiI'] = time.time() - tic
            #print("TETCI&DiI {:>7.3f} |".format(time.time() - tic), end=" ")
            
        
        else: # otherwise, compute only relevant idxi, idxj, xij, rij for this core+halo
            tic = time.time()
            dtypeTEST = molecule_whole.Z.dtype # torch.long
            atom_index = torch.arange(molecule_whole.nmol*molecule_whole.molsize, device=P_contr.device,dtype=torch.int64)
            len_block_indices = len(block_indices)

            # Prepare lists to hold the indices that will form iii and jjj
            iii_list = []
            jjj_list = []
            pos = torch.searchsorted(block_indices, atom_index)
            pos = torch.clamp(pos, max=len_block_indices - 1)
            mask_atom_index_in_block_indices = (pos < len_block_indices) & (block_indices[pos] == atom_index)

            if sdc.ijMethod == 'Vec':
                # Create pairwise indices using broadcasting
                ii_matrix = atom_index.unsqueeze(1).expand(-1, atom_index.size(0))  # Rows represent i
                jj_matrix = atom_index.unsqueeze(0).expand(atom_index.size(0), -1)  # Columns represent j

                # Mask for valid pairs
                mask_upper_triangle = jj_matrix > ii_matrix  # Enforce jj > ii
                mask_valid_i = mask_atom_index_in_block_indices.unsqueeze(1)  # Valid i in block_indices
                mask_valid_j = mask_atom_index_in_block_indices.unsqueeze(0)  # Valid j in block_indices
                mask_pairs = mask_upper_triangle * (mask_valid_i + mask_valid_j)  # Combine all conditions

                # Flatten the indices of valid pairs
                iii = ii_matrix[mask_pairs]
                jjj = jj_matrix[mask_pairs]
            else:
                # Loop over atom_index and handle vectorized operations within each iteration
                for i in range(len(mask_atom_index_in_block_indices)): ### $$$ needs vectorization
                    jj = atom_index[i+1:]
                    ii = torch.full_like(jj, i)  # Create a tensor of `i` repeated for each `j`
                    # If `i` is in block_indices, add all pairs (i, jj)
                    if mask_atom_index_in_block_indices[i]:
                        iii_list.append(ii)
                        jjj_list.append(jj)
                    else:
                        # If `i` is not in block_indices, use binary search for checking presence in sorted `block_indices`
                        # Ensure indices are within bounds of block_indices
                        valid_idx_in_block = pos[i+1:][mask_atom_index_in_block_indices[i+1:]]
                        valid_jj = jj[mask_atom_index_in_block_indices[i+1:]]

                        # Now check if the values at valid indices match the elements in jj
                        mask_j_in_block = block_indices[valid_idx_in_block] == valid_jj

                        # Append only the values where jj is in block_indices
                        iii_list.append(ii[mask_atom_index_in_block_indices[i+1:]][mask_j_in_block])
                        jjj_list.append(valid_jj[mask_j_in_block])
                        del valid_idx_in_block, valid_jj, mask_j_in_block
                    del ii, jj
                # Concatenate all the lists to form the final iii and jjj tensors
                iii = torch.cat(iii_list) if iii_list else torch.tensor([], dtype=dtypeTEST)
                jjj = torch.cat(jjj_list) if jjj_list else torch.tensor([], dtype=dtypeTEST)
                del iii_list, jjj_list, pos, mask_atom_index_in_block_indices, atom_index

            paircoord = molecule_whole.coordinates[0,iii] - molecule_whole.coordinates[0,jjj]
            pairdist = torch.sqrt(torch.square(paircoord).sum(dim=1))

            r_ij = pairdist * molecule_whole.const.length_conversion_factor
            x_ij = -paircoord / pairdist.unsqueeze(1)
            del paircoord, pairdist
            
            ham_timing['idxi&idxj'] = time.time() - tic
            #print("idxi&idxj {:>7.3f} |".format(time.time() - tic), end=" ")

            tic = time.time() # compute 2c2e and diagonal h1elec
            coulInts_test, e1b, e2a, _, _ = TETCI(molecule_whole.const, iii, jjj,
                    molecule_whole.Z[iii], molecule_whole.Z[jjj], x_ij, r_ij, molecule_whole.Z,\
                    molecule_whole.parameters['zeta_s'], molecule_whole.parameters['zeta_p'], molecule_whole.parameters['zeta_d'],\
                    molecule_whole.parameters['s_orb_exp_tail'], molecule_whole.parameters['p_orb_exp_tail'], molecule_whole.parameters['d_orb_exp_tail'],\
                    molecule_whole.parameters['g_ss'], molecule_whole.parameters['g_pp'], molecule_whole.parameters['g_p2'], molecule_whole.parameters['h_sp'],\
                    molecule_whole.parameters['F0SD'], molecule_whole.parameters['G2SD'], molecule_whole.parameters['rho_core'],\
                    molecule_whole.alp, molecule_whole.chi, molecule_whole.method)

            ### Checkpointed version
            # def tetci_function(const, idxi, idxj, ni, nj, xij, rij, Z,
            #         zeta_s, zeta_p, zeta_d, s_orb_exp_tail,
            #         p_orb_exp_tail, d_orb_exp_tail, g_ss,
            #         g_pp, g_p2, h_sp, F0SD, G2SD, rho_core,
            #         alp, chi, method):
            #     # Call your TETCI function logic here
            #     return TETCI(const, idxi, idxj, ni, nj, xij, rij, Z,
            #                 zeta_s, zeta_p, zeta_d, s_orb_exp_tail,
            #                 p_orb_exp_tail, d_orb_exp_tail, g_ss,
            #                 g_pp, g_p2, h_sp, F0SD, G2SD, rho_core,
            #                 alp, chi, method)

            # # Use checkpointing to call the function
            # coulInts_test, e1b, e2a, _, _ = checkpoint(
            #     tetci_function,
            #     molecule_whole.const,
            #     iii,
            #     jjj,
            #     molecule_whole.Z[iii],
            #     molecule_whole.Z[jjj],
            #     x_ij,
            #     r_ij,
            #     molecule_whole.Z,
            #     molecule_whole.parameters['zeta_s'],
            #     molecule_whole.parameters['zeta_p'],
            #     molecule_whole.parameters['zeta_d'],
            #     molecule_whole.parameters['s_orb_exp_tail'],
            #     molecule_whole.parameters['p_orb_exp_tail'],
            #     molecule_whole.parameters['d_orb_exp_tail'],
            #     molecule_whole.parameters['g_ss'],
            #     molecule_whole.parameters['g_pp'],
            #     molecule_whole.parameters['g_p2'],
            #     molecule_whole.parameters['h_sp'],
            #     molecule_whole.parameters['F0SD'],
            #     molecule_whole.parameters['G2SD'],
            #     molecule_whole.parameters['rho_core'],
            #     molecule_whole.alp,
            #     molecule_whole.chi,
            #     molecule_whole.method,
            #     use_reentrant=True
            # )
            ham_timing['TETCI&DiI'] = time.time() - tic
            #print("TETCI&DiI {:>7.3f} |".format(time.time() - tic), end=" ")
        
        tic = time.time()
        idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
        max_key = max(idx_to_idx_mapping.keys())
        lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long, device = P_contr.device)
        # Populate the lookup tensor
        for key, value in idx_to_idx_mapping.items():
            lookup_tensor[key] = value
        if eng.use_pyseqm_lt:
            in_block_mask = torch.zeros(molecule_whole.molsize, dtype=torch.bool, device = P_contr.device)
            in_block_mask[block_indices]=True
            new_idxi = lookup_tensor[molecule_whole.idxi[in_block_mask[molecule_whole.idxi].to(torch.bool)]]
            new_idxj = lookup_tensor[molecule_whole.idxj[in_block_mask[molecule_whole.idxj].to(torch.bool)]]
            ham_timing['diIndsExp'] = time.time() - tic
            #print("diIndsExp {:>7.3f} |".format(time.time() - tic), end=" ")
            tic = time.time()
            # add diagonal to h1elec
            M_sub.index_add_(0,molSub.maskd[new_idxi], e1b[torch.isin(molecule_whole.idxi[subIndsUnion], block_indices)])
            M_sub.index_add_(0,molSub.maskd[new_idxj], e2a[torch.isin(molecule_whole.idxj[subIndsUnion], block_indices)])
            ham_timing['h1elDiUpd'] = time.time() - tic
            #print("h1elDiUpd {:>7.3f} |".format(time.time() - tic), end=" ")
        else:
            # Calculate the repeated counts for each index in block_indices
            repeats = molecule_whole.molsize - 1 - block_indices
            # Use `torch.repeat_interleave` to create the final tensor without needing a for loop
            new_iii = torch.repeat_interleave(torch.arange(len(block_indices), device = P_contr.device), repeats)

            new_jjj_list = []
            top_row = torch.arange(0, molSub.molsize, dtype=block_indices.dtype)
            start_indices = torch.cumsum((lookup_tensor[:] != 0).to(dtype=torch.long), dim=0) + 1
            start_indices[:block_indices[0]] = 0
            # Generate slices from top_row based on start_indices for each row
            new_jjj_list = ([top_row[start:] for start in start_indices])
            new_jjj = torch.cat(new_jjj_list)
            ham_timing['diIndsExp'] = time.time() - tic
            #print("diIndsExp {:>7.3f} |".format(time.time() - tic), end=" ")
            ### $$$ index_add_ is very slow!

            tic = time.time()
            pos = torch.searchsorted(block_indices, iii)
            # Ensure the indices are within bounds
            pos = torch.clamp(pos, max=len(block_indices) - 1)
            # Check if the positions are valid and match
            idxi_sub_ovrlp_with_rest = (pos < len(block_indices)) & (block_indices[pos] == iii)

            #M_sub.index_add_(0,molSub.maskd[new_iii], e1b[torch.isin(iii, block_indices)])
            M_sub.index_add_(0,molSub.maskd[new_iii], e1b[idxi_sub_ovrlp_with_rest])
            M_sub.index_add_(0,molSub.maskd[new_jjj], e2a[torch.isin(jjj, block_indices)])
            del repeats, new_iii, new_jjj_list, new_jjj, top_row, start_indices, pos, idxi_sub_ovrlp_with_rest
            ham_timing['h1elDiUpd'] = time.time() - tic
            #print("h1elDiUpd {:>7.3f} |".format(time.time() - tic), end=" ")
            
        del e1b, e2a, _
        
        tic = time.time()
        graph_for_pairs = torch.from_numpy(graph_for_pairs).to(P_contr.device, dtype=eng.torch_int_dt)
        
        if sdc.UHF: # open shell
            P_sub_from_contr = torch.zeros(2, len(block_indices)*len(block_indices),4,4, device = P_contr.device, dtype=eng.torch_dt)
            P_sub_from_contr = P_sub_from_contr.reshape(2, len(block_indices), len(block_indices), 4,4)

            parts_mask = torch.isin(block_indices, torch.tensor(partsIndex, device=block_indices.device))
            max_len = graph_for_pairs[partsIndex[0]][0]
            P_sub_from_contr[:,:,parts_mask] = P_contr[:,:max_len, block_indices[parts_mask]]

            for i in range(len(parts_mask)): ### $$$ needs vecorization
                if not parts_mask[i]:
                    tmp = graph_for_pairs[block_indices[i]][1:graph_for_pairs[block_indices[i]][0]+1]
                    pos = torch.searchsorted(block_indices, tmp)
                    # Ensure the indices are within bounds
                    pos = torch.clamp(pos, max=len(block_indices) - 1)
                    # Check if the positions are valid and match
                    mask_for_lookup = (pos < len(block_indices)) & (block_indices[pos] == tmp)
                    P_sub_from_contr[:,lookup_tensor[tmp[mask_for_lookup]],i] = \
                        P_contr[:,:graph_for_pairs[block_indices[i]][0],block_indices[i]][:,mask_for_lookup]
                    del pos, tmp, mask_for_lookup
            del parts_mask

            P_sub_from_contr = P_sub_from_contr.reshape(2,len(block_indices)*len(block_indices), 4,4)
            P_diag_contr = P_contr.transpose(1,2).reshape(2,molecule_whole.molsize*(len(graph_for_pairs[0])-1), 4,4)[:,graph_maskd]#.transpose(1,2)

        else: # closed shell
            P_sub_from_contr = torch.zeros(len(block_indices)*len(block_indices),4,4, device = P_contr.device, dtype=eng.torch_dt)
            P_sub_from_contr = P_sub_from_contr.reshape(len(block_indices), len(block_indices), 4,4)

            parts_mask = torch.isin(block_indices, torch.tensor(partsIndex, device=block_indices.device))
            max_len = graph_for_pairs[partsIndex[0]][0]
            P_sub_from_contr[:,parts_mask] = P_contr[:max_len, block_indices[parts_mask]]

            for i in range(len(parts_mask)): ### $$$ needs vecorization
                if not parts_mask[i]:
                    tmp = graph_for_pairs[block_indices[i]][1:graph_for_pairs[block_indices[i]][0]+1]
                    pos = torch.searchsorted(block_indices, tmp)
                    # Ensure the indices are within bounds
                    pos = torch.clamp(pos, max=len(block_indices) - 1)
                    # Check if the positions are valid and match
                    mask_for_lookup = (pos < len(block_indices)) & (block_indices[pos] == tmp)
                    P_sub_from_contr[lookup_tensor[tmp[mask_for_lookup]],i] = \
                        P_contr[:graph_for_pairs[block_indices[i]][0],block_indices[i]][mask_for_lookup]
                    del pos, tmp, mask_for_lookup
            del parts_mask

            P_sub_from_contr = P_sub_from_contr.reshape(len(block_indices)*len(block_indices), 4,4)
            #P_sub_from_contr = P_sub_from_contr.reshape(len(block_indices)*4, len(block_indices)*4)
            P_diag_contr = P_contr.transpose(0,1).reshape(molecule_whole.molsize*(len(graph_for_pairs[0])-1), 4,4)[graph_maskd]#.transpose(0,1)
        ham_timing['P_sub_from_contr'] = time.time() - tic
        tic = time.time()

        if sdc.UHF: # open shell
            if eng.use_pyseqm_lt:
                ham_contr = get_fock_pyseqm_u(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, block_indices,
                        molecule_whole.nmol, molecule_whole.idxi[subIndsUnion], molecule_whole.idxj[subIndsUnion], molSub.rij,
                        molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
            else:
                ham_contr = get_fock_pyseqm_u(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, block_indices,
                        molecule_whole.nmol, iii, jjj, molSub.rij,
                        molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
        else: # closed shell
            if eng.use_pyseqm_lt:
                ham_contr = get_fock_pyseqm(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, block_indices,
                        molecule_whole.nmol, molecule_whole.idxi[subIndsUnion], molecule_whole.idxj[subIndsUnion], molSub.rij,
                        molecule_whole.parameters, maskd_sub, mask_sub) # slowest part
            else:
                ham_contr = get_fock_pyseqm(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, block_indices,
                        molecule_whole.nmol, iii, jjj, molSub.rij,
                        molecule_whole.parameters, maskd_sub, mask_sub) # slowest part

        # #Define a wrapper function for the checkpointed function with required gradients
        # def checkpointed_get_fock(M_sub, coulInts_test, idxi_grad, idxj_grad, rij_grad, 
        #                         P_diag_contr, P_sub_from_contr, block_indices, nmol, 
        #                         parameters, maskd_sub, mask_sub):
        #     return get_fock_pyseqm(P_diag_contr, P_sub_from_contr, M_sub, coulInts_test, 
        #                             block_indices, nmol, idxi_grad, idxj_grad, rij_grad, 
        #                             parameters, maskd_sub, mask_sub)
        # # Use checkpoint with only the tensors requiring gradients
        
        # ham_contr = checkpoint(
        #     checkpointed_get_fock,
        #     M_sub, coulInts_test, 
        #     iii,  # requires grad
        #     jjj,  # requires grad
        #     molSub.rij,  # requires grad
        #     P_diag_contr, P_sub_from_contr, block_indices,
        #     molecule_whole.nmol,
        #     molecule_whole.parameters,
        #     maskd_sub,
        #     mask_sub,
        #     use_reentrant=True
        #     )

        del coulInts_test, P_diag_contr, maskd_sub, mask_sub, lookup_tensor, max_key, idx_to_idx_mapping
        if eng.use_pyseqm_lt:
            del subIndsUnion, new_idxi, new_idxj, in_block_mask
        else:
            del iii, jjj, r_ij, x_ij
        ham_timing['FulSubFock'] = time.time() - tic
        #print("FulSubFock {:>7.3f} |".format(time.time() - tic), end=" ")

        ### CALC Eelec ###
        tic = time.time()
        h1elec_sub = M_sub.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                .transpose(2,3) \
                .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize) 
        h1elec_sub = h1elec_sub.triu()+h1elec_sub.triu(1).transpose(1,2)
        #print(ham_contr)

        if sdc.UHF:
            dm_contr = P_sub_from_contr.reshape(2, molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                .transpose(3,4) \
                .reshape(2, molSub.nmol, 4*molSub.molsize, 4*molSub.molsize).transpose(0,1)
            eElec_contr  = 0.5*((dm_contr[:,0,:,core_indices_in_sub_expanded]+dm_contr[:,1,:,core_indices_in_sub_expanded])*h1elec_sub[:,:,core_indices_in_sub_expanded] + \
                                dm_contr[:,0,:,core_indices_in_sub_expanded]*ham_contr[:,0,:,core_indices_in_sub_expanded] + dm_contr[:,1,:,core_indices_in_sub_expanded]*ham_contr[:,1,:,core_indices_in_sub_expanded]).sum()
        else:
            dm_contr = P_sub_from_contr.reshape(molSub.nmol, molSub.molsize, molSub.molsize,4,4) \
                .transpose(2,3) \
                .reshape(molSub.nmol, 4*molSub.molsize, 4*molSub.molsize)
            eElec_contr  = 0.5*(dm_contr[:,:,core_indices_in_sub_expanded]*(h1elec_sub[:,:,core_indices_in_sub_expanded]+ham_contr[:,:,core_indices_in_sub_expanded])).sum()
        del dm_contr, h1elec_sub, P_sub_from_contr, M_sub, molSub
        ham_timing['En'] = time.time() - tic
        #print("En {:>7.3f} (s)|".format(time.time() - tic), end=" ")
        ### END CALC Eelec ###

        if doForces:            
            tic = time.time()
            L = eElec_contr.sum()
            torch.cuda.empty_cache()
            if molecule_whole.coordinates.requires_grad:
                L.backward()
                force = -molecule_whole.coordinates.grad.detach()[0].cpu().numpy()
                molecule_whole.coordinates.grad.zero_()
            else:
                force = molecule_whole.coordinates[0].cpu().numpy() * 0.0
            del  L, ham_contr
            ham_timing['Force'] = time.time() - tic
            #print("Force {:>7.3f} |".format(time.time() - tic), end=" ")
            return force, eElec_contr.detach().cpu().numpy()
        else:
            #print_memory_usage(len(partsIndex), 99, "HAM memory usage")
            return ham_contr, eElec_contr.detach().cpu().numpy()

    raise ValueError(f"ERROR!!!: Interface type not recognized: '{eng.interface}'. " +
                     f"Use any of the following: Module,File,Socket,MDI")

