"""Graph adaptive self-consistenf charge solver"""
import warnings
warnings.simplefilter("ignore", FutureWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"
import time
import torch
print('num_threads',torch.get_num_threads())
from sedacs.density_matrix import get_density_matrix, get_initDM, get_dmErrs, get_dmTrace
from sedacs.density_matrix_renorm import get_density_matrix_renorm
from sedacs.energy import get_eNuc, get_eTot
from sedacs.mol_sys_data import get_molSysData
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.graph_partition import get_coreHaloIndices, graph_partition, get_coreHaloIndicesPYSEQM

from sedacs.hamiltonian import get_hamiltonian
from sedacs.density_matrix import get_density_matrix
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import collect_and_sum_matrices
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
import numpy as np
from sedacs.evals import get_eVals
from sedacs.chemical_potential import get_mu
from sedacs.graph import get_initial_graph, update_dm_contraction, get_ch_graph, get_maskd, collect_graph_from_rho, collect_graph_from_rho_PYSEQM, add_graphs, print_graph, add_mult_graphs
from sedacs.interface.pyseqm import get_coreHalo_ham_inds, get_diag_guess_pyseqm, pyseqmObjects, get_molecule_pyseqm
import itertools
import sys
import psutil
import pickle
import socket
import copy
from seqm.seqm_functions.pack import pack
import gc
import numpy as np
from sedacs.engine import Engine

try:
    from mpi4py import MPI
    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False
__all__ = ["get_occ_singlePoint", "get_adaptiveSCFDM"]

def get_singlePoint(sdc,
                    eng,
                    partsPerGPU,
                    partsPerNode,
                    node_id,
                    node_rank,
                    rank,
                    gpu_comm,
                    parts,
                    partsCoreHalo,
                    sy,
                    hindex,
                    mu0,
                    molecule_whole,
                    P_contr,
                    graph_for_pairs,
                    graph_maskd):
    """
    Function calculates CH hamiltonians (ham), then eVals and dVals on each
    rank of gpu_comm. Then it gathers everything on global rank 0, computes
    chemical potential mu0.

    Note that in this context:
    eVals -> Eigenvalues
    Q     -> Eigenvectors
    dVals -> Norm over the *CORE PART* of the Eigenvectors, Q.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng: Engine
        The SEDACS engine.

    partsPerGPU:
        number of CH processed by one rank.
    partsPerNode:
        number of CH processed on one node.
    node_id:
        Current node identifier.
    node_rank:
        Local rank on a node. E.g., for [0,1,2,3] [4,5,6,7],
        global rank 4 is local rank 0.
    rank:
        Global rank.
    gpu_comm:
        Global communicator for ranks with GPU.
        If on CPU, all ranks are involved. gpu_comm is identical to master comm
        If on GPU and num_gpus (per node) == node_numranks, gpu_comm is
        identical to master comm.
        If on GPU and num_gpus (per node) <= node_numranks, gpu_comm is
        different form master comm. For example, 8 ranks on two nodes [0,1,2,3]
        and [4,5,6,7] with 2 GPUs per node. In that case, only ranks [0,1] and
        [4,5] are involved. They, however, become [0,1] [2,3] within gpu_comm.
    parts:
        List of core indices.
    partsCoreHalo:
        List of core+halo indices.
    sy:
        System object.
    hindex:
        Atom->orbtial index mapping.
    mu0:
        The chemical potential.
    molecule_whole:
        PYSEQM Molecule object.
    P_contr:
        Contracted density matrix. (sy.nats, sdc.maxDeg, 4, 4)
    graph_for_pairs:
        Graph of communities. E.g. graph_for_pairs[i] is a whole CH community
        in which atom i is a core atom, including itself. graph_for_pairs[i][0]
        is a community size.
    graph_maskd:
        Diagonal mask for the contracted density matrix, P_contr.

    Returns
    -------
    EELEC:
        The total electronic energy.
    eVal_LIST:
        List of the eigenvalues by partition.
    Q_LIST:
        List of the eigenvectors by partition.
    NH_Nh_Hs_LIST:
        List of NHeavy, NHydrogen, Hydrogen indices by partition.
    core_indices_in_sub_expanded_LIST:
        List of core indices within each partition, using the expanded indexing
        convention required by PYSEQM.
    Nocc_LIST:
        List of occupation numbyers by partition.
    mu0:
        Global chemical potential.
    """
    # Handles the scenario where N_GPU < N_ranks_per_node
    partIndex1 = (node_rank) * partsPerGPU + node_id*partsPerNode
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id*partsPerNode

    # All arrays in thsi code block are flattened 1D numpy arrays.
    if sdc.UHF:  # Open shell calculation.
        dValOnRank = np.empty((2, 0))
        eValOnRank = np.empty((2, 0))
    else:  # Closed shell calculation.
        dValOnRank = np.array([])
        eValOnRank = np.array([])

    # eVals and dVals arranged for each Core-Halo. These are given as a List of
    # torch Tensors. These are needed for mu0, and later for the density matrix
    eValOnRank_list = []
    dValOnRank_list = []

    # Eigenvectors for each partition
    Q_list = []

    # Number of occupied orbitals for each partition.
    # NOTE: This is not used for thermal HF.
    Nocc_list = []

    # Indices of core hamiltonian in core+halo hamiltonian. Might be useful
    # when core and halo atoms are shuffled to stay sorted, like in PySEQM.
    core_indices_in_sub_expanded_list = []

    # List -> [number_heavy_atoms, number_hydrogens, dim_H_C+H, N_occupied.
    NH_Nh_Hs_list = []

    # Result for the summed eletronic energies of C+Hs on the current rank.
    EELEC = 0.0

    # Loop over the partitions allocated to the current rank.
    for partIndex in range(partIndex1, partIndex2):

        ticHam = time.perf_counter()

        # Extract the subsystem information. (CORE + HALO).
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords,
                                                      sy.types,
                                                      sy.symbols,
                                                      partsCoreHalo[partIndex])

        # Extract the subsystem information. (CORE ONLY).
        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords, subSyCore.types = extract_subsystem(sy.coords,
                                                              sy.types,
                                                              sy.symbols,
                                                              parts[partIndex])

        if sdc.writeGeom:

            # Write subsystems to PDB and XYZ (Core+Halo)
            partFileName = "subSy"+str(rank)+"_"+str(partIndex)+".pdb"
            write_pdb_coordinates(partFileName,
                                  subSy.coords,
                                  subSy.types,
                                  subSy.symbols)
            write_xyz_coordinates("subSy"+str(rank)+"_"+str(partIndex)+".xyz",
                                  subSy.coords,
                                  subSy.types,
                                  subSy.symbols)

            # Write cores to PDB and XYZ
            partCoreFileName = "CoreSubSy"+str(rank)+"_"+str(partIndex)+".pdb"
            write_pdb_coordinates(partCoreFileName,
                                  subSyCore.coords,
                                  subSyCore.types,
                                  subSyCore.symbols)
            write_xyz_coordinates("CoreSubSy"+str(rank)+"_"+str(partIndex)+".xyz",
                                  subSyCore.coords,
                                  subSyCore.types,
                                  subSyCore.symbols)

        # Get the core indices, expandide core indices, and orbital indicies
        # for the current subsystem.
        core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub = \
            get_coreHalo_ham_inds(parts[partIndex],
                                  partsCoreHalo[partIndex],
                                  sdc,
                                  sy,
                                  subSy,
                                  device=P_contr.device)

        # Get CH hamiltonian. If on GPU, temporarily redefine molecule_whole
        # on GPU (its faster than transfering) and send P_contr to GPU. The
        # idea is to have P_contr in shared memory on each node, then update in
        # parallel on node 0. If initialized on GPU, then well need to transfer
        # it back into shared memory CPU to update in parallel. Ideally,
        # this needs to be fixed.
        ham_timing = {}
        if sdc.scfDevice == 'cuda':
            device = 'cuda:{}'.format(node_rank)

            # Get the temporary PYSEQM Molecule object for this subsystem.
            # Simply redefine on the proper device rather than transferring
            # molecule_whole from CPU.
            tmp_molecule_whole = get_molecule_pyseqm(sdc,
                                                     sy.coords,
                                                     sy.symbols,
                                                     sy.types,
                                                     do_large_tensors=sdc.use_pyseqm_lt,
                                                     device=device)[0]

            # Get the Hamiltonian using the PYSEQM Molecule object for this
            # Core+Halo subsystem.
            ham, eElec = get_hamiltonian(sdc,
                                         eng,
                                         subSy.coords,
                                         subSy.types,
                                         subSy.symbols,
                                         parts[partIndex],
                                         partsCoreHalo[partIndex],
                                         tmp_molecule_whole,
                                         P_contr.to(device),
                                         graph_for_pairs,
                                         graph_maskd,
                                         core_indices_in_sub_expanded,
                                         ham_timing,
                                         verbose=False)
            del tmp_molecule_whole

        else:
            device = 'cpu'

            # Get the Hamiltonian using the PYSEQM Molecule object for this
            # Core+Halo subsystem.
            ham, eElec = get_hamiltonian(sdc,
                                         eng,
                                         subSy.coords,
                                         subSy.types,
                                         subSy.symbols,
                                         parts[partIndex],
                                         partsCoreHalo[partIndex],
                                         molecule_whole,
                                         P_contr,
                                         graph_for_pairs,
                                         graph_maskd,
                                         core_indices_in_sub_expanded,
                                         ham_timing,
                                         verbose=False)

        EELEC += eElec

        # Get eVals, dVals
        tic = time.perf_counter()
        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals. Not used.
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs = get_eVals(eng,
                                              sdc,
                                              sy,
                                              ham,
                                              subSy.coords,
                                              subSy.symbols,
                                              subSy.types,
                                              sdc.Tel,
                                              mu0,
                                              core_indices_in_sub,
                                              core_indices_in_sub_expanded,
                                              hindex_sub,
                                              coreSize,
                                              subSy,
                                              subSyCore,
                                              parts[partIndex],
                                              partsCoreHalo[partIndex],
                                              verbose=False)
        del ham

        # Append eVals/eVals depending on if open/closed shell.
        if sdc.UHF:
            dValOnRank = np.append(dValOnRank, dVals, axis=1)
            eValOnRank = np.append(eValOnRank, eVals.cpu().numpy(), axis=1)
        else:
            dValOnRank = np.append(dValOnRank, dVals)
            eValOnRank = np.append(eValOnRank, eVals.cpu().numpy())

        # Append eVals/dVals/eigenvectors local to this rank.
        eValOnRank_list.append(eVals.cpu())
        dValOnRank_list.append(dVals)

        # See above documentation for the explanation of all of these lists.
        # These effective collect the current rank's ifnromation for the above-
        # mentioned quantities.
        Q_list.append(Q.cpu())
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        # Timing on getting the Hamiltonians.
        ham_timing['eVals/dVals'] = time.perf_counter() - tic
        ham_timing['TOT'] = time.perf_counter() - ticHam
        formatted_string = " | ".join(f"{key} {value:8.3f}" for key, value in ham_timing.items())
        print('Rank', rank, 'part', partIndex, ':', formatted_string)

    # Time the MPI collection of eigenvectors.
    tic = time.perf_counter()
    if rank != 0:
        gpu_comm.send(Q_list, dest=0, tag=0)
    else:
        Q_LIST = [Q_list]
        for i in range(1, gpu_comm.Get_size()):
            Q_LIST.append(gpu_comm.recv(source=i, tag=0))
    print("Time Q_LIST send/recv {:>9.4f} (s)"
          .format(time.perf_counter() - tic), rank)

    tic = time.perf_counter()
    full_dVals = None
    full_eVals = None
    eValOnRank_size = np.array(eValOnRank.shape[-1], dtype=int)
    eValOnRank_SIZES = None
    recvcounts = None
    if rank == 0:
        eValOnRank_SIZES = np.empty(gpu_comm.Get_size(), dtype=int)

    # Gather eigenvalues.
    gpu_comm.Gather(eValOnRank_size, eValOnRank_SIZES, root=0)

    # Set up full containers on Rank 0.
    if rank == 0:

        if sdc.UHF:
            full_dVals = np.empty((2, np.sum(eValOnRank_SIZES)),
                                  dtype=eValOnRank.dtype)
            recvcounts = [2 * size for size in eValOnRank_SIZES]

        else:
            full_dVals = np.empty(np.sum(eValOnRank_SIZES),
                                  dtype=eValOnRank.dtype)
            full_eVals = np.empty(np.sum(eValOnRank_SIZES),
                                  dtype=eValOnRank.dtype)

    # Gather the actual values.
    if sdc.UHF:
        dValOnRank_flat = dValOnRank.flatten()
        gpu_comm.Gatherv(
            sendbuf=dValOnRank_flat,  # Flattened 1D send buffer
            recvbuf=(full_dVals, recvcounts),
            root=0)
    else:
        gpu_comm.Gatherv(dValOnRank, [full_dVals, eValOnRank_SIZES], root=0)
        gpu_comm.Gatherv(eValOnRank, [full_eVals, eValOnRank_SIZES], root=0)

    # These are the FULL versions of the corresponding lists above. See the 
    # above documentation for which  all of these lists are explained.
    eVal_LIST = gpu_comm.gather(eValOnRank_list, root=0)
    dVal_LIST = gpu_comm.gather(dValOnRank_list, root=0)
    NH_Nh_Hs_LIST = gpu_comm.gather(NH_Nh_Hs_list, root=0)
    core_indices_in_sub_expanded_LIST = gpu_comm.gather(core_indices_in_sub_expanded_list, root=0)
    Nocc_LIST = gpu_comm.gather(Nocc_list, root=0)

    # Flatten the nested list of lists into a single list of tensors.
    # There is one tensor per CH partition.
    if rank == 0:
        eVal_LIST = list(itertools.chain(*eVal_LIST))
        dVal_LIST = list(itertools.chain(*dVal_LIST))
        Q_LIST = list(itertools.chain(*Q_LIST))
        NH_Nh_Hs_LIST = list(itertools.chain(*NH_Nh_Hs_LIST))
        core_indices_in_sub_expanded_LIST = list(itertools.chain(*core_indices_in_sub_expanded_LIST))
        Nocc_LIST = list(itertools.chain(*Nocc_LIST))
    else:
        Q_LIST = None

    if node_rank == 0:
        print("| t commLists {:>9.4f} (s)"
              .format(time.perf_counter() - tic), rank)

    if rank == 0:
        tic = time.perf_counter()
        mu0 = get_mu(mu0, full_dVals, full_eVals, sdc.Tel, sy.numel/2)  # oldR
        print("Time mu0 {:>9.4f} (s)".format(time.perf_counter() - tic))

    return (EELEC,
            eVal_LIST,
            Q_LIST,
            NH_Nh_Hs_LIST,
            core_indices_in_sub_expanded_LIST,
            Nocc_LIST,
            mu0)


def get_singlePoint_charges(sdc,
                            eng,
                            partsPerGPU,
                            partsPerNode,
                            node_id,
                            node_rank,
                            rank,
                            gpu_comm,
                            parts,
                            partsCoreHalo,
                            sy,
                            hindex,
                            gscf,
                            mu0,
                            molecule_whole,
                            P_contr,
                            graph_for_pairs,
                            graph_maskd):
    """
    Function calculates CH hamiltonians (ham), then eVals and dVals on each
    rank of gpu_comm. Then it gathers everything on global rank 0, computes
    chemical potential mu0.

    Note that in this context:
    eVals -> Eigenvalues
    Q     -> Eigenvectors
    dVals -> Norm over the *CORE PART* of the Eigenvectors, Q.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng: Engine
        The SEDACS engine.

    partsPerGPU:
        number of CH processed by one rank.
    partsPerNode:
        number of CH processed on one node.
    node_id:
        Current node identifier.
    node_rank:
        Local rank on a node. E.g., for [0,1,2,3] [4,5,6,7],
        global rank 4 is local rank 0.
    rank:
        Global rank.
    gpu_comm:
        Global communicator for ranks with GPU.
        If on CPU, all ranks are involved. gpu_comm is identical to master comm
        If on GPU and num_gpus (per node) == node_numranks, gpu_comm is
        identical to master comm.
        If on GPU and num_gpus (per node) <= node_numranks, gpu_comm is
        different form master comm. For example, 8 ranks on two nodes [0,1,2,3]
        and [4,5,6,7] with 2 GPUs per node. In that case, only ranks [0,1] and
        [4,5] are involved. They, however, become [0,1] [2,3] within gpu_comm.
    parts:
        List of core indices.
    partsCoreHalo:
        List of core+halo indices.
    sy:
        System object.
    hindex:
        Atom->orbtial index mapping.
    mu0:
        The chemical potential.
    molecule_whole:
        PYSEQM Molecule object.
    P_contr:
        Contracted density matrix. (sy.nats, sdc.maxDeg, 4, 4)
    graph_for_pairs:
        Graph of communities. E.g. graph_for_pairs[i] is a whole CH community
        in which atom i is a core atom, including itself. graph_for_pairs[i][0]
        is a community size.
    graph_maskd:
        Diagonal mask for the contracted density matrix, P_contr.

    Returns
    -------
    EELEC:
        The total electronic energy.
    eVal_LIST:
        List of the eigenvalues by partition.
    Q_LIST:
        List of the eigenvectors by partition.
    NH_Nh_Hs_LIST:
        List of NHeavy, NHydrogen, Hydrogen indices by partition.
    core_indices_in_sub_expanded_LIST:
        List of core indices within each partition, using the expanded indexing
        convention required by PYSEQM.
    Nocc_LIST:
        List of occupation numbyers by partition.
    mu0:
        Global chemical potential.
    """
    # Handles the scenario where N_GPU < N_ranks_per_node
    partIndex1 = (node_rank) * partsPerGPU + node_id*partsPerNode
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id*partsPerNode

    # All arrays in thsi code block are flattened 1D numpy arrays.
    if sdc.UHF:  # Open shell.
        dValOnRank = np.empty((2, 0))
        eValOnRank = np.empty((2, 0))
    else:  # Closed shell.
        dValOnRank = np.array([])
        eValOnRank = np.array([])
    eValOnRank_list = []
    dValOnRank_list = []

    # Eigenvectors for each partition
    Q_list = []

    # Number of occupied orbitals for each partition.
    # NOTE: This is not used for thermal HF.
    Nocc_list = []

    # Indices of core hamiltonian in core+halo hamiltonian. Might be useful
    # when core and halo atoms are shuffled to stay sorted, like in PySEQM.
    core_indices_in_sub_expanded_list = []

    # List -> [number_heavy_atoms, number_hydrogens, dim_H_C+H, N_occupied.
    NH_Nh_Hs_list = []

    # Result for the summed eletronic energies of C+Hs on the current rank.
    EELEC = 0.0

    chargesOnRank = None
    subSysOnRank = []

    # Loop over the partitions allocated to the current rank.
    for partIndex in range(partIndex1, partIndex2):

        ticHam = time.perf_counter()

        # Extract the subsystem information. (CORE + HALO).
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords,
                                                      sy.types,
                                                      sy.symbols,
                                                      partsCoreHalo[partIndex])

        if (gscf == 0):
            subSy.charges = np.zeros(len(subSy.types))

        # Extract the subsystem information. (CORE ONLY).
        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords, subSyCore.types = extract_subsystem(sy.coords,
                                                              sy.types,
                                                              sy.symbols,
                                                              parts[partIndex])

        if sdc.writeGeom:

            # Write subsytem to PDB and XYZ. (CORE + HALO).
            partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
            write_pdb_coordinates(partFileName,
                                  subSy.coords,
                                  subSy.types,
                                  subSy.symbols)
            write_xyz_coordinates("subSy" + str(rank) + "_" + str(partIndex) + ".xyz",
                                  subSy.coords,
                                  subSy.types,
                                  subSy.symbols)

            partCoreFileName = "CoreSubSy"+str(rank)+"_"+str(partIndex)+".pdb"

            # Write subsytem to PDB and XYZ. (CORE ONLY).
            write_pdb_coordinates(partCoreFileName,
                                  subSyCore.coords,
                                  subSyCore.types,
                                  subSyCore.symbols)

            write_xyz_coordinates("CoreSubSy"+str(rank)+"_"+str(partIndex)+".xyz",
                                  subSyCore.coords,
                                  subSyCore.types,
                                  subSyCore.symbols)

        core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub = \
            get_coreHalo_ham_inds(parts[partIndex],
                                  partsCoreHalo[partIndex],
                                  sdc,
                                  sy,
                                  subSy,
                                  device=P_contr.device)

        # Get CH hamiltonian. If on GPU, temporarily redefine molecule_whole on
        # GPU (its faster than transfering) and send P_contr to GPU. The idea
        # is to have P_contr in shared memory on each node, then update input
        # in parallel on node 0. If initialized on GPU, then we'll need to
        # transfer it back into shared memory CPU to update in parallel.
        # Ideally, this needs to be fixed.
        ham_timing = {}
        if sdc.scfDevice == 'cuda':
            device = 'cuda:{}'.format(node_rank)

            # Get the temporary PYSEQM Molecule object for this subsystem.
            # Simply redefine on the proper device rather than transferring
            # molecule_whole from CPU.
            tmp_molecule_whole = get_molecule_pyseqm(sdc,
                                                     sy.coords,
                                                     sy.symbols,
                                                     sy.types,
                                                     do_large_tensors=sdc.use_pyseqm_lt,
                                                     device=device)[0]
            # Get the Hamiltonian using the PYSEQM Molecule object for this
            # Core+Halo subsystem.
            ham, eElec = get_hamiltonian(sdc,
                                         eng,
                                         subSy.coords,
                                         subSy.types,
                                         subSy.symbols,
                                         parts[partIndex],
                                         partsCoreHalo[partIndex],
                                         tmp_molecule_whole,
                                         P_contr.to(device),
                                         graph_for_pairs,
                                         graph_maskd,
                                         core_indices_in_sub_expanded,
                                         ham_timing,
                                         verbose=False)
            del tmp_molecule_whole

        else:
            device = 'cpu'

            # Get the Hamiltonian using the PYSEQM Molecule object for this
            # Core+Halo subsystem.
            ham, eElec = get_hamiltonian(sdc, eng,
                                         subSy.coords,
                                         subSy.types,
                                         subSy.symbols,
                                         parts[partIndex],
                                         partsCoreHalo[partIndex],
                                         molecule_whole,
                                         P_contr,
                                         graph_for_pairs,
                                         graph_maskd,
                                         core_indices_in_sub_expanded,
                                         ham_timing,
                                         verbose=False)

        EELEC += eElec

        # Get eVals, dVals
        tic = time.perf_counter()
        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals. Not used.
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs = get_eVals(eng,
                                              sdc,
                                              sy,
                                              ham,
                                              subSy.coords,
                                              subSy.symbols,
                                              subSy.types,
                                              sdc.Tel,
                                              mu0,
                                              core_indices_in_sub,
                                              core_indices_in_sub_expanded,
                                              hindex_sub,
                                              coreSize,
                                              subSy,
                                              subSyCore,
                                              parts[partIndex],
                                              partsCoreHalo[partIndex],
                                              verbose=False)
        del ham
        
        # Append eVals/eVals depending on if open/closed shell.
        if sdc.UHF:
            dValOnRank = np.append(dValOnRank, dVals, axis=1)
            eValOnRank = np.append(eValOnRank, eVals.cpu().numpy(), axis=1)
        else:
            dValOnRank = np.append(dValOnRank, dVals)
            eValOnRank = np.append(eValOnRank, eVals.cpu().numpy())

        # Append eVals/dVals/eigenvectors local to this rank.
        eValOnRank_list.append(eVals.cpu())
        dValOnRank_list.append(dVals)

        # See above documentation for the explanation of all of these lists.
        # These effective collect the current rank's ifnromation for the above-
        # mentioned quantities.
        Q_list.append(Q.cpu())
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        # Timing on getting the Hamiltonians.
        ham_timing['eVals/dVals'] = time.perf_counter() - tic
        ham_timing['TOT'] = time.perf_counter() - ticHam
        formatted_string = " | ".join(f"{key} {value:8.3f}" for key, value in ham_timing.items())
        print('Rank', rank, 'part', partIndex, ':', formatted_string)


    # Time the MPI collection of eigenvectors.
    tic = time.perf_counter()
    if rank != 0:
        gpu_comm.send(Q_list, dest=0, tag=0)
    else:
        Q_LIST = [Q_list]
        for i in range(1, gpu_comm.Get_size()):
            Q_LIST.append(gpu_comm.recv(source=i, tag=0))
    print("Time Q_LIST send/recv {:>9.4f} (s)"
          .format(time.perf_counter() - tic), rank)

    tic = time.perf_counter()
    full_dVals = None
    full_eVals = None
    eValOnRank_size = np.array(eValOnRank.shape[-1], dtype=int)
    eValOnRank_SIZES = None
    recvcounts = None
    if rank == 0:
        eValOnRank_SIZES = np.empty(gpu_comm.Get_size(), dtype=int)

    # Gather eigenvalues.
    gpu_comm.Gather(eValOnRank_size, eValOnRank_SIZES, root=0)

    # Set up full containers on Rank 0.
    if rank == 0:

        if sdc.UHF:
            full_dVals = np.empty((2, np.sum(eValOnRank_SIZES)), dtype=eValOnRank.dtype)
            recvcounts = [2 * size for size in eValOnRank_SIZES]

        else:
            full_dVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)
            full_eVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype) #### oldR

    # Gather the actual values.
    if sdc.UHF:
        dValOnRank_flat = dValOnRank.flatten()
        gpu_comm.Gatherv(
            sendbuf=dValOnRank_flat,  # Flattened 1D send buffer
            recvbuf=(full_dVals, recvcounts),
            root=0)
    else:
        gpu_comm.Gatherv(dValOnRank, [full_dVals, eValOnRank_SIZES], root=0)
        gpu_comm.Gatherv(eValOnRank, [full_eVals, eValOnRank_SIZES], root=0) #### oldR

    # These are the FULL versions of the corresponding lists above. See the 
    # above documentation for which  all of these lists are explained.
    eVal_LIST = gpu_comm.gather(eValOnRank_list, root=0)
    dVal_LIST = gpu_comm.gather(dValOnRank_list, root=0)
    NH_Nh_Hs_LIST = gpu_comm.gather(NH_Nh_Hs_list, root=0)
    core_indices_in_sub_expanded_LIST = gpu_comm.gather(core_indices_in_sub_expanded_list, root=0)
    Nocc_LIST = gpu_comm.gather(Nocc_list, root=0)

    # Flatten the nested list of lists into a single list of tensors.
    # There is one tensor per CH partition.
    if rank == 0:
        eVal_LIST = list(itertools.chain(*eVal_LIST))
        dVal_LIST = list(itertools.chain(*dVal_LIST))
        Q_LIST = list(itertools.chain(*Q_LIST))
        NH_Nh_Hs_LIST = list(itertools.chain(*NH_Nh_Hs_LIST))
        core_indices_in_sub_expanded_LIST = list(itertools.chain(*core_indices_in_sub_expanded_LIST))
        Nocc_LIST = list(itertools.chain(*Nocc_LIST))
    else:
        Q_LIST = None

    if node_rank == 0:
        print("| t commLists {:>9.4f} (s)"
              .format(time.perf_counter() - tic), rank)

    if rank == 0:
        tic = time.perf_counter()
        mu0 = get_mu(mu0, full_dVals, full_eVals, sdc.Tel, sy.numel/2)
        print("Time mu0 {:>9.4f} (s)".format(time.perf_counter() - tic))

    return (EELEC,
            eVal_LIST,
            Q_LIST,
            NH_Nh_Hs_LIST,
            core_indices_in_sub_expanded_LIST,
            Nocc_LIST,
            mu0)


def get_singlePointForces(sdc,
                          eng,
                          partsPerGPU,
                          partsPerNode,
                          node_id,
                          node_rank,
                          rank,
                          parts,
                          partsCoreHalo,
                          sy,
                          hindex,
                          forces,
                          molecule_whole,
                          P,
                          P_contr,
                          graph_for_pairs,
                          graph_maskd):
    '''
    Function calculates forces on ALL atoms via backprop through an electronic
    energy a CH. Electronic energy is obtained from a "rectangular" hamiltonian

    ***UPDATES FORCES IN-PLACE***

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng: Engine
        The SEDACS engine.

    partsPerGPU:
        number of CH processed by one rank.
    partsPerNode:
        number of CH processed on one node.
    node_id:
        Current node identifier.
    node_rank:
        Local rank on a node. E.g., for [0,1,2,3] [4,5,6,7],
        global rank 4 is local rank 0.
    rank:
        Global rank.
    gpu_comm:
        Global communicator for ranks with GPU.
        If on CPU, all ranks are involved. gpu_comm is identical to master comm
        If on GPU and num_gpus (per node) == node_numranks, gpu_comm is
        identical to master comm.
        If on GPU and num_gpus (per node) <= node_numranks, gpu_comm is
        different form master comm. For example, 8 ranks on two nodes [0,1,2,3]
        and [4,5,6,7] with 2 GPUs per node. In that case, only ranks [0,1] and
        [4,5] are involved. They, however, become [0,1] [2,3] within gpu_comm.
    parts:
        List of core indices.
    partsCoreHalo:
        List of core+halo indices.
    sy:
        System object.
    hindex:
        Atom->orbtial index mapping.
    mu0:
        The chemical potential.
    molecule_whole:
        PYSEQM Molecule object.
    P_contr:
        Contracted density matrix. (sy.nats, sdc.maxDeg, 4, 4)
    graph_for_pairs:
        Graph of communities. E.g. graph_for_pairs[i] is a whole CH community
        in which atom i is a core atom, including itself. graph_for_pairs[i][0]
        is a community size.
    graph_maskd:
        Diagonal mask for the contracted density matrix, P_contr.

    Returns
    -------
    EELEC:
        The total electronic energy.
    '''
    partIndex1 = (node_rank) * partsPerGPU + node_id*partsPerNode
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id*partsPerNode
    EELEC = 0.0

    # Loop over the partitions allocated to the current rank.
    for partIndex in range(partIndex1, partIndex2):

        # Extract the subsystem information. (CORE + HALO).
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords,
                                                      sy.types,
                                                      sy.symbols,
                                                      partsCoreHalo[partIndex])

        # Extract indices needed for generating the Hamiltonian.
        _, core_indices_in_sub_expanded, _ = \
            get_coreHalo_ham_inds(parts[partIndex],
                                  partsCoreHalo[partIndex],
                                  sdc,
                                  sy,
                                  subSy)

        tic = time.perf_counter()
        ham_timing = {}
        tmp_molecule_whole = copy.deepcopy(molecule_whole)
        if sdc.doForces:
            tmp_molecule_whole.coordinates.requires_grad_(True)

        # Get hamiltonian with forces.
        f, eElec = get_hamiltonian(sdc,
                                   eng,
                                   subSy.coords,
                                   subSy.types,
                                   subSy.symbols,
                                   parts[partIndex],
                                   partsCoreHalo[partIndex],
                                   tmp_molecule_whole,
                                   P_contr,
                                   graph_for_pairs,
                                   graph_maskd,
                                   core_indices_in_sub_expanded,
                                   ham_timing,
                                   doForces=True,
                                   verbose=False)

        del tmp_molecule_whole

        # Modify forces in-place.
        forces += f
        
        # Sum electronic energy.
        EELEC += eElec

        # Print timing.
        ham_timing['TOT'] = time.perf_counter() - tic
        formatted_string = " | ".join(f"{key} {value:8.3f}" for key, value in ham_timing.items())
        print('Rank', rank, 'part', partIndex, ':', formatted_string,
              "|| EelecCH {:>7.3f} eV ||".format(eElec.item()))
        del eElec, subSy, f

    return EELEC


def get_singlePointDM(sdc,
                      eng,
                      rank,
                      node_numranks,
                      node_comm,
                      parts,
                      partsCoreHalo,
                      sy,
                      hindex,
                      mu0,
                      P_contr,
                      graph_for_pairs,
                      eValOnRank_list,
                      Q_list,
                      NH_Nh_Hs_list,
                      core_indices_in_sub_expanded_list):
    """
    Function updates P_contr with core columns of CH dm. This is done in
    parallel on ALL local ranks of node 0 on CPU. TODO: Improve the efficiency.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng: Engine
        The SEDACS engine.
    rank:
        Current global rank.
    node_numranks:
        Number of ranks on a node (local ranks).
    node_comm:
        Local comminicator on a node.
    parts:
        List of core indices.
    partsCoreHalo:
        List of core+halo indices.
    sy:
        System object.
    hindex:
        Atom->orbtial index mapping.
    mu0:
        The chemical potential.
    P_contr:
        Contracted density matrix. Shape: (sy.nats, sdc.maxDeg, 4,4)
    graph_for_pairs:
        Graph of communities. E.g. graph_for_pairs[i] is a whole CH community
        in which atom i is a core atom, including itself. graph_for_pairs[i][0]
        is a community size.
    eValOnRank_list:
        Eigenvalues of CHs. Here, for all CHs.
    Q_list:
        Eigenvectors of CHs. Here, only those used by this rank are present.
    NH_Nh_Hs_list:
        List of [number_of_heavy_atoms, number_of_hydrogens,
        dim_of_coreHalo_ham]. Here, for all CHs.
    core_indices_in_sub_expanded_list:
        Indices of core columns in CH. E.g., CH[i] contains atoms [0,1,2,3],
        core atoms are [1,3], 4 AOs per atom. Then,
        core_indices_in_sub_expanded_list[i] is [4,5,6,7, 12,13,14,15].

    Returns
    -------
    graphOnRank:
        Get connectivity graph for the dm of current CH.
    """
    if rank == 0:
        print("eElec:   {:>10.8f} | \u0394E| {:>10.8f}"
              .format(sdc.EelecNew, abs(sdc.EelecNew-sdc.EelecOld)))

    sdc.EelecOld = sdc.EelecNew

    # Parititon per rank and determine the partitions the current rank is
    # responsible for.
    partsPerRank = int(sdc.nparts / node_numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    maxDifList = []
    sumDifTot = 0
    P_contr_maxDifList = []
    P_contr_sumDifTot = 0


    for partIndex, i in zip(range(partIndex1, partIndex2), range(partsPerRank)):

        # This will calculate the DM in subsys and update the whole DM
        # rho_ren is a dm contructed with electronic temperature. It's shaped
        # into 4x4 blocks, even for hydrogen atoms, as required by PYSEQM.
        rho_ren, maxDif, sumDif = get_density_matrix_renorm(sdc,
                                                            eng,
                                                            sdc.Tel,
                                                            mu0,
                                                            P_contr,
                                                            graph_for_pairs,
                                                            eValOnRank_list[partIndex],
                                                            Q_list[i],
                                                            NH_Nh_Hs_list[partIndex],
                                                            core_indices_in_sub_expanded_list[partIndex])

        indices_in_sub = np.linspace(0,
                                     len(partsCoreHalo[partIndex])-1,
                                     len(partsCoreHalo[partIndex]),
                                     dtype=eng.np_int_dt)  # Indices for CH DM.

        # Core column blocks in CH DM (assuming its shaped as:
        # [n_atoms, n_atoms, 4, 4])
        core_indices_in_sub = indices_in_sub[np.isin(partsCoreHalo[partIndex],
                                                     parts[partIndex])]
        P_contr_maxDif = []
        P_contr_sumDif = 0
        if sdc.UHF:  # Open shell.

            # Vectorized. Faster for larger cores.
            max_len = graph_for_pairs[parts[partIndex][0]][0]

            # Get part of P_contr that corresponds to cores of current CH
            TMP1 = P_contr[:, :max_len, parts[partIndex]]

            # Get core column blocks of CH
            TMP2 = rho_ren.reshape((1,
                                    2,
                                    NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],
                                    4,
                                    NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],
                                    4)) \
                                    .transpose(3,4).reshape(2,
                                                            (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),
                                                            (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),
                                                            4,
                                                            4).transpose(3,4).transpose(1,2)[:,:,core_indices_in_sub] 

            # Max difference in DM elements.
            P_contr_maxDif.append(torch.max(torch.abs(TMP1 - TMP2)).cpu().numpy())

            # Sum of abs differences between new and old DM.
            P_contr_sumDif += torch.sum(torch.abs(TMP1 - TMP2)).cpu().numpy()

            # Update DM.
            P_contr[:, :max_len, parts[partIndex]] = (1-sdc.alpha)*TMP1 + sdc.alpha * TMP2

            # Packing rho_ren from 4x4 blocks into normal form based on number
            # of AOs per atom.
            rho_ren = pack(rho_ren[0]+rho_ren[1],
                           NH_Nh_Hs_list[partIndex][0],
                           NH_Nh_Hs_list[partIndex][1])

        else:  # Closed shell. See documentation in open-shell.
            max_len = graph_for_pairs[parts[partIndex][0]][0]
            TMP1 = P_contr[:max_len, parts[partIndex]]
            TMP2 = rho_ren.reshape((1,
                                    NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],
                                    4,
                                    NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],
                                    4)) \
                                    .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),
                                                            (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),
                                                            4,
                                                            4).transpose(2, 3).transpose(0, 1)[:, core_indices_in_sub]

            # Max difference in DM elements.
            P_contr_maxDif.append(torch.max(torch.abs(TMP1 - TMP2)).cpu().numpy())

            # Sum of abs differences between new and old DM.
            P_contr_sumDif += torch.sum(torch.abs(TMP1 - TMP2)).cpu().numpy()

            # Update DM.
            P_contr[:max_len,parts[partIndex]] = (1-sdc.alpha)*TMP1 + sdc.alpha * TMP2

            # Packing rho_ren from 4x4 blocks into normal form based on number
            # of AOs per atom.
            rho_ren = pack(rho_ren,
                           NH_Nh_Hs_list[partIndex][0],
                           NH_Nh_Hs_list[partIndex][1])


        # Total max difference.
        P_contr_maxDif = max(P_contr_maxDif)

        # Store max differences.
        P_contr_maxDifList.append(P_contr_maxDif)

        # Total of the summed differences.
        P_contr_sumDifTot += P_contr_sumDif

        maxDifList.append(maxDif)
        try:
            sumDifTot += sumDif
        except:
            sumDifTot += 0

        # Get connectivity graph for the dm of current CH
        graphOnRank = collect_graph_from_rho_PYSEQM(graphOnRank,
                                                    rho_ren,
                                                    sdc.gthresh,
                                                    sy.nats,
                                                    sdc.maxDeg,
                                                    partsCoreHalo[partIndex],
                                                    hindex,
                                                    verb=False)
        del rho_ren

    # Logging.
    print(" MAX |\u0394DM_ij|: {:>10.7f} at SubSy {:>5d}".format(max(P_contr_maxDifList), np.argmax(P_contr_maxDifList)))
    print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(P_contr_sumDifTot))

    return graphOnRank


def print_memory_usage(rank, node_rank, message) -> None:
    """
    Prints the memory usage on the desired rank.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message} | Rank: {rank}, Node Rank: {node_rank}, Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")


def tensor_size(tensor):
    """
    Returns the size of the desired tensor.
    """
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)


def get_tensors() -> torch.Tensor:
    """
    Generator which collects all tensors in the current environment.
    Relies on the Python garbage collector for persistence. Yields tensors
    one-by-one.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                yield obj
        except Exception as e:
            pass


def print_attribute_sizes(obj) -> None:
    """
    Prints size of attributes of the input object.
    """
    for attr in dir(obj):
        # Skip private or callable attributes
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue

        attribute = getattr(obj, attr)

        # Calculate this size in memory of attributable to the reference of the
        # object attribute.
        size_bytes = attribute.nbytes if isinstance(attribute, np.ndarray) else attribute.element_size() * attribute.nelement() if isinstance(attribute, torch.Tensor) else sys.getsizeof(attribute)
        size_mb = size_bytes / (1024 ** 2)  # Convert bytes to MB
        print(f"{attr}: {size_mb:.2f} MB")


def get_adaptiveDM_PYSEQM(sdc,
                          eng,
                          comm,
                          rank,
                          numranks,
                          sy,
                          hindex,
                          graphNL):
    """
    The main driver function. It initializes supplementary comms, dm, graphs,
    performs scf cycle with graph and dm updates, and then computes forces.

    sdc:
        SEDACS driver.
    eng:
        SEDACS Engine.
    comm:
        Master MPI communicator.
    rank:
        The global rank.
    numranks:
        The number of global ranks.
    sy:
        The full SEDACS System.
    hindex:
        Orbital indices for each atom in the system.
    graphNL:
        Initial connectivity graph
    """

    # SCF Initialization time.
    t_INIT = time.perf_counter()
    tic = time.perf_counter()
    sdc.EelecOld = 0.0

    # Whether or not to use large tensors. Prohibitively large for big systems.
    eng.use_pyseqm_lt = False

    # If False, pyseqm won't compute them and sedacs will coumpute only
    # the necessary subparts.
    sdc.use_pyseqm_lt = eng.use_pyseqm_lt

    # Reconstructs the full density matrix for debugging purpose.
    eng.reconstruct_dm = False
    sdc.reconstruct_dm = eng.reconstruct_dm

    # //Set up the MPI communicators.

    # Local communicator for ranks on a given node.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    # The local rank on a given node.
    node_rank = node_comm.Get_rank()

    # The size of a communicator on a given node.
    node_numranks = node_comm.Get_size()

    # Get relevant information specific to the nodes the job has been
    # allocated.
    node_name = socket.gethostname()
    node_names = comm.allgather(node_name)

    unique_nodes = list(set(node_names))
    num_nodes = len(unique_nodes)
    node_id = int(rank/node_numranks)

    # Get primary ranks on each node.
    # E.g. when running 16 ranks on two nodes, these are ranks 0 and 8.
    # [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]
    primary_rank = None
    if node_rank == 0:
        primary_rank = rank  # Global rank of the primary rank on each node

    # Gather the primary ranks from each node
    primary_ranks = comm.allgather(primary_rank)

    # Filter out Nones.
    primary_ranks = [r for r in primary_ranks if r is not None]
    color = 0 if rank in primary_ranks else MPI.UNDEFINED

    # Communicator for primary ranks.
    primary_comm = comm.Split(color=color, key=rank)

    device = 'cpu'
    if sdc.scfDevice == 'cuda' and sdc.numGPU == -1:
        num_gpus = torch.cuda.device_count()
    elif sdc.scfDevice == 'cuda':
        num_gpus = sdc.numGPU
    else:
        num_gpus = node_numranks

    if num_gpus > node_numranks:
        num_gpus = node_numranks

    color = 0 if node_rank < num_gpus else MPI.UNDEFINED

    # Global communicator for ranks with GPU.
    # Identical to comm if running on CPU.
    gpu_comm = comm.Split(color=color, key=rank)

    # Assume all nodes have same number of GPUs!
    partsPerGPU = int(sdc.nparts / (num_gpus*num_nodes))

    # How many CH are processed by each node.
    partsPerNode = int(sdc.nparts / num_nodes)
    
    gpu_global_rank = None

    # Global rank of the primary rank on each node.
    if node_rank < num_gpus:
        gpu_global_rank = rank  

    # Gather the primary ranks from each node.
    gpu_global_ranks = comm.allgather(gpu_global_rank)

    # Filter out Nones.
    gpu_global_ranks = [r for r in gpu_global_ranks if r is not None]

    # //Communicator setup finished.

    # Some data type info for numpy and torch.
    # Double precision is necessary for pyseqm.
    if torch.get_default_dtype() == torch.float32:
        eng.torch_dt = torch.float32
        sdc.torch_dt = eng.torch_dt

        eng.torch_int_dt = torch.int32
        sdc.torch_int_dt = eng.torch_int_dt

        eng.np_dt = np.float32
        sdc.np_dt = eng.np_dt

        eng.np_int_dt = np.int32
        sdc.np_int_dt = eng.np_int_dt
    else:
        eng.torch_dt = torch.float64
        sdc.torch_dt = eng.torch_dt

        eng.torch_int_dt = torch.int64
        sdc.torch_int_dt = eng.torch_int_dt

        eng.np_dt = np.float64
        sdc.np_dt = eng.np_dt

        eng.np_int_dt = np.int64
        sdc.np_int_dt = eng.np_int_dt

    fullGraph = graphNL.copy()

    # Get PYSEQM Molecule object.
    tic = time.perf_counter()
    with torch.no_grad():
        molecule_whole = get_molecule_pyseqm(sdc,
                                             sy.coords,
                                             sy.symbols,
                                             sy.types,
                                             do_large_tensors=sdc.use_pyseqm_lt,
                                             device=device)[0]                

    if rank == 0:
        print("Time to init molSysData {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

    # //Initialization for SCF

    # Things that are calculated on each primary rank of each node.
    if node_rank == 0:
        tic = time.perf_counter()
        if rank == 0:
            print('Computing cores.')

        # Core partitions.
        parts = graph_partition(sdc,
                                eng,
                                fullGraph,
                                sdc.partitionType,
                                sdc.nparts,
                                sy.coords,
                                sdc.verb)
        if rank == 0:
            print("Time to compute cores {:>7.2f} (s)"
                  .format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        if rank == 0:
            print('Loading the molecule and parameters.')

        partsCoreHalo = []
        if rank == 0:
            print('\n|||| Adaptive iter:', 0, '||||')
            print("Core and halos indices for every part:")
        for i in range(sdc.nparts):
            # Halos.
            coreHalo, nc = get_coreHaloIndicesPYSEQM(eng,
                                                     parts[i],
                                                     fullGraph,
                                                     sdc.numJumps,
                                                     sdc,
                                                     sy)
            partsCoreHalo.append(coreHalo)
            if sdc.verb:
                print("coreHalo for part", i, "=", coreHalo)
            if rank == 0:
                print('  N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'
                      .format(i, len(parts[i]), len(coreHalo)))

        print("Time to compute halos {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()

        # Graph where new_graph_for_pairs[i] is a CH in which atom i is a core
        # atom. new_graph_for_pairs[i][0] is the size of CH.
        new_graph_for_pairs = get_ch_graph(sdc,
                                           sy,
                                           fullGraph,
                                           parts,
                                           partsCoreHalo)

        # Here, same as new_graph_for_pairs.
        graph_for_pairs = new_graph_for_pairs

        # Mask for diagonal block in contracted density matrix.
        graph_maskd = get_maskd(sdc, sy, graph_for_pairs)

        print("Time to init mod graphs {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        if sdc.UHF:

            # Contracted density matrix.
            P_contr = torch.zeros(2,
                                  sy.nats*sdc.maxDeg,
                                  4,
                                  4,
                                  dtype=eng.torch_dt,
                                  device=device)

            # Diagonal initial guess.
            P_contr[:, graph_maskd] = 0.5*get_diag_guess_pyseqm(molecule_whole,
                                                                sy)

            # Shape is: (2, n_atoms, max_deg, 4, 4).
            # A rectangle of 4x4 square blocks.
            P_contr = P_contr.reshape(2,
                                      sy.nats,
                                      sdc.maxDeg,
                                      4,
                                      4).transpose(1, 2)
            P_contr_size = P_contr.size()
            P_contr_nbytes = P_contr.numel() * P_contr.element_size()

        else:
            # Contracted density matrix.
            P_contr = torch.zeros(sy.nats*sdc.maxDeg,
                                  4,
                                  4,
                                  dtype=eng.torch_dt,
                                  device=device)

            # Diagonal initial guess.
            P_contr[graph_maskd] = get_diag_guess_pyseqm(molecule_whole, sy)

            # Shape is: (n_atoms, max_deg, 4, 4).
            # A rectangle of 4x4 square blocks.
            P_contr = P_contr.reshape(sy.nats,
                                      sdc.maxDeg,
                                      4,
                                      4).transpose(0, 1)

            P_contr_size = P_contr.size()
            P_contr_nbytes = P_contr.numel() * P_contr.element_size()

        print("Time to init DM {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

        del graphNL

    else:

        parts = None
        sdc.nparts = None

        fullGraph = None
        coreHalo = None
        partsCoreHalo = None

        new_graph_for_pairs = None
        graph_for_pairs = None
        graph_maskd = None

        P_contr = None
        P_contr_size = None
        P_contr_nbytes = 0

    tic = time.perf_counter()
    parts = node_comm.bcast(parts, root=0)
    sdc.nparts = node_comm.bcast(sdc.nparts, root=0)

    if rank == 0:
        print("BCST1 {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

    tic = time.perf_counter()
    # P_contr is in shared memory between ranks on one node
    # but each node has its own copy.
    P_contr_size = node_comm.bcast(P_contr_size, root=0)

    # 8 is the size of torch.float64
    P_contr_win = MPI.Win.Allocate_shared(P_contr_nbytes,
                                          torch.tensor(0,
                                                       dtype=eng.torch_dt)
                                          .element_size(), comm=node_comm)

    P_contr_buf, P_contr_itemsize = P_contr_win.Shared_query(0)

    P_contr_ary = np.ndarray(buffer=P_contr_buf,
                             dtype=eng.np_dt,
                             shape=(P_contr_size))

    if node_rank == 0:
        P_contr_ary[:] = P_contr.cpu().numpy()

    comm.Barrier()

    del P_contr

    P_contr = torch.from_numpy(P_contr_ary).to(device)

    if rank == 0:
        print("BCST2 {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

    tic = time.perf_counter()
    fullGraph = node_comm.bcast(fullGraph, root=0)
    coreHalo = node_comm.bcast(coreHalo, root=0)
    partsCoreHalo = node_comm.bcast(partsCoreHalo, root=0)
    graph_maskd = node_comm.bcast(graph_maskd, root=0)
    graph_for_pairs = node_comm.bcast(graph_for_pairs, root=0)
    if rank == 0:
        print("BCST3 {:>7.2f} (s)"
              .format(time.perf_counter() - tic), rank)

    print("Time to init bcast and share DM {:>7.2f} (s)"
          .format(time.perf_counter() - tic), rank)

    if rank == 0:
        print("Time INIT {:>7.2f} (s)".format(time.perf_counter() - t_INIT))

    # //Initilziation for SCF finished.

    # //Begin SCF Cycle.

    # Initial chemical potential guess. TODO: This should probably not be 
    # hard-coded.
    mu0 = -5.5
    if sdc.UHF:
        mu0 = np.array([mu0+0.1, mu0-0.1])
        mu0 = np.array([-1.3, -5.5])

    # Iteration loop.
    for gsc in range(sdc.numAdaptIter):

        if rank == 0: print('\n\n|||| Adaptive iter:', gsc, '||||')
        TIC_iter = time.perf_counter()
        tic = time.perf_counter()

        # Broadcasts dm from root rank (assuming its rank 0 on node 0) to
        # primary ranks on other nodes. E.g. for ranks arranged as 
        # {node0:[0,1,2,3] node1:[4,5,6,7]}, dm is broadcates from 0 to 4.
        # One of the major bottlenecks.
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)

        if rank == 0:
            print("Time to  bcast DM_cpu_np {:>7.2f} (s)"
                  .format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()

        # Lots of things have been done during initialization,
        # so after iteration 0 we can proceed right to get_singlePoint.
        if gsc > 0:

            # Halos, dm contraction, and graphs are performed on primary ranks
            # of each node and then broadcasted locally to other ranks.
            if node_rank == 0:

                # //Begin HALOS
                tic = time.perf_counter()
                partsCoreHalo = []

                if rank == 0:
                    print("Core and halos indices for every part:")

                for i in range(sdc.nparts):

                    coreHalo, nc, nch = get_coreHaloIndices(parts[i], fullGraph, sdc.numJumps, eng=eng)
                    partsCoreHalo.append(coreHalo)

                    if sdc.verb and rank == 0:
                        print("coreHalo for part", i, "=", coreHalo)

                    if rank == 0:
                        print('  N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'
                              .format(i, len(parts[i]), len(coreHalo)))

                if rank == 0:
                    print("Time to compute halos {:>7.2f} (s)"
                          .format(time.perf_counter() - tic))
                # //End HALOS

                tic = time.perf_counter()

                # Graph where new_graph_for_pairs[i] is a CH in which atom i
                # is a core atom. new_graph_for_pairs[i][0] is the size of CH.
                new_graph_for_pairs = get_ch_graph(sdc,
                                                   sy,
                                                   fullGraph,
                                                   parts,
                                                   partsCoreHalo)

                if rank == 0:
                    print("Time to updt DM and mod graphs {:>7.2f} (s)"
                          .format(time.perf_counter() - tic))

                tic = time.perf_counter()

                # Update dm contraction based on new_graph_for_pairs.
                update_dm_contraction(sdc,
                                      sy,
                                      P_contr,
                                      graph_for_pairs,
                                      new_graph_for_pairs,
                                      device)

                # Reset graph_for_pairs.
                graph_for_pairs = new_graph_for_pairs
                if rank == 0:
                    print("Time to updt DM and mod graphs {:>7.2f} (s)"
                          .format(time.perf_counter() - tic))

                tic = time.perf_counter()

                # Diagonal mask of contracted density matrix.
                graph_maskd = get_maskd(sdc, sy, graph_for_pairs)

                if rank == 0:
                    print("Time to updt DM and mod graphs {:>7.2f} (s)"
                          .format(time.perf_counter() - tic))

            else:
                coreHalo = None
                partsCoreHalo = None
                graph_for_pairs = None
                graph_maskd = None

            tic = time.perf_counter()
            coreHalo = node_comm.bcast(coreHalo, root=0)
            partsCoreHalo = node_comm.bcast(partsCoreHalo, root=0)
            graph_for_pairs = node_comm.bcast(graph_for_pairs, root=0)
            graph_maskd = node_comm.bcast(graph_maskd, root=0)
            if node_rank == 0:
                print("Time to bcast DM and mod graphs {:>7.2f} (s)"
                      .format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()

        # Single point part. For efficiency, the PySEQM density matrix
        # needs to be reshaped in 4x4 blocks.

        # This will sum electronic energy from CHs on ranks, giving total Eelec
        global_Eelec = np.zeros(1, dtype=np.float64)

        # TODO: All of this is so PYSEQM specific already, this should probably
        # be removed.
        if eng.interface == "PySEQM":
            with torch.no_grad():
                # This condition is for GPU jobs only because sometimes there
                # are fewer GPUs per node than ranks per nodes.

                if node_rank < num_gpus:
                    # We want more ranks per node because dm update always
                    # happens on CPU, on node 0, in parallel.
                    (eElec,
                     eValOnRank_list,
                     Q_list,
                     NH_Nh_Hs_list,
                     core_indices_in_sub_expanded_list,
                     Nocc_list, mu0) = get_singlePoint(sdc,
                                                       eng,
                                                       partsPerGPU,
                                                       partsPerNode,
                                                       node_id,
                                                       node_rank,
                                                       rank,
                                                       gpu_comm,
                                                       parts,
                                                       partsCoreHalo,
                                                       sy,
                                                       hindex,
                                                       mu0,
                                                       molecule_whole,
                                                       P_contr,
                                                       graph_for_pairs,
                                                       graph_maskd)

                    gpu_comm.Allreduce(eElec, global_Eelec, op=MPI.SUM)

                else:
                    (eElec,
                     eValOnRank_list,
                     Q_list,
                     NH_Nh_Hs_list,
                     core_indices_in_sub_expanded_list,
                     Nocc_list,
                     mu0) = 0, None, None, None, None, None, None

            comm.Barrier()

        else:
            raise ValueError(f"ERROR!!!: Interface type not recognized: '{eng.interface}'. " +
                             f"Use any of the following: Module,File,Socket,MDI")

        if gsc == 0 and sdc.UHF:
            print('sym break')

            for I in range(len(Q_list)):
                orb_idx = NH_Nh_Hs_list[I][3][0]
                Q_list[I][0,:,orb_idx] = 0.9*Q_list[I][0,:,orb_idx-1] + 0.1*Q_list[I][0,:,orb_idx]

        sdc.EelecNew = global_Eelec[0]

        if rank == 0:
            print("Time to get_singlePoint {:>7.2f} (s)"
                  .format(time.perf_counter() - tic))

        # If True, these files will be read and used instead as default initial guess.
        if sdc.restartLoad: 
            sdc.restartLoad = False
            if node_rank == 0:
                P_contr[:] = torch.load('P_contr.pt')
            with open('parts.pkl', 'rb') as f:
                parts = pickle.load(f)
            with open('partsCoreHalo.pkl', 'rb') as f:
                partsCoreHalo = pickle.load(f)
            with open('fullGraph.pkl', 'rb') as f:
                fullGraph = pickle.load(f)

            mu0 = np.load('mu0.npy')
            graph_for_pairs = np.load('graph_for_pairs.npy')
            graph_maskd = np.load('graph_maskd.npy')

            if rank == 0:
                eValOnRank_list = torch.load('eValOnRank_list.pt')
                Q_list = torch.load('Q_list.pt')
                NH_Nh_Hs_list = torch.load('NH_Nh_Hs_list.pt')
                core_indices_in_sub_expanded_list = torch.load('core_indices_in_sub_expanded_list.pt')
                Nocc_list = torch.load('Nocc_list.pt')

        # Save for future restart. Slows things down significantly.
        if rank == 0 and sdc.restartSave:
            torch.save(eValOnRank_list, 'eValOnRank_list.pt')
            torch.save(Q_list, 'Q_list.pt')
            torch.save(NH_Nh_Hs_list, 'NH_Nh_Hs_list.pt')
            torch.save(core_indices_in_sub_expanded_list,
                       'core_indices_in_sub_expanded_list.pt')
            torch.save(Nocc_list, 'Nocc_list.pt')
            torch.save(P_contr, 'P_contr.pt')
            with open('parts.pkl', 'wb') as f:
                pickle.dump(parts, f)
            with open('partsCoreHalo.pkl', 'wb') as f:
                pickle.dump(partsCoreHalo, f)
            with open('fullGraph.pkl', 'wb') as f:
                pickle.dump(fullGraph, f)
            np.save('mu0', mu0)
            np.save('graph_for_pairs', graph_for_pairs)
            np.save('graph_maskd', graph_maskd)

        # This defines what part of density matrix will be
        # updated by each rank on node 0.
        if rank == 0:
            tic = time.perf_counter()
            partsPerRank = int(sdc.nparts / node_numranks)
            partIndex1 = 0 * partsPerRank
            partIndex2 = (0 + 1) * partsPerRank
            # Root rank processes its own part
            Q_list_on_rank = Q_list[partIndex1:partIndex2]

            for r in range(1, node_numranks):
                partIndex1 = r * partsPerRank
                partIndex2 = (r + 1) * partsPerRank

                # Send only the necessary slice to each rank
                node_comm.send(Q_list[partIndex1:partIndex2], dest=r, tag=0)

            print("Time send Q_list slice {:>7.2f} (s)"
                  .format(time.perf_counter() - tic))

        if rank < node_numranks and rank != 0:
            Q_list_on_rank = node_comm.recv(source=0, tag=0)

        if rank < node_numranks:
            tic = time.perf_counter()

            # Broadcast data across ranks on node 0.
            eValOnRank_list = node_comm.bcast(eValOnRank_list, root=0)
            NH_Nh_Hs_list = node_comm.bcast(NH_Nh_Hs_list, root=0)
            core_indices_in_sub_expanded_list = node_comm.bcast(core_indices_in_sub_expanded_list, root=0)
            Nocc_list = node_comm.bcast(Nocc_list, root=0)
            mu0 = node_comm.bcast(mu0, root=0)
            node_comm.Barrier()

            # Density matrix update and the graph from the DM.
            with torch.no_grad():
                fullGraphRho = get_singlePointDM(sdc,
                                                 eng,
                                                 rank,
                                                 node_numranks,
                                                 node_comm,
                                                 parts,
                                                 partsCoreHalo,
                                                 sy,
                                                 hindex,
                                                 mu0,
                                                 P_contr,
                                                 graph_for_pairs,
                                                 eValOnRank_list,
                                                 Q_list_on_rank,
                                                 NH_Nh_Hs_list,
                                                 core_indices_in_sub_expanded_list)

            if rank == 0:
                print("Time to updt DM {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

            node_comm.Barrier()

            tic = time.perf_counter()

            # Get graph derived from the density matrix (on each rank.)
            fullGraphRho_LIST = node_comm.gather(fullGraphRho,
                                                 root=0)
            if rank == 0:

                # Adds the graph we got on the previous iteration. NOTE:, when
                # doing SCF, the graph keeps growing, no nodes from previous
                # iterations are removed.
                fullGraphRho_LIST.append(fullGraph)

                # Combines graphs from the Python List.
                fullGraph = add_mult_graphs(fullGraphRho_LIST)

                print("Time to add graphs {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

            del fullGraphRho

            if rank == 0:
                tic = time.perf_counter()
                if sdc.UHF:
                    trace = torch.sum(P_contr.transpose(1, 2)
                                      .reshape(2,
                                               molecule_whole.molsize*(len(graph_for_pairs[0])-1),
                                               4,
                                               4)[:, graph_maskd].diagonal(dim1=-2, dim2=-1), dim=(1, 2))

                    print("DM TRACE: {:>10.8f}, {:>10.8f}"
                          .format(trace[0], trace[1]))

                else:
                    trace = torch.sum(P_contr.transpose(0, 1)
                                      .reshape(molecule_whole.molsize*(len(graph_for_pairs[0])-1),
                                               4,
                                               4)[graph_maskd].diagonal(dim1=-2, dim2=-1))

                    print("DM TRACE: {:>10.7f}".format(trace))

                print("Time to get trace {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

        else:
            fullGraph = None

        tic = time.perf_counter()

        # Broadcast the new graph across ALL ranks.
        fullGraph = comm.bcast(fullGraph, root=0)

        if rank == 0:
            print("Time to bcast fullGraph {:>7.2f} (s)"
                  .format(time.perf_counter() - tic))

        del eValOnRank_list, Q_list, NH_Nh_Hs_list, Nocc_list

        torch.cuda.empty_cache()
        if rank == 0:
            print("t Iter {:>8.2f} (s)"
                  .format(time.perf_counter() - TIC_iter))

    # //SCF Cycle complete.

    # //Forces begin
    tic_F_INIT = time.perf_counter()

    if node_rank < num_gpus:

        # Broadcast density matrix from root rank (assuming rank=node=0.)
        # to primary ranks on other nodes.
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)
            forces = np.zeros((sy.coords.shape))
            partsCoreHalo = []
            if rank == 0:
                print("\nCore and halos indices for every part:")

            for i in range(sdc.nparts):
                coreHalo, nc, nch = get_coreHaloIndices(parts[i],
                                                        fullGraph,
                                                        sdc.numJumps,
                                                        eng=eng)

                partsCoreHalo.append(coreHalo)
                if sdc.verb:
                    print("coreHalo for part", i, "=", coreHalo)
                if rank == 0:
                    print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'
                          .format(i, len(parts[i]), len(coreHalo)))

            tic = time.perf_counter()
            new_graph_for_pairs = get_ch_graph(sdc, sy, fullGraph, parts, partsCoreHalo)
            if rank == 0:
                print("Time to updt DM and mod graphs {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

            tic = time.perf_counter()
            update_dm_contraction(sdc,
                                  sy,
                                  P_contr,
                                  graph_for_pairs,
                                  new_graph_for_pairs,
                                  device)

            graph_for_pairs = new_graph_for_pairs
            if rank == 0:
                print("Time to updt DM and mod graphs {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

            tic = time.perf_counter()
            graph_maskd = get_maskd(sdc, sy, graph_for_pairs)

            if rank == 0:
                print("Time to updt DM and mod graphs {:>7.2f} (s)"
                      .format(time.perf_counter() - tic))

        else:
            forces = None
            partsCoreHalo = None
            new_graph_for_pairs = None
            graph_for_pairs = None
            graph_maskd = None

        if sdc.scfDevice == 'cuda':
            device = 'cuda:{}'.format(node_rank)
        else:
            device = 'cpu'

        molecule_whole = get_molecule_pyseqm(sdc,
                                             sy.coords,
                                             sy.symbols,
                                             sy.types,
                                             do_large_tensors=sdc.use_pyseqm_lt,
                                             device=device)[0]

        forces = gpu_comm.bcast(forces, root=0)
        partsCoreHalo = gpu_comm.bcast(partsCoreHalo, root=0)
        gpu_comm.Barrier()
        graph_for_pairs = gpu_comm.bcast(graph_for_pairs, root=0)
        new_graph_for_pairs = gpu_comm.bcast(new_graph_for_pairs, root=0)
        graph_maskd = gpu_comm.bcast(graph_maskd, root=0)
        if rank == 0:
            forces[:] = .0
        gpu_comm.Barrier()

        if rank == 0:
            print("Time init forces {:>8.2f} (s)"
                  .format(time.perf_counter() - tic_F_INIT))

        tic = time.perf_counter()
        if eng.interface == "PySEQM":

            if sdc.doForces:
                eElec = get_singlePointForces(sdc,
                                              eng,
                                              partsPerGPU,
                                              partsPerNode,
                                              node_id,
                                              node_rank,
                                              rank,
                                              parts,
                                              partsCoreHalo,
                                              sy,
                                              hindex,
                                              forces,
                                              molecule_whole,
                                              None,
                                              P_contr.to(device),
                                              graph_for_pairs,
                                              graph_maskd)
            else:
                with torch.no_grad():
                    eElec = get_singlePointForces(sdc,
                                                  eng,
                                                  partsPerGPU,
                                                  partsPerNode,
                                                  node_id,
                                                  node_rank,
                                                  rank,
                                                  parts,
                                                  partsCoreHalo,
                                                  sy,
                                                  hindex,
                                                  forces,
                                                  molecule_whole,
                                                  None,
                                                  P_contr.to(device),
                                                  graph_for_pairs,
                                                  graph_maskd)

            global_Eelec = np.zeros(1, dtype=np.float64)
            gpu_comm.Barrier()
            gpu_comm.Allreduce(MPI.IN_PLACE, forces, op=MPI.SUM)

            # Primary communicator.
            gpu_comm.Allreduce(eElec, global_Eelec, op=MPI.SUM)

        if rank == 0:
            print("Time to get electron forces {:>8.2f} (s)"
                  .format(time.perf_counter() - tic))

            print("eElec:   {:>10.12f}"
                  .format(global_Eelec[0]),)

        # Nuclear energy and forces. For now, done on one cpu/gpu,
        # for the whole system at once (pyseqm style). Hence, do_large_tensors
        # = True. Needs to be fixed.
        if rank == 0:
            # Object with whatever initial parameters and tensors.
            molSysData = pyseqmObjects(sdc,
                                       sy.coords,
                                       sy.symbols,
                                       sy.types,
                                       do_large_tensors=True,
                                       device=device)

            tic = time.perf_counter()
            eNucAB = get_eNuc(eng, molSysData)
            eTot, eNuc = get_eTot(eng, molSysData, eNucAB, 0)
            print("Enuc:   {:>10.12f}".format(eNuc),)
            L = eNuc.sum()
            L.backward()
            forceNuc = -molSysData.molecule_whole.coordinates.grad.detach()
            molSysData.molecule_whole.coordinates.grad.zero_()
            print("Time to get nuclear forces {:>8.2f} (s)".format(time.perf_counter() - tic))
            np.save('forces', (forces+forceNuc.cpu().numpy()[0]), )

    # //Forces finished.


def get_adaptiveDM(sdc,
                   eng,
                   comm,
                   rank,
                   numranks,
                   sy,
                   hindex,
                   graphNL):
    """

    Gets the adaptive denisty matrix at the current iteration.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng:
        The SEDACS Engine.
    comm:
        The MPI communicator.
    rank:
        The MPI rank.
    numranks:
        Number of global MPI ranks.
    sy:
        The full system SEDACS System object.
    hindex:
        Atom-wise orbital indices.
    graphNL:
        The thresholded graph neighborlist.

    Returns
    -------
    fullGraph:
        The full graph for the entire system.
    charges:
        Charges for the atoms in the system.
    parts:
        The core partitions.
    subSysOnRank:
        Rank indices for a given subsystem.
    """

    if eng.interface == "PySEQM":
        get_adaptiveDM_PYSEQM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL)
        return

    fullGraph = graphNL

    # Initial guess for the excess ocupation vector. This is the negative of
    # the charge!
    charges = np.zeros(sy.nats)
    chargesOld = np.zeros(sy.nats)

    # TODO: Remove?
    chargesIn = None
    chargesOld = None
    chargesOut = None

    # Loop over number of iterations.
    for gscf in range(sdc.numAdaptIter):
        msg = "Graph-adaptive iteration" + str(gscf)
        status_at("get_adaptiveSCFDM", msg)

        # Partition the graph
        parts = graph_partition(fullGraph,
                                sdc.partitionType,
                                sdc.nparts,
                                False)

        njumps = 1
        partsCoreHalo = []
        numCores = []
        print("\nCore and halos indices for every part:")

        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i],
                                                   fullGraph,
                                                   njumps)

            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("coreHalo for part", i, "=", coreHalo)

        (fullGraphRho,
         charges,
         subSysOnRank) = get_singlePoint_charges(sdc,
                                                 eng,
                                                 rank,
                                                 numranks,
                                                 comm,
                                                 parts,
                                                 partsCoreHalo,
                                                 sy,
                                                 hindex,
                                                 gscf)

        print("Collected charges", charges)

        fullGraph = add_graphs(fullGraphRho, graphNL)
        for i in range(sy.nats):
            print("Charges:", i, sy.symbols[sy.types[i]], charges[i])

        # FIXME: scfError where?
        print("SCF ERR =", scfError)
        print("TotalCharge", sum(charges))

        if (scfError < sdc.scfTol):
            status_at("get_adaptiveSCFDM",
                      "SCF converged with SCF error = "+str(scfError))
            break

        if (gscf == sdc.numAdaptIter - 1):
            warning_at("get_adaptiveSCFDM",
                       "SCF did not converged ... ")

    AtToPrint = 0

    subSy = System(fullGraphRho[AtToPrint, 0])
    subSy.symbols = sy.symbols
    subSy.coords, subSy.types = extract_subsystem(sy.coords,
                                                  sy.types,
                                                  sy.symbols,
                                                  fullGraph[AtToPrint, 1 : fullGraph[AtToPrint, 0] + 1])

    if rank == 0:
        write_pdb_coordinates("subSyG_fin.pdb", subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates("subSyG_fin.xyz", subSy.coords, subSy.types, subSy.symbols)

    return fullGraph, charges, parts, subSysOnRank
