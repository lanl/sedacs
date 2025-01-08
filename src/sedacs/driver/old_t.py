"""Graph adaptive solver"""

import time
import torch
print('num_threads',torch.get_num_threads())
#torch.set_num_threads(20)  

from sedacs.density_matrix import get_density_matrix, get_initDM, get_dmErrs, get_dmTrace
from sedacs.density_matrix_renorm import get_density_matrix_renorm
from sedacs.energy import get_eElec, get_eNuc, get_eTot
from sedacs.forces import get_forces
from sedacs.molSysData import get_molSysData
from sedacs.fock import get_fock
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph, add_mult_graphs
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.hamiltonian import get_hamiltonian, get_force
from sedacs.mpi import collect_and_sum_matrices
from sedacs.system import System, extract_subsystem
from sedacs.evals import get_eVals
from sedacs.chemical_potential import get_mu
from sedacs.graph import get_initial_graph
from sedacs.overlap import get_overlap
from sedacs.interface_pyseqm import get_coreHalo_ham_inds, get_diag_guess_pyseqm, ParamContainer, pyseqmObjects, get_molecule_pyseqm
import itertools
import sys
import psutil
import pickle
import socket
import copy

from seqm.seqm_functions.pack import pack

import gc

import numpy as np

try:
    from mpi4py import MPI

    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False

is_mpi_available = False

mpiOnDebugFlag = True


__all__ = ["get_singlePoint", "get_adaptiveDM"]

## Single point calculation
# @brief Construct a connectivity graph based on constructing density matrices
# of parts of the system.
#
def get_singlePoint(sdc, eng,  partsPerGPU, partsPerNode, node_id, node_rank, rank, numranks, comm, gpu_global_comm, parts, partsCoreHalo, sy, hindex, mu0,
                    molecule_whole, P, P_contr, graph_for_pairs, graph_maskd):
    # computing DM for core+halo part
    partsPerRank = int(sdc.nparts / numranks)
    # partIndex1 = rank * partsPerRank
    # partIndex2 = (rank + 1) * partsPerRank
    partIndex1 = (node_rank) * partsPerGPU + node_id*partsPerNode #+ node_id * num_nodes
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id*partsPerNode #+ node_id * num_nodes


    graphOnRank = None
    dValOnRank = np.array([])
    eValOnRank = np.array([])
    eValOnRank_list = []
    Q_list = [] # Eigenvectors for each part
    I_list = [] # Indices for updating the columns in total DM
    I_halo_list = [] # indices of coreHalo in whole
    Nocc_list = [] # Number of occupied orbitals for each part
    core_indices_in_sub_expanded_list = [] # Indices of core hamiltonian in core+halo hamiltonian. Might be useful when core and halo atoms are shuffled, like in PySEQM.
    NH_Nh_Hs_list = [] # list of [number_of_heavy_atoms, number_of_hydrogens, dim_of_coreHalo_ham]
    Tel = sdc.Tel

    for partIndex in range(partIndex1, partIndex2):
        tic = time.perf_counter()
        print("\n Rank, part", rank, partIndex)
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])
        #partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        #write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        #write_xyz_coordinates("subSy" + str(rank) + "_" + str(partIndex) + ".xyz", subSy.coords, subSy.types, subSy.symbols)

        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords,subSyCore.types = extract_subsystem(sy.coords,sy.types,sy.symbols,parts[partIndex])
        #partCoreFileName = "CoreSubSy"+str(rank)+"_"+str(partIndex)+".pdb"
        #write_pdb_coordinates(partCoreFileName,subSyCore.coords,subSyCore.types,subSyCore.symbols)
        #write_xyz_coordinates("CoreSubSy"+str(rank)+"_"+str(partIndex)+".xyz",subSyCore.coords,subSyCore.types,subSyCore.symbols)

        if sdc.scfDevice == 'cuda':
            device = 'cuda:{}'.format(node_rank)
            tmp_molecule_whole = get_molecule_pyseqm(sdc, sy.coords, sy.symbols, sy.types, do_large_tensors = sdc.use_pyseqm_lt, device=device)[0]
            ham = get_hamiltonian(sdc, eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], tmp_molecule_whole, P, P_contr.to(device), graph_for_pairs, graph_maskd, None,
                              verbose=False)
            del tmp_molecule_whole
        else:
            device = 'cpu'
            ham = get_hamiltonian(sdc, eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], molecule_whole, P, P_contr, graph_for_pairs, graph_maskd, None,
                              verbose=False)
        print("TOT {:>8.3f} (s)".format(time.perf_counter() - tic))

        tic = time.perf_counter()
        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs, I, I_halo, core_indices_in_sub_expanded = get_eVals(eng, sdc, sy, occ, ham, subSy.coords, subSy.symbols, subSy.types, Tel, mu0,
                        coreSize, subSy, subSyCore, parts[partIndex], partsCoreHalo[partIndex],
                        verbose=False)

        del ham
        
        dValOnRank = np.append(dValOnRank, dVals)
        eValOnRank = np.append(eValOnRank, eVals.cpu().numpy())

        eValOnRank_list.append(eVals.cpu())
        Q_list.append(Q.cpu()#.to(torch.float32)
                      )
        I_list.append(I)
        I_halo_list.append(I_halo)
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        print("| t eVals/dVals {:>9.4f} (s)".format(time.perf_counter() - tic))

    
    tic = time.perf_counter()
    torch.save(Q_list, 'Q/Q_list_{}.pt'.format(rank))
    print("Time to save Q_list {:>9.4f} (s)".format(time.perf_counter() - tic))

    tic = time.perf_counter()
    full_dVals = None
    full_eVals = None
    eValOnRank_size = np.array(len(eValOnRank), dtype=int)
    eValOnRank_SIZES = None

    if mpiOnDebugFlag:
        if rank == 0:
            eValOnRank_SIZES = np.empty(gpu_global_comm.Get_size(), dtype=int)
            
        gpu_global_comm.Gather(eValOnRank_size, eValOnRank_SIZES, root=0)
        if rank == 0:
            full_dVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)
            full_eVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)

        gpu_global_comm.Gatherv(dValOnRank, [full_dVals, eValOnRank_SIZES], root=0)
        gpu_global_comm.Gatherv(eValOnRank, [full_eVals, eValOnRank_SIZES], root=0)
        eVal_LIST = gpu_global_comm.gather(eValOnRank_list, root=0)
        #Q_LIST = comm.gather(Q_list, root=0)
        NH_Nh_Hs_LIST = gpu_global_comm.gather(NH_Nh_Hs_list, root=0)
        I_LIST = gpu_global_comm.gather(I_list, root=0)
        I_halo_LIST = gpu_global_comm.gather(I_halo_list, root=0)
        core_indices_in_sub_expanded_LIST = gpu_global_comm.gather(core_indices_in_sub_expanded_list, root=0)
        Nocc_LIST = gpu_global_comm.gather(Nocc_list, root=0)

        if rank == 0:

            Q_LIST = []
            for i in range(numranks):
                Q_LIST.append(torch.load('Q/Q_list_{}.pt'.format(i)))

            # Flatten the nested list of lists into a single list of tensors
            eVal_LIST = list(itertools.chain(*eVal_LIST))
            Q_LIST = list(itertools.chain(*Q_LIST))
            NH_Nh_Hs_LIST = list(itertools.chain(*NH_Nh_Hs_LIST))
            I_LIST = list(itertools.chain(*I_LIST))
            I_halo_LIST = list(itertools.chain(*I_halo_LIST))
            core_indices_in_sub_expanded_LIST = list(itertools.chain(*core_indices_in_sub_expanded_LIST))
            Nocc_LIST = list(itertools.chain(*Nocc_LIST))
        else:
            Q_LIST = None


    if node_rank == 0: print("| t commLists {:>9.4f} (s)".format(time.perf_counter() - tic), rank)
    if rank == 0:
        mu0 = get_mu(mu0, full_dVals, full_eVals, Tel, sy.numel/2)

    return eVal_LIST, Q_LIST, NH_Nh_Hs_LIST, I_LIST, I_halo_LIST, core_indices_in_sub_expanded_LIST, Nocc_LIST, mu0

def get_singlePointForces(sdc, eng, partsPerGPU, partsPerNode, node_id, node_rank, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData, P, P_contr, graph_for_pairs, graph_maskd):
    # partsPerRank = int(sdc.nparts / numranks)
    # partIndex1 = rank * partsPerRank
    # partIndex2 = (rank + 1) * partsPerRank
    partIndex1 = (node_rank) * partsPerGPU + node_id*partsPerNode #+ node_id * num_nodes
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id*partsPerNode #+ node_id * num_nodes

    EELEC = 0.0
    for partIndex in range(partIndex1, partIndex2):
        print("Rank, part", rank, partIndex)
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])

        _, core_indices_in_sub_expanded, _, _, _ = \
            get_coreHalo_ham_inds(parts[partIndex], partsCoreHalo[partIndex], sdc, sy, subSy)

        tic = time.perf_counter()        
        tmp_molecule_whole = copy.deepcopy(molSysData.molecule_whole)
        if sdc.doForces:
            tmp_molecule_whole.coordinates.requires_grad_(True)

        # get_force
        # get_hamiltonian
        f, eElec = get_hamiltonian(sdc, eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], tmp_molecule_whole, P, P_contr, graph_for_pairs, graph_maskd, core_indices_in_sub_expanded, doForces = True,
                              verbose=False)
        del tmp_molecule_whole
        # if mpiOnDebugFlag:
        #     comm.Allreduce(f, forces, op=MPI.SUM)
        # else:
        #     forces += f

        forces += f
        EELEC += eElec
        print("EelecCH {:>7.3f} |".format(eElec.item()), end=" ")
        del eElec, subSy, f
        print("TOT", time.perf_counter() - tic, "(s)")
    #print("eElec_SUM: {:>10.7f}".format(EELEC),)    
    return EELEC

def get_singlePointDM(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, dm, P_contr, graph_for_pairs,
                      eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list):
    
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None

    Tel = sdc.Tel
    maxDifList = []
    sumDifTot = 0
    P_contr_maxDifList = []
    P_contr_sumDifTot = 0

    for partIndex in range(partIndex1,partIndex2):
        #tic = time.perf_counter()
        # this will calculate the DM in subsys and update the whole DM
        rho_ren, maxDif, sumDif = get_density_matrix_renorm(eng, Tel, mu0, dm, P_contr, graph_for_pairs,
                                            eValOnRank_list[partIndex], Q_list[partIndex].to(torch.float64), NH_Nh_Hs_list[partIndex], I_list[partIndex], core_indices_in_sub_expanded_list[partIndex], Nocc_list[partIndex])

        indices_in_sub = np.linspace(0,len(partsCoreHalo[partIndex])-1, len(partsCoreHalo[partIndex]), dtype = eng.np_int_dt)
        core_indices_in_sub = indices_in_sub[np.isin(partsCoreHalo[partIndex], parts[partIndex])]
        
        #print("t DM1 {:>8.3f} (s)".format(time.perf_counter() - tic))
        alpha = sdc.alpha
        P_contr_maxDif = []
        P_contr_sumDif = 0

        ### vectorized loop. Faster for larger cores.
        max_len = graph_for_pairs[parts[partIndex][0]][0]
        TMP1 = P_contr[:max_len,parts[partIndex]]#.clone()
        TMP2 = rho_ren.reshape((1, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4)) \
                                .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]), (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),4,4).transpose(2,3).transpose(0,1)[:,core_indices_in_sub]#.clone()
        P_contr_maxDif.append(torch.max(torch.abs(TMP1 - TMP2)).cpu().numpy())
        P_contr_sumDif += torch.sum(torch.abs(TMP1 - TMP2)).cpu().numpy()
        P_contr[:max_len,parts[partIndex]] = (1-alpha)*TMP1 + alpha * TMP2

        # for i in range(len(parts[partIndex])):
        #     tmp1 = P_contr[:graph_for_pairs[parts[partIndex][i]][0],parts[partIndex][i]]
        #     tmp2 = rho_ren.reshape((1, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4)) \
        #                         .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]), (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),4,4)[core_indices_in_sub[i]].transpose(1,2)

        #     P_contr_maxDif.append(torch.max(torch.abs(tmp1 - tmp2)).cpu().numpy())
        #     P_contr_sumDif += torch.sum(torch.abs(tmp1 - tmp2)).cpu().numpy()
        #     P_contr[:graph_for_pairs[parts[partIndex][i]][0],parts[partIndex][i]] = (1-alpha)*tmp1 + alpha*tmp2
        #     del tmp1, tmp2
            
        rho_ren = pack(rho_ren, NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])

        P_contr_maxDif = max(P_contr_maxDif)
        P_contr_maxDifList.append(P_contr_maxDif)
        P_contr_sumDifTot += P_contr_sumDif

        maxDifList.append(maxDif)
        try:
            sumDifTot += sumDif
        except:
            sumDifTot += 0
        graphOnRank = collect_graph_from_rho(graphOnRank, rho_ren, sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False)
        # graphOnRank = collect_graph_from_rho(graphOnRank,
        #                                      pack(dm[:,I_halo_list[partIndex][0], I_halo_list[partIndex][1]], NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])[0],
        #                                      sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False)
        del rho_ren

    print('HERE_DM_1')
    if eng.reconstruct_dm:
        print(" MAX |\u0394DM_ij|: {:>10.7f} at SubSy {:>5d}".format(max(maxDifList), np.argmax(maxDifList)))
        print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(sumDifTot))

    print(" MAX |\u0394DM_ij|: {:>10.7f} at SubSy {:>5d}".format(max(P_contr_maxDifList), np.argmax(P_contr_maxDifList)))
    print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(P_contr_sumDifTot))


    return graphOnRank

def print_memory_usage(rank, node_rank, message):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message} | Rank: {rank}, Node Rank: {node_rank}, Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")
def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)
# Collect all tensors in the current environment
def get_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                yield obj
        except Exception as e:
            pass

def print_attribute_sizes(obj):
    for attr in dir(obj):
        # Skip private or callable attributes
        if attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        attribute = getattr(obj, attr)
        size_bytes = attribute.nbytes if isinstance(attribute, np.ndarray) else attribute.element_size() * attribute.nelement() if isinstance(attribute, torch.Tensor) else sys.getsizeof(attribute)
        size_mb = size_bytes / (1024 ** 2)  # Convert bytes to MB
        print(f"{attr}: {size_mb:.2f} MB")

class MyClass:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"MyClass(data={self.data})"


def get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL):
    t_INIT = time.perf_counter()
    tic = time.perf_counter()
    eng.use_pyseqm_lt = False
    sdc.use_pyseqm_lt = eng.use_pyseqm_lt

    eng.reconstruct_dm = False
    sdc.reconstruct_dm = eng.reconstruct_dm

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # local communicator on a node
    node_rank = node_comm.Get_rank()  # Rank within the node
    node_numranks = node_comm.Get_size()

    node_name = socket.gethostname()
    node_names = comm.allgather(node_name)

    unique_nodes = list(set(node_names))
    num_nodes = len(unique_nodes)         # Total number of unique nodes
    node_id = int(rank/node_numranks)

    primary_rank = None
    if node_rank == 0:
        primary_rank = rank  # Global rank of the primary rank on each node

    # Gather the primary ranks from each node
    primary_ranks = comm.allgather(primary_rank)
    primary_ranks = [r for r in primary_ranks if r is not None]  # Filter out None values

    color = 0 if rank in primary_ranks else MPI.UNDEFINED
    primary_comm = comm.Split(color=color, key=rank)

    device = 'cpu'

    if sdc.fDevice == 'cuda' and sdc.numGPU == -1:
        num_gpus = torch.cuda.device_count()
    elif sdc.fDevice == 'cuda':
        num_gpus = sdc.numGPU
    else:
        num_gpus = node_numranks

    if num_gpus > node_numranks:
        num_gpus = node_numranks

    color = 0 if node_rank < num_gpus else MPI.UNDEFINED
    gpu_comm = comm.Split(color=color, key=rank) # comm for local GPUs. Identical to node_comm if running on CPU.
    partsPerGPU = int(sdc.nparts / (num_gpus*num_nodes)) # assume all nodes have same number of GPUs
    partsPerNode = int(sdc.nparts / num_nodes)

    color = 0 if node_rank < num_gpus else MPI.UNDEFINED
    gpu_global_comm = comm.Split(color=color, key=rank) # global communicator for ranks with GPU. Identical to comm if running on CPU.
    #gpu_global_rank = gpu_global_comm.Get_rank()  # Rank within the node
    #gpu_global_numranks = gpu_global_comm.Get_size()

    #print('gpu_global_rank', gpu_global_rank, gpu_global_numranks)



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

    njumps = sdc.numJumps

    tic = time.perf_counter()
    fullGraph = graphNL.copy()

    with torch.no_grad():
        molecule_whole = get_molecule_pyseqm(sdc, sy.coords, sy.symbols, sy.types, do_large_tensors = sdc.use_pyseqm_lt, device=device)[0]
                                             
    #print_attribute_sizes(molSysData.molecule_whole)
    if rank == 0: print("Time to init molSysData {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

    if node_rank == 0:
        tic = time.perf_counter()
        print('Computing cores.')
        parts = graph_partition(eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, sdc.verb)
        print("Time to compute cores {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        print('Loading the molecule and parameters.')
        if eng.reconstruct_dm:
            dm = get_initDM(eng, sdc, sy.coords, sy.symbols, sy.types, molecule_whole)#.share_memory_()
            dm_size = dm.size()
            nbytes = dm.numel() * dm.element_size()
        
        partsCoreHalo = []
        if rank == 0:
            print('\n\n|||| Adaptive iter:', 0, '||||')
            print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
            partsCoreHalo.append(coreHalo)
            if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
            if rank == 0: print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')
        print("Time to compute halos {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        new_graph_for_pairs = np.array(fullGraph.copy())
        for i in range(sy.nats):
            for sublist_idx in range(sdc.nparts):
                if i in parts[sublist_idx]:
                    new_graph_for_pairs[i, 0] = len(partsCoreHalo[sublist_idx])
                    new_graph_for_pairs[i, 1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
                    break

        graph_for_pairs = new_graph_for_pairs
        graph_maskd = []
        counter = 0
        for j in range(sy.nats):
            for i in graph_for_pairs[j][1:graph_for_pairs[j][0]+1]: 
                if i==j:
                    graph_maskd.append(counter)
                counter +=1
            counter += int(sdc.maxDeg - graph_for_pairs[j][0])
        print("Time to init mod graphs {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        P_contr = torch.zeros(sy.nats*sdc.maxDeg,4,4, dtype=eng.torch_dt, device=device)  # density matrix
        P_contr[graph_maskd] = get_diag_guess_pyseqm(molecule_whole, sy)
        P_contr = P_contr.reshape(sy.nats, sdc.maxDeg, 4,4).transpose(0,1)
        graph_maskd = np.array(graph_maskd)
        P_contr_size = P_contr.size()
        P_contr_nbytes = P_contr.numel() * P_contr.element_size()
        
        print("Time to init DM {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

        # graphNL = collect_graph_from_rho(None, sdc.overlap_whole, sdc.gthreshinit, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
        del graphNL
    else:
        parts = None
        sdc.nparts = None
        if eng.reconstruct_dm:
            dm = None
            dm_size = None
            nbytes = 0

        fullGraph = None
        coreHalo = None
        partsCoreHalo = None

        new_graph_for_pairs = None
        graph_for_pairs = None
        graph_maskd = None

        P_contr = None
        P_contr_size = None
        P_contr_nbytes = 0
    
    if mpiOnDebugFlag:
        tic = time.perf_counter()
        parts = node_comm.bcast(parts, root=0)
        sdc.nparts = node_comm.bcast(sdc.nparts, root=0)
        if rank == 0: print("BCST1 {:>7.2f} (s)".format(time.perf_counter() - tic), rank)


        tic = time.perf_counter()
        if eng.reconstruct_dm:
            dm_size = comm.bcast(dm_size, root=0)
            win = MPI.Win.Allocate_shared(nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=comm) # 8 is the size of torch.float64
            buf, itemsize = win.Shared_query(0) 
            #assert itemsize == MPI.DOUBLE.Get_size() 
            ary = np.ndarray(buffer=buf, dtype=eng.np_dt, shape=(dm_size))
            if rank == 0:
                ary[:] = dm.numpy()   
            del dm
            dm = torch.from_numpy(ary)
            print(ary.shape)
            print(dm.shape)

        P_contr_size = node_comm.bcast(P_contr_size, root=0)
        P_contr_win = MPI.Win.Allocate_shared(P_contr_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
        P_contr_buf, P_contr_itemsize = P_contr_win.Shared_query(0) 
        #assert P_contr_itemsize == MPI.DOUBLE.Get_size() 
        P_contr_ary = np.ndarray(buffer=P_contr_buf, dtype=eng.np_dt, shape=(P_contr_size))
        if node_rank == 0:
            P_contr_ary[:] = P_contr.cpu().numpy()   
        comm.Barrier()

        del P_contr
        P_contr = torch.from_numpy(P_contr_ary).to(device)
        if rank == 0: print("BCST2 {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

        tic = time.perf_counter()
        fullGraph = node_comm.bcast(fullGraph, root=0)
        coreHalo = node_comm.bcast(coreHalo, root=0)
        partsCoreHalo = node_comm.bcast(partsCoreHalo, root=0)
        #new_graph_for_pairs = node_comm.bcast(new_graph_for_pairs, root=0)
        graph_maskd = node_comm.bcast(graph_maskd, root=0)
        graph_for_pairs = node_comm.bcast(graph_for_pairs, root=0)
        if rank == 0: print("BCST3 {:>7.2f} (s)".format(time.perf_counter() - tic), rank)
    
    print("Time to init bcast and share DM {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

    if rank == 0:
        print("Time INIT {:>7.2f} (s)".format(time.perf_counter() - t_INIT))


    dmOld = None
    mu0 = -5.5
    for gsc in range(sdc.numAdaptIter):
        if rank == 0: print('\n\n|||| Adaptive iter:', gsc, '||||')
        #print_memory_usage(rank, node_rank, "Memory usage")
        TIC_iter = time.perf_counter()
        tic = time.perf_counter()
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)
        if rank == 0:print("Time to  bcast DM_cpu_np {:>7.2f} (s)".format(time.perf_counter() - tic), rank)
        # Partition the graph
        tic = time.perf_counter()
        if gsc > 0:

            if node_rank == 0:
                tic = time.perf_counter()
                partsCoreHalo = []
                if rank == 0:print("\nCore and halos indices for every part:")
                for i in range(sdc.nparts):
                    coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
                    partsCoreHalo.append(coreHalo)
                    if sdc.verb and rank == 0: print("coreHalo for part", i, "=", coreHalo)
                    if rank == 0: print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')
                    
                if rank == 0: print("Time to compute halos {:>7.2f} (s)".format(time.perf_counter() - tic))

                tic = time.perf_counter()
                new_graph_for_pairs = np.array(fullGraph.copy())
                for i in range(sy.nats):
                    for sublist_idx in range(sdc.nparts):
                        if i in parts[sublist_idx]:
                            new_graph_for_pairs[i, 0] = len(partsCoreHalo[sublist_idx])
                            new_graph_for_pairs[i, 1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
                            break

                if rank == 0: print("Time to updt DM and mod graphs {:>7.2f} (s)".format(time.perf_counter() - tic))
                tic = time.perf_counter()
                #### THIS IS BAD. NEEDS TO BE FIXEd $$$
                P_contr_new = torch.zeros_like(P_contr, device=device)
                for i in range(sy.nats):
                    tmp1 = graph_for_pairs[i][1:graph_for_pairs[i][0]+1]
                    tmp2 = new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1]
                    pos = np.searchsorted(tmp1, tmp2)
                    # Ensure the indices are within bounds
                    pos = np.clip(pos, a_min=0, a_max=len(tmp1) - 1)
                    # Check if the positions are valid and match
                    mask_isin_n_in_o = (pos < len(tmp1)) & (tmp1[pos] == tmp2)
                    #print('isin',(np.isin(tmp2, tmp1) == mask_isin_n_in_o).all())

                    pos = np.searchsorted(tmp2, tmp1)
                    # Ensure the indices are within bounds
                    #pos = np.clip(pos, max=len(tmp2) - 1)
                    # Check if the positions are valid and match
                    mask_isin_o_in_n = (pos < len(tmp2)) & (tmp2[pos] == tmp1)
                    #print('PC', (np.isin(tmp1, tmp2) == mask_isin_o_in_n).all())

                    P_contr_new[:,i][  :new_graph_for_pairs[i][0]  ][   mask_isin_n_in_o   ] = \
                        P_contr[:,i][:graph_for_pairs[i][0]][   mask_isin_o_in_n   ] 
                P_contr[:] = P_contr_new[:]
                del P_contr_new

                if rank == 0: print("Time to updt DM and mod graphs {:>7.2f} (s)".format(time.perf_counter() - tic))
                tic = time.perf_counter()
                graph_for_pairs = new_graph_for_pairs
                # Initialize an array to hold graph_maskd values
                graph_maskd = []
                # Track the position counter across rows in a vectorized way
                counter = 0
                for j in range(sy.nats):
                    # Get neighbors for node j from graph_for_pairs
                    neighbors = graph_for_pairs[j][1:graph_for_pairs[j][0] + 1]
                    # Find positions where `i == j` (self-loops) in the neighbors list
                    mask = np.where(neighbors == j)[0]
                    # Calculate the absolute position for masked values and store them
                    graph_maskd.extend(counter + mask)
                    # Update the counter for the next row, adding the degree difference
                    counter += len(neighbors) + int(sdc.maxDeg - graph_for_pairs[j][0])
                # Convert graph_maskd to a NumPy array
                graph_maskd = np.array(graph_maskd)

                if rank == 0: print("Time to updt DM and mod graphs {:>7.2f} (s)".format(time.perf_counter() - tic))
            else:
                coreHalo = None
                partsCoreHalo = None
                graph_for_pairs = None
                #new_graph_for_pairs = None
                graph_maskd = None

            tic = time.perf_counter()
            if mpiOnDebugFlag:
                coreHalo = node_comm.bcast(coreHalo, root=0)
                partsCoreHalo = node_comm.bcast(partsCoreHalo, root=0)
                graph_for_pairs = node_comm.bcast(graph_for_pairs, root=0)
                #new_graph_for_pairs = node_comm.bcast(new_graph_for_pairs, root=0)
                graph_maskd = node_comm.bcast(graph_maskd, root=0)
            if node_rank == 0: print("Time to bcast DM and mod graphs {:>7.2f} (s)".format(time.perf_counter() - tic), rank)

            
        tic = time.perf_counter()
        # for efficiency, the PySEQM dm needs to be reshaped in 4x4 blocks.
        if eng.interface == "PySEQM":
            with torch.no_grad():
                if eng.reconstruct_dm:
                    eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                    get_singlePoint(sdc, eng, rank, node_rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molecule_whole,
                                    dm.reshape((molecule_whole.nmol, molecule_whole.molsize,4, molecule_whole.molsize,4)) \
                                    .transpose(2,3).reshape(molecule_whole.nmol*molecule_whole.molsize*molecule_whole.molsize,4,4), P_contr, graph_for_pairs, graph_maskd)
                else:
                    if node_rank < num_gpus:
                        eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                        get_singlePoint(sdc, eng, partsPerGPU, partsPerNode, node_id, node_rank, rank, numranks, comm, gpu_global_comm, parts, partsCoreHalo, sy, hindex, mu0, molecule_whole,
                                        None, P_contr, graph_for_pairs, graph_maskd)
                    else:
                        eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = None, None, None, None, None, None, None, None

            if mpiOnDebugFlag: comm.Barrier()
        else:
            eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                get_singlePoint(sdc, eng, rank, node_rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molecule_whole, dm)
        

        if rank == 0: print("Time to get_singlePoint {:>7.2f} (s)".format(time.perf_counter() - tic))

        if sdc.restartLoad:
            sdc.restartLoad = False
            if node_rank == 0:
                P_contr[:] = torch.load('P_contr.pt')
            with open('parts.pkl','rb') as f:
                parts = pickle.load(f)
            with open('partsCoreHalo.pkl','rb') as f:
                partsCoreHalo = pickle.load(f)
            with open('fullGraph.pkl','rb') as f:
                fullGraph = pickle.load(f)
            mu0 = np.load('mu0.npy')
            graph_for_pairs = np.load('graph_for_pairs.npy')
            graph_maskd = np.load('graph_maskd.npy')
            if rank == 0:
                eValOnRank_list = torch.load('eValOnRank_list.pt')
                Q_list = torch.load('Q_list.pt')
                NH_Nh_Hs_list = torch.load('NH_Nh_Hs_list.pt')
                I_list = torch.load('I_list.pt')
                I_halo_list = torch.load('I_halo_list.pt')
                core_indices_in_sub_expanded_list = torch.load('core_indices_in_sub_expanded_list.pt')
                Nocc_list = torch.load('Nocc_list.pt')


        if rank == 0 and sdc.restartSave:
            torch.save(eValOnRank_list, 'eValOnRank_list.pt')
            torch.save(Q_list, 'Q_list.pt')
            torch.save(NH_Nh_Hs_list, 'NH_Nh_Hs_list.pt')
            torch.save(I_list, 'I_list.pt')
            torch.save(I_halo_list, 'I_halo_list.pt')
            torch.save(core_indices_in_sub_expanded_list, 'core_indices_in_sub_expanded_list.pt')
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

        
        #if rank == 0:
        if rank < node_numranks:
            tic = time.perf_counter()
            eValOnRank_list = node_comm.bcast(eValOnRank_list, root=0)
            Q_list = node_comm.bcast(Q_list, root=0)
            NH_Nh_Hs_list = node_comm.bcast(NH_Nh_Hs_list, root=0)
            I_list = node_comm.bcast(I_list, root=0)
            I_halo_list = node_comm.bcast(I_halo_list, root=0)
            core_indices_in_sub_expanded_list = node_comm.bcast(core_indices_in_sub_expanded_list, root=0)
            Nocc_list = node_comm.bcast(Nocc_list, root=0)
            mu0 = node_comm.bcast(mu0, root=0)

            

                  

            with torch.no_grad():
                if eng.reconstruct_dm:
                    fullGraphRho = get_singlePointDM(sdc, eng, rank, 1, comm, parts, partsCoreHalo, sy, hindex, mu0, dm, P_contr, graph_for_pairs,
                                         eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list)
                else:
                    fullGraphRho = get_singlePointDM(sdc, eng, rank, node_numranks, node_comm, parts, partsCoreHalo, sy, hindex, mu0, None, P_contr, graph_for_pairs,
                                         eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list)
            
            
            ########### GATHER fullGraphRho ###########

            #np.save( 'graphs/fullGraphRho_{}'.format(rank), fullGraphRho,)

            if rank == 0: print("Time to updt DM {:>7.2f} (s)".format(time.perf_counter() - tic))
            
            node_comm.Barrier()

            tic = time.perf_counter()
            fullGraphRho_LIST = node_comm.gather(fullGraphRho, root=0)
            if rank == 0:

                # fullGraphRho_LIST = []
                # for i in range(node_numranks):
                #     fullGraphRho_LIST.append(np.load('graphs/fullGraphRho_{}.npy'.format(i)))

                fullGraphRho_LIST.append(fullGraph)
                fullGraph = add_mult_graphs(fullGraphRho_LIST)
                print("Time to add graphs {:>7.2f} (s)".format(time.perf_counter() - tic))
            #fullGraph = add_graphs(fullGraph, fullGraphRho, )
            del fullGraphRho
            
            
            if eng.reconstruct_dm:
                trace = get_dmTrace(eng, dm)
                print("DM TRACE: {:>10.7f}".format(trace))
            if rank == 0:
                tic = time.perf_counter()
                trace = torch.sum(P_contr.transpose(0,1).reshape(molecule_whole.molsize*(len(graph_for_pairs[0])-1), 4,4)[graph_maskd].diagonal(dim1=-2, dim2=-1))
                print("DM TRACE: {:>10.7f}".format(trace))
                print("Time to get trace {:>7.2f} (s)".format(time.perf_counter() - tic))

        else:
            fullGraph = None
            
        if mpiOnDebugFlag:
            tic = time.perf_counter()
            #comm.Barrier()
            fullGraph = comm.bcast(fullGraph, root=0)
            if rank == 0: print("Time to bcast fullGraph {:>7.2f} (s)".format(time.perf_counter() - tic))

        del eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, Nocc_list
        torch.cuda.empty_cache()

        # Function to calculate tensor size in megabytes (MB)
        # if rank == 0:
            # # Sort tensors by size and print them
            # tensors = list(get_tensors())
            # tensors.sort(key=lambda x: tensor_size(x), reverse=True)
            # print("Top memory-consuming tensors:")
            # for tensor in tensors:
                # if tensor_size(tensor) > 0.1:
                    # print(f"Tensor size: {tensor_size(tensor):.2f} MB | Shape: {tensor.shape} | Dtype: {tensor.dtype}")

        if rank == 0: print("t Iter {:>8.2f} (s)".format(time.perf_counter() - TIC_iter))

    ### forces calculation ###
    tic = time.perf_counter()    
    if node_rank < num_gpus:
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)
            forces = np.zeros((sy.coords.shape))
            partsCoreHalo = []
            if rank == 0: print("\nCore and halos indices for every part:")
            for i in range(sdc.nparts):
                coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
                partsCoreHalo.append(coreHalo)
                if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
                if rank == 0: print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')

            new_graph_for_pairs = np.array(fullGraph.copy())
            for i in range(sy.nats):
                for sublist_idx in range(sdc.nparts):
                    if i in parts[sublist_idx]:
                        new_graph_for_pairs[i, 0] = len(partsCoreHalo[sublist_idx])
                        new_graph_for_pairs[i, 1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
                        break

            #### THIS IS BAD. NEEDS TO BE FIXEd $$$
            P_contr_new = torch.zeros_like(P_contr, device=device)
            for i in range(len(new_graph_for_pairs)):
                P_contr_new[:,i][  :new_graph_for_pairs[i][0]  ][   np.isin(new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1], graph_for_pairs[i][1:graph_for_pairs[i][0]+1])   ] = \
                    P_contr[:,i][:graph_for_pairs[i][0]][   np.isin(graph_for_pairs[i][1:graph_for_pairs[i][0]+1], new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1])   ]
            P_contr[:] = P_contr_new[:]
            del P_contr_new

            graph_for_pairs = new_graph_for_pairs
            graph_maskd = []
            counter = 0
            for j in range(len(graph_for_pairs)):
                sub_counter = 0
                for i in graph_for_pairs[j][1:graph_for_pairs[j][0]+1]: 
                    if i==j:
                        graph_maskd.append(counter)
                    counter +=1
                    sub_counter += 1 
                counter += int(sdc.maxDeg - graph_for_pairs[j][0])
        else:
            forces = None
            partsCoreHalo = None
            new_graph_for_pairs = None
            graph_for_pairs = None
            graph_maskd = None

        if sdc.fDevice == 'cuda':
            device = 'cuda:{}'.format(node_rank)
        else:
            device = 'cpu'

        molSysData = pyseqmObjects(sdc, sy.coords, sy.symbols, sy.types, do_large_tensors = sdc.use_pyseqm_lt, device=device) #object with whatever initial parameters and tensors
        #molSysData.molecule_whole.coordinates.requires_grad_(True)
        
        if mpiOnDebugFlag:
            forces = gpu_comm.bcast(forces, root=0)
            print('HERE1')
            partsCoreHalo = gpu_comm.bcast(partsCoreHalo, root=0)
            gpu_comm.Barrier()
            graph_for_pairs = gpu_comm.bcast(graph_for_pairs, root=0)
            new_graph_for_pairs = gpu_comm.bcast(new_graph_for_pairs, root=0)
            graph_maskd = gpu_comm.bcast(graph_maskd, root=0)

            if rank == 0:
                forces[:] = .0
            gpu_comm.Barrier()
        else:
            forces = np.zeros((sy.coords.shape))
        
        if rank == 0: print("Time init forces {:>8.2f} (s)".format(time.perf_counter() - tic))

        tic = time.perf_counter()
        if eng.interface == "PySEQM":
            if eng.reconstruct_dm:
                eElec = get_singlePointForces(sdc, eng, partsPerGPU, partsPerNode, node_id, node_rank, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
                                dm.reshape((molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize,4, molSysData.molecule_whole.molsize,4)) \
                                .transpose(2,3).reshape(molSysData.molecule_whole.nmol*molSysData.molecule_whole.molsize*molSysData.molecule_whole.molsize,4,4),P_contr, graph_for_pairs, graph_maskd)
            else:
                if sdc.doForces:
                    eElec = get_singlePointForces(sdc, eng, partsPerGPU, partsPerNode, node_id, node_rank, rank, num_gpus, gpu_comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
                                None, P_contr.to(device), graph_for_pairs, graph_maskd)
                else:
                    with torch.no_grad():
                        eElec = get_singlePointForces(sdc, eng, partsPerGPU, partsPerNode, node_id, node_rank, rank, num_gpus, gpu_comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
                                None, P_contr.to(device), graph_for_pairs, graph_maskd)

            if mpiOnDebugFlag:
                global_Eelec = np.zeros(1, dtype=np.float64)
                gpu_comm.Barrier()
                gpu_comm.Allreduce(MPI.IN_PLACE, forces, op=MPI.SUM)
                #eElec_LIST = gpu_comm.gather(eElec, root=0)
                #comm.Allreduce(eElec, global_Eelec, op=MPI.SUM)
                gpu_comm.Allreduce(eElec, global_Eelec, op=MPI.SUM) #primary_comm
            else:
                eElec_LIST = eElec
        else:
            get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData, dm)
        if rank == 0: print("Time to get electron forces {:>8.2f} (s)".format(time.perf_counter() - tic))
        
        if rank == 0:
            del molSysData
            molSysData = pyseqmObjects(sdc, sy.coords, sy.symbols, sy.types, do_large_tensors = True, device=device) #object with whatever initial parameters and tensors

            if mpiOnDebugFlag:
                print("eElec:   {:>10.12f}".format(global_Eelec[0]),)
            else:
                print("eElec:   {:>10.12f}".format(eElec[0]),)
            
            tic = time.perf_counter()
            eNucAB = get_eNuc(eng, molSysData)
            eTot, eNuc = get_eTot(eng, molSysData, eNucAB, 0)
            print("Enuc:   {:>10.12f}".format(eNuc),)
            L = eNuc.sum()
            L.backward()
            forceNuc = -molSysData.molecule_whole.coordinates.grad.detach()
            molSysData.molecule_whole.coordinates.grad.zero_()
            print("Time to get nuclear forces {:>8.2f} (s)".format(time.perf_counter() - tic))
            #print(forceNuc)
            np.save('forces_test.np', (forces+forceNuc.cpu().numpy()[0]), )
            #np.save('forces_test.np', (forces), )
