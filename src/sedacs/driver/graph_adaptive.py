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
from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.hamiltonian import get_hamiltonian
from sedacs.mpi import collect_and_sum_matrices
from sedacs.system import System, extract_subsystem
from sedacs.evals import get_eVals
from sedacs.chemical_potential import get_mu
from sedacs.graph import get_initial_graph
from sedacs.overlap import get_overlap
from sedacs.interface_pyseqm import get_coreHalo_ham_inds, get_diag_guess_pyseqm, ParamContainer
import itertools
import sys
import psutil
import pickle


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
def get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0,
                    molSysData, P, P_contr, graph_for_pairs, graph_maskd):
    # computing DM for core+halo part
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank

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
        partFileName = "subSy" + str(rank) + "_" + str(partIndex) + ".pdb"
        write_pdb_coordinates(partFileName, subSy.coords, subSy.types, subSy.symbols)
        write_xyz_coordinates("subSy" + str(rank) + "_" + str(partIndex) + ".xyz", subSy.coords, subSy.types, subSy.symbols)

        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords,subSyCore.types = extract_subsystem(sy.coords,sy.types,sy.symbols,parts[partIndex])
        partCoreFileName = "CoreSubSy"+str(rank)+"_"+str(partIndex)+".pdb"
        write_pdb_coordinates(partCoreFileName,subSyCore.coords,subSyCore.types,subSyCore.symbols)
        write_xyz_coordinates("CoreSubSy"+str(rank)+"_"+str(partIndex)+".xyz",subSyCore.coords,subSyCore.types,subSyCore.symbols)

        ham = get_hamiltonian(eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], molSysData, P, P_contr, graph_for_pairs, graph_maskd, None,
                              verbose=False)
        print("TOT {:>8.3f} (s)".format(time.perf_counter() - tic))

        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals

        tic = time.perf_counter()
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs, I, I_halo, core_indices_in_sub_expanded = get_eVals(eng, sdc, sy, occ, ham, subSy.coords, subSy.symbols, subSy.types, Tel, mu0,
                        coreSize, subSy, subSyCore, parts[partIndex], partsCoreHalo[partIndex],
                        verbose=False)

        del ham
        
        dValOnRank = np.append(dValOnRank, dVals)
        eValOnRank = np.append(eValOnRank, eVals.detach().cpu().numpy())

        eValOnRank_list.append(eVals)
        Q_list.append(Q)
        I_list.append(I)
        I_halo_list.append(I_halo)
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        print("| t eVals/dVals {:>9.4f} (s)".format(time.perf_counter() - tic))

    full_dVals = None
    full_eVals = None
    eValOnRank_size = np.array(len(eValOnRank), dtype=int)

    eValOnRank_SIZES = None
    if mpiOnDebugFlag:
        if rank == 0:
            eValOnRank_SIZES = np.empty(comm.Get_size(), dtype=int)
            
        comm.Gather(eValOnRank_size, eValOnRank_SIZES, root=0)

        comm.Barrier()
        if rank == 0:
            full_dVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)
            full_eVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)
        comm.Gatherv(dValOnRank, [full_dVals, eValOnRank_SIZES], root=0)
        comm.Gatherv(eValOnRank, [full_eVals, eValOnRank_SIZES], root=0)
        eVal_LIST = comm.gather(eValOnRank_list, root=0)
        Q_LIST = comm.gather(Q_list, root=0)
        NH_Nh_Hs_LIST = comm.gather(NH_Nh_Hs_list, root=0)
        I_LIST = comm.gather(I_list, root=0)
        I_halo_LIST = comm.gather(I_halo_list, root=0)
        core_indices_in_sub_expanded_LIST = comm.gather(core_indices_in_sub_expanded_list, root=0)
        Nocc_LIST = comm.gather(Nocc_list, root=0)

        if rank == 0:
        #     # Flatten the nested list of lists into a single list of tensors
            eVal_LIST = list(itertools.chain(*eVal_LIST))
            Q_LIST = list(itertools.chain(*Q_LIST))
            NH_Nh_Hs_LIST = list(itertools.chain(*NH_Nh_Hs_LIST))
            I_LIST = list(itertools.chain(*I_LIST))
            I_halo_LIST = list(itertools.chain(*I_halo_LIST))
            core_indices_in_sub_expanded_LIST = list(itertools.chain(*core_indices_in_sub_expanded_LIST))
            Nocc_LIST = list(itertools.chain(*Nocc_LIST))

    else:
        full_dVals = dValOnRank
        full_eVals = eValOnRank
        eVal_LIST = eValOnRank_list
        Q_LIST = Q_list
        NH_Nh_Hs_LIST = NH_Nh_Hs_list
        I_LIST = I_list
        I_halo_LIST = I_halo_list
        core_indices_in_sub_expanded_LIST = core_indices_in_sub_expanded_list
        Nocc_LIST = Nocc_list


    
    if rank == 0:
        mu0 = get_mu(mu0, full_dVals, full_eVals, Tel, sy.numel/2)

    return eVal_LIST, Q_LIST, NH_Nh_Hs_LIST, I_LIST, I_halo_LIST, core_indices_in_sub_expanded_LIST, Nocc_LIST, mu0

def get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData, P, P_contr, graph_for_pairs, graph_maskd):
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    EELEC = torch.tensor([0.0], dtype = eng.torch_dt)
    for partIndex in range(partIndex1, partIndex2):
        print("Rank, part", rank, partIndex)
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])

        _, core_indices_in_sub_expanded, _, _, _ = \
            get_coreHalo_ham_inds(parts[partIndex], partsCoreHalo[partIndex], sdc, sy, subSy)

        tic = time.perf_counter()        
        f, eElec = get_hamiltonian(eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], molSysData, P, P_contr, graph_for_pairs, graph_maskd, core_indices_in_sub_expanded, doForces = True,
                              verbose=False)
        # if mpiOnDebugFlag:
        #     comm.Allreduce(f, forces, op=MPI.SUM)
        # else:
        #     forces += f

        forces += f
        EELEC += eElec
        print("EelecCH {:>7.3f} |".format(eElec.item()), end=" ")
        del eElec, subSy, f
        print("TOT", time.perf_counter() - tic, "(s)")
    print("eElec_SUM: {:>10.7f}".format(EELEC[0]),)    
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
        #print(partIndex)
        tic = time.perf_counter()
        # this will calculate the DM in subsys and update the whole DM
        rho_ren, maxDif, sumDif = get_density_matrix_renorm(eng, Tel, mu0, dm, P_contr, graph_for_pairs,
                                            eValOnRank_list[partIndex], Q_list[partIndex], NH_Nh_Hs_list[partIndex], I_list[partIndex], core_indices_in_sub_expanded_list[partIndex], Nocc_list[partIndex])
        
        indices_in_sub = np.linspace(0,len(partsCoreHalo[partIndex])-1, len(partsCoreHalo[partIndex]), dtype = eng.np_int_dt)
        core_indices_in_sub = indices_in_sub[np.isin(partsCoreHalo[partIndex], parts[partIndex])]
        
        alpha = 0.6
        P_contr_maxDif = []
        P_contr_sumDif = 0
        for i in range(len(parts[partIndex])):
            #print(P_contr[:,i][:graph_for_pairs[parts[partIndex][i]][0]].shape)
            P_contr_maxDif.append(torch.max(torch.abs(P_contr[:,parts[partIndex][i]][:graph_for_pairs[parts[partIndex][i]][0]] - \
                                rho_ren.reshape((1, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4)) \
                                .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]), (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),4,4)[core_indices_in_sub[i]].transpose(1,2))).cpu().numpy()
)
            P_contr_sumDif += torch.sum(torch.abs(P_contr[:,parts[partIndex][i]][:graph_for_pairs[parts[partIndex][i]][0]] - \
                                rho_ren.reshape((1, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4)) \
                                .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]), (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),4,4)[core_indices_in_sub[i]].transpose(1,2))).cpu().numpy()
            
            P_contr[:,parts[partIndex][i]][:graph_for_pairs[parts[partIndex][i]][0]] = (1-alpha)*P_contr[:,parts[partIndex][i]][:graph_for_pairs[parts[partIndex][i]][0]] + \
            alpha*rho_ren.reshape((1, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4, NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1],4)) \
                                .transpose(2,3).reshape((NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]), (NH_Nh_Hs_list[partIndex][0]+NH_Nh_Hs_list[partIndex][1]),4,4)[core_indices_in_sub[i]].transpose(1,2)
            

        rho_ren = pack(rho_ren, NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])

        P_contr_maxDif = max(P_contr_maxDif)
        P_contr_maxDifList.append(P_contr_maxDif)
        P_contr_sumDifTot += P_contr_sumDif
        print(" MAX |\u0394DM_ij|: {:>10.7f}".format(P_contr_maxDif))
        print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(P_contr_sumDif))

        maxDifList.append(maxDif)
        try:
            sumDifTot += sumDif
        except:
            sumDifTot += 0
        graphOnRank = collect_graph_from_rho(graphOnRank, rho_ren.cpu(), sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False)
        # graphOnRank = collect_graph_from_rho(graphOnRank,
        #                                      pack(dm[:,I_halo_list[partIndex][0], I_halo_list[partIndex][1]], NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])[0],
        #                                      sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False)
        del rho_ren
        print("t DM {:>8.3f} (s)".format(time.perf_counter() - tic))

    print('HERE_DM_1')
    if eng.reconstruct_dm:
        print(" MAX |\u0394DM_ij|: {:>10.7f} at SubSy {:>5d}".format(max(maxDifList), np.argmax(maxDifList)))
        print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(sumDifTot))

    print(" MAX |\u0394DM_ij|: {:>10.7f} at SubSy {:>5d}".format(max(P_contr_maxDifList), np.argmax(P_contr_maxDifList)))
    print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(P_contr_sumDifTot))



    if is_mpi_available:
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        # dValsFull = collect_dValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???
        # eValsFull = collect_eValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???
        comm.Barrier()
        return fullGraphRho
    else:
        #fullGraphRho = graphOnRank
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
    eng.use_pyseqm_lt = False
    sdc.use_pyseqm_lt = eng.use_pyseqm_lt

    eng.reconstruct_dm = False
    sdc.reconstruct_dm = eng.reconstruct_dm


    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()  # Rank within the node

    primary_rank = None
    if node_rank == 0:
        primary_rank = rank  # Global rank of the primary rank on each node

    # Gather the primary ranks from each node
    primary_ranks = comm.allgather(primary_rank)
    primary_ranks = [r for r in primary_ranks if r is not None]  # Filter out None values


    color = 0 if rank in primary_ranks else MPI.UNDEFINED
    primary_comm = comm.Split(color=color, key=rank)
    comm.Barrier()

    device = 'cpu'

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

    njumps = 1

    tic = time.perf_counter()
    fullGraph = graphNL.copy()

    with torch.no_grad(): molSysData = get_molSysData(eng, sdc, sy.coords, sy.symbols, sy.types, do_large_tensors = sdc.use_pyseqm_lt, device=device) #object with whatever initial parameters and tensors
    
    # if rank == 0:

    #     with torch.no_grad(): molSysData = get_molSysData(eng, sdc, sy.coords, sy.symbols, sy.types, device=device) #object with whatever initial parameters and tensors
        
    #     idxi_size = molSysData.molecule_whole.idxi.size()
    #     idxi_nbytes = molSysData.molecule_whole.idxi.size().numel() * molSysData.molecule_whole.idxi.element_size()

    #     idxj_size = idxi_size
    #     idxj_nbytes = idxi_nbytes

    #     rij_size = molSysData.molecule_whole.rij.size()
    #     rij_nbytes = molSysData.molecule_whole.rij.size().numel() * molSysData.molecule_whole.rij.element_size()

    #     xij_size = molSysData.molecule_whole.xij.size()
    #     xij_nbytes = molSysData.molecule_whole.xij.size().numel() * molSysData.molecule_whole.xij.element_size()

    #     ni_size = molSysData.molecule_whole.ni.size()
    #     ni_nbytes = molSysData.molecule_whole.ni.size().numel() * molSysData.molecule_whole.ni.element_size()

    #     nj_size = molSysData.molecule_whole.nj.size()
    #     nj_nbytes = molSysData.molecule_whole.nj.size().numel() * molSysData.molecule_whole.nj.element_size()

    #     mask_size = molSysData.molecule_whole.mask.size()
    #     mask_nbytes = molSysData.molecule_whole.mask.size().numel() * molSysData.molecule_whole.mask.element_size()

    #     mask_l_size = molSysData.molecule_whole.mask_l.size()
    #     mask_l_nbytes = molSysData.molecule_whole.mask_l.size().numel() * molSysData.molecule_whole.mask_l.element_size()

    #     pair_molid_size = molSysData.molecule_whole.pair_molid.size()
    #     pair_molid_nbytes = molSysData.molecule_whole.pair_molid.size().numel() * molSysData.molecule_whole.pair_molid.element_size()

    #     ###
    #     w_ssss_size = molSysData.w_ssss.size()
    #     w_ssss_nbytes = molSysData.w_ssss.size().numel() * molSysData.w_ssss.element_size()

    #     rho0xi_whole_size = molSysData.rho0xi_whole.size()
    #     rho0xi_whole_nbytes = molSysData.rho0xi_whole.size().numel() * molSysData.rho0xi_whole.element_size()

    #     rho0xj_whole_size = molSysData.rho0xj_whole.size()
    #     rho0xj_whole_nbytes = molSysData.rho0xj_whole.size().numel() * molSysData.rho0xj_whole.element_size()
    # else:
    #     molSysData = ParamContainer()

    #     molSysData.molecule_whole = ParamContainer()

    #     molSysData.molecule_whole.idxi = None
    #     idxi_size = None
    #     idxi_nbytes = 0

    #     molSysData.molecule_whole.idxj = None
    #     idxj_size = None
    #     idxj_nbytes = 0

    #     molSysData.molecule_whole.rij = None
    #     rij_size = None
    #     rij_nbytes = 0

    #     molSysData.molecule_whole.xij = None
    #     xij_size = None
    #     xij_nbytes = 0

    #     molSysData.molecule_whole.ni = None
    #     ni_size = None
    #     ni_nbytes = 0

    #     molSysData.molecule_whole.nj = None
    #     nj_size = None
    #     nj_nbytes = 0

    #     molSysData.molecule_whole.mask = None
    #     mask_size = None
    #     mask_nbytes = 0

    #     molSysData.molecule_whole.mask_l = None
    #     mask_l_size = None
    #     mask_l_nbytes = 0

    #     molSysData.molecule_whole.pair_molid = None
    #     pair_molid_size = None
    #     pair_molid_nbytes = 0

    #     ###
    #     molSysData.w_ssss = None
    #     w_ssss_size = None
    #     w_ssss_nbytes = 0

    #     molSysData.rho0xi_whole = None
    #     rho0xi_whole_size = None
    #     rho0xi_whole_nbytes = 0

    #     molSysData.rho0xj_whole = None
    #     rho0xj_whole_size = None
    #     rho0xj_whole_nbytes = 0

    #     ###
    #     molSysData.molecule_whole.const = None
    #     molSysData.molecule_whole.Z = None
    #     molSysData.molecule_whole.alp = None
    #     molSysData.molecule_whole.chi = None
    #     molSysData.molecule_whole.atom_molid = None
    #     molSysData.molecule_whole.coordinates = None
    #     molSysData.molecule_whole.species = None
    #     molSysData.molecule_whole.maskd = None
    #     molSysData.molecule_whole.mass = None
    #     molSysData.molecule_whole.method = None
    #     molSysData.molecule_whole.molsize = None
    #     molSysData.molecule_whole.mult = None
    #     molSysData.molecule_whole.nHeavy = None
    #     molSysData.molecule_whole.nHydro = None
    #     molSysData.molecule_whole.nSuperHeavy = None
    #     molSysData.molecule_whole.nmol = None
    #     molSysData.molecule_whole.nocc = None
    #     molSysData.molecule_whole.parameters = None
    #     molSysData.molecule_whole.seqm_parameters = None
    #     molSysData.molecule_whole.tot_charge = None

    # if mpiOnDebugFlag:
    #     molSysData.molecule_whole.const= comm.bcast(molSysData.molecule_whole.const, root=0)
    #     molSysData.molecule_whole.Z = comm.bcast(molSysData.molecule_whole.Z, root=0)
    #     molSysData.molecule_whole.alp = comm.bcast(molSysData.molecule_whole.alp, root=0)
    #     molSysData.molecule_whole.chi = comm.bcast(molSysData.molecule_whole.chi, root=0)
    #     molSysData.molecule_whole.atom_molid = comm.bcast(molSysData.molecule_whole.atom_molid, root=0)
    #     molSysData.molecule_whole.coordinates = comm.bcast(molSysData.molecule_whole.coordinates, root=0)
    #     molSysData.molecule_whole.species = comm.bcast(molSysData.molecule_whole.species, root=0)
    #     molSysData.molecule_whole.maskd = comm.bcast(molSysData.molecule_whole.maskd, root=0)
    #     molSysData.molecule_whole.mass = comm.bcast(molSysData.molecule_whole.mass, root=0)
    #     molSysData.molecule_whole.method = comm.bcast(molSysData.molecule_whole.method, root=0)
    #     molSysData.molecule_whole.molsize = comm.bcast(molSysData.molecule_whole.molsize, root=0)
    #     molSysData.molecule_whole.mult = comm.bcast(molSysData.molecule_whole.mult, root=0)
    #     molSysData.molecule_whole.nHeavy = comm.bcast(molSysData.molecule_whole.nHeavy, root=0)
    #     molSysData.molecule_whole.nHydro = comm.bcast(molSysData.molecule_whole.nHydro, root=0)
    #     molSysData.molecule_whole.nSuperHeavy = comm.bcast(molSysData.molecule_whole.nSuperHeavy, root=0)
    #     molSysData.molecule_whole.nmol = comm.bcast(molSysData.molecule_whole.nmol, root=0)
    #     molSysData.molecule_whole.nocc = comm.bcast(molSysData.molecule_whole.nocc, root=0)
    #     molSysData.molecule_whole.parameters = comm.bcast(molSysData.molecule_whole.parameters, root=0)
    #     molSysData.molecule_whole.seqm_parameters = comm.bcast(molSysData.molecule_whole.seqm_parameters, root=0)
    #     molSysData.molecule_whole.tot_charge = comm.bcast(molSysData.molecule_whole.tot_charge, root=0)


    
    #     comm.Barrier()
    #     if rank in primary_ranks:
    #         molSysData.molecule_whole.idxi = primary_comm.bcast(molSysData.molecule_whole.idxi, root=0)
    #         molSysData.molecule_whole.idxj = primary_comm.bcast(molSysData.molecule_whole.idxj, root=0)
    #         molSysData.molecule_whole.rij = primary_comm.bcast(molSysData.molecule_whole.rij, root=0)
    #         molSysData.molecule_whole.xij = primary_comm.bcast(molSysData.molecule_whole.xij, root=0)
    #         molSysData.molecule_whole.ni = primary_comm.bcast(molSysData.molecule_whole.ni, root=0)
    #         molSysData.molecule_whole.nj = primary_comm.bcast(molSysData.molecule_whole.nj, root=0)
    #         molSysData.molecule_whole.mask = primary_comm.bcast(molSysData.molecule_whole.mask, root=0)
    #         molSysData.molecule_whole.mask_l = primary_comm.bcast(molSysData.molecule_whole.mask_l, root=0)
    #         molSysData.molecule_whole.pair_molid = primary_comm.bcast(molSysData.molecule_whole.pair_molid, root=0)
    #         ###
    #         molSysData.w_ssss = primary_comm.bcast(molSysData.w_ssss, root=0)
    #         molSysData.rho0xi_whole = primary_comm.bcast(molSysData.rho0xi_whole, root=0)
    #         molSysData.rho0xj_whole = primary_comm.bcast(molSysData.rho0xj_whole, root=0)

    #         idxi_nbytes = primary_comm.bcast(idxi_nbytes, root=0)
    #         idxj_nbytes = primary_comm.bcast(idxj_nbytes, root=0)
    #         rij_nbytes = primary_comm.bcast(rij_nbytes, root=0)
    #         xij_nbytes = primary_comm.bcast(xij_nbytes, root=0)
    #         ni_nbytes = primary_comm.bcast(ni_nbytes, root=0)
    #         nj_nbytes = primary_comm.bcast(nj_nbytes, root=0)
    #         mask_nbytes = primary_comm.bcast(mask_nbytes, root=0)
    #         mask_l_nbytes = primary_comm.bcast(mask_l_nbytes, root=0)
    #         pair_molid_nbytes = primary_comm.bcast(pair_molid_nbytes, root=0)
    #         ###
    #         w_ssss_nbytes = primary_comm.bcast(w_ssss_nbytes, root=0)
    #         rho0xi_whole_nbytes = primary_comm.bcast(rho0xi_whole_nbytes, root=0)
    #         rho0xj_whole_nbytes = primary_comm.bcast(rho0xj_whole_nbytes, root=0)

    #     idxi_size = comm.bcast(idxi_size, root=0)
    #     idxj_size = comm.bcast(idxj_size, root=0)
    #     rij_size = comm.bcast(rij_size, root=0)
    #     xij_size = comm.bcast(xij_size, root=0)
    #     ni_size = comm.bcast(ni_size, root=0)
    #     nj_size = comm.bcast(nj_size, root=0)
    #     mask_size = comm.bcast(mask_size, root=0)
    #     mask_l_size = comm.bcast(mask_l_size, root=0)
    #     pair_molid_size = comm.bcast(pair_molid_size, root=0)
    #     ###
    #     w_ssss_size = comm.bcast(w_ssss_size, root=0)
    #     rho0xi_whole_size = comm.bcast(rho0xi_whole_size, root=0)
    #     rho0xj_whole_size = comm.bcast(rho0xj_whole_size, root=0)


    #     idxi_win = MPI.Win.Allocate_shared(idxi_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     idxi_buf, idxi_itemsize = idxi_win.Shared_query(0) 
    #     idxi_ary = np.ndarray(buffer=idxi_buf, dtype=eng.np_int_dt, shape=(idxi_size))
    #     if rank == 0:
    #         idxi_ary[:] = molSysData.molecule_whole.idxi.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.idxi
    #     molSysData.molecule_whole.idxi = torch.from_numpy(idxi_ary).to(device)

    #     idxj_win = MPI.Win.Allocate_shared(idxj_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     idxj_buf, idxj_itemsize = idxj_win.Shared_query(0) 
    #     idxj_ary = np.ndarray(buffer=idxj_buf, dtype=eng.np_int_dt, shape=(idxj_size))
    #     if rank == 0:
    #         idxj_ary[:] = molSysData.molecule_whole.idxj.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.idxj
    #     molSysData.molecule_whole.idxj = torch.from_numpy(idxj_ary).to(device)

    #     rij_win = MPI.Win.Allocate_shared(rij_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     rij_buf, rij_itemsize = rij_win.Shared_query(0) 
    #     rij_ary = np.ndarray(buffer=rij_buf, dtype=eng.np_dt, shape=(rij_size))
    #     if rank == 0:
    #         rij_ary[:] = molSysData.molecule_whole.rij.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.rij
    #     molSysData.molecule_whole.rij = torch.from_numpy(rij_ary).to(device)

    #     xij_win = MPI.Win.Allocate_shared(xij_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     xij_buf, xij_itemsize = xij_win.Shared_query(0) 
    #     xij_ary = np.ndarray(buffer=xij_buf, dtype=eng.np_dt, shape=(xij_size))
    #     if rank == 0:
    #         xij_ary[:] = molSysData.molecule_whole.xij.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.xij
    #     molSysData.molecule_whole.xij = torch.from_numpy(xij_ary).to(device)
    #     #xij_win.Free()

    #     ni_win = MPI.Win.Allocate_shared(ni_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     ni_buf, ni_itemsize = ni_win.Shared_query(0) 
    #     ni_ary = np.ndarray(buffer=ni_buf, dtype=eng.np_int_dt, shape=(ni_size))
    #     if rank == 0:
    #         ni_ary[:] = molSysData.molecule_whole.ni.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.ni
    #     molSysData.molecule_whole.ni = torch.from_numpy(ni_ary).to(device)

    #     nj_win = MPI.Win.Allocate_shared(nj_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     nj_buf, nj_itemsize = nj_win.Shared_query(0) 
    #     nj_ary = np.ndarray(buffer=nj_buf, dtype=eng.np_int_dt, shape=(nj_size))
    #     if rank == 0:
    #         nj_ary[:] = molSysData.molecule_whole.nj.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.nj
    #     molSysData.molecule_whole.nj = torch.from_numpy(nj_ary).to(device)

    #     mask_win = MPI.Win.Allocate_shared(mask_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     mask_buf, mask_itemsize = mask_win.Shared_query(0) 
    #     mask_ary = np.ndarray(buffer=mask_buf, dtype=eng.np_int_dt, shape=(mask_size))
    #     if rank == 0:
    #         mask_ary[:] = molSysData.molecule_whole.mask.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.mask
    #     molSysData.molecule_whole.mask = torch.from_numpy(mask_ary).to(device)

    #     mask_l_win = MPI.Win.Allocate_shared(mask_l_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     mask_l_buf, mask_l_itemsize = mask_l_win.Shared_query(0) 
    #     mask_l_ary = np.ndarray(buffer=mask_l_buf, dtype=eng.np_int_dt, shape=(mask_l_size))
    #     if rank == 0:
    #         mask_l_ary[:] = molSysData.molecule_whole.mask_l.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.mask_l
    #     molSysData.molecule_whole.mask_l = torch.from_numpy(mask_l_ary).to(device)

    #     pair_molid_win = MPI.Win.Allocate_shared(pair_molid_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     pair_molid_buf, pair_molid_itemsize = pair_molid_win.Shared_query(0) 
    #     pair_molid_ary = np.ndarray(buffer=pair_molid_buf, dtype=eng.np_int_dt, shape=(pair_molid_size))
    #     if rank == 0:
    #         pair_molid_ary[:] = molSysData.molecule_whole.pair_molid.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.molecule_whole.pair_molid
    #     molSysData.molecule_whole.pair_molid = torch.from_numpy(pair_molid_ary).to(device)

    #     w_ssss_win = MPI.Win.Allocate_shared(w_ssss_nbytes, torch.tensor(0, dtype=eng.torch_int_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     w_ssss_buf, w_ssss_itemsize = w_ssss_win.Shared_query(0) 
    #     w_ssss_ary = np.ndarray(buffer=w_ssss_buf, dtype=eng.np_int_dt, shape=(w_ssss_size))
    #     if rank == 0:
    #         w_ssss_ary[:] = molSysData.w_ssss.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.w_ssss
    #     molSysData.w_ssss = torch.from_numpy(w_ssss_ary).to(device)

    #     rho0xi_whole_win = MPI.Win.Allocate_shared(rho0xi_whole_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     rho0xi_whole_buf, rho0xi_whole_itemsize = rho0xi_whole_win.Shared_query(0) 
    #     rho0xi_whole_ary = np.ndarray(buffer=rho0xi_whole_buf, dtype=eng.np_int_dt, shape=(rho0xi_whole_size))
    #     if rank == 0:
    #         rho0xi_whole_ary[:] = molSysData.rho0xi_whole.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.rho0xi_whole
    #     molSysData.rho0xi_whole = torch.from_numpy(rho0xi_whole_ary).to(device)

    #     rho0xj_whole_win = MPI.Win.Allocate_shared(rho0xj_whole_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
    #     rho0xj_whole_buf, rho0xj_whole_itemsize = rho0xj_whole_win.Shared_query(0) 
    #     rho0xj_whole_ary = np.ndarray(buffer=rho0xj_whole_buf, dtype=eng.np_int_dt, shape=(rho0xj_whole_size))
    #     if rank == 0:
    #         rho0xj_whole_ary[:] = molSysData.rho0xj_whole.cpu().numpy()   
    #     comm.Barrier()
    #     del molSysData.rho0xj_whole
    #     molSysData.rho0xj_whole = torch.from_numpy(rho0xj_whole_ary).to(device)

    #     #del idxi_ary, idxj_ary, rij_ary, xij_ary, ni_ary, nj_ary, mask_ary, mask_l_ary, pair_molid_ary



    # print(obj_size)
    # print_attribute_sizes(molSysData)
    print_attribute_sizes(molSysData.molecule_whole)
    #exit()

    if rank == 0:
        print('Computing cores.')
        parts = graph_partition(eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, sdc.verb)
        sdc.nparts = len(parts)
        print('New nparts:', sdc.nparts)
        print("Time to compute cores {:>7.2f} (s)".format(time.perf_counter() - tic))
        num_elements = 0

        for i in range(sdc.nparts):
            subSyCore = System(len(parts[i]))
            subSyCore.symbols = sy.symbols
            subSyCore.coords,subSyCore.types = extract_subsystem(sy.coords,sy.types,sy.symbols,parts[i])
            partCoreFileName = "CoreSubSy"+str(rank)+"_"+str(i)+".pdb"
            write_pdb_coordinates(partCoreFileName,subSyCore.coords,subSyCore.types,subSyCore.symbols)
            write_xyz_coordinates("CoreSubSy"+str(rank)+"_"+str(i)+".xyz",subSyCore.coords,subSyCore.types,subSyCore.symbols)
            print('N atoms in core {:>6d} : {:>6d}'.format(i, len(parts[i])))
            num_elements += len(parts[i])
            del subSyCore
        print('NUMBER OF ELEMENTS', num_elements)
        print('Loading the molecule and parameters.')
        if eng.reconstruct_dm:
            dm = get_initDM(eng, sdc, sy.coords, sy.symbols, sy.types, molSysData)#.share_memory_()
            dm_size = dm.size()
            nbytes = dm.numel() * dm.element_size()
        
        print('\n\n|||| Adaptive iter:', 0, '||||')
        partsCoreHalo = []
        print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
            partsCoreHalo.append(coreHalo)
            if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
            print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')

        new_graph_for_pairs = fullGraph.copy()
        for i in range(sy.nats):
            for sublist_idx in range(len(parts)):
                if i in parts[sublist_idx]:
                    new_graph_for_pairs[i][0] = len(partsCoreHalo[sublist_idx])
                    new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
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

        P_contr = torch.zeros(sy.nats*sdc.maxDeg,4,4, dtype=eng.torch_dt, device=device)  # density matrix
        P_contr[graph_maskd] = get_diag_guess_pyseqm(molSysData.molecule_whole, sy)
        P_contr = P_contr.reshape(sy.nats, sdc.maxDeg, 4,4).transpose(0,1)
        graph_maskd = np.array(graph_maskd)
        P_contr_size = P_contr.size()
        P_contr_nbytes = P_contr.numel() * P_contr.element_size()
        

        # print('collect_graph_from_rho S.')
        # graphNL = collect_graph_from_rho(None, sdc.overlap_whole,
        #                                   sdc.gthreshinit, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
        # del sdc.overlap_whole
        # print('collect_graph_from_rho dm.')
        # graphNL_dm = collect_graph_from_rho(None, pack(dm, molSysData.molecule_whole.nHeavy, molSysData.molecule_whole.nHydro)[0],
        #                               sdc.gthresh, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
        # fullGraph = add_graphs(graphNL, graphNL_dm )
        #fullGraph = graphNL_dm
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
        comm.Barrier()
        parts = comm.bcast(parts, root=0)
        sdc.nparts = comm.bcast(sdc.nparts, root=0)
        if rank in primary_ranks and rank != 0:
            P_contr = primary_comm.bcast(P_contr, root=0)
            P_contr_nbytes = primary_comm.bcast(P_contr_nbytes, root=0)


        if eng.reconstruct_dm:
            dm_size = comm.bcast(dm_size, root=0)
            win = MPI.Win.Allocate_shared(nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=comm) # 8 is the size of torch.float64
            buf, itemsize = win.Shared_query(0) 
            #assert itemsize == MPI.DOUBLE.Get_size() 
            ary = np.ndarray(buffer=buf, dtype=eng.np_dt, shape=(dm_size))
            if rank == 0:
                ary[:] = dm.numpy()   
            comm.Barrier()
            del dm
            dm = torch.from_numpy(ary)
            print(ary.shape)
            print(dm.shape)

        P_contr_size = comm.bcast(P_contr_size, root=0)
        P_contr_win = MPI.Win.Allocate_shared(P_contr_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm) # 8 is the size of torch.float64
        P_contr_buf, P_contr_itemsize = P_contr_win.Shared_query(0) 
        #assert P_contr_itemsize == MPI.DOUBLE.Get_size() 
        P_contr_ary = np.ndarray(buffer=P_contr_buf, dtype=eng.np_dt, shape=(P_contr_size))
        #if rank == 0:
        if rank == 0:
            P_contr_ary[:] = P_contr.cpu().numpy()   
        comm.Barrier()
        del P_contr
        P_contr = torch.from_numpy(P_contr_ary).to(device)

        fullGraph = comm.bcast(fullGraph, root=0)

        coreHalo = comm.bcast(coreHalo, root=0)
        partsCoreHalo = comm.bcast(partsCoreHalo, root=0)
        new_graph_for_pairs = comm.bcast(new_graph_for_pairs, root=0)
        graph_maskd = comm.bcast(graph_maskd, root=0)
        graph_for_pairs = comm.bcast(graph_for_pairs, root=0)


    dmOld = None
    mu0 = -5.5
    for gsc in range(sdc.numAdaptIter):
        print_memory_usage(rank, node_rank, "Memory usage")
        TIC_iter = time.perf_counter()
        # Partition the graph
        tic = time.perf_counter()
        if gsc > 0:
            if rank == 0:
                print('\n\n|||| Adaptive iter:', gsc, '||||')
                partsCoreHalo = []
                print("\nCore and halos indices for every part:")
                #print(fullGraph[parts[0]])
                for i in range(sdc.nparts):
                    coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
                    partsCoreHalo.append(coreHalo)
                    if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
                    print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')
                print("Time to compute halos {:>7.2f} (s)".format(time.perf_counter() - tic))

                new_graph_for_pairs = fullGraph.copy()
                for i in range(sy.nats):
                    for sublist_idx in range(len(parts)):
                        if i in parts[sublist_idx]:
                            new_graph_for_pairs[i][0] = len(partsCoreHalo[sublist_idx])
                            new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]
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
                graph_maskd = np.array(graph_maskd)
            else:
                coreHalo = None
                partsCoreHalo = None
                graph_for_pairs = None
                new_graph_for_pairs = None
                graph_maskd = None


            if mpiOnDebugFlag:
                coreHalo = comm.bcast(coreHalo, root=0)
                partsCoreHalo = comm.bcast(partsCoreHalo, root=0)
                graph_for_pairs = comm.bcast(graph_for_pairs, root=0)
                new_graph_for_pairs = comm.bcast(new_graph_for_pairs, root=0)
                graph_maskd = comm.bcast(graph_maskd, root=0)
            
        # for efficiency, the PySEQM dm needs to be reshaped in 4x4 blocks.
        if eng.interface == "PySEQM":
            with torch.no_grad():
                if eng.reconstruct_dm:
                    eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                    get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molSysData,
                                    dm.reshape((molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize,4, molSysData.molecule_whole.molsize,4)) \
                                    .transpose(2,3).reshape(molSysData.molecule_whole.nmol*molSysData.molecule_whole.molsize*molSysData.molecule_whole.molsize,4,4), P_contr, graph_for_pairs, graph_maskd)
                else:
                    eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                    get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molSysData,
                                    None, P_contr, graph_for_pairs, graph_maskd)

            if mpiOnDebugFlag: comm.Barrier()
        else:
            eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molSysData, dm)

        if rank ==0:
            with torch.no_grad():
                if eng.reconstruct_dm:
                    fullGraphRho = get_singlePointDM(sdc, eng, rank, 1, comm, parts, partsCoreHalo, sy, hindex, mu0, dm, P_contr, graph_for_pairs,
                                         eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list)
                else:
                    fullGraphRho = get_singlePointDM(sdc, eng, rank, 1, comm, parts, partsCoreHalo, sy, hindex, mu0, None, P_contr, graph_for_pairs,
                                         eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, core_indices_in_sub_expanded_list, Nocc_list)
            fullGraph = add_graphs(fullGraph, fullGraphRho, )
            del fullGraphRho
            if eng.reconstruct_dm:
                trace = get_dmTrace(eng, dm)
                print("DM TRACE: {:>10.7f}".format(trace))
            trace = torch.sum(P_contr.transpose(0,1).reshape(molSysData.molecule_whole.molsize*(len(graph_for_pairs[0])-1), 4,4)[graph_maskd].diagonal(dim1=-2, dim2=-1))
            print("DM TRACE: {:>10.7f}".format(trace))

        else:
            fullGraph = None
            
        if mpiOnDebugFlag:
            comm.Barrier()
            fullGraph = comm.bcast(fullGraph, root=0)

        del eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, I_halo_list, Nocc_list
        torch.cuda.empty_cache()
        #torch.save(dm, 'gs_solvated_cell_dm.pt')
        #torch.save(dm, 'gs_10k_dm_128.pt')
        #torch.save(dm, 'w32_4_dm.pt')
        #torch.save(dm, 'nanostar_dm.pt')        
        #torch.save(dm, 'w_4_dm.pt')  

        # Function to calculate tensor size in megabytes (MB)
        if rank == 0:
            # Sort tensors by size and print them
            tensors = list(get_tensors())
            tensors.sort(key=lambda x: tensor_size(x), reverse=True)
            print("Top memory-consuming tensors:")
            for tensor in tensors:
                if tensor_size(tensor) > 0.1:
                    print(f"Tensor size: {tensor_size(tensor):.2f} MB | Shape: {tensor.shape} | Dtype: {tensor.dtype}")

        print("t Iter {:>8.2f} (s)".format(time.perf_counter() - TIC_iter))


    ### forces calculation
    num_gpus = torch.cuda.device_count()
    color = 0 if rank < num_gpus else 1
    gpu_comm = comm.Split(color=color, key=rank)

    
    if rank < num_gpus:
        device = 'cuda:{}'.format(rank)
        del molSysData
        molSysData = get_molSysData(eng, sdc, sy.coords, sy.symbols, sy.types, device=device) #object with whatever initial parameters and tensors
        P_contr = P_contr.to(device)
        if rank == 0:
            forces = np.zeros((sy.coords.shape))
            #f_size = [sy.nats,3]
            #f_nbytes = sy.nats*3 * np.float64(0).nbytes

            tic = time.perf_counter()

            partsCoreHalo = []
            print("\nCore and halos indices for every part:")
            for i in range(sdc.nparts):
                coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
                partsCoreHalo.append(coreHalo)
                if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
                print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')

            new_graph_for_pairs = fullGraph.copy()
            for i in range(sy.nats):
                for sublist_idx in range(len(parts)):
                    if i in parts[sublist_idx]:
                        new_graph_for_pairs[i][0] = len(partsCoreHalo[sublist_idx])
                        new_graph_for_pairs[i][1:new_graph_for_pairs[i][0]+1] = partsCoreHalo[sublist_idx]

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
            #f_size = None
            #f_nbytes = 0
            partsCoreHalo = None
            new_graph_for_pairs = None
            graph_for_pairs = None
            graph_maskd = None

        
        if mpiOnDebugFlag:
            #f_size = comm.bcast(f_size, root=0)
            #f_nbytes = comm.bcast(f_nbytes, root=0)
            forces = gpu_comm.bcast(forces, root=0)
            print('HERE1')
            partsCoreHalo = gpu_comm.bcast(partsCoreHalo, root=0)
            print('BARR')
            gpu_comm.Barrier()
        
            #f_win = MPI.Win.Allocate_shared(f_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=comm) # 8 is the size of torch.float64
            #f_buf, f_itemsize = f_win.Shared_query(0) 
            #assert f_itemsize == MPI.DOUBLE.Get_size() 
            #forces = np.ndarray(buffer=f_buf, dtype='d', shape=(f_size))

            graph_for_pairs = gpu_comm.bcast(graph_for_pairs, root=0)
            new_graph_for_pairs = gpu_comm.bcast(new_graph_for_pairs, root=0)
            graph_maskd = gpu_comm.bcast(graph_maskd, root=0)

            if rank == 0:
                forces[:] = .0
            gpu_comm.Barrier()
        else:
            forces = np.zeros((sy.coords.shape))
        print('rank', rank)
        if eng.interface == "PySEQM":
            if eng.reconstruct_dm:
                print()
                eElec = get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
                                dm.reshape((molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize,4, molSysData.molecule_whole.molsize,4)) \
                                .transpose(2,3).reshape(molSysData.molecule_whole.nmol*molSysData.molecule_whole.molsize*molSysData.molecule_whole.molsize,4,4),P_contr, graph_for_pairs, graph_maskd)
            else:
                eElec = get_singlePointForces(sdc, eng, rank, num_gpus, gpu_comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
                                None, P_contr, graph_for_pairs, graph_maskd)

            if mpiOnDebugFlag:
                gpu_comm.Barrier()
                gpu_comm.Allreduce(MPI.IN_PLACE, forces, op=MPI.SUM)

                eElec_LIST = gpu_comm.gather(eElec, root=0)
            else:
                eElec_LIST = eElec
        else:
            get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData, dm)
        print("Time to get electron forces", time.perf_counter() - tic,"(s)")
        
        if rank == 0:
            if mpiOnDebugFlag:
                print("eElec:   {:>10.7f}".format(sum(eElec_LIST)[0]),)
            else:
                print("eElec:   {:>10.7f}".format(eElec[0]),)
            
            tic = time.perf_counter()
            eNucAB = get_eNuc(eng, molSysData)
            eTot, eNuc = get_eTot(eng, molSysData, eNucAB, 0)
            print("Enuc:   {:>10.7f}".format(eNuc),)
            L = eNuc.sum()
            L.backward()
            forceNuc = -molSysData.molecule_whole.coordinates.grad.detach()
            molSysData.molecule_whole.coordinates.grad.zero_()
            print("Time to get nuclear forces", time.perf_counter() - tic,"(s)")
            #print(forceNuc)
            np.save('forces_test.np', (forces+forceNuc.cpu().numpy()[0]), )
            #np.save('forces_test.np', (forces), )
            #np.save('forces_test.np', (forceNuc.cpu().numpy()[0]), )


    # fockFull = get_fock(eng, molSysData)
    # Hcore_whole = molSysData.M_whole.reshape(molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize, molSysData.molecule_whole.molsize,4,4) \
    #              .transpose(2,3) \
    #              .reshape(molSysData.molecule_whole.nmol, 4*molSysData.molecule_whole.molsize, 4*molSysData.molecule_whole.molsize)
    # eElec = get_eElec(eng, dm, fockFull, Hcore_whole, doTriu=True)
    # print("Eelec: {:>10.7f}".format(eElec[0]),)
    # tic = time.perf_counter()
    # L = eElec.sum()
    # L.backward(
    #     retain_graph=True
    # )
    # force = -molSysData.molecule_whole.coordinates.grad.detach()
    # molSysData.molecule_whole.coordinates.grad.zero_()
    # print("Time to get forces", time.perf_counter() - tic,"(s)")

    # eElec_reconstr = get_eElec(eng, dm, molSysData.h2elec_test, molSysData.h1elec_test, doTriu=False)
    # print("Eelec_Reconstr: {:>10.7f}".format(eElec_reconstr[0]),)
    # tic = time.perf_counter()
    # L_reconstr = eElec_reconstr.sum()
    # L_reconstr.backward(
    #     retain_graph=True
    #     )
    # force_reconstr = -molSysData.molecule_whole.coordinates.grad.detach()
    # molSysData.molecule_whole.coordinates.grad.zero_()
    # print("Time to get forces", time.perf_counter() - tic,"(s)")

    # maxDif = torch.max(torch.abs(Hcore_whole - molSysData.h1elec_test)).detach().numpy()
    # sumDif = torch.sum(torch.abs(Hcore_whole - molSysData.h1elec_test)).detach().numpy()
    # print(maxDif, sumDif)
    # print(((torch.abs(Hcore_whole - molSysData.h1elec_test))[0]==torch.max((torch.abs(Hcore_whole - molSysData.h1elec_test))[0])).nonzero())


    # h = molSysData.h1elec_test#.triu()+molSysData.h1elec_test.triu(1).transpose(1,2)


    
    # print('whole\n', (force+forceNuc)[0])
    # print('reconstr\n', (force_reconstr+forceNuc)[0])

    
    # print("Etot:  {:>10.7f}".format(eTot),)

    # forces = get_forces(eng, molSysData, eTot)

    AtToPrint = 0
    #print("graphNL", graphNL[AtToPrint])
    #print("fullGraphRho:", fullGraphRho[AtToPrint])

    # print(graphNL)
    # Get the neighbors of atom 1234 (by the graph)
