"""Graph adaptive solver"""

import time
import torch
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

from seqm.seqm_functions.pack import pack

import gc

import numpy as np

try:
    from mpi4py import MPI

    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False

is_mpi_available = False

__all__ = ["get_singlePoint", "get_adaptiveDM"]

## Single point calculation
# @brief Construct a connectivity graph based on constructing density matrices
# of parts of the system.
#
def get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0,
                    molSysData, P):
    # computing DM for core+halo part
    #
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    dValOnRank = np.array([])
    eValOnRank = np.array([])
    eValOnRank_list = []
    Q_list = [] # Eigenvectors for each part
    I_list = [] # Indices for updating the columns in total DM
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


        # for kk in range(subSy.nats):
        #    subSy.coords[0,kk] = subSy.coords[0,kk] + sy.latticeVectors[0,:] * nlTrX[partsCoreHalo[partIndex][kk]]

        ham = get_hamiltonian(eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], molSysData, P,
                              verbose=False)
        #print("Time for get_hamiltonian", time.perf_counter() - tic, "(s)")
        print("TOT {:>8.3f} (s)".format(time.perf_counter() - tic))


        norbs = subSy.nats
        occ = int(float(norbs) / 2.0)  # Get the total occupied orbitals

        tic = time.perf_counter()
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs, I, core_indices_in_sub_expanded = get_eVals(eng, sdc, sy, occ, ham, subSy.coords, subSy.symbols, subSy.types, Tel, mu0,
                        coreSize, subSy, subSyCore, parts[partIndex], partsCoreHalo[partIndex],
                        verbose=False)

        dValOnRank = np.append(dValOnRank, dVals)
        eValOnRank = np.append(eValOnRank, eVals.detach().numpy())

        eValOnRank_list.append(eVals)
        Q_list.append(Q)
        I_list.append(I)
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        #print("Time to get eVals/dVals", time.perf_counter() - tic, "(s)")
        print("| t eVals/dVals {:>9.4f} (s)".format(time.perf_counter() - tic))

        # Gather all the eval and dvals from all the coreHalos within this execution rank
        # dvalsOnRank = collect_dValsOnRank(dVals)
        # evalsOnRank = collect_eValsOnRank(eVals)

    full_dVals = dValOnRank
    full_eVals = eValOnRank
    mu0 = get_mu(mu0, full_dVals, full_eVals, Tel, sy.numel/2)

    return eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list, mu0

def get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces,
                    molSysData, P):
    # computing DM for core+halo part
    #
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    for partIndex in range(partIndex1, partIndex2):
        print("Rank, part", rank, partIndex)
        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])

        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords,subSyCore.types = extract_subsystem(sy.coords,sy.types,sy.symbols,parts[partIndex])

        tic = time.perf_counter()

        # for kk in range(subSy.nats):
        #    subSy.coords[0,kk] = subSy.coords[0,kk] + sy.latticeVectors[0,:] * nlTrX[partsCoreHalo[partIndex][kk]]

        forces[parts[partIndex]] = get_hamiltonian(eng,subSy.coords,subSy.types,subSy.symbols, 
                              parts[partIndex], partsCoreHalo[partIndex], molSysData, P, doForces = True,
                              verbose=False)
        toc = time.perf_counter()
        print("Time for get_hamiltonian", toc - tic, "(s)")

    #return forces

def get_singlePointDM(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, dm,
                      eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list):
    
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None

    Tel = sdc.Tel
    for partIndex in range(partIndex1,partIndex2):
        tic = time.perf_counter()
        # this will calculate the DM in subsys and update the whole DM
        rho_ren = get_density_matrix_renorm(eng, Tel, mu0, dm,
                                            eValOnRank_list[partIndex], Q_list[partIndex], NH_Nh_Hs_list[partIndex], I_list[partIndex], core_indices_in_sub_expanded_list[partIndex], Nocc_list[partIndex])
        graphOnRank = collect_graph_from_rho(graphOnRank, rho_ren, sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False)
        del rho_ren
        #print('Time to get DM', time.perf_counter() - tic)
        print("t DM {:>8.3f} (s)".format(time.perf_counter() - tic))

    print('HERE_DM_1')
    if is_mpi_available:
        fullGraphRho = collect_and_sum_matrices(graphOnRank, rank, numranks, comm)
        # dValsFull = collect_dValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???
        # eValsFull = collect_eValsFull(dValsOnRank) #MPI functions # Newton-Raphosn from graph paper???

        comm.Barrier()
        return fullGraphRho
    else:
        #fullGraphRho = graphOnRank
        return graphOnRank


def get_adaptiveDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL):
    tic = time.perf_counter()
        
    fullGraph = graphNL.copy()
    del graphNL

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

        #print('N atoms in core', i, ':', len(parts[i]))
        print('N atoms in core {:>6d} : {:>6d}'.format(i, len(parts[i])))
        num_elements += len(parts[i])
        del subSyCore

    print('NUMBER OF ELEMENTS', num_elements)
    print('Loading the molecule and parameters.')
    molSysData = get_molSysData(eng, sdc, sy.coords, sy.symbols, sy.types) #object with whatever initial parameters and tensors
    dm = get_initDM(eng, sdc, sy.coords, sy.symbols, sy.types, molSysData)
    # graphNL = collect_graph_from_rho(None, pack(dm, molSysData.molecule_whole.nHeavy, molSysData.molecule_whole.nHydro)[0],
    #                                   sdc.gthresh, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
    print('collect_graph_from_rho S.')
    graphNL = collect_graph_from_rho(None, sdc.overlap_whole,
                                      sdc.gthreshinit, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
    print('collect_graph_from_rho dm.')
    graphNL_dm = collect_graph_from_rho(None, pack(dm, molSysData.molecule_whole.nHeavy, molSysData.molecule_whole.nHydro)[0],
                                      sdc.gthresh, sy.nats, sdc.maxDeg, [i for i in range(0,sy.nats)],hindex)
    
    fullGraph = add_graphs(graphNL_dm, graphNL)
    del graphNL_dm

    # del fullGraph
    # fullGraph = graphNL.copy()
    del sdc.overlap_whole

    dmOld = None
    njumps = 2
    mu0 = -5.5
    for gsc in range(sdc.numAdaptIter):
        TIC_iter = time.perf_counter()
        # Partition the graph
        print('\n\n|||| Adaptive iter:', gsc, '||||')
        tic = time.perf_counter()
        partsCoreHalo = []
        print("\nCore and halos indices for every part:")
        for i in range(sdc.nparts):
            coreHalo, nc = get_coreHaloIndices(eng, parts[i], fullGraph, njumps, sdc, sy)
            partsCoreHalo.append(coreHalo)
            if sdc.verb: print("coreHalo for part", i, "=", coreHalo)
            print('N atoms in core/coreHalo {:>6d} : {:>6d} {:>6d}'.format(i, len(parts[i]), len(coreHalo)), '\n')

        print("Time to compute halos {:>7.2f} (s)".format(time.perf_counter() - tic))
        dmOld = dm.clone()
        torch.save(dmOld, 'gs_10k_dmOld_128.pt')
        del dmOld


        # for efficiency, the PySEQM dm needs to be reshaped in 4x4 blocks.
        if eng.interface == "PySEQM":
            with torch.no_grad():
                eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molSysData,
                                dm.reshape((molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize,4, molSysData.molecule_whole.molsize,4)) \
                                .transpose(2,3).reshape(molSysData.molecule_whole.nmol*molSysData.molecule_whole.molsize*molSysData.molecule_whole.molsize,4,4))
        else:
            eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list, mu0 = \
                get_singlePoint(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, molSysData, dm)

        with torch.no_grad():
            fullGraphRho = get_singlePointDM(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, mu0, dm,
                                         eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list)
        del eValOnRank_list, Q_list, NH_Nh_Hs_list, I_list, core_indices_in_sub_expanded_list, Nocc_list

        torch.cuda.empty_cache()
        torch.save(dm, 'gs_10k_dm_128.pt')

        fullGraph = add_graphs(fullGraphRho, graphNL)
        del fullGraphRho
        trace = get_dmTrace(eng, dm)
        print("DM TRACE: {:>10.7f}".format(trace))
        dmOld = torch.load('gs_10k_dmOld_128.pt')
        maxDif, sumDif = get_dmErrs(eng, dmOld, dm)
        del dmOld
        print('HERE_after_2')

        print('ERROR:')
        print(" MAX |\u0394DM_ij|: {:>10.7f}".format(maxDif))
        print(" \u03A3   |\u0394DM_ij|: {:>10.7f}".format(sumDif))

        # if gsc >=6:
        #     njumps = 2
        

        # Function to calculate tensor size in megabytes (MB)
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

        # Sort tensors by size and print them
        tensors = list(get_tensors())
        tensors.sort(key=lambda x: tensor_size(x), reverse=True)

        print("Top memory-consuming tensors:")
        for tensor in tensors:
            if tensor_size(tensor) > 0.1:
                print(f"Tensor size: {tensor_size(tensor):.2f} MB | Shape: {tensor.shape} | Dtype: {tensor.dtype}")

        print("t Iter {:>8.2f} (s)".format(time.perf_counter() - TIC_iter))
        

    ### forces calculation
    # forces = np.zeros((sy.coords.shape))
    # if eng.interface == "PySEQM":
    #         get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData,
    #                             dm.reshape((molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize,4, molSysData.molecule_whole.molsize,4)) \
    #                             .transpose(2,3).reshape(molSysData.molecule_whole.nmol*molSysData.molecule_whole.molsize*molSysData.molecule_whole.molsize,4,4))
    # else:
    #         get_singlePointForces(sdc, eng, rank, numranks, comm, parts, partsCoreHalo, sy, hindex, forces, molSysData, dm)

    # print(forces)
    

    # fockFull = get_fock(eng, molSysData)
    # Hcore_whole = molSysData.M_whole.reshape(molSysData.molecule_whole.nmol, molSysData.molecule_whole.molsize, molSysData.molecule_whole.molsize,4,4) \
    #              .transpose(2,3) \
    #              .reshape(molSysData.molecule_whole.nmol, 4*molSysData.molecule_whole.molsize, 4*molSysData.molecule_whole.molsize)

    # eElec = get_eElec(eng, molSysData.molecule_whole.dm, fockFull, Hcore_whole)
    # eNucAB = get_eNuc(eng, molSysData)
    # eTot, eNuc = get_eTot(eng, molSysData, eNucAB, eElec)
    # print("Eelec: {:>10.7f}".format(eElec[0]),)
    # print("Enuc:   {:>10.7f}".format(eNuc),)
    # print("Etot:  {:>10.7f}".format(eTot),)

    # forces = get_forces(eng, molSysData, eTot)

    AtToPrint = 0
    #print("graphNL", graphNL[AtToPrint])
    #print("fullGraphRho:", fullGraphRho[AtToPrint])
    #print("fullGraph", fullGraph[AtToPrint])

    # print(graphNL)
    # print(fullGraph)
    # Get the neighbors of atom 1234 (by the graph)
