"""
graph_kernel_byparts.py
====================================
Utility functions for computing the kernel preconditioner 

"""

import time
import numpy as np
from scipy.linalg import inv
from pathlib import Path

from sedacs.graph import add_graphs, collect_graph_from_rho, print_graph
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.sdc_hamiltonian import get_hamiltonian
from sedacs.sdc_density_matrix import get_density_matrix
from sedacs.sdc_evals_dvals import get_evals_dvals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.mpi import (
    collect_and_sum_matrices,
    collect_and_sum_vectors_float,
    collect_and_concatenate_vectors,
    collect_and_sum_matrices_float,
    collect_and_sum_vectors_int,
)
from sedacs.system import System, extract_subsystem, get_hindex
from sedacs.coulombic import get_PME_coulvs, build_coul_ham
from sedacs.charges import get_charges, collect_charges
from sedacs.evals_dvals import collect_evals, collect_dvals
from sedacs.message import status_at, error_at, warning_at
from sedacs.mixer import diis_mix, linear_mix
from sedacs.chemical_potential import get_mu
from sedacs.file_io import read_latte_tbparams

try:
    from mpi4py import MPI

    is_mpi_available = True
except ModuleNotFoundError as e:
    is_mpi_available = False
    error_at(
        "get_adaptiveSCFDM_scf",
        "mpi4py not found, parallelization will not be available",
    )
    raise e

__all__ = ["get_kernel_byParts", "Canon_Response_dPdMu", "apply_kernel_byParts"]


def get_kernel_byParts(
    sdc, rank, numranks, parts, partsCoreHalo, sy, mu=0.0
):
    """
    Compute the kernel preconditioner for each subsystem in parallel with MPI support.

    Parameters
    ----------
    sdc : sedacs driver object
        Refer to driver/init.py for detailed information.
    rank: int
        Rank of the current process in the MPI communicator.
    numranks: int
        Total number of processes in the MPI communicator.
    parts: list of lists of int
        List of partitions of the full system.
    partsCoreHalo: list of lists of int
        List of core and halo indices for each partition.
    sy: System object
        Refer to system.py for detailed information.
    mu: float
        Chemical potential for the full system. Default is 0.0.
    
    Returns
    -------
    None
    """
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize charge pertubation vector
    chargePertVect = np.zeros(sy.nats)
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        # Get the number of atoms in the core region for the current part
        numberOfCoreAtoms = len(parts[partIndex])
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Get the number of orbitals in the subsystem
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        # Get the number of orbitals in the core region
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]
        norbsInCore = int(np.sum(tmpArray))
        print("Number of orbitals in the core =", norbsInCore)
        # Initialize Kernel preconditioner
        subSy.ker = np.zeros((numberOfCoreAtoms, numberOfCoreAtoms))
        # Initialize Jacobian matrix
        Jacobian = np.zeros((numberOfCoreAtoms, numberOfCoreAtoms))        
        # Iterate through all atoms in the core region
        for i in range(numberOfCoreAtoms):
            # Set the charge perturbation vector to zeros each time before starting each iteration
            chargePertVect[:] = 0.0
            # Get the index of the atom in the full system and set the corresponding charge to 1.0
            atom_index = parts[partIndex][i]
            chargePertVect[atom_index] = 1.0
            # Compute the Coulomb potential from charge perturbation vector  
            # Note that the Hubbard U correction is included in the computed Coulomb potential
            coulvs, ewald_e = get_PME_coulvs(
                    chargePertVect, sy.hubbard_u, sy.coords, sy.types, sy.latticeVectors
                )
            # Get the Coulomb potential and charges for the Core+Halo part 
            coulvsInPart = coulvs[partsCoreHalo[partIndex]]    
            # Build the Hamiltonian from Coulomb potential and charges from charge perturbation
            H_dq_v = np.zeros((norbs, norbs))

            for j in range(subSy.nats):
                start = subSy.hindex[j]
                end = subSy.hindex[j + 1]
                H_dq_v[start:end, start:end] = np.diag(coulvsInPart[j] * np.ones(end - start))

            H_dq_v = 0.5 * (np.matmul(subSy.over, H_dq_v) + np.matmul(H_dq_v, subSy.over)) 

            # Precompute ZQ and (ZQ)^t for the forward and backward transform
            ZQ = np.matmul(subSy.zmat, subSy.evects)
            ZQ_T = ZQ.T

            # H1 = Q'Z'*H_dq_v*ZQ  Forward transform
            X = np.matmul(ZQ_T, H_dq_v)
            H1 = np.matmul(X, ZQ)
            
            # Compute canonical quantum perturbation
            dPdMu, P1 = Canon_Response_dPdMu(H1, sdc.etemp, subSy.evals, mu, 12)   
            
            # Initialize dPdMuAO matrix with diagonal elements from dPdMu
            dPdMuAO = np.zeros((norbs, norbs))
            dPdMuAO[np.diag_indices_from(dPdMuAO)] = dPdMu

            # Transform P1 back to the nonortho-canonical basis set.
            X = np.matmul(ZQ, P1)
            P1 = np.matmul(X, ZQ_T)
            # Multiply P1 with the overlap matrix
            p1S = np.matmul(P1, subSy.over)
            # Transform dPdMu back to the nonortho-canonical basis set
            X = np.matmul(ZQ, dPdMuAO)
            dPdMuAO = np.matmul(X, ZQ_T)
            # Multiply dPdMuAO with the overlap matrix
            dPdMuAOS = np.matmul(dPdMuAO, subSy.over)
            # Get the diagonal elements of dPdMuAO and p1S only from the core region
            dPdMuAO_dia = np.diag(dPdMuAOS)
            p1_dia = np.diag(p1S)
            trP1 = np.sum(p1_dia[0:norbsInCore])
            trdPdMuAO = np.sum(dPdMuAO_dia[0:norbsInCore])
            # Compute the chemical potential
            mu1 = -trP1 / trdPdMuAO if abs(trdPdMuAO) > 1e-12 else 0.0
            # Adjust P1 with the repsonse to get the density matrix
            ptrho = 2 * (P1 + mu1 * dPdMuAO)
            ptrho = np.matmul(ptrho, subSy.over)
            # Get charges from the density matrix
            fullDiag = np.diag(ptrho)
            pt_charges = np.zeros(subSy.nats)

            for j in range(subSy.nats):
                pt_charges[j] = 0.0
                for jj in range(subSy.hindex[j], subSy.hindex[j + 1]):
                    pt_charges[j] = pt_charges[j] + fullDiag[jj]
            
            # Compute the Jacobian matrix 
            for j in range(numberOfCoreAtoms):
                val = pt_charges[j]
                if i == j:
                    val -= 1.0
                Jacobian[j, i] = val

        # Matrix inversion using SciPy
        # ipiv = lu_factor(Jacobian)
        # subSy.ker[:,:] = lu_solve(ipiv, np.eye(numberOfCoreAtoms))
        subSy.ker[:,:] = inv(Jacobian)
        # Rescale summation of each column of the sub kernel to -1 for maintaining charge neutrality
        subSy.ker = subSy.ker / subSy.ker.sum(axis=0)[None, :] * -1


def Canon_Response_dPdMu(H1, etemp, evals, mu, m):
    """
    Compute the canonical quantum perturbation and its derivative with respect to the chemical potential.

    Parameters
    ----------
    H1 : 2D numpy array, dtype: float
        The Hamiltonian matrix in the ortho-eigen basis.
    etemp : float
        The electronic temperature in Kelvin.
    evals : 1D numpy array, dtype: float
        The eigenvalues of the Hamiltonian matrix, H0.
    mu : float
        Chemical potential for the full system.
    m : int
        The number of recursion steps. 
    
    Returns
    -------
    dPdMu : 1D numpy array, dtype: float
        The derivative of the density matrix with respect to the chemical potential.
    P1 : 2D numpy array, dtype: float
        The canonical quantum perturbation.
    """
    kB = 8.61739e-5 # (eV/K)
    h_0 = evals     # Diagonal Hamiltonian H0 respresented in the eigenbasis Q
    beta = 1.0 / (kB * etemp) 
    cnst = beta / (1.0 * 2**(m+2)) # Scaling constant
    p_0 = 0.5 - cnst * (h_0 - mu)
    P1 = -cnst * H1
    # Loop over m recursion steps
    for i in range(m):
        # Compute denominators for broadcasting
        denom_j = 2.0 * p_0 * (p_0 - 1.0) + 1.0  # shape (HDIM,)
        denom_k = 2.0 * p_0 * (p_0 - 1.0) + 1.0  # shape (HDIM,)

        # Broadcast p_0 vectors to 2D
        p0_j = p_0[:, np.newaxis]  # shape (HDIM, 1)
        p0_k = p_0[np.newaxis, :]  # shape (1, HDIM)

        denom_j_2D = denom_j[:, np.newaxis]  # shape (HDIM, 1)
        denom_k_2D = denom_k[np.newaxis, :]  # shape (1, HDIM)

        # Compute updated P1
        factor = 1.0 / denom_j_2D  # shape (HDIM, 1)
        correction = 2.0 * (P1 - (p0_j + p0_k) * P1) * (1.0 / denom_k_2D) * (p0_k ** 2)
        P1 = factor * ((p0_j + p0_k) * P1 + correction)

        # Update p_0
        p_0 = (1.0 / (2.0 * (p_0 * p_0 - p_0) + 1.0)) * (p_0 * p_0)
     
    dPdMu = beta * p_0 * (1.0 - p_0)

    return dPdMu, P1


def apply_kernel_byParts(q_n, n, sdc, rank, numranks, comm, parts, sy):
    """
    Apply the kernel preconditioner to the residuals between q[n] and n for each subsystem.

    Parameters
    ----------
    q_n : 1D numpy array, dtype: float
        The charge vector q[n] for the full system.
    n : 1D numpy array, dtype: float
        The charge vector n for the full system.
    sdc : sedacs driver object
        Refer to driver/init.py for detailed information.
    rank: int
        Rank of the current process in the MPI communicator.
    numranks: int
        Total number of processes in the MPI communicator. 
    comm: MPI communicator
        The MPI communicator object.
    parts: list of lists of int
        List of partitions of the full system.
    sy: System object
        Refer to system.py for detailed information.
    
    Returns
    -------
    KK0Res : 1D numpy array, dtype: float
        The kernel preconditioner applied to the residuals between q[n] and n for each subsystem.
    """
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize KK0ResOnRank as None
    KK0ResOnRank = None
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        # Get the number of atoms in the core region for the current part
        numberOfCoreAtoms = len(parts[partIndex])
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Retrieve q[n] and n charge vectors for current part
        n_InPart = n[parts[partIndex]]
        q_n_InPart = q_n[parts[partIndex]]  
        # Initialize KK0ResInPart as a zero vector
        KK0ResInPart = np.zeros(numberOfCoreAtoms)
        # Compute the kernel preconditioner applied to the residuals and store it in KK0ResInPart
        KK0ResInPart[0:numberOfCoreAtoms] = np.dot(subSy.ker, (q_n_InPart - n_InPart))
        # Expand KK0ResInPart into KK0ResOnRank
        KK0ResOnRank = collect_charges(
            KK0ResOnRank, KK0ResInPart, parts[partIndex], sy.nats, verb=True
        )
    # If MPI is available and there are multiple ranks, collect and sum the KK0ResOnRank vector
    if is_mpi_available and numranks > 1:
        KK0Res = collect_and_sum_vectors_float(KK0ResOnRank, rank, numranks, comm)
        comm.Barrier() 
    else:
        KK0Res = KK0ResOnRank
    
    return KK0Res



def rankN_update_byParts(
    q_n, n, maxRanks, sdc, rank, numranks, comm, parts, partsCoreHalo, sy, mu=0.0
):
    """
    Perform the rank-N update for the kernel preconditioner and apply it to the residuals.

    Parameters
    ----------
    q_n : 1D numpy array, dtype: float
        The charge vector q[n] for the full system.
    n : 1D numpy array, dtype: float
        The charge vector n for the full system.
    maxRanks : int
        The maximum number of rank updates.
    sdc : sedacs driver object
        Refer to driver/init.py for detailed information.
    rank: int
        Rank of the current process in the MPI communicator.
    numranks: int
        Total number of processes in the MPI communicator.
    comm: MPI communicator
        The MPI communicator object.
    parts: list of lists of int
        List of partitions of the full system.
    partsCoreHalo: list of lists of int
        List of core and halo indices for each partition.
    sy: System object
        Refer to system.py for detailed information.
    mu: float
        Chemical potential for the full system. Default is 0.0.
    
    Returns
    -------
    KK0Res : 1D numpy array, dtype: float
        The kernel preconditioner applied to the preconditioned residuals between q[n] and n for each subsystem.
    """
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize the preconditioned residual vector K0ResOnRank as None
    K0ResOnRank = np.zeros(sy.nats)
    # Get the maximum number of atoms in the core region among all parts in the present MPI rank 
    maxCoresAmongParts = np.zeros(numranks, dtype=int)
    for partIndex in range(partIndex1, partIndex2):
        numberOfCoreAtoms = len(parts[partIndex])
        maxCoresAmongParts[rank] = max(maxCoresAmongParts[rank], numberOfCoreAtoms) 
    # Initialize K0ResPart to store the preconditioned residuals for each part
    K0ResPart = np.zeros((maxCoresAmongParts[rank], partsPerRank))
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        # Get the number of atoms in the core region for the current part
        numberOfCoreAtoms = len(parts[partIndex])
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Get the number of atoms in the core+halo region for the current part
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        # Get the number of orbitals in the core region
        tmpArray = np.zeros(numberOfCoreAtoms)
        tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]
        norbsInCore = int(np.sum(tmpArray))
        print("Number of orbitals in the core =", norbsInCore)
        # Retrieve q[n] and n charge vectors for current part
        q_nInPart = q_n[parts[partIndex]] 
        nInPart = n[parts[partIndex]] 
        # Calculate K0Res which is the product of the Preconditioner K with the residue q(n) - n
        K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1] = np.matmul(subSy.ker, (q_nInPart - nInPart)) 
        # Expand K0resPart into K0Res
        K0ResOnRank[parts[partIndex]] = K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1]
    # If MPI is available and there are multiple ranks, collect and sum the K0ResOnRank vector
    if is_mpi_available and numranks > 1:
        K0Res = collect_and_sum_vectors_float(K0ResOnRank, rank, numranks, comm)
        maxCoresAmongPartsAndRanks = collect_and_sum_vectors_int(maxCoresAmongParts, rank, numranks, comm)
        comm.Barrier()
    else:
        K0Res = K0ResOnRank
        maxCoresAmongPartsAndRanks = maxCoresAmongParts
    
    # Get the maximum number of atoms in the core region among all parts in all ranks
    maxCoresAmongPartsAndRanks = int(max(maxCoresAmongPartsAndRanks))
    # Initialize directional derivatives (dr) from the preconditioned residuals
    dr = np.zeros(sy.nats)
    dr[:] = K0Res[:]
    # Initial arrays for the rank updates
    vi = np.zeros((sy.nats, maxRanks))
    dr_save = np.zeros((sy.nats, maxRanks))
    v_core_i = np.zeros((maxCoresAmongPartsAndRanks, partsPerRank, maxRanks))
    c_i = np.zeros(maxRanks)
    ff = np.zeros((maxCoresAmongPartsAndRanks, partsPerRank, maxRanks))
    irank = -1
    # Here we enter the loop for the rank updates (do not confuse with MPI rank)
    # for irank in range(maxRanks):
    error = 1.0
    mRank = maxRanks
    while irank < mRank - 1 and error > 1e-6:
        irank = irank + 1
        # Construct Krylov subspace vector from previous directional derivative
        vi[:, irank] = dr / np.linalg.norm(dr)
        # Gram-Schmidt orthogonalization
        if irank > 0:
            for kk in range(irank):
                vi[:, irank] = vi[:, irank] - np.dot(vi[:, irank], vi[:, kk]) * vi[:, kk]
            # Normalize the vector
            vi[:, irank] = vi[:, irank] / np.linalg.norm(vi[:, irank])
        
        # Get the charge perturbation vector
        chargePertVect = vi[:, irank]
        # Get the Coulomb potential from charge perturbation vector
        # Note that the Hubbard U correction is included in the computed Coulomb potential
        coulvs, ewald_e = get_PME_coulvs(
                chargePertVect, sy.hubbard_u, sy.coords, sy.types, sy.latticeVectors
        )
        # Initialize the core part of the charge response (q1, dqdmu) by the derivative of subsystem
        # density matrix with respect to perturbation parameter (lambda) and chemical potential (mu).
        q1 = np.zeros((maxCoresAmongPartsAndRanks, partsPerRank))
        dqdmu = np.zeros((maxCoresAmongPartsAndRanks, partsPerRank))
        # Initialize the variables to sum up the partial traces
        trP1 = np.zeros(1); trdPdMu = np.zeros(1)
        for partIndex in range(partIndex1, partIndex2):
            numberofCoreHaloAtoms = len(partsCoreHalo[partIndex])
            numberOfCoreAtoms = len(parts[partIndex])
            subSy = sy.subSy_list[partIndex - partIndex1]
            assert numberofCoreHaloAtoms == subSy.nats, "Number of atoms in the core+halo region should be equal to the number of atoms in the subsystem"
            norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
            # tmpArray = np.zeros(numberOfCoreAtoms)
            # tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]
            # norbsInCore = int(np.sum(tmpArray))
            # print("Number of orbitals in the core =", norbsInCore)

            # Get the Coulomb potential and charges for the Core+Halo part 
            coulvsInPart = coulvs[partsCoreHalo[partIndex]]   
            # Extract the perturbation over the core part only
            v_core_i[0:numberOfCoreAtoms, partIndex - partIndex1, irank] = vi[parts[partIndex], irank]

            # Build the Hamiltonian from Coulomb potential and charges from charge perturbation
            H_dq_v = np.zeros((norbs, norbs))

            for j in range(subSy.nats):
                start = subSy.hindex[j]
                end = subSy.hindex[j + 1]
                H_dq_v[start:end, start:end] = np.diag(coulvsInPart[j] * np.ones(end - start))

            H_dq_v = 0.5 * (np.matmul(subSy.over, H_dq_v) + np.matmul(H_dq_v, subSy.over))
            
            # H1 = Q'Z'*H_dq_v*ZQ  Forward transform
            # Compute transformations ZQ and (ZQ)^t transformation that takes from the canonical nonorthogonal
            # to the orthogonal eigenbasis.
            ZQ = np.matmul(subSy.zmat, subSy.evects)
            ZQ_T = ZQ.T
            # Take H1 to the ortho-eigen basis set.
            X = np.matmul(ZQ_T, H_dq_v)
            H1 = np.matmul(X, ZQ)
            # Construct the "bare" response P1 and the derivative with respect to the
            # chemical potential (dPdMu). Everything in the ortho-eigen basis set
            dPdMu, P1 = Canon_Response_dPdMu(H1, sdc.etemp, subSy.evals, mu, 12)
            # Transform P1 back to the nonortho-canonical basis set.
            X = np.matmul(ZQ, P1)
            P1 = np.matmul(X, ZQ_T)
            # Convert dPdMu to matrix
            dPdMuAO = np.zeros((norbs, norbs))
            dPdMuAO[np.diag_indices_from(dPdMuAO)] = dPdMu
            # Transform dPdMu back to the nonortho-canonical basis set
            X = np.matmul(ZQ, dPdMuAO)
            dPdMuAO = np.matmul(X, ZQ_T)
            # Here we compute the charges response (q1) from P1 and we store it on 
            # a vector q1 that stores all the previous q1s from past iranks iterations
            # We also compute the partial trace contribution (trP1) from this mpi
            # execution and the current part (partIndex).
            P1 = 2 * P1
            ptrho = np.matmul(P1, subSy.over)
            fullDiag = np.diag(ptrho)
            pt_charges = np.zeros(numberofCoreHaloAtoms)

            for j in range(numberofCoreHaloAtoms):
                pt_charges[j] = 0.0
                for jj in range(subSy.hindex[j], subSy.hindex[j + 1]):
                    pt_charges[j] = pt_charges[j] + fullDiag[jj]
            # Collect the charge response from the core region
            q1[:numberOfCoreAtoms, partIndex - partIndex1] = pt_charges[:numberOfCoreAtoms]
            # Add up the partial trace contribution from the core region
            trP1 = trP1 + np.sum(pt_charges[:numberOfCoreAtoms])

            # Here we compute the charges response (dqdmu) from dPdMu and we store
            # them on a matrix dqdmu that stores all the previous dqdmus from past
            # irank iterations.
            # We also compute the partial trace contribution (trdPdMu) from this node
            # and the current part (partIndex).
            dPdMuAO = 2 * dPdMuAO
            ptrho = np.matmul(dPdMuAO, subSy.over)
            fullDiag = np.diag(ptrho)
            pt_charges = np.zeros(numberofCoreHaloAtoms)
            
            for j in range(numberofCoreHaloAtoms):
                pt_charges[j] = 0.0
                for jj in range(subSy.hindex[j], subSy.hindex[j + 1]):
                    pt_charges[j] = pt_charges[j] + fullDiag[jj]
            # Collect the charge response from the core region
            dqdmu[:numberOfCoreAtoms, partIndex - partIndex1] = pt_charges[:numberOfCoreAtoms]
            # Add up the partial trace contribution from the core region
            trdPdMu = trdPdMu + np.sum(pt_charges[:numberOfCoreAtoms])
        # If MPI is available and there are multiple ranks, collect and sum the partial traces
        if is_mpi_available and numranks > 1:
            trP1 = collect_and_sum_vectors_float(trP1, rank, numranks, comm)
            trdPdMu = collect_and_sum_vectors_float(trdPdMu, rank, numranks, comm)
            comm.Barrier()
        # Compute the response to the chemical potential (mu1) and adjust q1
        mu1_Global = - trP1 / trdPdMu if abs(trdPdMu) > 1e-12 else 0.0
        q1 = q1 + mu1_Global * dqdmu
        # Initialize f to store directional derivatives of the residual function
        f = np.zeros(maxCoresAmongPartsAndRanks)
        # Initialize dr to store the preconditioned directional derivatives
        dr[:] = 0.0
        c_i_temp = np.zeros(1)
        # Loop over all partitions in the current MPI rank
        for partIndex in range(partIndex1, partIndex2):
            # Get the number of atoms in the core region for the current part
            numberOfCoreAtoms = len(parts[partIndex])
            # Get the subsystem object for the current part
            subSy = sy.subSy_list[partIndex - partIndex1]
            # Get the number of orbitals in the subsystem 
            # norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
            # tmpArray = np.zeros(numberOfCoreAtoms)
            # tmpArray[:] = subSy.orbs[subSy.types[0:numberOfCoreAtoms]]
            # # # Get the number of orbitals in the core region
            # norbsInCore = int(np.sum(tmpArray))
            # print("Number of orbitals in the core =", norbsInCore)
            assert numberOfCoreAtoms == subSy.ker.shape[0], "Number of atoms in the core should be equal to the number of atoms in the kernel"
            # Compute the directional derivative of the residual function and store it in f
            f[0:numberOfCoreAtoms] = q1[0:numberOfCoreAtoms, partIndex - partIndex1] - v_core_i[0:numberOfCoreAtoms, partIndex - partIndex1, irank]
            # Compute the preconditioned directional derivative of the residual function
            ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank] = np.matmul(subSy.ker, f[0:numberOfCoreAtoms])
            # Preconiditioned Krylov subspace approximation
            # c_i[irank] = c_i[irank] + np.dot(ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank], K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1])
            c_i_temp = c_i_temp + np.dot(ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank], K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1])
            # Save the preconditioned directional derivative of the residual function for getting the resolution of identity
            dr[parts[partIndex]] = ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank]
        # If MPI is available and there are multiple ranks, collect and sum the preconditioned directional derivative
        if is_mpi_available and numranks > 1:
            dr = collect_and_sum_vectors_float(dr, rank, numranks, comm)
            comm.Barrier()
        # Save dr for the current rank update
        dr_save[:, irank] = dr

        maxRanks = irank + 1
        # If MPI is available and there are multiple ranks, collect and sum the preconditioned Krylov subspace approximation
        if is_mpi_available and numranks > 1:
            # c_i = collect_and_sum_vectors_float(c_i, rank, numranks, comm)
            c_i_temp = collect_and_sum_vectors_float(c_i_temp, rank, numranks, comm)
            comm.Barrier()
        c_i[irank] = c_i_temp
        # Compute elements of the overlap matrix
        auxVect = np.zeros(maxRanks * maxRanks)
        for i in range(maxRanks):
            for j in range(maxRanks):
                for k in range(partsPerRank):
                    auxVect[i * maxRanks + j] = auxVect[i * maxRanks + j] + np.dot(ff[:, k, i], ff[:, k, j])
        # auxVect[:] = np.einsum('api,apj->ij', ff, ff).reshape(-1) # This is the vectorized version for the code above
        # If MPI is available and there are multiple ranks, collect and sum the elements of the overlap matrix
        if is_mpi_available and numranks > 1:
            auxVect = collect_and_sum_vectors_float(auxVect, rank, numranks, comm)
            comm.Barrier()
        # Reshape the auxVect to get the overlap matrix
        oij = np.zeros((maxRanks, maxRanks))
        mMat = np.zeros((maxRanks, maxRanks))
        # for i in range(maxRanks):
        #     for j in range(maxRanks):
        #         oij[i, j] = auxVect[i * maxRanks + j]
        oij[:, :] = auxVect.reshape((maxRanks, maxRanks)) # This is the vectorized version for the code above
        # Compute the inverse of the overlap matrix
        mMat[:,:] = inv(oij)
        # Compute the resolution of identity
        KK0Res = np.zeros(sy.nats)
        IdK0Res = np.zeros(sy.nats)
        for i in range(maxRanks):
            for j in range(maxRanks):
                KK0Res = KK0Res + vi[:, i] * mMat[i, j] * c_i[j]
                IdK0Res = IdK0Res + dr_save[:, i] * mMat[i, j] * c_i[j]
        # The following is the vectorized version for the code above
        # KK0Res[:] = np.einsum('ni,ij,j->n', vi, mMat, c_i)
        # IdK0Res[:] = np.einsum('ni,ij,j->n', dr_save, mMat, c_i)

        error = np.linalg.norm(K0Res - IdK0Res) / np.linalg.norm(K0Res)
        if rank == 0:
            print("Error Rank-Update", error, irank, "\n")
            # print("Error Rank-Update", error, maxRanks, "\n")


    return KK0Res




