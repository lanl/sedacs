"""
graph_kernel_byparts.py
====================================
Utility functions for computing the kernel preconditioner 

"""

import time
import torch
import numpy as np
from pathlib import Path
import nvtx

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
torch.set_float32_matmul_precision("high")

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
    sdc, rank, numranks, parts, partsCoreHalo, sy, mu=0.0, nbr_inds=None, disps=None, dists=None, alpha=None, PME_data=None, device="cuda"
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
    device : str
        The device to use for computation (e.g., "cuda" or "cpu").
    dtype : torch.dtype
        The data type for the tensors (default is torch.float64).
    
    Returns
    -------
    None
    """
    dtype = torch.float32
    torch.cuda.synchronize()
    nvtx.push_range("get_kernel_byParts", color="blue", domain="get_kernel_byParts")
    torch.cuda.synchronize()
    nvtx.push_range("initialization", color="green", domain="get_kernel_byParts")
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize charge perturbation vector
    chargePertVect = torch.zeros(sy.nats, dtype=dtype, device=device)
    # Convert numpy array to torch tensor
    lattice_vecs = torch.from_numpy(sy.latticeVectors).to(device).to(dtype)
    coords = torch.from_numpy(sy.coords).to(device).to(dtype)
    hubbard_u = torch.from_numpy(sy.hubbard_u).to(device).to(dtype)
    atomtypes = torch.from_numpy(sy.types).to(device).to(torch.int64)
    torch.cuda.synchronize()
    nvtx.pop_range("get_kernel_byParts")
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        torch.cuda.synchronize()
        nvtx.push_range(f"part {partIndex} init", color="purple", domain="get_kernel_byParts")
        # Get the number of atoms in the core region for the current part
        numberOfCoreAtoms = len(parts[partIndex])
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Get the number of orbitals in the subsystem
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        # Get the number of orbitals in the core region
        tmpArray = torch.from_numpy(subSy.orbs[subSy.types[0:numberOfCoreAtoms]]).to(device).to(torch.int64)
        norbsInCore = int(torch.sum(tmpArray).item())
        #print("Number of orbitals in the core =", norbsInCore)
        # Get the overlap matrix 
        over = torch.from_numpy(subSy.over).to(device).to(dtype)
        # Get the Z matrix
        zmat = torch.from_numpy(subSy.zmat).to(device).to(dtype)
        # Get the eigenvectors 
        evects = torch.from_numpy(subSy.evects).to(device).to(dtype)
        # Get the eigenvalues
        evals = torch.from_numpy(subSy.evals).to(device).to(dtype)
        # Get the hindex 
        hindex = torch.from_numpy(subSy.hindex).to(device).to(torch.int64)
        # Initialize Kernel preconditioner
        subSy.ker = torch.zeros((numberOfCoreAtoms, numberOfCoreAtoms), device=device, dtype=dtype)
        # Initialize Jacobian matrix
        Jacobian = torch.zeros((numberOfCoreAtoms, numberOfCoreAtoms), device=device, dtype=dtype)
        # Precompute ZQ and (ZQ)^t for the forward and backward transform
        ZQ = torch.matmul(zmat, evects)
        ZQ_T = ZQ.T
        # Initialize H_dq_v matrix
        H_dq_v = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        H_indices = torch.arange(hindex[0], hindex[-1], device=device, dtype=torch.int64)
        # Initialize the charge vector for the subsystem
        pt_charges = torch.zeros(subSy.nats, device=device, dtype=dtype)
        pt_indices = torch.repeat_interleave(
            torch.arange(subSy.nats, device=device),
            torch.diff(hindex[:subSy.nats + 1])
        )
        coulvsInPart = torch.zeros(subSy.nats, device=device, dtype=dtype)
        H1 = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        dPdMuAO = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        dPdMu = torch.zeros(norbs, device=device, dtype=dtype)
        P1 = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        p1S = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        dPdMuAOS = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        dPdMuAO_dia = torch.zeros(norbs, device=device, dtype=dtype)
        p1_dia = torch.zeros(norbs, device=device, dtype=dtype)
        ptrho = torch.zeros((norbs, norbs), device=device, dtype=dtype)
        fullDiag = torch.zeros(norbs, device=device, dtype=dtype)
        
        coreindex = torch.tensor(parts[partIndex]).to(device).to(torch.int64)
        corehaloindex = torch.tensor(partsCoreHalo[partIndex]).to(device).to(torch.int64)
        torch.cuda.synchronize()
        nvtx.pop_range("get_kernel_byParts")
        torch.cuda.synchronize()
        nvtx.push_range(f"loop over core regions", color="orange", domain="get_kernel_byParts")
        # Iterate through all atoms in the core region
        for i in range(numberOfCoreAtoms):
            torch.cuda.synchronize()
            nvtx.push_range("charge perturbation", color="yellow", domain="get_kernel_byParts")
            # Set the charge perturbation vector to zeros each time before starting each iteration
            # chargePertVect.zero_()
            # Get the index of the atom in the full system and set the corresponding charge to 1.0
            # atom_index = coreindex[i]
            # chargePertVect[atom_index] = 1.0
            chargePertVect.index_fill_(0, coreindex[i], 1.0)
            torch.cuda.synchronize()
            nvtx.push_range("PME", color="yellow", domain="get_kernel_byParts")
            # Compute the Coulomb potential from charge perturbation vector
            coulvs, ewald_e, nbr_inds, disps, dists, alpha, PME_data = get_PME_coulvs(
                chargePertVect, hubbard_u, coords, atomtypes, lattice_vecs, nbr_inds=nbr_inds, disps=disps, dists=dists, alpha=alpha, PME_data=PME_data, device=device, use_torch=True, convert=False, 
            )
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            # Get the Coulomb potential and charges for the Core+Halo part
            # coulvsInPart[:] = torch.from_numpy(coulvs[partsCoreHalo[partIndex]]).to(device).to(dtype)
            coulvsInPart[:] = coulvs[corehaloindex]
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("build H_dq_v", color="yellow", domain="get_kernel_byParts")
            # Build the Hamiltonian from Coulomb potential and charges from charge perturbation
            H_dq_v.zero_()
            H_dq_v[H_indices, H_indices] = torch.repeat_interleave(coulvsInPart, torch.diff(hindex)) # Same as code below
            # for j in range(subSy.nats):
            #     start = subSy.hindex[j]
            #     end = subSy.hindex[j + 1]
            #     H_dq_v[start:end, start:end] = coulvsInPart[j] * np.eye(end - start)
            H_dq_v = 0.5 * (torch.matmul(over, H_dq_v) + torch.matmul(H_dq_v, over))
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("H forward transform", color="yellow", domain="get_kernel_byParts")
            # H1 = Q'Z'*H_dq_v*ZQ  Forward transform
            H1[:, :] = torch.matmul(torch.matmul(ZQ_T, H_dq_v), ZQ)
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("canonical response", color="yellow", domain="get_kernel_byParts")
            # Compute canonical quantum perturbation
            dPdMu[:], P1[:, :] = Canon_Response_dPdMu(H1, sdc.etemp, evals, mu, 12)
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("rho compute", color="yellow", domain="get_kernel_byParts")
            # Initialize dPdMuAO matrix with diagonal elements from dPdMu
            dPdMuAO.zero_()
            # dPdMuAO[torch.arange(norbs), torch.arange(norbs)] = dPdMu
            dPdMuAO.diagonal().copy_(dPdMu)
            torch.cuda.synchronize()
            nvtx.push_range("matmuls", color="yellow", domain="get_kernel_byParts")
            # Transform P1 back to the nonortho-canonical basis set.
            P1[:, :] = torch.matmul(torch.matmul(ZQ, P1), ZQ_T)
            # P1 = matrix_transform(ZQ, P1, ZQ_T)
            # P1 = torch.matmul(ZQ, P1)
            # P1 = torch.matmul(P1, ZQ_T)
            # Multiply P1 with the overlap matrix
            p1S[:, :] = torch.matmul(P1, over)
            # Transform dPdMu back to the nonortho-canonical basis set
            dPdMuAO[:, :] = torch.matmul(torch.matmul(ZQ, dPdMuAO), ZQ_T)
            # dPdMuAO = matrix_transform(ZQ, dPdMuAO, ZQ_T)
            # dPdMuAO = torch.matmul(ZQ, dPdMuAO)
            # dPdMuAO = torch.matmul(dPdMuAO, ZQ_T)
            # Multiply dPdMuAO with the overlap matrix
            dPdMuAOS[:, :] = torch.matmul(dPdMuAO, over)
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            # P1, p1S, dPdMuAOS = matmuls(ZQ, P1, ZQ_T, over, dPdMuAO)
            # Get the diagonal elements of dPdMuAO and p1S only from the core region
            # dPdMuAO_dia[:] = torch.diagonal(dPdMuAOS)
            dPdMuAO_dia.copy_(dPdMuAOS.as_strided((dPdMuAOS.size(0),), (dPdMuAOS.size(1) + 1,)))
            # del dPdMuAOS
            # p1_dia[:] = torch.diagonal(p1S)
            p1_dia.copy_(p1S.as_strided((p1S.size(0),), (p1S.size(1) + 1,)))
            # del p1S
            trP1 = torch.sum(p1_dia[0:norbsInCore])
            trdPdMuAO = torch.sum(dPdMuAO_dia[0:norbsInCore])
            # Compute the chemical potential
            # mu1 = -trP1 / trdPdMuAO if abs(trdPdMuAO) > 1e-12 else 0.0
            mu1 = torch.where(torch.abs(trdPdMuAO) > 1e-12, -trP1 / trdPdMuAO, torch.tensor(0.0, device=device, dtype=dtype))
            # Adjust P1 with the response to get the density matrix
            ptrho[:, :] = 2 * (P1 + mu1 * dPdMuAO)
            ptrho[:, :] = torch.matmul(ptrho, over)
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            # del P1
            # del dPdMuAO
            torch.cuda.synchronize()
            nvtx.push_range("charge from rho", color="yellow", domain="get_kernel_byParts")
            # Get charges from the density matrix
            fullDiag[:] = torch.diagonal(ptrho)
            pt_charges.zero_()
            pt_charges.index_add_(0, pt_indices, fullDiag) # same as code below
            torch.cuda.synchronize()
            nvtx.pop_range("get_kernel_byParts")
            # for j in range(subSy.nats):
            #    pt_charges[j] = 0.0
            #    for jj in range(hindex[j], hindex[j + 1]):
            #        pt_charges[j] = pt_charges[j] + fullDiag[jj]
            # Compute the Jacobian matrix
            Jacobian[:, i] = pt_charges[:numberOfCoreAtoms]
            Jacobian[i, i] -= 1.0
            # chargePertVect[atom_index] = 0.0
            chargePertVect.index_fill_(0, coreindex[i], 0.0)
            # del ptrho, fullDiag, dPdMuAO_dia, p1_dia, trP1, trdPdMuAO, mu1, H1, dPdMu, P1, H_dq_v, coulvsInPart, coulvs, ewald_e
        torch.cuda.synchronize()
        nvtx.push_range("kernel compute", color="orange", domain="get_kernel_byParts")
        # Matrix inversion using PyTorch
        subSy.ker[:, :] = torch.linalg.inv(Jacobian)
        # Rescale summation of each column of the sub kernel to -1 for maintaining charge neutrality
        subSy.ker = subSy.ker / subSy.ker.sum(dim=0, keepdim=True) * -1
        subSy.ker = subSy.ker.to(torch.float64)
        torch.cuda.synchronize()
        nvtx.pop_range("get_kernel_byParts")
        torch.cuda.synchronize()
        nvtx.pop_range("get_kernel_byParts")
    torch.cuda.synchronize()
    nvtx.pop_range("get_kernel_byParts")


def Canon_Response_dPdMu(H1, etemp, evals, mu, m):
    """
    Compute the canonical quantum perturbation and its derivative with respect to the chemical potential.

    Parameters
    ----------
    H1 : 2D torch tensor, dtype: float
        The Hamiltonian matrix in the ortho-eigen basis.
    etemp : float
        The electronic temperature in Kelvin.
    evals : 1D torch tensor, dtype: float
        The eigenvalues of the Hamiltonian matrix, H0.
    mu : float
        Chemical potential for the full system.
    m : int
        The number of recursion steps. 
    
    Returns
    -------
    dPdMu : 1D torch tensor, dtype: float
        The derivative of the density matrix with respect to the chemical potential.
    P1 : 2D torch tensor, dtype: float
        The canonical quantum perturbation.
    """
    kB = 8.61739e-5  # (eV/K)
    h_0 = evals  # Diagonal Hamiltonian H0 represented in the eigenbasis Q
    beta = 1.0 / (kB * etemp)
    cnst = beta / (1.0 * 2**(m + 2))  # Scaling constant
    p_0 = 0.5 - cnst * (h_0 - mu)
    P1 = -cnst * H1

    # Loop over m recursion steps
    for _ in range(m):
        # Compute denominators
        denom_j = 2.0 * p_0 * (p_0 - 1.0) + 1.0
        denom_k = 2.0 * p_0 * (p_0 - 1.0) + 1.0

        # Broadcast p_0 vectors to 2D
        p0_j = p_0.unsqueeze(1)  # shape (HDIM, 1)
        p0_k = p_0.unsqueeze(0)  # shape (1, HDIM)

        denom_j_2D = denom_j.unsqueeze(1)  # shape (HDIM, 1)
        denom_k_2D = denom_k.unsqueeze(0)  # shape (1, HDIM)

        # Compute updated P1
        factor = 1.0 / denom_j_2D
        correction = 2.0 * (P1 - (p0_j + p0_k) * P1) * (1.0 / denom_k_2D) * (p0_k**2)
        P1 = factor * ((p0_j + p0_k) * P1 + correction)

        # Update p_0
        p_0 = (1.0 / (2.0 * (p_0 * p_0 - p_0) + 1.0)) * (p_0 * p_0)

    dPdMu = beta * p_0 * (1.0 - p_0)

    return dPdMu, P1

def apply_kernel_byParts(q_n, n, sdc, rank, numranks, comm, parts, sy, device="cuda", dtype=torch.float64):
    """
    Apply the kernel preconditioner to the residuals between q[n] and n for each subsystem.

    Parameters
    ----------
    q_n : 1D torch tensor, dtype: float
        The charge vector q[n] for the full system.
    n : 1D torch tensor, dtype: float
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
    device : str
        The device to use for computation (e.g., "cuda" or "cpu").
    dtype : torch.dtype
        The data type for the tensors (default is torch.float64).
    
    Returns
    -------
    KK0Res : 1D torch tensor, dtype: float
        The kernel preconditioner applied to the residuals between q[n] and n for each subsystem.
    """
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize KK0ResOnRank as a zero tensor
    KK0ResOnRank = torch.zeros(sy.nats, device=device, dtype=dtype)
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Retrieve q[n] and n charge vectors for current part
        n_InPart = n[parts[partIndex]]
        q_n_InPart = q_n[parts[partIndex]]  
        # Compute the kernel preconditioner applied to the residuals
        KK0ResInPart = torch.matmul(subSy.ker, (q_n_InPart - n_InPart))
        # Expand KK0ResInPart into KK0ResOnRank
        KK0ResOnRank[parts[partIndex]] = KK0ResInPart
    # If MPI is available and there are multiple ranks, collect and sum the KK0ResOnRank tensor
    if is_mpi_available and numranks > 1:
        KK0Res = collect_and_sum_vectors_float(KK0ResOnRank, rank, numranks, comm)
        comm.Barrier() 
    else:
        KK0Res = KK0ResOnRank
    
    return KK0Res


def rankN_update_byParts(
    q_n, n, maxRanks, sdc, rank, numranks, comm, parts, partsCoreHalo, sy, mu=0.0, thresh=1e-2, nbr_inds=None, disps=None, dists=None, alpha=None, PME_data=None, device="cuda"
):
    """
    Perform the rank-N update for the kernel preconditioner and apply it to the residuals.

    Parameters
    ----------
    q_n : 1D torch tensor, dtype: float
        The charge vector q[n] for the full system.
    n : 1D torch tensor, dtype: float
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
    device : str
        The device to use for computation (e.g., "cuda" or "cpu").
    dtype : torch.dtype
        The data type for the tensors (default is torch.float64).
    
    Returns
    -------
    KK0Res : 1D torch tensor, dtype: float
        The kernel preconditioner applied to the preconditioned residuals between q[n] and n for each subsystem.
    """
    dtype=torch.float64
    q_n = q_n.to(device).to(dtype)
    n = n.to(device).to(dtype)
    nvtx.push_range("init", color="blue", domain="rankN_update_byParts")
    # Get the partition indices for the current MPI rank
    partsPerRank = int(sdc.nparts / numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    # Initialize the preconditioned residual vector K0ResOnRank as None
    K0ResOnRank = torch.zeros(sy.nats, device=device, dtype=dtype)
    # Get the maximum number of atoms in the core region among all parts in the present MPI rank 
    maxCoresAmongParts = np.zeros(numranks, dtype=int)
    for partIndex in range(partIndex1, partIndex2):
        numberOfCoreAtoms = len(parts[partIndex])
        maxCoresAmongParts[rank] = max(maxCoresAmongParts[rank], numberOfCoreAtoms) 
    # Initialize K0ResPart to store the preconditioned residuals for each part
    K0ResPart = torch.zeros((int(maxCoresAmongParts[rank]), partsPerRank), device=device, dtype=dtype)
    nvtx.pop_range("rankN_update_byParts")
    nvtx.push_range("K0Res compute", color="green", domain="rankN_update_byParts")
    # Loop over all partitions in the current MPI rank
    for partIndex in range(partIndex1, partIndex2):
        # Get the number of atoms in the core region for the current part
        numberOfCoreAtoms = len(parts[partIndex])
        # Get the subsystem for the current part
        subSy = sy.subSy_list[partIndex - partIndex1]
        # Get the number of atoms in the core+halo region for the current part
        norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
        # Retrieve q[n] and n charge vectors for current part
        q_nInPart = q_n[parts[partIndex]] 
        nInPart = n[parts[partIndex]] 
        # Calculate K0Res which is the product of the Preconditioner K with the residue q(n) - n
        K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1] = torch.matmul(subSy.ker, (q_nInPart - nInPart)) 
        # Expand K0resPart into K0Res
        K0ResOnRank[parts[partIndex]] = K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1]
    nvtx.pop_range("rankN_update_byParts")
    nvtx.push_range("K0Res collect", color="green", domain="rankN_update_byParts")
    # If MPI is available and there are multiple ranks, collect and sum the K0ResOnRank vector
    if is_mpi_available and numranks > 1:
        K0Res = collect_and_sum_vectors_float(K0ResOnRank.cpu().double().numpy(), rank, numranks, comm)
        K0Res = torch.from_numpy(K0Res).to(device).to(dtype)
        maxCoresAmongPartsAndRanks = collect_and_sum_vectors_int(maxCoresAmongParts, rank, numranks, comm)
        comm.Barrier()
    else:
        K0Res = K0ResOnRank
        maxCoresAmongPartsAndRanks = maxCoresAmongParts
    nvtx.pop_range("rankN_update_byParts")
    nvtx.push_range("rankN update init", color="green", domain="rankN_update_byParts")
    # Get the maximum number of atoms in the core region among all parts in all ranks
    maxCoresAmongPartsAndRanks = int(max(maxCoresAmongPartsAndRanks))
    # Initialize directional derivatives (dr) from the preconditioned residuals
    dr = torch.zeros(sy.nats, device=device, dtype=dtype)
    dr[:] = K0Res[:]
    # Initial arrays for the rank updates
    dr_save = torch.zeros((sy.nats, maxRanks), device=device, dtype=dtype)
    v_core_i = torch.zeros((maxCoresAmongPartsAndRanks, partsPerRank, maxRanks), device=device, dtype=dtype)
    c_i = torch.zeros(maxRanks, device=device, dtype=dtype)
    ff = torch.zeros((maxCoresAmongPartsAndRanks, partsPerRank, maxRanks), device=device, dtype=dtype)
    # Convert numpy array to torch tensor
    dtype = torch.float32
    vi = torch.zeros((sy.nats, maxRanks), device=device, dtype=dtype)
    lattice_vecs = torch.from_numpy(sy.latticeVectors).to(device).to(dtype)
    coords = torch.from_numpy(sy.coords).to(device).to(dtype)
    hubbard_u = torch.from_numpy(sy.hubbard_u).to(device).to(dtype)
    atomtypes = torch.from_numpy(sy.types).to(device).to(torch.int64)
    irank = -1
    # Here we enter the loop for the rank updates (do not confuse with MPI rank)
    # for irank in range(maxRanks):
    error = 1.0
    mRank = maxRanks

    nvtx.pop_range("rankN_update_byParts")
    nvtx.push_range("rankN update loop", color="green", domain="rankN_update_byParts")
    while irank < mRank - 1 and error > thresh:
        dtype = torch.float32
        nvtx.push_range(f"Gram-Schmidt", color="purple", domain="rankN_update_byParts")
        irank = irank + 1
        # Construct Krylov subspace vector from previous directional derivative
        vi[:, irank] = dr / torch.linalg.norm(dr)
        # Gram-Schmidt orthogonalization
        if irank > 0:
            for kk in range(irank):
                vi[:, irank] = vi[:, irank] - torch.dot(vi[:, irank], vi[:, kk]) * vi[:, kk]
            # Normalize the vector
            vi[:, irank] = vi[:, irank] / torch.linalg.norm(vi[:, irank])
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("charge perturbation", color="purple", domain="rankN_update_byParts")
        # Get the charge perturbation vector
        chargePertVect = vi[:, irank].clone() # cloning to avoid error in PME calculation below
        # Get the Coulomb potential from charge perturbation vector
        # Note that the Hubbard U correction is included in the computed Coulomb potential
        coulvs, ewald_e, nbr_inds, disps, dists, alpha, PME_data = get_PME_coulvs(
            chargePertVect, hubbard_u, coords, atomtypes, lattice_vecs, nbr_inds=nbr_inds, disps=disps, dists=dists, alpha=alpha, PME_data=PME_data, device=device, use_torch=True, convert=False,
        )
        nvtx.pop_range("rankN_update_byParts")
        # Initialize the core part of the charge response (q1, dqdmu) by the derivative of subsystem
        # density matrix with respect to perturbation parameter (lambda) and chemical potential (mu).
        q1 = torch.zeros((maxCoresAmongPartsAndRanks, partsPerRank), device=device, dtype=dtype)
        dqdmu = torch.zeros((maxCoresAmongPartsAndRanks, partsPerRank), device=device, dtype=dtype)
        # Initialize the variables to sum up the partial traces
        trP1 = torch.zeros(1, device=device, dtype=dtype)
        trdPdMu = torch.zeros(1, device=device, dtype=dtype)
        nvtx.push_range("loop over parts", color="purple", domain="rankN_update_byParts")
        for partIndex in range(partIndex1, partIndex2):
            torch.cuda.synchronize()
            nvtx.push_range(f"part {partIndex} init", color="orange", domain="rankN_update_byParts")
            numberofCoreHaloAtoms = len(partsCoreHalo[partIndex])
            numberOfCoreAtoms = len(parts[partIndex])
            subSy = sy.subSy_list[partIndex - partIndex1]
            assert numberofCoreHaloAtoms == subSy.nats, "Number of atoms in the core+halo region should be equal to the number of atoms in the subsystem"
            norbs = subSy.norbs  # We have as many orbitals as columns in the Hamiltonian
            # Get the overlap matrix 
            over = torch.from_numpy(subSy.over).to(device).to(dtype)
            # Get the Z matrix
            zmat = torch.from_numpy(subSy.zmat).to(device).to(dtype)
            # Get the eigenvectors 
            evects = torch.from_numpy(subSy.evects).to(device).to(dtype)
            # Get the eigenvalues
            evals = torch.from_numpy(subSy.evals).to(device).to(dtype)
            # Get the Coulomb potential and charges for the Core+Halo part 
            # coulvsInPart = torch.from_numpy(coulvs[partsCoreHalo[partIndex]]).to(device).to(dtype)
            corehaloindex = torch.tensor(partsCoreHalo[partIndex]).to(device).to(torch.int64)
            coulvsInPart = coulvs[corehaloindex]
            # Get the hindex 
            hindex = torch.from_numpy(subSy.hindex).to(device).to(torch.int64)
            # Extract the perturbation over the core part only
            v_core_i[0:numberOfCoreAtoms, partIndex - partIndex1, irank] = vi[parts[partIndex], irank]
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("build H_dq_v", color="orange", domain="rankN_update_byParts")
            # Build the Hamiltonian from Coulomb potential and charges from charge perturbation
            H_dq_v = torch.zeros((norbs, norbs), device=device, dtype=dtype)

            indices = torch.arange(hindex[0], hindex[-1], device=device, dtype=torch.int64)
            H_dq_v[indices, indices] = torch.repeat_interleave(coulvsInPart, torch.diff(hindex)) # same as code below
            # for j in range(subSy.nats):
            #     start = subSy.hindex[j]
            #     end = subSy.hindex[j + 1]
            #     H_dq_v[start:end, start:end] = np.diag(coulvsInPart[j] * np.ones(end - start))
            H_dq_v = 0.5 * (torch.matmul(over, H_dq_v) + torch.matmul(H_dq_v, over))
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("H forward transform", color="orange", domain="rankN_update_byParts")
            # H1 = Q'Z'*H_dq_v*ZQ  Forward transform
            # Compute transformations ZQ and (ZQ)^t transformation that takes from the canonical nonorthogonal
            # to the orthogonal eigenbasis.
            ZQ = torch.matmul(zmat, evects)
            ZQ_T = ZQ.T
            # Take H1 to the ortho-eigen basis set.
            H1 = torch.matmul(torch.matmul(ZQ_T, H_dq_v), ZQ)
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("canonical response", color="orange", domain="rankN_update_byParts")
            # Construct the "bare" response P1 and the derivative with respect to the
            # chemical potential (dPdMu). Everything in the ortho-eigen basis set
            dPdMu, P1 = Canon_Response_dPdMu(H1, sdc.etemp, evals, mu, 12)
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("rho compute", color="orange", domain="rankN_update_byParts")
            # Transform P1 back to the nonortho-canonical basis set.
            # P1 = torch.matmul(torch.matmul(ZQ, P1), ZQ_T)
            # P1 = matrix_transform(ZQ, P1, ZQ_T)
            P1 = torch.matmul(ZQ, P1)
            P1 = torch.matmul(P1, ZQ_T)
            # Convert dPdMu to matrix
            dPdMuAO = torch.zeros((norbs, norbs), device=device, dtype=dtype)
            # dPdMuAO[torch.arange(norbs), torch.arange(norbs)] = dPdMu
            dPdMuAO.diagonal().copy_(dPdMu)
            # Transform dPdMu back to the nonortho-canonical basis set
            # dPdMuAO = torch.matmul(torch.matmul(ZQ, dPdMuAO), ZQ_T)
            # dPdMuAO = matrix_transform(ZQ, dPdMuAO, ZQ_T)
            dPdMuAO = torch.matmul(ZQ, dPdMuAO)
            dPdMuAO = torch.matmul(dPdMuAO, ZQ_T)
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            torch.cuda.synchronize()
            nvtx.push_range("charge from rho", color="orange", domain="rankN_update_byParts")
            # Here we compute the charges response (q1) from P1 and we store it on 
            # a vector q1 that stores all the previous q1s from past iranks iterations
            # We also compute the partial trace contribution (trP1) from this mpi
            # execution and the current part (partIndex).
            P1 = 2 * P1
            ptrho = torch.matmul(P1, over)
            fullDiag = torch.diag(ptrho)
            pt_charges = torch.zeros(numberofCoreHaloAtoms, device=device, dtype=dtype)

            indices = torch.repeat_interleave(
                torch.arange(subSy.nats, device=device),
                torch.diff(hindex[:numberofCoreHaloAtoms + 1])
            )
            pt_charges.index_add_(0, indices, fullDiag) # same as code below
            # for j in range(numberofCoreHaloAtoms):
            #     pt_charges[j] = 0.0
            #     for jj in range(hindex[j], hindex[j + 1]):
            #         pt_charges[j] = pt_charges[j] + fullDiag[jj]

            # Collect the charge response from the core region
            q1[:numberOfCoreAtoms, partIndex - partIndex1] = pt_charges[:numberOfCoreAtoms]
            # Add up the partial trace contribution from the core region
            trP1 = trP1 + torch.sum(pt_charges[:numberOfCoreAtoms])

            # Here we compute the charges response (dqdmu) from dPdMu and we store
            # them on a matrix dqdmu that stores all the previous dqdmus from past
            # irank iterations.
            # We also compute the partial trace contribution (trdPdMu) from this node
            # and the current part (partIndex).
            dPdMuAO = 2 * dPdMuAO
            ptrho = torch.matmul(dPdMuAO, over)
            fullDiag = torch.diag(ptrho)
            pt_charges = torch.zeros(numberofCoreHaloAtoms, device=device, dtype=dtype)
            
            indices = torch.repeat_interleave(
                torch.arange(subSy.nats, device=device),
                torch.diff(hindex[:numberofCoreHaloAtoms + 1])
            )
            pt_charges.index_add_(0, indices, fullDiag) # same as code below
            # for j in range(numberofCoreHaloAtoms):
            #     pt_charges[j] = 0.0
            #     for jj in range(hindex[j], hindex[j + 1]):
            #         pt_charges[j] = pt_charges[j] + fullDiag[jj]
            # Collect the charge response from the core region
            dqdmu[:numberOfCoreAtoms, partIndex - partIndex1] = pt_charges[:numberOfCoreAtoms]
            # Add up the partial trace contribution from the core region
            trdPdMu = trdPdMu + torch.sum(pt_charges[:numberOfCoreAtoms])
            torch.cuda.synchronize()
            nvtx.pop_range("rankN_update_byParts")
            # gc.collect()
            # torch.cuda.empty_cache()
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("P1 trdPdMu collect", color="purple", domain="rankN_update_byParts")
        # If MPI is available and there are multiple ranks, collect and sum the partial traces
        if is_mpi_available and numranks > 1:
            trP1 = collect_and_sum_vectors_float(trP1.cpu().double().numpy(), rank, numranks, comm)
            trP1 = torch.from_numpy(trP1).to(device).to(dtype)
            trdPdMu = collect_and_sum_vectors_float(trdPdMu.cpu().double().numpy(), rank, numranks, comm)
            trdPdMu = torch.from_numpy(trdPdMu).to(device).to(dtype)
            comm.Barrier()
        nvtx.pop_range("rankN_update_byParts")
        # Compute the response to the chemical potential (mu1) and adjust q1
        mu1_Global = - trP1 / trdPdMu if abs(trdPdMu) > 1e-12 else 0.0
        
        q1 = q1 + mu1_Global * dqdmu
        dtype = torch.float64
        q1 = q1.to(dtype)
        # Initialize f to store directional derivatives of the residual function
        f = torch.zeros(maxCoresAmongPartsAndRanks, device=device, dtype=dtype)
        # Initialize dr to store the preconditioned directional derivatives
        #dr[:] = 0.0
        dr = torch.zeros(sy.nats, device=device, dtype=dtype)
        c_i_temp = torch.zeros(1, device=device, dtype=dtype)
        nvtx.push_range("loop over parts", color="purple", domain="rankN_update_byParts")
        # Loop over all partitions in the current MPI rank
        for partIndex in range(partIndex1, partIndex2):
            # Get the number of atoms in the core region for the current part
            numberOfCoreAtoms = len(parts[partIndex])
            # Get the subsystem object for the current part
            subSy = sy.subSy_list[partIndex - partIndex1]
            assert numberOfCoreAtoms == subSy.ker.shape[0], "Number of atoms in the core should be equal to the number of atoms in the kernel"
            # Compute the directional derivative of the residual function and store it in f
            f[0:numberOfCoreAtoms] = q1[0:numberOfCoreAtoms, partIndex - partIndex1] - v_core_i[0:numberOfCoreAtoms, partIndex - partIndex1, irank]
            # Compute the preconditioned directional derivative of the residual function
            ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank] = torch.matmul(subSy.ker, f[0:numberOfCoreAtoms])
            # Preconditioned Krylov subspace approximation
            c_i_temp = c_i_temp + torch.dot(ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank], K0ResPart[0:numberOfCoreAtoms, partIndex - partIndex1])
            # Save the preconditioned directional derivative of the residual function for getting the resolution of identity
            dr[parts[partIndex]] = ff[0:numberOfCoreAtoms, partIndex - partIndex1, irank]
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("dr, c_i collect", color="purple", domain="rankN_update_byParts")
        # If MPI is available and there are multiple ranks, collect and sum the preconditioned directional derivative
        if is_mpi_available and numranks > 1:
            dr = collect_and_sum_vectors_float(dr.cpu().double().numpy(), rank, numranks, comm)
            dr = torch.from_numpy(dr).to(device).to(dtype)
            comm.Barrier()
        nvtx.pop_range("rankN_update_byParts")
        # Save dr for the current rank update
        dr_save[:, irank] = dr
        dr = dr.to(torch.float32)

        maxRanks = irank + 1
        nvtx.push_range("c_i collect", color="purple", domain="rankN_update_byParts")
        # If MPI is available and there are multiple ranks, collect and sum the preconditioned Krylov subspace approximation
        if is_mpi_available and numranks > 1:
            c_i_temp = collect_and_sum_vectors_float(c_i_temp.cpu().double().numpy(), rank, numranks, comm)
            c_i_temp = torch.from_numpy(c_i_temp).to(device).to(dtype)
            comm.Barrier()
        c_i[irank] = c_i_temp
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("overlap matrix compute", color="purple", domain="rankN_update_byParts")
        # Compute elements of the overlap matrix
        auxVect = torch.zeros(maxRanks * maxRanks, device=device, dtype=dtype)
        for i in range(maxRanks):
            for j in range(maxRanks):
                for k in range(partsPerRank):
                    auxVect[i * maxRanks + j] = auxVect[i * maxRanks + j] + torch.dot(ff[:, k, i], ff[:, k, j])
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("overlap matrix collect", color="purple", domain="rankN_update_byParts")
        # If MPI is available and there are multiple ranks, collect and sum the elements of the overlap matrix
        if is_mpi_available and numranks > 1:
            auxVect = collect_and_sum_vectors_float(auxVect.cpu().double().numpy(), rank, numranks, comm)
            auxVect = torch.from_numpy(auxVect).to(device).to(dtype)
            comm.Barrier()
        nvtx.pop_range("rankN_update_byParts")
        nvtx.push_range("resolution of identity", color="purple", domain="rankN_update_byParts")
        # Reshape the auxVect to get the overlap matrix
        oij = torch.zeros((maxRanks, maxRanks), device=device, dtype=dtype)
        mMat = torch.zeros((maxRanks, maxRanks), device=device, dtype=dtype)
        oij[:, :] = auxVect.reshape((maxRanks, maxRanks))
        # Compute the inverse of the overlap matrix
        mMat[:, :] = torch.linalg.inv(oij)
        # Compute the resolution of identity
        KK0Res = torch.zeros(sy.nats, device=device, dtype=dtype)
        IdK0Res = torch.zeros(sy.nats, device=device, dtype=dtype)
        for i in range(maxRanks):
            for j in range(maxRanks):
                KK0Res = KK0Res + vi[:, i] * mMat[i, j] * c_i[j]
                IdK0Res = IdK0Res + dr_save[:, i] * mMat[i, j] * c_i[j]
        nvtx.pop_range("rankN_update_byParts")
        error = torch.linalg.norm(K0Res - IdK0Res) / torch.linalg.norm(K0Res)
        if rank == 0:
            print("Error Rank-Update", error.item(), irank, "\n")
    nvtx.pop_range("rankN_update_byParts")
    return KK0Res




