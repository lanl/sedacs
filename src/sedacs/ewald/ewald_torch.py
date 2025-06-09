import math
import torch
import numpy as np
import time
from typing import Optional, Tuple
from .util import CONV_FACTOR

@torch.compile  
def ewald_real(
    nbr_inds: torch.Tensor,
    nbr_diff_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the real-space contribution to the Ewald summation.

    This function calculates the electrostatic interaction energy in the real-space 
    portion of the Ewald summation. It also optionally computes forces and derivatives 
    with respect to charge if `calculate_forces` or `calculate_dq` are set.

    Args:
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `N` is the number of local atoms.
            - `K` is the maximum number of neighbors per atom.
        nbr_diff_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)`, where:
            - `3` represents the x, y, and z components of the displacement.
            - `N` is the number of local atoms.
            - `K` is the number of neighbors per atom.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]: 
            - **(float)** Real-space energy contribution (scalar).
            - **(torch.Tensor, shape `(3, N)`)** Forces on atoms if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(N,)`)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """
    # TODO: finalize DUMMY_ATOM_IND
    DUMMY_NBR_IND = -1
    N = len(charges)
    q_sq = charges[nbr_inds] * charges[:, None]
    qq_over_dist = q_sq / nbr_dists
    qq_over_dist = qq_over_dist * ((nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff))
    erfc = torch.erfc(alpha * nbr_dists)
    res = erfc * qq_over_dist
    #de_dq = erfc * charges[nbr_inds] * (nbr_inds != DUMMY_NBR_IND) / nbr_dists
    #de_dq = torch.sum(de_dq, dim=1)
    if calculate_forces:
        nbr_dists_sq = nbr_dists**2
        f = qq_over_dist * (erfc / nbr_dists_sq
                            + (2.0 * alpha / math.sqrt(torch.pi)) 
                            * torch.exp(-alpha * alpha * nbr_dists_sq) / nbr_dists)
        f = -1.0 * torch.sum(f[None, ...] * nbr_diff_vecs, dim=2)
    else:
        f = None
    
    if calculate_dq:
        de_dq = erfc * charges[nbr_inds] * (nbr_inds != DUMMY_NBR_IND) / nbr_dists
        de_dq = torch.sum(de_dq, dim=1)
    else:
        de_dq = None
        
    return torch.sum(res) / 2.0, f, de_dq

@torch.compile
def ewald_real_screening(nbr_inds, nbr_diff_vecs, nbr_dists, charges, hubbard_u, atomtypes,
               alpha: float, cutoff: float, calculate_forces: int,
               calculate_dq: int):
    """
    Computes the real-space contribution with the Hubbard-U screening correction to the Ewald summation.

    This function calculates the electrostatic interaction energy in the real-space 
    portion of the Ewald summation. It also optionally computes forces and derivatives 
    with respect to charge if `calculate_forces` or `calculate_dq` are set.

    Args:
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `N` is the number of local atoms.
            - `K` is the maximum number of neighbors per atom.
        nbr_diff_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)`, where:
            - `3` represents the x, y, and z components of the displacement.
            - `N` is the number of local atoms.
            - `K` is the number of neighbors per atom.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        hubbard_u (torch.Tensor): Hubbard U values for each atom. Shape: `(N,)`.
        atomtypes (torch.Tensor): Atomic types for each atom. Shape: `(N,)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]: 
            - **(float)** Real-space energy contribution (scalar).
            - **(torch.Tensor, shape `(3, N)`)** Forces on atoms if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(N,)`)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """
    # TODO: finalize DUMMY_ATOM_IND
    KECONST = 14.3996437701414
    device = nbr_dists.device
    dtype = nbr_dists.dtype

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    DUMMY_NBR_IND = -1
    # symbols = torch.Tensor(sy.symbols)[atomtypes]
    mask = ((nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff))
    same_element_mask = mask & (atomtypes.unsqueeze(1) == atomtypes[nbr_inds])  # (Nr_atoms, Max_Nr_Neigh)
    different_element_mask = mask & ~same_element_mask
    
    TFACT  = 16.0 / (5.0 * KECONST)
    #TI = TFACT * U.unsqueeze(1) * mask # (Nr_atoms, Max_Nr_Neigh)
    TI = torch.where(mask, TFACT * hubbard_u.unsqueeze(1) * mask, one)
    TI2 = TI * TI
    TI3 = TI2 * TI
    TI4 = TI2 * TI2
    TI6 = TI4 * TI2

    SSA = TI
    SSB = TI3 / 48.0
    SSC = 3.0 * TI2 / 16.0
    SSD = 11.0 * TI / 16.0
    SSE = 1.0

    MAGR = torch.where(mask, nbr_dists, one)
    MAGR2 = MAGR * MAGR
    Z = abs(alpha * MAGR)
    NUMREP_ERFC = torch.special.erfc(Z)
   
    J0 = torch.where(mask, NUMREP_ERFC / MAGR, zero)

    EXPTI = torch.exp(-TI * MAGR)

    J0[same_element_mask] = J0[same_element_mask] - (EXPTI * \
                    (SSB * MAGR2 + SSC * MAGR + SSD + SSE / MAGR))[same_element_mask]

    #TJ = TFACT * U[nbr_inds] * different_element_mask     # (Nr_atoms, Max_Nr_Neigh)
    TJ = torch.where(different_element_mask, TFACT * hubbard_u[nbr_inds] * different_element_mask, one)
    TJ2 = TJ * TJ
    TJ4 = TJ2 * TJ2
    TJ6 = TJ4 * TJ2
    EXPTJ = torch.exp(-TJ * MAGR) 
    TI2MTJ2 = TI2 - TJ2
    TI2MTJ2 = torch.where(different_element_mask, TI2MTJ2, one)
    SA = TI
    SB = EXPTI * TJ4 * TI / 2.0 / TI2MTJ2 / TI2MTJ2 
    SC = EXPTI * (TJ6 - 3.0 * TJ4 * TI2) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    SD = TJ
    SE = EXPTJ * TI4 * TJ / 2.0 / TI2MTJ2 / TI2MTJ2 
    SF = EXPTJ * (-(TI6 - 3.0 * TI4 * TJ2)) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2 
    J0[different_element_mask] = J0[different_element_mask] - (1.0 * (SB - SC / MAGR) + \
                1.0 * (SE - SF / MAGR))[different_element_mask]

    energy = charges[:, None] * J0 * charges[nbr_inds]

    if calculate_forces:
        nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 2).contiguous()
        nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 1).contiguous()
        alpha2 = alpha * alpha 
        DC = torch.where(mask.unsqueeze(2), nbr_diff_vecs / nbr_dists.unsqueeze(2), zero)
        CA = torch.where(mask, NUMREP_ERFC / MAGR, zero) 
        CA = CA + 2.0 * alpha * torch.exp(-alpha2 * MAGR2) / math.sqrt(math.pi)
        FORCE = -torch.sum((charges[:, None] * charges[nbr_inds] * \
                 torch.where(mask, CA / MAGR, zero)).unsqueeze(2) * \
                 DC * mask.unsqueeze(2), dim=1)

        FORCE = FORCE + torch.sum(((charges[:, None] * charges[nbr_inds] * EXPTI) * \
                    ((torch.where(same_element_mask, SSE / MAGR2, zero) - 2.0 * SSB * MAGR - SSC) \
                    + SSA * (SSB * MAGR2 + SSC * MAGR + SSD + torch.where(same_element_mask, SSE / MAGR, zero)))).unsqueeze(2) * \
                    DC * same_element_mask.unsqueeze(2), dim=1)
        FORCE = FORCE + torch.sum((charges[:, None] * charges[nbr_inds] * ((1.0 * (SA * (SB - torch.where(different_element_mask, SC / MAGR, zero)) - \
            torch.where(different_element_mask, SC / MAGR2, zero))) + (1.0 * (SD * (SE - torch.where(different_element_mask, SF / MAGR, zero)) - \
            torch.where(different_element_mask, SF / MAGR2, zero))))).unsqueeze(2) * DC * different_element_mask.unsqueeze(2), dim=1)

    else:
        FORCE = None

    if calculate_dq:
        COULOMBV = torch.sum(J0 * charges[nbr_inds], dim=1) 
    else:
        COULOMBV = None

    return torch.sum(energy) / 2.0, FORCE, COULOMBV



def ewald_real_matrix(my_inds, nbr_inds, nbr_diff_vecs, nbr_dists, charges, alpha: float):
    # TODO: finalize DUMMY_ATOM_IND
    DUMMY_NBR_IND = -1
    N = len(charges)
    q_sq = charges[nbr_inds] * charges[my_inds, None]
    qq_over_dist = q_sq / nbr_dists
    qq_over_dist = qq_over_dist * (nbr_inds != DUMMY_NBR_IND)
    erfc = torch.erfc(alpha * nbr_dists)
    res = erfc * qq_over_dist
    A = torch.vmap(lambda j: torch.vmap(lambda sub_res, sub_inds: torch.sum(sub_res * (sub_inds == j)))(res, nbr_inds))(torch.arange(N))
    # de/dq = erfc * charges[nbr_inds]/dist (double check, this is needed for the solver)
    return A

def ewald_kspace_matrix():
    pass

def ewald_self_energy(
    charges: torch.Tensor, 
    alpha: float, 
    calculate_dq: int = 0
) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Computes the self-energy contribution in the Ewald summation.

    The self-energy term accounts for the interaction of each charge with its own 
    periodic images in an Ewald summation.

    Args:
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`, where `N` is the number of atoms.
        alpha (float): Ewald screening parameter (scalar).
        calculate_dq (int, optional): Flag to compute charge derivatives. 
            - `1`: Compute derivatives (`dq`).
            - `0`: Do not compute (`dq` is `None`).
            Defaults to `0`.

    Returns:
        Tuple[float, Optional[torch.Tensor]]: 
            - Self-energy contribution (scalar, `float`).
            - Charge derivatives (`torch.Tensor` of shape `(N,)`) if `calculate_dq` is enabled, otherwise `None`.
    """
    en = -1.0 * alpha * torch.sum(charges**2) / math.sqrt(torch.pi)
    dq = None
    if calculate_dq == 1:
        dq = -2.0 * alpha * charges / math.sqrt(torch.pi)
    return en, dq



@torch.compile
def ewald_kspace_part1(positions, charges, kvecs):
    '''
    Part 1 of the ewald sum. Calculate intermediate values to share with other processes
    for reduction.
    '''
    # mmul is M x N, M: # kvectors and N: number of (local) atoms
    mmul = kvecs @ positions
    r_vals = torch.cos(mmul) * charges
    i_vals = torch.sin(mmul) * charges
    return r_vals, i_vals

@torch.compile
def ewald_kspace_part2(
    sum_r: torch.Tensor,
    sum_i: torch.Tensor,
    r_vals: torch.Tensor,
    i_vals: torch.Tensor,
    vol: float,
    kvecs: torch.Tensor,
    I: torch.Tensor,
    charges: torch.Tensor,
    positions: torch.Tensor,
    calculate_forces: int,
    calculate_dq: int
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the reciprocal-space contribution to the Ewald summation.

    This function calculates the electrostatic interaction energy in reciprocal space, 
    as well as optional force and charge derivative calculations.

    Args:
        sum_r (torch.Tensor): Real part of the structure factor sum. Shape: `(K,)`, where:
            - `K` is the number of k-vectors.
        sum_i (torch.Tensor): Imaginary part of the structure factor sum. Shape: `(K,)`.
        r_vals (torch.Tensor): Real part of exponential terms per atom. Shape: `(K, n)`, where:
            - `n` is the number of local atoms.
        i_vals (torch.Tensor): Imaginary part of exponential terms per atom. Shape: `(K, n)`.
        vol (float): Volume of the simulation box.
        kvecs (torch.Tensor): Reciprocal space vectors. Shape: `(K, 3)`, where:
            - `3` represents the x, y, and z components of each k-vector.
        I (torch.Tensor): Fourier-space prefactors. Shape: `(K,)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`, where:
            - `N` is the total number of atoms.
        positions (torch.Tensor): Atomic positions. Shape: `(3, N)`.
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False`).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Total k-space energy contribution.
            - **(torch.Tensor, shape `(3, n)`, optional)** Computed forces if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(n,)`, optional)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """

    # sum_r, sum_i: [M,]
    abs_fac_sq = sum_r**2 + sum_i**2
    '''
    dE/dq = (2 * 2 * pi * I * sum_r).reshape(-1, M) @ torch.cos(mmul)/vol
          + (2 * 2 * pi * I * sum_i).reshape(-1, M) @ torch.sin(mmul)/vol 
    We either need to return cos(mmul) and sin(mmul), or do a division to restore the values
    One potential issue: this can cause numerical issues 
    # if torch.cos(mmul) * charges / charges != torch.cos(mmul) (in terms of num. precision)
    '''
    #de_dq = (4.0 * torch.pi * I * sum_r).reshape(-1, M) @ (r_vals / charges)/vol
    #de_dq += (4.0 * torch.pi * I * sum_i).reshape(-1, M) @ (i_vals / charges)/vol

    if calculate_forces:
        # local N
        N = r_vals.shape[1]
        # cos_sin_ln is M x N
        cos_sin_ln = (r_vals * sum_i.reshape(-1,1))
        sin_cos_ln = (i_vals * sum_r.reshape(-1,1))

        prefac_ln = (I.reshape(-1,1) * (cos_sin_ln - sin_cos_ln))
        # convert from Nx3 to 3xN
        f_nc = torch.sum(kvecs.reshape(-1, 1, 3) * prefac_ln.reshape(-1, N, 1), dim=0).T
        forces = -4 * torch.pi * f_nc / vol
    else:
        forces = None
    
    if calculate_dq:
        M = len(abs_fac_sq)
        charges = charges.reshape(1, -1)
        # TODO: find better solution to this, 
        # fix zero charge issue
        charges = torch.where(charges != 0, charges, 1.0)
        de_dq = (4.0 * torch.pi * I * sum_r).reshape(-1, M) @ (r_vals / charges)/vol
        de_dq += (4.0 * torch.pi * I * sum_i).reshape(-1, M) @ (i_vals / charges)/vol
        de_dq = de_dq.flatten()
    else:
        de_dq = None, 
    return 2 * torch.pi * torch.sum(I * abs_fac_sq) / vol, forces, de_dq

def construct_kspace(cell, kcounts, cutoff, alpha, transpose_kvec=False):
    '''
    k-vectors: Mx3
    '''
    nx = torch.arange(-kcounts[0], kcounts[0] + 1)
    ny = torch.arange(-kcounts[1], kcounts[1] + 1)
    nz = torch.arange(-kcounts[2], kcounts[2] + 1)
    n_lc = torch.stack(torch.meshgrid(nx, ny, nz, indexing='xy'))
    n_lc = n_lc.permute(*torch.arange(n_lc.ndim - 1, -1, -1))
    n_lc = n_lc.reshape(-1,3).to(cell.device)
    k_lc = 2 * torch.pi * torch.matmul(torch.linalg.inv(cell), n_lc.T.type(cell.dtype)).T
    k = torch.linalg.norm(k_lc, dim=1)
    mask = torch.logical_and(k <= cutoff, k != 0)

    kvecs = k_lc[mask]
    if transpose_kvec:
        kvecs = kvecs.T.contiguous()

    return torch.exp(-((k[mask] / (2 * alpha)) ** 2)) / k[mask] ** 2, kvecs

@torch.compile
def ewald_kspace(positions, charges, vol, kvecs, I, calculate_forces=0, calculate_dq=0):
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, charges, kvecs)
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)
    en, out_f, out_dq =  ewald_kspace_part2(r_sum, i_sum, my_r_vals, my_i_vals, vol, kvecs, I,
                        charges, positions, calculate_forces, calculate_dq) 
    return en, out_f, out_dq

def ewald_benchmark(positions, charges, nbr_inds, nbr_disp_vecs, nbr_dists, alpha, 
                    cutoff,
                    vol, kvecs, I, calculate_forces=1, calculate_dq=0):
    
    my_e_real, my_f_real, my_dq_real = ewald_real(0, len(charges), nbr_inds, nbr_disp_vecs, 
                                                  nbr_dists, charges, alpha,
                                                  cutoff, 
                                                  calculate_forces, calculate_dq)
    my_charges = charges[0:0+len(charges)]
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, my_charges, kvecs)
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)
    en, out_f, out_dq =  ewald_kspace_part2(r_sum, i_sum, my_r_vals, my_i_vals, vol, kvecs, I,
                        my_charges, positions, calculate_forces, calculate_dq) 
    
    if calculate_forces:
        out_f = my_f_real + out_f
    if calculate_dq:
        out_dq = my_dq_real + out_dq 
    return en + my_e_real, out_f, out_dq

def ewald_energy(
    positions: torch.Tensor,
    cell: torch.Tensor,
    nbr_inds: torch.Tensor,
    nbr_disp_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    kvecs: torch.Tensor,
    I: torch.Tensor,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int = 0,
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the Ewald sum energy and forces in a distributed way.

    This function calculates the real-space and reciprocal-space contributions to the 
    Ewald summation, including optional force and charge derivative calculations.

    The computed forces will have the same shape as the positions

    Args:
        positions (torch.Tensor): Atomic positions. Shape: `(3, N)` or `(3, N)`, where:
            - `N` is the total number of atoms.
            - `3` represents x, y, z coordinates.
        cell (torch.Tensor): Simulation cell matrix. Shape: `(3, 3)`.
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `K` is the max number of neighbors per atom.
        nbr_disp_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)` or `(N, K, 3)`.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        charges (torch.Tensor): Charge per atom. Shape: `(N,)`.
        kvecs (torch.Tensor): Reciprocal space vectors. Shape: `(M, 3)`, where:
            - `M` is the number of k-space vectors.
        I (torch.Tensor): Fourier-space prefactors. Shape: `(M,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for real-space interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int, optional): Flag to compute charge derivatives (`1` for True, `0` for False`). Defaults to `0`.
    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **total_ewald_e (float)**: Total ewald energy.
            - **forces (torch.Tensor, shape `(3, N)`, optional)**: Computed forces if `calculate_forces` is enabled, otherwise `None`.
                If the positions are provided as `(N, 3)`, the forces will be also  `(N, 3)`.
            - **dq (torch.Tensor, shape `(N,)`, optional)**: Charge derivatives if `calculate_dq` is enabled, otherwise `None`.

    """
    # As the internal functions expects (3, N), transpose the position tensor as needed
    transpose = False
    if positions.shape[1] == 3:
        transpose = True
        positions = positions.T.contiguous()

    # transpose the disp. vectors as needed
    if nbr_disp_vecs.shape[2] == 3:
        nbr_disp_vecs = nbr_disp_vecs.permute(2, 0, 1).contiguous()

    device = positions.device
    N = positions.shape[1]
    my_e_real, my_f_real, my_dq_real = ewald_real(nbr_inds, nbr_disp_vecs, 
                                                nbr_dists, charges, alpha, 
                                                cutoff,
                                                calculate_forces, calculate_dq)
    vol = torch.det(cell)
    alpha = torch.tensor(alpha)
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, charges, kvecs) 
    # size K vectors
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)

    total_e_kspace, my_f_kspace, my_dq_kspace = ewald_kspace_part2(r_sum, i_sum, my_r_vals, my_i_vals, 
                                                    vol, kvecs, I, 
                                                    charges,
                                                    positions,
                                                    calculate_forces,
                                                    calculate_dq)
    
    self_e, self_dq = ewald_self_energy(charges, alpha, calculate_dq)

    if calculate_forces:
        forces = (my_f_real + my_f_kspace) * CONV_FACTOR
    else:
        forces = None
    if calculate_dq:
        dq = (my_dq_kspace + my_dq_real + self_dq) * CONV_FACTOR
    else:
        dq = None
    
    total_ewald_e = (my_e_real + total_e_kspace + self_e) * CONV_FACTOR

    # if user provided [N,3] positions, tranpose the forces to [N, 3]
    if transpose and calculate_forces:
        forces = forces.T.contiguous()

    return total_ewald_e, forces, dq

