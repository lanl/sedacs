import math
import numpy as np
import torch
from typing import Callable, Optional, Tuple, List

__all__ = ["calculate_num_kvecs_dynamic", "calculate_num_kvecs_ch_indep", 
           "determine_alpha", "determine_alpha_ch_indep", "calculate_alpha_and_num_grids",
           "CG"]

# Charges q are expressed as multiples of the elementary charge e: q = x*e
# e^2/(4*pi*epsilon0) = 14.399645 eV * Angström
CONV_FACTOR = 14.399645

@torch.compile
def mixed_precision_sum(data, dim=None):
    '''
    Util function for mixed precision summation (with double accumulation)
    '''
    return torch.sum(data, dtype=torch.float64, dim=dim).to(data.dtype)

@torch.compile
def __CG_update(x, r, p, q, rho_cur):
    alpha = rho_cur / torch.dot(p, q)
    x += alpha * p
    r -= alpha * q
    return x, r

def CG(mv: Callable[[torch.Tensor], torch.Tensor],
       b: torch.Tensor,
       x0: Optional[torch.Tensor] = None,
       max_iter: Optional[int] = None,
       rtol: float = 1e-5,
       atol: float = 0.0,
       M: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
       ) -> Tuple[torch.Tensor, int]:
    """
    Conjugate Gradient (CG) solver using a pure PyTorch implementation.

    This function solves the linear system Ax = b for x using the CG method,
    where A is represented implicitly via the matrix-vector product function `mv`.
    The preconditioner `M` is optional.

    Args:
        mv (Callable[[torch.Tensor], torch.Tensor]): Function that computes matrix-vector product Ax.
        b (torch.Tensor): Right-hand side vector.
        x0 (Optional[torch.Tensor], optional): Initial guess for the solution. Defaults to None.
        max_iter (Optional[int], optional): Maximum number of iterations. Defaults to len(b).
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 0.0.
        M (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): Preconditioner function. Defaults to identity.

    Returns:
        Tuple[torch.Tensor, int]: Approximate solution `x` and number of matrix-vector multiplications performed.
    """
    dtype = b.dtype
    b = b.to(torch.float64)
    matvec_count = 0
    if M == None:
        M_func = lambda x: x
    else:
        M_func = M
    atol2 = max(atol * atol, (rtol * rtol) * torch.dot(b, b).item())
    if max_iter is None:
        max_iter = len(b)
    if x0 is None:
        x = torch.zeros_like(b, dtype=dtype)
        r = b.clone()
    else:
        r = b - mv(x0.to(dtype)).to(torch.float64)
        x = x0
        matvec_count += 1
    rho_prev, p = None, None
    for i in range(max_iter):
        rdot = torch.dot(r, r).item()
        if rdot <= atol2:
            return x.to(dtype), matvec_count
        z = M_func(r)
        if M is None:
            rho_cur = rdot
        else:
            rho_cur = torch.dot(r, z).item()
        if i > 0:
            beta = rho_cur / rho_prev
            p = p * beta + z
        else:
            p = z.clone()
        
        q = mv(p.to(dtype)).to(torch.float64)
        matvec_count += 1
        '''
        alpha = rho_cur / torch.dot(p, q).item()
        x += alpha * p
        r -= alpha * q
        '''
        x, r = __CG_update(x, r, p, q, rho_cur)
        rho_prev = rho_cur

    return x.to(dtype), matvec_count


def has_only_prime_factors_2_3_5(num):
    # Check if a number has only 2, 3, and 5 as prime factors
    for prime in [2, 3, 5]:
        while num % prime == 0:
            num //= prime
    return num == 1

def find_available_number(n):
    '''
    Find the next available number that has only 2,3 and 5 as prime factors
    FFT is supposed to work better when the problem can be split into these factors.
    TODO: This should be tested, may not make any difference with torch.fft
    '''
    i = n
    while not has_only_prime_factors_2_3_5(i):
        i += 1
    return i

def calculate_alpha_and_num_grids(
    cell: np.ndarray, cutoff: float, t_err: float
) -> Tuple[float, List[int]]:
    """
    Compute alpha and the number of grid points based on the heuristics from OpenMM.
    This code is ported from openMM.
    Args:
        cell (np.ndarray): Simulation cell matrix.
        cutoff (float): Cutoff radius.
        t_err (float): Target error.

    Returns:
        Tuple[float, List[int]]: Computed alpha and the number of grid points.
    """
    alpha = math.sqrt(-math.log(2*t_err))/cutoff
    denom = 3.0 * math.pow(t_err, 0.2)
    nmesh1 = math.ceil(2.0 * alpha * cell[0][0] / denom)
    nmesh2 = math.ceil(2.0 * alpha * cell[1][1] / denom)
    nmesh3 = math.ceil(2.0 * alpha * cell[2][2] / denom)

    return alpha, [find_available_number(nmesh1), find_available_number(nmesh2), find_available_number(nmesh3)]


def calculate_num_kvecs_dynamic(
    charges: np.ndarray, cell: np.ndarray, accuracy: float, alpha: float
) -> Tuple[float, List[int]]:
    """
    Determine the maximal number of points in reciprocal space for each direction.

    Args:
        charges (np.ndarray): Charge distribution.
        cell (np.ndarray): Simulation cell matrix.
        accuracy (float): Target accuracy.
        alpha (float): Ewald parameter.

    Returns:
        Tuple[float, List[int]]: Maximum reciprocal space cutoff and number of k-vectors.
    """
    # The kspace rms error is computed relative to the force that two unit point
    # charges exert on each other at a distance of 1 Angström
    accuracy_relative = accuracy * CONV_FACTOR

    nxmax = 1
    nymax = 1
    nzmax = 1
    
    natoms = len(charges)

    q_sq_sum = CONV_FACTOR * np.sum(charges**2)

    error = rms_kspace(nxmax, cell[0, 0], natoms, alpha, q_sq_sum)
    #TODO: turn this into binary search
    while error > (accuracy_relative):
        nxmax += 1
        error = rms_kspace(nxmax, cell[0, 0], natoms, alpha, q_sq_sum)

    error = rms_kspace(nymax, cell[1, 1], natoms, alpha, q_sq_sum)
    while error > (accuracy_relative):
        nymax += 1
        error = rms_kspace(nymax, cell[1, 1], natoms, alpha, q_sq_sum)

    error = rms_kspace(nzmax, cell[2, 2], natoms, alpha, q_sq_sum)
    while error > (accuracy_relative):
        nzmax += 1
        error = rms_kspace(nzmax, cell[2, 2], natoms, alpha, q_sq_sum)

    kxmax = 2 * np.pi / cell[0, 0] * nxmax
    kymax = 2 * np.pi / cell[1, 1] * nymax
    kzmax = 2 * np.pi / cell[2, 2] * nzmax

    kmax = max(kxmax, kymax, kzmax)

    # Check if box is triclinic --> Scale lattice vectors for triclinic skew
    if np.count_nonzero(cell - np.diag(np.diagonal(cell))) != 9:
        vector = np.array(
            [nxmax / cell[0, 0], nymax / cell[1, 1], nzmax / cell[2, 2]]
        )
        scaled_nbk = np.dot(np.array(np.abs(cell)), vector)
        nxmax = max(1, int(scaled_nbk[0]))
        nymax = max(1, int(scaled_nbk[1]))
        nzmax = max(1, int(scaled_nbk[2]))

    return kmax, [nxmax, nymax, nzmax]

def determine_alpha_ch_indep(t_err: float, cutoff: float) -> float:
    """
    Compute alpha in a charge-independent way.

    Args:
        t_err (float): Target error.
        cutoff (float): Cutoff distance.

    Returns:
        float: Computed alpha.
    """
    return math.sqrt(-math.log(2 * t_err)) / cutoff

def calculate_error_ch_indep(alpha, kmax, d):
    error = kmax * math.sqrt(d*alpha)/20 * math.exp(-((math.pi * kmax) / (d*alpha))**2)
    return error

def calculate_num_kvecs_ch_indep(
    t_err: float, cutoff: float, cell: np.ndarray
) -> Tuple[List[int], float]:
    """
    Compute the number of k-vectors and alpha in a charge-independent way.
     This code is ported from openMM.
    Args:
        t_err (float): Target error.
        cutoff (float): Cutoff distance.
        cell (np.ndarray): Simulation cell matrix.

    Returns:
        Tuple[List[int], float]: Number of k-vectors and computed alpha.
    """
    alpha = determine_alpha_ch_indep(t_err, cutoff)
    kmax_vals = []
    for i in range(3):
        sub_max = search_single_kmax_ch_indep(t_err, alpha, cell[i][i])
        kmax_vals.append(sub_max)
    return kmax_vals, alpha

def search_single_kmax_ch_indep(t_err, alpha, d):
    kmax = 1

    err = calculate_error_ch_indep(alpha, kmax, d)

    while err > t_err:
        kmax += 1
        err = calculate_error_ch_indep(alpha, kmax, d)   
    return kmax

def rms_kspace(km, l, n, a, q2):
    """
    Compute the root mean square error of the force in reciprocal space

    Reference
    ------------------
    Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
    """

    return (
        2
        * q2
        * a
        / l
        * np.sqrt(1 / (np.pi * km * n))
        * np.exp(-((np.pi * km / (a * l)) ** 2))
    )

def determine_alpha(
    charge: np.ndarray, acc: float, cutoff: float, cell: np.ndarray
) -> float:
    """
    Estimate alpha based on charge, accuracy, and simulation cell.
    (Adopted from LAMMPS)
    Args:
        charge (np.ndarray): Charge distribution.
        acc (float): Accuracy.
        cutoff (float): Cutoff distance.
        cell (np.ndarray): Simulation cell matrix.

    Returns:
        float: Estimated alpha.
    """

    # The kspace rms error is computed relative to the force that two unit point
    # charges exert on each other at a distance of 1 Angström
    accuracy_relative = acc * CONV_FACTOR

    qsqsum = CONV_FACTOR * np.sum(charge**2)

    a = (
        accuracy_relative
        * np.sqrt(
            len(charge) * cutoff * cell[0, 0] * cell[1, 1] * cell[2, 2]
        )
        / (2 * qsqsum)
    )

    if a >= 1.0:
        return (1.35 - 0.15 * np.log(accuracy_relative)) / cutoff

    else:
        return np.sqrt(-np.log(a)) / cutoff