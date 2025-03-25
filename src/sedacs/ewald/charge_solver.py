from sedacs.ewald import ewald_energy
from sedacs.ewald import CONV_FACTOR
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.sparse.linalg import LinearOperator, cg, gmres
from typing import Callable, Optional, Tuple

__all__ = ["QEQ_solver"]

def QEQ_solver(
    positions: torch.Tensor,
    ewald_vec_func: Callable[[torch.Tensor], torch.Tensor],
    b: np.ndarray,
    h_u: np.ndarray,
    init_charges: Optional[np.ndarray] = None,
    A_inv: Optional[np.ndarray] = None,
    rtol: float = 1e-5,
    maxiter: int = 100
) -> Tuple[torch.Tensor, int]:
    """
    Solve the QEq (Charge Equilibration) equations using the GMRES solver.

    This function solves the system for atomic partial charges that minimize 
    electrostatic energy using an iterative linear solver.

    Args:
        positions (torch.Tensor): Atomic positions tensor ([3, N]).
        ewald_vec_func (Callable[[torch.Tensor], torch.Tensor]): Function computing Ewald vector.
        b (np.ndarray): Negative electronegativity values and total charge constraint.
        h_u (np.ndarray): Hardness values.
        init_charges (Optional[np.ndarray], optional): Initial charge guess. Defaults to None.
        A_inv (Optional[np.ndarray], optional): 1/diag(matrix), for jacobi preconditioner. Defaults to None.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
        maxiter (int, optional): Maximum number of solver iterations. Defaults to 100.

    Returns:
        Tuple[torch.Tensor, int]: Computed charges and iteration count.
    """
    dtype = positions.dtype
    device = positions.device
    np_dtype = np.float32
    if dtype == torch.float64:
        np_dtype = np.float64

    N = positions.shape[1]
    res = np.zeros(N+1, dtype=np_dtype)
    iter_cnt = 0
    def mv(x):
        nonlocal res, iter_cnt
        iter_cnt += 1
        q = x[:-1]
        elect = x[-1]
        total_ch = np.sum(q)
        charges = torch.from_numpy(q).type(dtype).to(device)
        dq = ewald_vec_func(charges)
        dq = dq.cpu().numpy()
        res[:-1] = dq + elect + h_u * q
        res[-1] = total_ch
        return res
    '''
    #TODO: Torch based solver is needed to not do data transfer between CPU and GPU.
    #The torch based CG solver is already available but not GMRES.
    res_torch = torch.zeros(N+1, dtype=dtype, device=device)
    def mv_torch(x):
        nonlocal res_torch
        q = x[:-1]
        elect = x[-1]
        total_ch = mixed_precision_sum(q)
        dq = ewald_vec_func(q)
        res_torch[:-1] = dq + elect + h_u * q
        res_torch[-1] = total_ch
        return res_torch
    '''

    A = LinearOperator((N+1,N+1), matvec=mv)
    if init_charges is not None:
        init_charges = np.concatenate((init_charges, np.array([0.0])))
        
    M = None
    if A_inv is not None:
        M = LinearOperator((N+1,N+1), matvec=lambda x: A_inv * x)
    x, exit_code = gmres(A, b, x0=init_charges, rtol=rtol, maxiter=maxiter, M=M)
    if exit_code > 0:
        print(f"[WARNING] No QEQ convergence after {exit_code} iterations")
    return torch.from_numpy(x).to(dtype).to(device), iter_cnt