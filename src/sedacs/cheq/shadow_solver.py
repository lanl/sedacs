import torch
from scipy.sparse.linalg import LinearOperator, cg, gmres
import numpy as np
from typing import Callable, Optional, Tuple

def calculate_fixed_parts_linear_ChEQ_solver(xi: torch.Tensor, u: torch.Tensor):
    """
    Compute fixed (rank-2 update) components for a linearized Charge Equilibration system.
    This function calculates terms that remain fixed throughout the ChEQ. 
    Args:
        xi (torch.Tensor): Tensor of size [N] for electronegativity values.
        u (torch.Tensor): Tensor of size [N], representing hardness-like parameters 
            (diagonal components in the ChEQ problem).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - fixed1 (torch.Tensor): A tensor of shape [N+1, 2].
            - fixed2 (torch.Tensor): A tensor of shape [2, N+1].
    """
    orig_dtype = xi.dtype
    xi = xi.to(torch.float64)
    u = u.to(torch.float64)
    N = len(xi)
    A0_inv = torch.ones(N+1,device=u.device, dtype=u.dtype)
    A0_inv[:N] = 1/u
    MInv = torch.tensor([[0.0,1.0], [1.0, 0.0]], device=u.device, dtype=u.dtype)
    v = torch.zeros((N+1, 2), dtype=u.dtype, device=u.device)
    v[:N,0] = 1
    v[N,0] = -0.5
    v[N,1] = 1.0
    G = torch.linalg.inv(MInv + (v.T * A0_inv) @ v) # (2,N) (N,N) * (N,2) = (2,2)  
    fixed1 = (A0_inv * v.T).T @ G # [N+1, 2]
    fixed2 = (v.T * A0_inv) # [2, N+1]
    return (fixed1.to(orig_dtype), fixed2.to(orig_dtype))


@torch.compile
def linear_ChEQ_solver(
    ewald_p: torch.Tensor,
    xi: torch.Tensor,
    u: torch.Tensor,
    q_tot: float,
    fixed_parts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the linear Charge Equilibration (ChEQ) equations for partial charges and
    Lagrange multiplier under a linear approximation.

    This solver follows the shadow approach that treats the hardness as diagonal (u) 
    and introduces coupling and constraints through rank-2 corrections. 
    The solution can be found in closed form.

    Defined in Eq. (17) of:
    Li, Cheng-Han, et al. Shadow Molecular Dynamics with a Machine Learned Flexible Charge Potential
    (https://doi.org/10.26434/chemrxiv-2025-x8b23)


    Args:
        ewald_p (torch.Tensor): Tensor of length N, representing Ewald potential
            contributions at each site.
        xi (torch.Tensor): Tensor of length N, for electronegativity values.
        u (torch.Tensor): Tensor of length N, representing diagonal hardness-like
            parameters for each site.
        q_tot (float): Desired total charge across all sites.
        fixed_parts (Optional[Tuple[torch.Tensor, torch.Tensor]], optional):
            Precomputed fixed auxiliary components. If None, they are calculated 
            internally. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - q_p (torch.Tensor): A tensor of length N containing the solved partial
              charges on each site.
            - mu (torch.Tensor): A scalar representing the Lagrange multiplier enforcing
              the total charge constraint.
    """
    N = len(xi)
    b = torch.zeros(N+1, dtype=u.dtype, device=u.device)
    b[N] = q_tot
    b[:N] = - xi - ewald_p
    # store the diagonal matrix as a vector
    A0_inv = torch.ones(N+1,device=u.device, dtype=u.dtype)
    A0_inv[:N] = 1/u
    if fixed_parts == None:
        fixed_parts = calculate_fixed_parts_linear_ChEQ_solver(xi, u)
    x = A0_inv * b - fixed_parts[0] @ (fixed_parts[1] @ b)
    q_p = x[:N]
    mu = x[N]

    return (q_p, mu)

@torch.compile
def calculate_jacob_vec(
    p: torch.Tensor,
    ewald_vec: torch.Tensor,
    xi: torch.Tensor,
    u: torch.Tensor,
    q_tot: float,
    vec: torch.Tensor,
    fixed_parts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Compute the Jacobian-vector product.

    Defined in Eq. (21) of:
    Li, Cheng-Han, et al. Shadow Molecular Dynamics with a Machine Learned Flexible Charge Potential
    (https://doi.org/10.26434/chemrxiv-2025-x8b23)

    Args:
        p (torch.Tensor): Tensor of length N, representing current partial charges.
        ewald_vec (torch.Tensor): Tensor of length N, representing differential Ewald
            potential contributions.
        xi (torch.Tensor): Tensor of length N, for electronegativity values.
        u (torch.Tensor): Tensor of length N, representing diagonal hardness-like
            parameters for each site.
        q_tot (float): Desired total charge across all sites.
        vec (torch.Tensor): The vector to be multiplied by the Jacobian. This typically
            appears in iterative solvers (e.g., GMRES).
        fixed_parts (Optional[Tuple[torch.Tensor, torch.Tensor]], optional):
            Precomputed auxiliary components for the linear ChEQ system. If None, they
            are calculated using `calculate_fixed_parts_linear_ChEQ_solver`. Defaults to None.

    Returns:
        torch.Tensor:
            A tensor of length N representing the result of the Jacobian-vector product.
    """
    N = len(p) 
    C_vec = ewald_vec
    part1 = (-(1/u) * C_vec)
    if fixed_parts == None:
        fixed_parts = calculate_fixed_parts_linear_ChEQ_solver(xi, u)
    part2 = fixed_parts[0][:-1] @ (fixed_parts[1][:,:-1] @ (-C_vec))
    J_vec = part1 - part2        
    return J_vec - vec

def shadow_QEQ_solver(
    jacob_mat_vec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    precond: Optional[torch.Tensor] = None,
    init: Optional[torch.Tensor] = None,
    rtol: float = 1e-5
) -> Tuple[torch.Tensor, int]:
    """
    Solve a linear 'shadow' equation, Jx = (q_p - p), using a Krylov solver
    (e.g., GMRES), where J is the linearized operator from the extended Lagrangian approach.

    This routine is often employed in shadow molecular dynamics to update
    auxiliary variables without requiring a fully converged solution at each step.
    The iteration tolerance may be relaxed while still preserving good energy
    conservation over many MD steps.

    Solves Eq. (41) of:
    Li, Cheng-Han, et al. Shadow Molecular Dynamics with a Machine Learned Flexible Charge Potential
    (https://doi.org/10.26434/chemrxiv-2025-x8b23)

    Args:
        jacob_mat_vec (Callable[[torch.Tensor], torch.Tensor]):
            A function that computes the product of the Jacobian matrix J with an arbitrary vector.
        b (torch.Tensor):
            Right-hand side vector of length N, corresponding to (q_p - p).
        precond (Optional[torch.Tensor], optional):
            A diagonal or approximate inverse of J, used as a preconditioner in GMRES.
            If provided, it should match the shape [N] or [N, N] (diagonal or dense). Defaults to None.
        init (Optional[torch.Tensor], optional):
            Initial guess for the GMRES solver. If None, a zero vector is used internally. Defaults to None.
        rtol (float, optional):
            Relative tolerance for convergence in GMRES. Defaults to 1e-5.

    Returns:
        Tuple[torch.Tensor, int]:
            - x (torch.Tensor): The solution vector of length N in the same device and dtype as `b`.
            - iter_cnt (int): The total number of times the Jacobian-vector product was called.
    """
    np_dtype = np.float64
    if b.dtype == torch.float32:
        np_dtype = np.float32
    iter_cnt = 0
    def mv(vec):
        nonlocal iter_cnt
        vec = torch.from_numpy(vec).to(dtype).to(device)
        res = jacob_mat_vec(vec)
        iter_cnt += 1
        return res.cpu().detach().numpy().astype(np_dtype)
    device = b.device
    dtype = b.dtype
    N = len(b)
    b_np = b.detach().cpu().numpy().astype(np_dtype)
    A = LinearOperator((N,N), matvec=mv)
    if precond != None:
        precond = precond.cpu().numpy().astype(np_dtype)
    if init != None:
        init = init.cpu().numpy().astype(np_dtype)
    
    x, exit_code = gmres(A, b_np, rtol=rtol, M=precond, x0=init)
    if exit_code > 0:
        print(f"[WARNING] No shadow solver convergence after {exit_code} iterations")
    x = torch.tensor(x, device=device, dtype=b.dtype)
    
    #b = b.cpu().numpy().astype(np_dtype)
    #x, exit_code = cg(A, b, rtol=rtol, M=precond, x0=init)
    #x, iter_cnt = CG(jacob_mat_vec, b, rtol=rtol, x0=init)
    #x = torch.tensor(x, device=device, dtype=dtype)
    
    return x, iter_cnt