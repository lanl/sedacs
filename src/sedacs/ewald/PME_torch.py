import torch
import math
from sedacs.ewald import ewald_real, CONV_FACTOR, ewald_self_energy, ewald_real_screening
from typing import Optional, Tuple, List, Union

def b(m, order=4):
    k = torch.arange(order - 1, device=m.device).reshape(1,1,1,order-1)
    M = compute_spline_coefficients(torch.tensor(1.0), order)[1:].flip(dims=(0,))
    M = M.reshape(1,1,1,order-1)
    M = M.to(m.dtype).to(m.device)
    
    prefix = torch.exp(2 * torch.pi * 1j * (order - 1) * m)
    m = m[..., None]
    
    bot = torch.sum(M * torch.exp(2 * torch.pi * 1j * m * k), dim=-1)

    res = prefix / bot
    return res.real * res.real + res.imag * res.imag

def B(mx, my, mz, order=4):
    b_x = b(mx,order)
    b_y = b(my,order)
    b_z = b(mz,order)
    return b_x * b_y * b_z

def compute_spline_coefficients(w, order: int = 4):
    shape = w.shape
    w = w.flatten()
    scale = 1.0 / (order - 1)
    data = [torch.empty_like(w) for _ in range(order)]
    #data = torch.zeros(order, *shape, dtype=w.dtype, device=w.device)
    data[order - 1] = torch.zeros_like(w)
    data[1] = w
    data[0] = 1.0 - w

    for j in range(3, order):
        div = 1.0 / (j - 1)
        data[j - 1] = div * w * data[j - 2]

        for k in range(1, j - 1):
            data[j - k - 1] = div * ((w + k) * data[j - k - 2] + (j - k - w) * data[j - k - 1])

        data[0] = div * (1.0 - w) * data[0]

    data[order - 1] = scale * w * data[order - 2]

    for j in range(1, order - 1):
        data[order - j - 1] = scale * ((w + j) * data[order - j - 2] + (order - j - w) * data[order - j - 1])

    data[0] = scale * (1.0 - w) * data[0]
    data = torch.stack(data, dim=0)
    return data.reshape(order, *shape)


def grid_position(u, K, order=4):
    base_grid = torch.floor(u).to(torch.int32) 
    base_grid = base_grid.unsqueeze(0) # shape: [1, 3, N]
    offsets = torch.arange(0, order, device=u.device).view(order, 1, 1)  
    grid = base_grid + offsets # shape: [order, 3, N]
    return grid % K.view(1, 3, 1)

def map_charges_to_grid(position, charge, inv_box, grid_dimensions, order=4):
    """
    Maps particle charges to a grid via B-spline spreading.
    """
    grid_dimensions_torch = torch.tensor(grid_dimensions, device=position.device, dtype=torch.int32)
    Q = torch.zeros(tuple(grid_dimensions), device=position.device, dtype=position.dtype)
    
    u = (inv_box @ position) * grid_dimensions_torch.view(3, 1)
    
    # Compute the fractional part
    w = u - torch.floor(u)  # shape: [3, N]
    
    coeffs = compute_spline_coefficients(w, order)
    
    # [order, 3, N].
    grid_pos = grid_position(u, grid_dimensions_torch, order)

    # Multiply by particle charge.
    accum = charge.reshape(1, 1, 1, -1) * (coeffs[:, 0, None, None, :] 
                                           * coeffs[None, :, 1, None, :] 
                                           * coeffs[None, None,:, 2, :])
    
    gp_x = grid_pos[:, 0, None, None, :] # [order, 1, 1, N]
    gp_y = grid_pos[None, :, 1, None, :] # [1, order, 1, N]
    gp_z = grid_pos[None, None,:, 2, :]  # [1, 1, order, N]
    
    
    grid_positions = torch.stack((gp_x.expand(order, order, order, -1),
                                  gp_y.expand(order, order, order, -1),
                                  gp_z.expand(order, order, order, -1)), dim=0)
    
    # Flatten over particles and all order dimensions.
    gp_new = grid_positions.view(3, -1)  # [N * order^3, 3]
    ac_new = accum.view(-1)              # [N * order^3]
    
    gp_new_flat = (gp_new[0] * (grid_dimensions[1] * grid_dimensions[2]) +
                   gp_new[1] * grid_dimensions[2] +
                   gp_new[2])
    
    Q_flat = Q.flatten()
    Q_flat.index_add_(0, gp_new_flat, ac_new)
    Q = Q_flat.reshape(tuple(grid_dimensions))
    return Q

def init_PME_data(grid_dimensions, box, alpha, order):
    inverse_box = torch.linalg.inv(box)

    res = torch.meshgrid(*[torch.fft.fftfreq(g) for g in grid_dimensions], indexing="ij")
    mx, my, mz = [r.to(box.dtype).to(box.device) for r in res]
    m = (inverse_box[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
         inverse_box[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
         inverse_box[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
    m_2 = torch.sum(m**2, dim=-1)
    V = torch.linalg.det(box)
    mask = m_2 != 0
    m_2 = torch.where(mask == 1, m_2, 1)
    
    exp_m = 1 / (2 * torch.pi * V) * torch.exp(-torch.pi**2 * m_2 / alpha**2) / m_2
    grid_multip = mask * exp_m * B(mx, my, mz, order)
    return (grid_dimensions, grid_multip, inverse_box, order)

@torch.compile(dynamic=False, fullgraph=True, options={"max_autotune":True})
def calculate_PME_energy(position, charge, box, alpha: float, PME_init_data:tuple):
    grid_dimensions, grid_multip, inverse_box, order = PME_init_data
    
    grid_new = map_charges_to_grid(position, charge, inverse_box, grid_dimensions, order)
    Fgrid = torch.fft.fftn(grid_new)
 
    e = torch.sum(
       grid_multip * (Fgrid.real ** 2 + Fgrid.imag ** 2))
    return e

def calculate_PME_ewald(
    positions: torch.Tensor,
    charges: torch.Tensor,
    box: torch.Tensor,
    nbr_inds: torch.Tensor,
    nbr_disp_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    alpha: float,
    cutoff: float,
    PME_init_data: Tuple[Union[List, Tuple], torch.Tensor, torch.Tensor, int],
    hubbard_u: torch.Tensor = None,
    atomtypes: torch.Tensor = None,
    calculate_forces: int = 0,
    calculate_dq: int = 0,
    screening: int = 0
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the total Ewald sum energy using Particle Mesh Ewald (PME).

    This function calculates the real-space and PME-based reciprocal-space 
    contributions to the Ewald summation. It also optionally computes forces 
    and charge derivatives.

    Args:
        positions (torch.Tensor): Atomic positions. Shape: `(N, 3)` or  `(3, N)`, where:
            - `N` is the number of atoms.
            - `3` represents the x, y, and z coordinates.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        box (torch.Tensor): Simulation box dimensions. Shape: `(3, 3)`.
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `K` is the maximum number of neighbors per atom.
        nbr_disp_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)` or `(N, K, 3)`.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for real-space interactions (scalar).
        PME_init_data (tuple): Precomputed PME initialization data.
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int, optional): Flag to compute charge derivatives (`1` for True, `0` for False`). Defaults to `0`.

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Total Ewald energy contribution.
            - **(torch.Tensor, shape `(3, N)`, optional)** Computed forces if `calculate_forces` is enabled, otherwise `None`.
                If the positions are provided as `(N, 3)`, the forces will be also  `(N, 3)`.

            - **(torch.Tensor, shape `(N,)`, optional)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.

    Notes:
        - Forces and charge derivatives for PME are computed via automatic differentiation.
    """
    N = len(charges)
    # As the internal functions expects (3, N), transpose the position tensor as needed
    transpose = False
    if positions.shape[1] == 3:
        transpose = True
        positions = positions.T.contiguous()

    # transpose the disp. vectors as needed
    if nbr_disp_vecs.shape[2] == 3:
        nbr_disp_vecs = nbr_disp_vecs.permute(2, 0, 1).contiguous()
    
    dq = None
    forces = None
    if screening:
        my_e_real, my_f_real, my_dq_real = ewald_real_screening(nbr_inds, nbr_disp_vecs, 
                                                    nbr_dists, charges, hubbard_u, atomtypes, alpha, 
                                                    cutoff,
                                                    calculate_forces, calculate_dq)
        
        if my_f_real is not None and my_f_real.device.type == 'cuda':
            my_f_real = my_f_real.T.contiguous() 

    else:
        my_e_real, my_f_real, my_dq_real = ewald_real(nbr_inds, nbr_disp_vecs, 
                                                    nbr_dists, charges, alpha, 
                                                    cutoff,
                                                    calculate_forces, calculate_dq)
    if calculate_dq:
        charges.grad = None
        charges.requires_grad = True
        
    if calculate_forces:
        positions.grad = None
        positions.requires_grad = True

    pme_e = calculate_PME_energy(positions, charges, box, alpha, PME_init_data)
    self_e, self_dq = ewald_self_energy(charges, alpha, calculate_dq)

    if calculate_dq or calculate_forces:
        pme_e.backward()

    if calculate_forces:
        forces = (-1.0 * positions.grad + my_f_real).detach() * CONV_FACTOR
        if transpose:
            forces = forces.T 
        positions.requires_grad = False
    if calculate_dq:
        dq = (charges.grad + my_dq_real + self_dq).detach() * CONV_FACTOR  
        charges.requires_grad = False
    
    total_ewald_e = (my_e_real + pme_e + self_e) * CONV_FACTOR
    return total_ewald_e, forces, dq

    
    

    