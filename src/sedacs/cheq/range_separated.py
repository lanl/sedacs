from scipy.optimize import
import torch


def taper(value,
          lo_rad: float,
          hi_rad: float):
    '''
    Tapering function for which all numerical values were taken from the JAX-MD
    code without modifications. This function tapers smoothly from 1-0 over the
    range specified by lo_rad to hi_rad. Beyond hi_rad, the function is
    strictly zero. From lo_rad to hi_rad, the output deviates from the
    functional form used to determine value.

    Parameters
    ----------
    value:
        This can be ArrayLike or a single numerical value.
    lo_rad: float
        The onset of the tapering.
    hi_rad: float
        The hard cutoff for the tapering, beyond which the function is strictly
        zero.

    Returns
    -------
    tapered_value:
        The tapered version of the output, matching in type.
    '''
    R = value - lo_rad
    hi_rad = hi_rad - lo_rad
    lo_rad = 0.0

    R2 = R**2
    R3 = R**3

    aB = hi_rad
    aA = lo_rad

    D1 = aB - aA
    D7 = D1 ** 7.0
    aA2 = aA * aA
    aA3 = aA2 * aA
    aB2 = aB * aB
    aB3 = aB2 * aB

    aC7 = 20.0
    aC6 = -70.0 * (aA + aB)
    aC5 = 84.0 * (aA2 + 3.0 * aA * aB + aB2)
    aC4 = -35.0 * (aA3 + 9.0 * aA2 * aB + 9.0 * aA * aB2 + aB3)
    aC3 = 140.0 * (aA3 * aB + 3.0 * aA2 * aB2 + aA * aB3)
    aC2 = -210.0 * (aA3 * aB2 + aA2 * aB3)
    aC1 = 140.0 * aA3 * aB3
    aC0 = (-35.0 * aA3 * aB2 * aB2 + 21.0 * aA2 * aB3 * aB2 - 7.0 * aA * aB3 *
           aB3 + aB3 * aB3 * aB)

    a = (aC7 * R3 * R3 * R + aC6 * R3 * R3 + aC5 * R3 * R2 + aC4 * R2 * R2 +
         aC3 * R3 + aC2 * R2 - aC1 * R + aC0) / D7

    tapered_values = torch.where(R < lo_rad,
                                 torch.tensor(1.0, device=value.device),
                                 torch.where(R < hi_rad, a,
                                             torch.tensor(0.0,
                                                          device=value.device))
                                 )

    return tapered_values


def tapered_inverse_distance_matrix(dist_matrix: torch.Tensor,
                                    lo_rad: float = 6.0,
                                    hi_rad: float = 8.0,
                                    eps:float = 1e-4,
                                    scale_fac: float = 14.399) -> torch.Tensor:
    """
    Tapered inverse distance matrix computation for solving the Range-Separated
    Coulomb matrix with tapering and thresholding. Input should be the *dense*
    Coulomb matrix. Modify the inv_dist term to add support for a neighbor-
    list form on the input.

    Applies the tapering function to the input inverse distance matrix. This is
    by default scaled using the appropriate units for usage in an MD/simulation
    framework using input units of eV and Angstrom. For atomic units, input
    distances should be in bohr, and scale_fac should be set to 1.

    Parameters
    ----------
    dist_matrix: torch.Tensor
        The input *inverse* distance matrix.
    lo_rad: float
        The onset of the tapering function.
    hi_rad: float
        The hard cutoff beyond which the tapering sends values -> zero.
    eps: float
        Small value beyond which values will be taken to be zero and masked
        from the tapering function.
    scale_fac: float
        Value for unit conversion. 14.399 is appropriate for units using the
        eV/A system common to most ASE calculators, etc. For atomic units,
        one must input distances as 1/bohr, and use scale_fac=1.
    Returns
    -------
    tapered_inv_distances: torch.Tensor
        The matrix of tapered inverse distances returned as a torch Tensor.
    """
    # TODO: Add support for neighborlist input format.
    inv_dist = torch.where(dist_matrix > eps,
                           1.0 / dist_matrix,
                           torch.tensor(0.0,
                                        device=dist_matrix.device))

    taper_weights = taper(dist_matrix, lo_rad, hi_rad)

    tapered_inv_distance = scale_fac * inv_dist * taper_weights

    return tapered_inv_distance


def sr_QEQ_charges(pos_T: torch.Tensor,
                   sr_nbr_inds: torch.Tensor,
                   sr_dists: torch.Tensor,
                   sr_cutoff: float,
                   chi: torch.Tensor,
                   hu: torch.Tensor,
                   q_tot: float,
                   dtype,
                   device):
    """
    Computes the charges from the short-range, tapered Coulomb matrix. Some
    comments on the requisite form for the input neighbor list information:

    sr_nbr_inds and sr_dists are the indices and distances respectively of the
    neighbors of atom i. These are to be obtained by the 

    Parameters
    ----------
    pos_T: Tensor
        Atomic positions.
    sr_nbr_inds: Tensor
        Neighbor indices for the short-range Coulomb matrix.
    sr_cutoff: float
        Cutoff for the short-range neighborlist.
    chi: Tensor
        Atom-wise electronegativities.
    hu: Tensor
        Atom-wise Hubbard U values on the diagonal of the Coulomb matrix.
    q_tot: float
        Total charge, 0.0 as default.
    dtype:
        Torch datatype for floating point values (electronegativities,
        distances, Coulomb matrix.)
    device:
        Torch device.

    Returns
    -------
    x: Tensor
        Torch tensor of shape: [q0, q1, ..., qN, mu] where mu
        is the chemical potential.

    TODO
    ----
    Remove the construction of the dense distance matrix in favor of the one
    with the shape of the neighborlist. This also requires changing how the
    diagonal of the inverse distance matrix is constructed in
    tapered_inverse_distance_matrix.
    """
    # Correct treatment of PBCs
    Rij = torch.zeros(pos_T.shape[1],
                      pos_T.shape[1],
                      dtype=dtype,
                      device=device)

    nbr_mask = sr_nbr_inds != -1
    row_idx = (torch.arange(Rij.shape[0], device=device)
               .unsqueeze(1)
               .expand_as(sr_nbr_inds)
               )

    #      Valid rows          Valid Columns     Valid distances.
    Rij[row_idx[nbr_mask], sr_nbr_inds[nbr_mask]] = sr_dists[nbr_mask]

    # Construct the tapered SR Coulomb matrix. (Dense)
    CL = tapered_inverse_distance_matrix(Rij,
                                         sr_cutoff-1.0,
                                         sr_cutoff)
    # print(CL)

    n_atoms = pos_T.shape[1]

    # Construct the b-vector.
    b = torch.zeros(n_atoms+1, dtype=dtype, device=device)
    b[:-1] = -chi
    b[-1] = q_tot

    # Diagonal of thresholded Coulomb, this part is not efficient.
    CS = torch.eye(n_atoms, device=device) * hu
    AS = torch.zeros((n_atoms+1, n_atoms+1),
                     device=device,
                     dtype=dtype)

    # Full A-matrix
    AS[:-1, :-1] = CS  # SR Coulomb
    AS[-1, :-1] = 1    # 1-vectors
    AS[:-1, -1] = 1    # 1-vectors
    AL = torch.zeros_like(AS, device=device)
    AL[:-1, :-1] = CL  # On-site part of Coulomb
    A = AS + AL        # Full A-matrix. (Dense)

    # x = torch.linalg.inv(A)@b
    # print(x)
    # Solve system of equations for RS_QEQ with MINRES
    x = solve_minres(A, b)

    return x

