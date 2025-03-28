import triton
import triton.language as tl
import torch
import time
from .ewald_torch import ewald_energy as torch_ewald_energy
from .ewald_torch import ewald_real as torch_ewald_real
from .ewald_torch import ewald_kspace_part1 as torch_ewald_kspace_part1
from .ewald_torch import ewald_kspace_part2 as torch_ewald_kspace_part2
from .ewald_torch import ewald_self_energy
from .util import CONV_FACTOR
from typing import Optional, Tuple

def get_autotune_config():
    """
    Create a list of config options for the kernels
    TODO: Need to spend time actually figuring out more reasonable options
    targeted for modern GPUs
    """
    return [
        triton.Config({"BLOCK_SIZE_N": 4, "BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 8}),

        triton.Config({"BLOCK_SIZE_N": 4, "BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 16}),

        triton.Config({"BLOCK_SIZE_N": 4, "BLOCK_SIZE_M": 32}),
        triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_M": 32}),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32}),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32}),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32}),

        triton.Config({"BLOCK_SIZE_N": 4, "BLOCK_SIZE_M": 64}),
        triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_M": 64}),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 64}),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64}),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64}),
    ]

@triton.autotune(configs=get_autotune_config(), key=["N", "M"])
@triton.jit
def ewald_kspace_part2_kernel_low_memory(out_f_ptr,
                                        out_dq_ptr,
                                        sum_r_vals_ptr,
                                        sum_i_vals_ptr,
                                        pos_ptr,
                                        kvecs_ptr,
                                        I_ptr,
                                        q_ptr,
                                        N: int,
                                        M: int,
                                        calculate_forces: tl.constexpr,
                                        calculate_dq: tl.constexpr,
                                        BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_M: tl.constexpr,
                                        dtype: tl.constexpr = tl.float32):
    '''
    Assume at least one of calculate_dq or calculate_forces set to 1
    '''
    row_block_idx = tl.program_id(axis=0)
    num_cols_M = tl.cdiv(M, BLOCK_SIZE_M)
    
    N_offsets = tl.arange(0, BLOCK_SIZE_N) + (BLOCK_SIZE_N * row_block_idx)
    pos_ptrs = pos_ptr + N_offsets
    atom_mask = N_offsets < N
    if calculate_forces:
        f_x = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
        f_y = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
        f_z = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
    if calculate_dq:
        sub_dq = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)

    xs = tl.load(pos_ptrs, mask=atom_mask, other=0.0)
    ys = tl.load(pos_ptrs + N, mask=atom_mask, other=0.0)
    zs = tl.load(pos_ptrs + 2*N, mask=atom_mask, other=0.0)
    qs = tl.load(q_ptr + N_offsets, mask=atom_mask, other=0.0)

    sub_col_offsets = tl.arange(0, BLOCK_SIZE_M)
    # go over M, block by block
    for col_idx in range(num_cols_M):
        col_offsets = sub_col_offsets + (BLOCK_SIZE_M * col_idx)
        col_mask = col_offsets < M
        # load multiple kvecs
        k_start_ptr = kvecs_ptr + col_offsets
        kx = tl.load(k_start_ptr, mask=col_mask, other=0.0)
        ky = tl.load(k_start_ptr + M, mask=col_mask, other=0.0)
        kz = tl.load(k_start_ptr + 2*M, mask=col_mask, other=0.0)
        # mmul: [BLOCK_SIZE_M x BLOCK_SIZE_N]
        mmul = kx[:, None] * xs[None, :] + ky[:, None] * ys[None, :] + kz[:, None] * zs[None, :]
        if calculate_dq:
            cos_mmul = tl.cos(mmul)
            sin_mmul = tl.sin(mmul)
            r_vals = cos_mmul * qs
            i_vals = sin_mmul * qs
        else:
            r_vals = tl.cos(mmul) * qs
            i_vals = tl.sin(mmul) * qs

        I = tl.load(I_ptr + col_offsets, mask=col_offsets < M, other=0.0)
        sum_r = tl.load(sum_r_vals_ptr + col_offsets, mask=col_offsets < M, other=0.0)
        sum_i = tl.load(sum_i_vals_ptr + col_offsets, mask=col_offsets < M, other=0.0)
       
        if calculate_forces:
            # BLOCK_SIZE_M X BLOCK_SIZE_N
            cos_sin_ln = (r_vals * sum_i[:, None])
            sin_cos_ln = (i_vals * sum_r[:, None])
            prefac_ln = I[:, None] * (cos_sin_ln - sin_cos_ln)
            # BLOCK_SIZE_M X 1  
            k_start_ptr = kvecs_ptr + col_offsets
            f_x += tl.sum(prefac_ln * kx[:, None], axis=0)
            f_y += tl.sum(prefac_ln * ky[:, None], axis=0)
            f_z += tl.sum(prefac_ln * kz[:, None], axis=0)   
        if calculate_dq:
            sub_dq += tl.sum((I * sum_r)[:, None] * cos_mmul, axis=0) \
                    + tl.sum((I * sum_i)[:, None] * sin_mmul, axis=0)

    if calculate_forces:
        out_fx_ptrs = out_f_ptr + N_offsets
        tl.store(out_fx_ptrs, f_x, mask=N_offsets < N)
        tl.store(out_fx_ptrs + N, f_y, mask=N_offsets < N)
        tl.store(out_fx_ptrs + 2*N, f_z, mask=N_offsets < N)
    if calculate_dq:
        out_dq_ptrs = out_dq_ptr + N_offsets
        tl.store(out_dq_ptrs, sub_dq, mask=N_offsets < N)

@triton.autotune(configs=get_autotune_config(), key=["N", "M"])
@triton.jit
def ewald_kspace_part2_kernel(out_f_ptr,
                            out_dq_ptr,
                            sum_r_vals_ptr,
                            sum_i_vals_ptr,
                            r_vals_ptr,
                            i_vals_ptr,
                            kvecs_ptr,
                            I_ptr,
                            q_ptr,
                            N: int,
                            M: int,
                            calculate_forces: tl.constexpr,
                            calculate_dq: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr,
                            BLOCK_SIZE_M: tl.constexpr,
                            dtype: tl.constexpr = tl.float32):
    '''
    Assume at least one of calculate_dq or calculate_forces set to 1
    '''
    row_block_idx = tl.program_id(axis=0)
    num_cols_M = tl.cdiv(M, BLOCK_SIZE_M)
    
    N_offsets = tl.arange(0, BLOCK_SIZE_N) + (BLOCK_SIZE_N * row_block_idx)
    if calculate_forces:
        f_x = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
        f_y = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
        f_z = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
    if calculate_dq:
        my_q = tl.load(q_ptr + N_offsets, mask=N_offsets < N, other=0.0)
        sub_dq = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
    sub_col_offsets = tl.arange(0, BLOCK_SIZE_M)
    # go over M, block by block
    for col_idx in range(num_cols_M):
        col_offsets = sub_col_offsets + (BLOCK_SIZE_M * col_idx)

        I = tl.load(I_ptr + col_offsets, mask=col_offsets < M, other=0.0)
        sum_r = tl.load(sum_r_vals_ptr + col_offsets, mask=col_offsets < M, other=0.0)
        sum_i = tl.load(sum_i_vals_ptr + col_offsets, mask=col_offsets < M, other=0.0)
        # BLOCK_SIZE_M X BLOCK_SIZE_N
        block_offsets = col_offsets[:, None] * N + N_offsets[None, :]
        block_mask = (col_offsets[:, None] < M) & (N_offsets[None, :] < N)
        r_vals = tl.load(r_vals_ptr + block_offsets, mask=block_mask, other=0)
        i_vals = tl.load(i_vals_ptr + block_offsets, mask=block_mask, other=0)
        # BLOCK_SIZE_M X BLOCK_SIZE_N
        cos_sin_ln = (r_vals * sum_i[:, None])
        sin_cos_ln = (i_vals * sum_r[:, None])
       
        if calculate_forces:
            # BLOCK_SIZE_M X BLOCK_SIZE_N
            prefac_ln = I[:, None] * (cos_sin_ln - sin_cos_ln)
            # BLOCK_SIZE_M X 1  
            k_start_ptr = kvecs_ptr + col_offsets
            kx = tl.load(k_start_ptr, mask=col_offsets < M, other=0.0)
            ky = tl.load(k_start_ptr + M, mask=col_offsets < M, other=0.0)
            kz = tl.load(k_start_ptr + 2*M, mask=col_offsets < M, other=0.0)
            f_x += tl.sum(prefac_ln * kx[:, None], axis=0)
            f_y += tl.sum(prefac_ln * ky[:, None], axis=0)
            f_z += tl.sum(prefac_ln * kz[:, None], axis=0)
        #de_dq = (4.0 * torch.pi * I * sum_r).reshape(-1, M) @ (r_vals / charges)/vol
        #de_dq += (4.0 * torch.pi * I * sum_i).reshape(-1, M) @ (i_vals / charges)/vol     
        if calculate_dq:
            sub_dq += tl.sum((I * sum_r)[:, None] * (r_vals / my_q[None, :]), axis=0) \
                    + tl.sum((I * sum_i)[:, None] * (i_vals / my_q[None, :]), axis=0)

    if calculate_forces:
        out_fx_ptrs = out_f_ptr + N_offsets
        tl.store(out_fx_ptrs, f_x, mask=N_offsets < N)
        tl.store(out_fx_ptrs + N, f_y, mask=N_offsets < N)
        tl.store(out_fx_ptrs + 2*N, f_z, mask=N_offsets < N)
    if calculate_dq:
        out_dq_ptrs = out_dq_ptr + N_offsets
        tl.store(out_dq_ptrs, sub_dq, mask=N_offsets < N)

@torch.compile
def __calc_en(sum_r, sum_i, I, vol):
    # internal function to calculate the energy
    abs_fac_sq = sum_r**2 + sum_i**2
    out_en = 2.0 * torch.pi * torch.sum(I * abs_fac_sq) / vol
    return out_en

def ewald_kspace_part2(sum_r, sum_i, r_vals, i_vals, vol, kvecs, I,
                       charges, positions, calculate_forces, calculate_dq,
                       out_f=None, out_dq=None,
                       low_memory_mode=False):
    # call the pytorch function if the tensors are on CPU
    if kvecs.device.type == 'cpu':
        # triton and pytorch uses different memory layout for kvecs
        # TODO: this assumes # kvecs will be > 3
        if kvecs.shape[0] == 3:
            kvecs = kvecs.T
        return torch_ewald_kspace_part2(sum_r, sum_i, r_vals, 
                                        i_vals, vol, kvecs, I, 
                                        charges,
                                        positions,
                                        calculate_forces,
                                        calculate_dq)
    
    N = len(charges)
    M = kvecs.shape[1]
    # positions expected to be 3xN
    out_f = out_dq = None
    if out_f == None and calculate_forces:
        out_f = torch.empty((3,N), device=charges.device).type(charges.dtype)
    if out_dq == None and calculate_dq:
        out_dq = torch.empty((N,), device=charges.device).type(charges.dtype) 
    # no need to move this part to triton as it is simple
    out_en = __calc_en(sum_r, sum_i, I, vol)
    dtype = tl.float32
    if charges.dtype == torch.float64:
        dtype = tl.float64
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    calc_forces_or_dq = calculate_forces or calculate_dq
    if low_memory_mode == False and calc_forces_or_dq:
        ewald_kspace_part2_kernel[grid](
            out_f, 
            out_dq,
            sum_r, sum_i,
            r_vals, i_vals,
            kvecs, I,
            charges,
            N, M,
            calculate_forces,
            calculate_dq,
            dtype=dtype)
    elif low_memory_mode == True and calc_forces_or_dq:
        ewald_kspace_part2_kernel_low_memory[grid](
            out_f, 
            out_dq,
            sum_r, sum_i,
            positions,
            kvecs, I,
            charges,
            N, M,
            calculate_forces,
            calculate_dq,
            dtype=dtype)     
    if out_f != None:
        out_f = out_f * (-4.0 * torch.pi / vol)
    if out_dq != None:
        out_dq = out_dq * (4.0 * torch.pi / vol)
    return out_en, out_f, out_dq 

@triton.autotune(configs=get_autotune_config(), key=["N", "M"])
@triton.jit
def ewald_kspace_part1_kernel(out_r_vals_ptr,
                      out_i_vals_ptr,
                      out_r_sum_ptr,
                      out_i_sum_ptr,
                      pos_ptr,
                      q_ptr, k_ptr,
                      N: int,
                      M: int,
                      BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_M: tl.constexpr,
                      dtype: tl.constexpr = tl.float32,
                      low_memory_mode: tl.constexpr = False):
    '''
    positions: 3xN
    k vecs: 3xM
    I: M
    '''
    row_block_idx = tl.program_id(axis=0)
    num_cols_N = tl.cdiv(N, BLOCK_SIZE_N)

    kvec_offsets = tl.arange(0, BLOCK_SIZE_M) + (BLOCK_SIZE_M * row_block_idx)
    k_start_ptr = k_ptr + kvec_offsets
    valid_kvec_mask = kvec_offsets < M
    # load multiple kvecs
    kx = tl.load(k_start_ptr, mask=valid_kvec_mask, other=0.0)
    ky = tl.load(k_start_ptr + M, mask=valid_kvec_mask, other=0.0)
    kz = tl.load(k_start_ptr + 2*M, mask=valid_kvec_mask, other=0.0)
    # iterate through positions block by block
    sub_col_offsets = tl.arange(0, BLOCK_SIZE_N)

    tmp_r_vals_sum = tl.zeros((BLOCK_SIZE_M,), dtype=dtype)
    tmp_i_vals_sum = tl.zeros((BLOCK_SIZE_M,), dtype=dtype)

    for col_idx in range(num_cols_N):
        col_offsets = sub_col_offsets + (BLOCK_SIZE_N * col_idx)
        valid_col_offsets_mask = col_offsets < N
        pos_ptrs = pos_ptr + col_offsets

        xs = tl.load(pos_ptrs, mask=valid_col_offsets_mask, other=0.0)
        ys = tl.load(pos_ptrs + N, mask=valid_col_offsets_mask, other=0.0)
        zs = tl.load(pos_ptrs + 2*N, mask=valid_col_offsets_mask, other=0.0)
        qs = tl.load(q_ptr + col_offsets, mask=valid_col_offsets_mask, other=0.0)

        # mmul: [BLOCK_SIZE_M x BLOCK_SIZE_N]
        mmul = kx[:, None] * xs[None, :] + ky[:, None] * ys[None, :] + kz[:, None] * zs[None, :]
        r_vals = tl.cos(mmul) * qs
        i_vals = tl.sin(mmul) * qs
        # update the tensors if the low memory mode is not active
        if low_memory_mode == False:
            out_r_vals_ptrs = out_r_vals_ptr + (kvec_offsets[:, None] * N + col_offsets[None, :])
            out_i_vals_ptrs = out_i_vals_ptr + (kvec_offsets[:, None] * N + col_offsets[None, :])
            mask = (kvec_offsets[:, None] < M) & (col_offsets[None, :] < N)
            tl.store(out_r_vals_ptrs, r_vals, mask=mask)
            tl.store(out_i_vals_ptrs, i_vals, mask=mask)
        # update the sums
        #size: [BLOCK_SIZE_M,]
        tmp_r_vals_sum += tl.sum(r_vals, axis=1)
        tmp_i_vals_sum += tl.sum(i_vals, axis=1)
    # write back to the final M sized sum vector
    out_sum_r_vals_ptrs = out_r_sum_ptr + kvec_offsets
    out_sum_i_vals_ptrs = out_i_sum_ptr + kvec_offsets
    tl.store(out_sum_r_vals_ptrs, tmp_r_vals_sum, mask=valid_kvec_mask)
    tl.store(out_sum_i_vals_ptrs, tmp_i_vals_sum, mask=valid_kvec_mask)





def ewald_kspace_part1(positions, charges, kvecs, my_r_vals=None, my_i_vals=None, 
                       my_sum_r=None, my_sum_i=None,
                       low_memory_mode=False):
    '''
    If low memory mode is active, it will not create the r_vals and i_vals tensors.
    It will directly compute the sum vectors.
    '''
    # call the pytorch function if the tensors are on CPU
    if charges.device.type == 'cpu':
        # triton and pytorch uses different memory layout for kvecs
        if kvecs.shape[0] == 3:
            kvecs = kvecs.T
        return torch_ewald_kspace_part1(positions, charges, kvecs)
    
    N = len(charges)
    M = kvecs.shape[1]
    # positions expected to be 3xN
    #BLOCK_SIZE_N = min(64, triton.next_power_of_2(N))
    #BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))
    #num_m_blocks = math.ceil(M/BLOCK_SIZE_M)
    if low_memory_mode == False:
        if my_r_vals == None:
            my_r_vals = torch.empty((M,N), device=positions.device).type(positions.dtype)
        if my_i_vals == None:
            my_i_vals = torch.empty((M,N), device=positions.device).type(positions.dtype)

    if my_sum_r == None:
        my_sum_r = torch.empty((M,), device=positions.device).type(positions.dtype)
    if my_sum_i == None:
        my_sum_i = torch.empty((M,), device=positions.device).type(positions.dtype)

    dtype = tl.float32
    if positions.dtype == torch.float64:
        dtype = tl.float64

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    ewald_kspace_part1_kernel[grid](
        my_r_vals,
        my_i_vals,
        my_sum_r,
        my_sum_i,
        positions,
        charges,
        kvecs,
        N,
        M,
        dtype=dtype,
        low_memory_mode=low_memory_mode)
      
    return my_r_vals, my_i_vals, my_sum_r, my_sum_i

@triton.jit
def ewald_real_space_kernel(out_en_ptr, out_f_ptr,
                            out_dq_ptr,
                            nbrs_ptr, nbr_dists_ptr,
                            dx_ptr, dy_ptr, dz_ptr,
                            q_ptr,
                            alpha: float,
                            cutoff: float,
                            N: int, M: int,
                            calculate_forces: tl.constexpr,
                            calculate_dq: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    '''
    Assumes tensors to be allocated in a contiguous manner
    '''
    row_idx = tl.program_id(0)
    pi = 3.141592653589793
    row_stride = M
    dist_start_ptr = nbr_dists_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    dist_ptrs = dist_start_ptr + col_offsets
    my_dists = tl.load(dist_ptrs, mask=col_offsets < M, other=1.0) # will be multiplied by 0 if out of bound
    my_dists_sq = my_dists * my_dists
    nbr_start_ptr = nbrs_ptr + row_idx * row_stride
    nbr_ptrs = nbr_start_ptr + col_offsets
    my_nbrs = tl.load(nbr_ptrs, mask=col_offsets < M, other=N)

    nbr_qs = tl.load(q_ptr + my_nbrs, mask=(my_nbrs < N) & (my_nbrs > -1) & (my_dists < cutoff), other=0.0)
    my_q = tl.load(q_ptr + row_idx, mask=row_idx < N, other=0.0)
    nbr_qs_over_dist = nbr_qs / my_dists
    qq_over_dist = nbr_qs_over_dist * my_q
    one_over_dists_sq = 1.0 / my_dists_sq
    erfc = 1.0 - tl.math.erf(alpha * my_dists)
    res = qq_over_dist * erfc
    # create [N,K] matrix with ewald real components
    #out_ptrs = output_ptr + row_idx * row_stride + col_offsets
    #tl.store(out_ptrs, res, mask=col_offsets < n_cols)

    # Force calculation
    if calculate_forces:
        col_disp_offsets = tl.arange(0, BLOCK_SIZE)
        disp_x_ptrs = dx_ptr + row_idx * row_stride + col_disp_offsets
        disp_y_ptrs = dy_ptr + row_idx * row_stride + col_disp_offsets
        disp_z_ptrs = dz_ptr + row_idx * row_stride + col_disp_offsets
        dx_disps = tl.load(disp_x_ptrs, mask=col_disp_offsets < M)
        dy_disps = tl.load(disp_y_ptrs, mask=col_disp_offsets < M)
        dz_disps = tl.load(disp_z_ptrs, mask=col_disp_offsets < M)

        two_over_sqrt_pi = 2.0 * tl.math.rsqrt(pi)
        alpha_sq = alpha * alpha
        f = -1.0 * qq_over_dist * (erfc * one_over_dists_sq + (two_over_sqrt_pi * alpha) * tl.exp(-alpha_sq * my_dists_sq) / my_dists)
        fx = tl.sum(f * dx_disps)
        fy = tl.sum(f * dy_disps)
        fz = tl.sum(f * dz_disps)

        out_fx_ptrs = out_f_ptr + row_idx
        out_fy_ptrs = out_f_ptr + N + row_idx
        out_fz_ptrs = out_f_ptr + 2*N + row_idx
        #TODO: do we really need atomic add here if the block size covers all columns?
        #      test this to make sure
        tl.store(out_fx_ptrs, fx, mask=row_idx < N)
        tl.store(out_fy_ptrs, fy, mask=row_idx < N)
        tl.store(out_fz_ptrs, fz, mask=row_idx < N)
    
    if calculate_dq:
        '''
        de_dq = erfc * charges[nbr_inds] * (nbr_inds != DUMMY_NBR_IND) / nbr_dists
        de_dq = torch.sum(de_dq, dim=1)
        '''
        de_dq = tl.sum(erfc * nbr_qs_over_dist)
        out_dq_ptrs = out_dq_ptr + row_idx
        tl.store(out_dq_ptrs, de_dq, mask=row_idx < N)

    #Energy calculation
    real_energy = tl.sum(res) / 2.0
    tl.atomic_add(out_en_ptr, real_energy)

def ewald_real(
    nbr_inds: torch.Tensor,
    nbr_diff_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int,
    out_f: Optional[torch.Tensor] = None, 
    out_dq: Optional[torch.Tensor] = None
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the real-space contribution to the Ewald summation.

    This function calculates the electrostatic interaction energy in the real-space 
    portion of the Ewald summation. It also optionally computes forces and derivatives 
    with respect to charge if `calculate_forces` or `calculate_dq` are set.

    Args:
        my_start_ind (int): Start index for local atoms.
        my_lcl_N (int): Number of local atoms being processed.
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
        out_f (Optional[torch.Tensor], optional): Preallocated tensor to store forces. Shape: `(3, N)`, default is `None`.
        out_dq (Optional[torch.Tensor], optional): Preallocated tensor to store charge derivatives. Shape: `(N,)`, default is `None`.

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]: 
            - **(float)** Real-space energy contribution (scalar).
            - **(torch.Tensor, shape `(3, N)`)** Forces on atoms if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(N,)`)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """
    # call the pytorch function if the tensors are on CPU
    if nbr_dists.device.type == 'cpu':
        return torch_ewald_real(nbr_inds, nbr_diff_vecs, nbr_dists, charges, 
               alpha, cutoff, calculate_forces,
               calculate_dq)
    
    _, M = nbr_inds.shape
    N = len(charges)
    BLOCK_SIZE = triton.next_power_of_2(M)
    if calculate_forces and out_f == None:
        out_f = torch.zeros((3, N), device=nbr_dists.device).type(nbr_dists.dtype)
    if calculate_dq and out_dq == None:
        out_dq = torch.zeros((N, ), device=nbr_dists.device).type(nbr_dists.dtype)
    y = torch.zeros((1,), device=nbr_dists.device).type(nbr_dists.dtype)
    ewald_real_space_kernel[(N,)](
        y,
        out_f,
        out_dq,
        nbr_inds,
        nbr_dists,
        nbr_diff_vecs[0],
        nbr_diff_vecs[1],
        nbr_diff_vecs[2],
        charges,
        alpha,
        cutoff,
        N,
        M,
        calculate_forces,
        calculate_dq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.item(), out_f, out_dq

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
    low_memory_mode: bool = True
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the Ewald sum energy and forces in a distributed way.

    This function calculates the real-space and reciprocal-space contributions to the 
    Ewald summation, including optional force and charge derivative calculations.

    Args:
        positions (torch.Tensor): Atomic positions. Shape: `(3, N)` or `(3, N)`, where:
            - `N` is the total number of atoms.
            - `3` represents x, y, z coordinates.
        cell (torch.Tensor): Simulation cell matrix. Shape: `(3, 3)`.
        my_nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `K` is the max number of neighbors per atom.
        nbr_disp_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)` or `(N, K, 3)`.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        charges (torch.Tensor): Charge per atom. Shape: `(N,)`.
        kvecs (torch.Tensor): Reciprocal space vectors. Shape: `(3, M)`, where:
            - `M` is the number of k-space vectors.
        I (torch.Tensor): Fourier-space prefactors. Shape: `(M,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for real-space interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int, optional): Flag to compute charge derivatives (`1` for True, `0` for False`). Defaults to `0`.
        low_memory_mode (bool, optional): Flag to control low memory mode for the Triton kernel. Defaults to `True`.
    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **total_ewald_e (float)**: Total ewald energy.
            - **forces (torch.Tensor, shape `(3, N)`, optional)**: Computed forces if `calculate_forces` is enabled, otherwise `None`.
                If the positions are provided as `(N, 3)`, the forces will be also  `(N, 3)`.
            - **dq (torch.Tensor, shape `(N,)`, optional)**: Charge derivatives if `calculate_dq` is enabled, otherwise `None`.

    """
    device = positions.device
    # As the internal functions expects (3, N), transpose the position tensor as needed
    transpose = False
    if positions.shape[1] == 3:
        transpose = True
        positions = positions.T.contiguous()

    # transpose the disp. vectors as needed
    if nbr_disp_vecs.shape[2] == 3:
        nbr_disp_vecs = nbr_disp_vecs.permute(2, 0, 1).contiguous()
    N = positions.shape[1]

    # call the pytorch function if the tensors are on CPU
    if positions.device.type == 'cpu':
        # triton and pytorch uses different memory layout for kvecs
        if kvecs.shape[0] == 3:
            kvecs = kvecs.T
        return torch_ewald_energy(positions, cell, 
                                    nbr_inds, nbr_disp_vecs, nbr_dists, 
                                    charges, kvecs, I, 
                                    alpha,
                                    cutoff, 
                                    calculate_forces,
                                    calculate_dq)
    
    my_e_real, my_f_real, my_dq_real = ewald_real(nbr_inds, nbr_disp_vecs, 
                                                nbr_dists, charges, alpha, cutoff,
                                                calculate_forces, calculate_dq)
    vol = torch.det(cell)
    alpha = torch.tensor(alpha)
    r_vals, i_vals, r_sum, i_sum = ewald_kspace_part1(positions, charges, kvecs,low_memory_mode=low_memory_mode)
    total_e_kspace, my_f_kspace, my_dq_kspace = ewald_kspace_part2(r_sum, i_sum, r_vals, i_vals, 
                                                    vol, kvecs, I, 
                                                    charges,
                                                    positions,
                                                    calculate_forces,
                                                    calculate_dq,
                                                    low_memory_mode=low_memory_mode)
    self_e, self_dq = ewald_self_energy(charges, alpha, calculate_dq)
    if calculate_forces:
        forces = (my_f_real + my_f_kspace) * CONV_FACTOR
    else:
        forces = None
    if calculate_dq:
        dq = (my_dq_kspace + my_dq_real + self_dq) * CONV_FACTOR
    else:
        dq = None

    # if user provided [N,3] positions, tranpose the forces to [N, 3]
    if transpose and calculate_forces:
        forces = forces.T.contiguous()
    
    total_ewald_e = (my_e_real + total_e_kspace + self_e) * CONV_FACTOR
    return total_ewald_e, forces, dq



