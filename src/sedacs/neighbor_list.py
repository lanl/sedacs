import torch
from torch import Tensor
import math 
import itertools
from typing import Union

DUMMY_IND = -1 # dummy neighbor index TODO: find a way do store this info

__all__ = ["generate_neighbor_list"]


def fractional_cell_size(lattice_vecs: Tensor, cutoff: float):
    xx = lattice_vecs[0, 0]
    yy = lattice_vecs[1, 1]
    zz = lattice_vecs[2, 2]
    xy = lattice_vecs[0, 1] / yy
    xz = lattice_vecs[0, 2] / zz
    yz = lattice_vecs[1, 2] / zz
    
    nx = xx / torch.sqrt(1 + xy**2 + (xy * yz - xz)**2)
    ny = yy / torch.sqrt(1 + yz**2)
    nz = zz
    
    nmin = torch.floor(torch.min(torch.tensor([nx, ny, nz])) / cutoff)
    nmin = torch.where(nmin == 0, 1, nmin)
    return 1 / nmin

def coords_cart_to_frac(cart_coords: Tensor, lattice_vecs: Tensor):
    '''
    Convert cart. coordinates to fractional ones
    '''
    A_transpose = lattice_vecs
    A_transpose_inv = torch.linalg.inv(A_transpose)
    frac_coords = torch.matmul(cart_coords, A_transpose_inv)
    return frac_coords

def coords_frac_to_cart(frac_coords: Tensor, lattice_vecs: Tensor):
    '''
    Convert frac. coordinates to cart. ones
    '''
    A_transpose = lattice_vecs
    cart_coords = torch.matmul(frac_coords, A_transpose)
    return cart_coords

def calculate_distance(coords: Tensor, candid_ids: Tensor, lattice_lengths: Tensor):
    '''
    Calculate distance to each candidate
    '''
    N = coords.shape[0]
    lattice_lengths = lattice_lengths[None,:]
    neigh_position = coords[candid_ids]
    disp = coords[:, None, :] - neigh_position
    # displacement trick (based on minumum image convention)
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    dists = torch.linalg.norm(disp, dim=2)
    return dists

def calculate_mask(candid_ids: Tensor, coords: Tensor, lattice_lengths: Tensor, cutoff: float):
    dists = calculate_distance(coords, candid_ids, lattice_lengths)
    mask = (dists < cutoff) & (candid_ids != -1)
    return mask

def self_mask(idx):
    '''
    Mask the self interactions
    '''
    self_mask = idx == torch.reshape(torch.arange(idx.shape[0], dtype=torch.int32, device=idx.device),
                                   (idx.shape[0], 1))
    return torch.where(self_mask, DUMMY_IND, idx)



def unflatten_cell_buffer(arr: Tensor,
                           cells_per_side: Tensor,
                           dim: int):  
    #cells_per_side = tuple([int(x) for x in torch.flip(cells_per_side,dims=(0,))])
    cells_per_side = tuple([int(x) for x in cells_per_side])

    return torch.reshape(arr, cells_per_side + (-1,))


def calculate_cell_dimensions(lattice_lengths: Tensor, min_cell_size: float):
    '''
    We need to modify the cell size to have a balanced clean split
    '''
    cells_per_side = torch.floor(lattice_lengths / min_cell_size).int()
    cell_size_per_dim = lattice_lengths / cells_per_side
    cell_count = torch.prod(cells_per_side)
    return cell_size_per_dim, cells_per_side, cell_count
    

def calculate_flattened_cell_offset(cells_per_side: Tensor):
    '''
    First increment is 1, second one is size of prev. dim
    last one is the mult. of prev 2 dims
    '''
    offsets = torch.ones_like(cells_per_side)
    offsets[1] = cells_per_side[0]
    offsets[0] = cells_per_side[0] * cells_per_side[1]
    return offsets.to(torch.int32)

@torch.compile(dynamic=True)
def count_flattened_cell_sizes(cell_inds: Tensor, lattice_lengths: Tensor, cell_size: float):
    '''
    Count # atoms per cell in a flattened fashion
    '''
    
    [cell_size_per_dim, 
     cells_per_side, 
     cell_count] = calculate_cell_dimensions(lattice_lengths, cell_size)
    # count the atom size in each box
    # to be able to use index add in one go, use the flattened cells (3d -> 1d)
    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # calculate the flat. cell ind. for each atom
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals, dtype=torch.int32, dim=1)
    flat_cell_sizes = torch.zeros(cell_count, dtype=torch.int32, device=cell_inds.device)
    # reduce the counts
    flat_cell_sizes = flat_cell_sizes.index_add_(0, particle_flat_cell_inds, 
                                                 torch.ones_like(particle_flat_cell_inds))    
    return flat_cell_sizes

def populate_cells(N: int, cell_inds: Tensor, cells_per_side: Tensor, cell_count: int, max_cell_capacity: int):
    '''
    Assign atoms to their cells, each cell stores the indices of the atoms it holds
    '''
    device=cell_inds.device
    atom_ids = torch.arange(N, device=device, dtype=torch.int32)
    
    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # atom to flat cell id
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals, dtype=torch.int32, dim=1)
    # sort to group the atoms which belong to the same cell together
    sorted_flat_cell_ids, sorted_flat_cell_id_map = torch.sort(particle_flat_cell_inds)
    # empty ones are DUMMY_IND, flat version of the cells
    cells = DUMMY_IND + torch.zeros((cell_count * max_cell_capacity,), dtype=torch.int32,
                             device=device)
    
    sorted_atom_ids = atom_ids[sorted_flat_cell_id_map]
    # find the exact spot for each atom in the cell index
    # Here we get the column indices using mod, it is collision free as we know
    # no cell has more atoms than the max capacity.
    sorted_cell_ids = atom_ids % max_cell_capacity 
    # for a matrix with [N,K]:
    # to go from 2d index (i, j) to flat index: i * K + j
    sorted_cell_ids = sorted_flat_cell_ids * max_cell_capacity + sorted_cell_ids
    
    cells[sorted_cell_ids] = sorted_atom_ids
    cells = unflatten_cell_buffer(cells, cells_per_side, 3)
    
    return cells


def shift_array(arr: Tensor, dindex: tuple):
    '''
    For each dimension, shift +1, -1 to concatanate neighbor cells
    '''
    dx, dy, dz = dindex
    
    if dx > 0:
        arr = torch.concatenate((arr[1:], arr[:1]))
    elif dx < 0:
        arr = torch.concatenate((arr[-1:], arr[:-1]))
    
    if dy > 0:
        arr = torch.concatenate((arr[:, 1:], arr[:, :1]), axis=1)
    elif dy < 0:
        arr = torch.concatenate((arr[:, -1:], arr[:, :-1]), axis=1)
    
    if dz > 0:
        arr = torch.concatenate((arr[:, :, 1:], arr[:, :, :1]), axis=2)
    elif dz < 0:
        arr = torch.concatenate((arr[:, :, -1:], arr[:, :, :-1]), axis=2)
    
    return arr

@torch.compile(dynamic=True)
def generate_candidates(cells: Tensor, N: int):
    '''
    Generate the candidate neighbors for each atom
    '''
    # go through 27 neighbors for each cell and concat. the neighboring cells together
    all_shifts = list(itertools.product(range(-1,2,1), repeat=3))
    all_shifts.remove((0,0,0))
    cell_nbr_candidates = [cells,]
    for (dx, dy, dz) in all_shifts:
        cell_nbr_candidates += [shift_array(cells, (dx, dy, dz))]
    # cx, cy, cz, num of candids
    # where num of candids = 27 * max cell capacity
    cell_nbr_candidates = torch.concatenate(cell_nbr_candidates, axis=-1)
    num_candids = cell_nbr_candidates.shape[-1]
    cell_dims, max_cell_capacity = cells.shape[:3], cells.shape[3]
    target_shape = (*cell_dims, max_cell_capacity, num_candids)
    # add new dimension for "max cell capacity"
    cell_nbr_candidates = cell_nbr_candidates[..., None, :]
    cell_nbr_candidates = torch.broadcast_to(cell_nbr_candidates, target_shape)
    # N+1 because of the "-1" values used for padding 
    neighbor_idx = DUMMY_IND + torch.zeros((N+1, num_candids), dtype=torch.int32, device=cells.device)
    scatter_indices = torch.reshape(cells, (-1,))
    nbr_candidates = torch.reshape(cell_nbr_candidates, (-1, num_candids))
    neighbor_idx[scatter_indices] = nbr_candidates
    #remove the extra row
    candid_ids = neighbor_idx[:-1]
    # mask out the self interactions
    candid_ids = self_mask(candid_ids)
    
    return candid_ids

@torch.compile(dynamic=True)
def generate_candidates_v2(cell_inds: Tensor, cells: Tensor, N: int):
    '''
    Generate the candidate neighbors for each atom
    '''
    # go through 27 neighbors for each cell and concat. the neighboring cells together
    all_shifts = list(itertools.product(range(-1,2,1), repeat=3))
    all_shifts.remove((0,0,0))
    cell_nbr_candidates = [cells,]
    for (dx, dy, dz) in all_shifts:
        cell_nbr_candidates += [shift_array(cells, (dx, dy, dz))]

    # cx, cy, cz, num of candids
    # where num of candids = 27 * max cell capacity
    cell_nbr_candidates = torch.concatenate(cell_nbr_candidates, axis=-1)
    candid_ids = cell_nbr_candidates[cell_inds[:,0],cell_inds[:,1],cell_inds[:,2], :]
    # mask out the self interactions
    candid_ids = self_mask(candid_ids)
    
    return candid_ids


@torch.compile(dynamic=True)
def create_sparse_neighbor_list(coords: Tensor, lattice_lengths: Tensor, candid_ids: Tensor, cutoff: float):
    '''
    Create COO based sparse neighbor list
    '''
    mask = calculate_mask(candid_ids, coords, lattice_lengths, cutoff)
    cumsum = torch.cumsum(mask, dim=1)
    max_occupancy = torch.max(cumsum[:, -1])
    
    index = torch.argwhere(mask)
    source, target = index[:,0], candid_ids[index[:,0], index[:,1]]
    return source, target

@torch.compile(dynamic=True)
def create_dense_neighbor_list(coords: Tensor, lattice_lengths: Tensor, candid_ids: Tensor, cutoff: float):
    '''
    Create ELLPACK based dense neighbor list
    '''
    mask = calculate_mask(candid_ids, coords, lattice_lengths, cutoff)
    cumsum = torch.cumsum(mask, dim=1)
    max_occupancy = torch.max(cumsum[:, -1])
    DUMMY_IND = -1
    
    out_idx = DUMMY_IND + torch.zeros(candid_ids.shape, dtype=torch.int32, device=coords.device)
    # This assumes the max_occupancy < # candidates, never equal
    # which should be the case
    index = torch.where(mask, cumsum - 1, candid_ids.shape[1] - 1)
    p_index = torch.arange(candid_ids.shape[0])[:, None]
    out_idx[p_index, index] = candid_ids
    
    return out_idx[:, :max_occupancy]

def generate_neighbor_list(coords: Tensor, lattice_vectors: Union[Tensor, None], cutoff: float, 
                           is_dense: bool = True,
                           rank: int = 0, num_ranks: int = 1):  
    '''
    Main function to generate neighbor list
    For single GPU/CPU: rank = 0, num_ranks = 1.
    For MPI, every process will hold the neighbors for its own atoms

    TODO: nonperiodic box part needs to be improved for small systems (if it is needed?)
    '''
    nats = len(coords)

    nats_per_rank = nats // num_ranks
    my_start = rank * nats_per_rank
    my_end = my_start + nats_per_rank
    if rank == num_ranks - 1:
        my_end = nats

    # shift the coords towards [0,0,0]
    coords = coords - coords.min(dim=0, keepdim=True)[0]
    if lattice_vectors != None:
        lattice_lengths = torch.linalg.norm(lattice_vectors, dim=1)
        is_periodic = True
    else:
        lattice_lengths = coords.max(dim=0)[0] + 1.0 # add some buffer room
        is_periodic = False

    cell_size_per_dim, cells_per_side, cell_count = calculate_cell_dimensions(lattice_lengths, cutoff)
    cell_inds = (coords / cell_size_per_dim[None, :]).to(torch.int32)
    
    cell_sizes = count_flattened_cell_sizes(cell_inds, lattice_lengths, cutoff)
    max_cell_capacity = torch.max(cell_sizes)
    multiple_of = 8
    max_cell_capacity = math.ceil(max_cell_capacity / multiple_of) * multiple_of
    
    cells = populate_cells(nats, cell_inds, cells_per_side, cell_count, max_cell_capacity)
    candid_ids = generate_candidates_v2(cell_inds[my_start:my_end], cells, nats)
    # trick to make the the nonperitodic work
    if is_periodic == False:
        lattice_lengths = lattice_lengths + (cutoff + 1.0)
    if is_dense:
        return create_dense_neighbor_list(coords, lattice_lengths, candid_ids, cutoff) 
    else:
        return create_sparse_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)