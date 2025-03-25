import torch
from torch import Tensor
import math
import itertools
from typing import Union
from dataclasses import dataclass
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def cell_shift(val, limit):
        if val == -1:
            return limit - 1
        elif val == limit:
            return 0
        else:
            return val


    @triton.jit
    def neighbor_list_kernel(
        positions_ptr,
        cell_list_ptr,
        neighbor_list_ptr,
        neighbor_count_ptr,
        N,
        cx_dim, cy_dim, cz_dim,
        num_candids,
        cutoff_sq,
        max_num_neighbors,
        cell_sizes_ptr,
        lattice_lengths_ptr,
        BLOCK_SIZE: tl.constexpr,
        only_count: tl.constexpr = False
    ):
        atom_id = tl.program_id(0)
        mask = atom_id < N

        # Load positions of atoms
        pos_x = tl.load(positions_ptr + atom_id, mask=mask)
        pos_y = tl.load(positions_ptr + N + atom_id, mask=mask)
        pos_z = tl.load(positions_ptr + 2*N + atom_id, mask=mask)

        cell_size_x = tl.load(cell_sizes_ptr)
        cell_size_y = tl.load(cell_sizes_ptr + 1)
        cell_size_z = tl.load(cell_sizes_ptr + 2)

        box_x = tl.load(lattice_lengths_ptr)
        box_y = tl.load(lattice_lengths_ptr + 1)
        box_z = tl.load(lattice_lengths_ptr + 2)

        # Compute cell indices for each atom
        cell_x = tl.cast(pos_x / cell_size_x, tl.int32)
        cell_y = tl.cast(pos_y / cell_size_y, tl.int32)
        cell_z = tl.cast(pos_z / cell_size_z, tl.int32)

        candid_offsets = tl.arange(0, BLOCK_SIZE)
        # Initialize neighbor counts
        neighbor_count = 0
        for s_x in range(-1, min(cx_dim - 1, 2)):
            for s_y in range(-1, min(cy_dim - 1, 2)):
                for s_z in range(-1, min(cz_dim - 1, 2)):
                    nx = cell_shift(cell_x + s_x, cx_dim)
                    ny = cell_shift(cell_y + s_y, cy_dim)
                    nz = cell_shift(cell_z + s_z, cz_dim)

                    # Compute cell index in flattened array
                    cell_flat_idx = ((nx * cy_dim * cz_dim) + (ny * cz_dim) + nz) * num_candids

                    # Load candidate indices
                    candid_ids = tl.load(cell_list_ptr + cell_flat_idx + candid_offsets, mask=candid_offsets < num_candids, other=-1)

                    valid_mask = (candid_ids != -1) & mask & (candid_ids != atom_id)

                    # Load candidate positions
                    cand_pos_x = tl.load(positions_ptr + candid_ids, mask=valid_mask, other=0.0)
                    cand_pos_y = tl.load(positions_ptr + N + candid_ids, mask=valid_mask, other=0.0)
                    cand_pos_z = tl.load(positions_ptr + 2*N + candid_ids, mask=valid_mask, other=0.0)

                    # Compute squared distances
                    dx = ((pos_x - cand_pos_x) + 0.5 * box_x) % box_x - 0.5 * box_x
                    dy = ((pos_y - cand_pos_y) + 0.5 * box_y) % box_y - 0.5 * box_y
                    dz = ((pos_z - cand_pos_z) + 0.5 * box_z) % box_z - 0.5 * box_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    within_cutoff = (dist_sq <= cutoff_sq) & valid_mask
                    within_cutoff_int = tl.cast(within_cutoff, tl.int32)
                    target_inds = tl.cumsum(within_cutoff_int) - 1
                    if only_count == False:
                        nbr_count_shift = neighbor_count + target_inds
                        nbr_ids = atom_id * max_num_neighbors + nbr_count_shift
                        store_mask = within_cutoff & (nbr_count_shift < max_num_neighbors)
                        tl.store(neighbor_list_ptr + nbr_ids, candid_ids, store_mask)
                    neighbor_count += tl.max(target_inds) + 1
        tl.store(neighbor_count_ptr + atom_id, neighbor_count)

def compute_neighbor_list_triton(positions, lattice_lengths, cell_list, cutoff, cell_lengths_per_dim, max_num_neighbors, only_count=False):
    N = positions.shape[1]

    cx_dim, cy_dim, cz_dim, num_candids = cell_list.shape

    cutoff_sq = cutoff ** 2
    if only_count == False:
        neighbor_list = torch.full((N, max_num_neighbors), -1, dtype=torch.int32, device=positions.device)
    else:
        neighbor_list = None
    neighbor_counts = torch.zeros(N, dtype=torch.int32, device=positions.device)
    #TODO: I assume the BLOCK_SIZE will not be large enough to not fit into the shared memory.
    # but this needs to be generalized using autotuning
    BLOCK_SIZE = triton.next_power_of_2(num_candids)

    neighbor_list_kernel[(N,)](
        positions,
        cell_list,
        neighbor_list,
        neighbor_counts,
        N,
        cx_dim, cy_dim, cz_dim,
        num_candids,
        cutoff_sq,
        max_num_neighbors,
        cell_lengths_per_dim,
        lattice_lengths,
        BLOCK_SIZE=BLOCK_SIZE,
        only_count=only_count
    )

    return neighbor_list, neighbor_counts



DUMMY_IND = -1 # dummy neighbor index TODO: find a way do store this info

__all__ = ["generate_neighbor_list", "NeighborState", "calculate_displacement"]

@torch.compile
def neigh_state_dist_check(coords1, coords2, lattice_lengths, buff):
    disp = (coords1 - coords2)
    lattice_lengths = lattice_lengths[:, None]
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    return torch.any(torch.sum(disp**2, dim=0) > (buff * 0.5)**2)

@torch.compile
def map2central(coordinates, cell, inv_cell):
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    coordinates_cell = torch.matmul(inv_cell, coordinates)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor()
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(cell, coordinates_cell)

@dataclass
class NeighborState:
    '''
    Packs the neighbor state.
    '''
    coords: Tensor
    lattice_vecs: Union[None, Tensor]
    nbr_inds: Union[None, Tensor]
    cutoff: float
    is_dense: bool = True
    buffer: float = 0.0
    reneighbor_cnt: int = 0
    use_triton: bool = False

    def __post_init__(self):
        # TODO: I think this is not necessary since we have map to central for position changes
        #self.coords = map2central(self.coords, self.lattice_vecs, torch.linalg.inv(self.lattice_vecs))
        
        # create the neighbor list using cutoff with buffer
        nbr_inds = generate_neighbor_list(self.coords,
                                        self.lattice_vecs,
                                        self.cutoff + self.buffer,
                                        self.is_dense,
                                        self.use_triton)
        self.nbr_inds = nbr_inds
        # TODO: trick to make displacement works, improve later
        if self.lattice_vecs != None:
            self.lattice_lengths = torch.linalg.norm(self.lattice_vecs, dim=1)
        else:
            self.lattice_lengths = self.coords.max(dim=1)[0] - self.coords.min(dim=1)[0] + 1.0
        self.reneighbor_cnt += 1
        
    def update(self, new_coords):
        # TODO: I think this is not necessary since we have map to central for position changes
        #new_coords = map2central(new_coords, self.lattice_vecs, torch.linalg.inv(self.lattice_vecs))
        
        #print(dist.abs().max(), self.reneighbor_cnt)
        # if any atom moved more than half "buffer", update the nbr list
        #if torch.any(torch.sum((self.coords - new_coords)**2, dim=0) > (self.buffer * 0.5)**2):
        #if True:
        if neigh_state_dist_check(self.coords, new_coords, self.lattice_lengths, self.buffer):
            nbr_inds = generate_neighbor_list(new_coords,
                                              self.lattice_vecs,
                                              self.cutoff + self.buffer,
                                              self.is_dense,
                                              self.use_triton)
            self.nbr_inds = nbr_inds
            self.coords = new_coords
            self.reneighbor_cnt += 1

    def calculate_displacement(self, coords):
        if self.is_dense:
            return calculate_displacement(coords, self.nbr_inds, self.lattice_lengths)
        else:
            return calculate_displacement_sparse(coords, self.nbr_inds, self.lattice_lengths)

    def calculate_distance(self, coords):
        disp = self.calculate_displacement(coords)
        return torch.linalg.norm(disp, dim=0)

    def calculate_shift(self, coords):
        if self.is_dense:
            return calculate_shift(coords, self.nbr_inds, self.lattice_lengths)
        else:
            return calculate_shift_sparse(coords, self.nbr_inds, self.lattice_lengths)

@torch.compile
def calculate_displacement(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    '''
    Calculate the displacement vectors for each neighbor (provided as NxK tensor)
    '''
    lattice_lengths = lattice_lengths[:, None, None]
    neigh_position = coords[:, nbr_ids]
    disp = neigh_position - coords[:, :, None]
    # displacement trick (based on minumum image convention)
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    return disp

@torch.compile
def calculate_displacement_sparse(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    '''
    Calculate the displacement vectors for each neighbor (provided as 2xK tensor)
    '''
    lattice_lengths = lattice_lengths[:, None]
    my_position = coords[:, nbr_ids[0]]
    neigh_position = coords[:, nbr_ids[1]]
    disp = neigh_position - my_position
    # displacement trick (based on minumum image convention)
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    return disp

@torch.compile
def calculate_shift(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    lattice_lengths = lattice_lengths[:, None, None]
    neigh_position = coords[:, nbr_ids]
    disp = neigh_position - coords[:, :, None]
    box_offset = -torch.floor((disp + 0.5 * lattice_lengths) / lattice_lengths)
    return box_offset

@torch.compile
def calculate_shift_sparse(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    lattice_lengths = lattice_lengths[:, None]
    my_position = coords[:, nbr_ids[0]]
    neigh_position = coords[:, nbr_ids[1]]
    disp = neigh_position - my_position
    box_offset = -torch.floor((disp + 0.5 * lattice_lengths) / lattice_lengths)
    return box_offset

@torch.compile
def calculate_distance(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    '''
    Calculate distance to each neighbor (provided as NxK tensor)
    '''
    disp = calculate_displacement(coords, nbr_ids, lattice_lengths)
    dists = torch.linalg.norm(disp, dim=0)
    return dists

@torch.compile
def calculate_distance_sparse(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor):
    '''
    Calculate distance to each neighbor (provided as 2xK tensor)
    '''
    disp = calculate_displacement_sparse(coords, nbr_ids, lattice_lengths)
    dists = torch.linalg.norm(disp, dim=0)
    return dists

@torch.compile
def calculate_mask(coords: Tensor, nbr_ids: Tensor, lattice_lengths: Tensor, cutoff: float):
    dists = calculate_distance(coords, nbr_ids, lattice_lengths)
    mask = (dists < cutoff) & (nbr_ids != -1)
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
    offsets[1] = cells_per_side[2]
    offsets[0] = cells_per_side[1] * cells_per_side[2]
    return offsets.to(torch.int32)

#@torch.compile(dynamic=True)
#TODO: torch compile fails with pytorch 2.5 for some reason
# investigate
def count_flattened_cell_sizes(cell_inds: Tensor, lattice_lengths: Tensor, cell_size: float):
    '''
    Count # atoms per cell in a flattened fashion
    cell_inds: 3xN Tensor, maps atom to specific cell
    lattice_lengths: [3,] Tensor
    '''

    [cell_size_per_dim,
     cells_per_side,
     cell_count] = calculate_cell_dimensions(lattice_lengths, cell_size)
    # count the atom size in each box
    # to be able to use index add in one go, use the flattened cells (3d -> 1d)
    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # calculate the flat. cell ind. for each atom
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals[:, None], dtype=torch.int32, dim=0)
    flat_cell_sizes = torch.zeros(cell_count, dtype=torch.int32, device=cell_inds.device)
    # reduce the counts
    flat_cell_sizes = flat_cell_sizes.index_add_(0, particle_flat_cell_inds,
                                                 torch.ones_like(particle_flat_cell_inds))
    return flat_cell_sizes

def populate_cells(cell_inds: Tensor, cells_per_side: Tensor, cell_count: int, max_cell_capacity: int):
    '''
    Assign atoms to their cells, each cell stores the indices of the atoms it holds
    cell_inds: 3xN Tensor, maps atom to specific cell
    cells_per_side: [3,] Tensor, contains # cells per dim
    cell_count: # cells
    '''
    device=cell_inds.device
    N = cell_inds.shape[1]
    atom_ids = torch.arange(N, device=device, dtype=torch.int32)

    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # atom to flat cell id
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals[:,None], dtype=torch.int32, dim=0)
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
def generate_candidates(cells: Tensor, N: int, all_shifts: list):
    '''
    Generate the candidate neighbors for each atom
    cells: [nx,ny,nz,max cell capacity] where nx,ny,nz is number of cells in each dimension
    '''
    # go through 27 neighbors for each cell and concat. the neighboring cells together
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
def generate_candidates_v2(cell_inds: Tensor, cells: Tensor, all_shifts: list):
    '''
    Generate the candidate neighbors for each atom
    cell_inds: 3xN Tensor, maps atom to specific cell
    cells: [nx,ny,nz,max cell capacity] where nx,ny,nz is number of cells in each dimension
    '''
    # go through 27 neighbors for each cell and concat. the neighboring cells together
    cell_nbr_candidates = [cells,]
    for (dx, dy, dz) in all_shifts:
        cell_nbr_candidates += [shift_array(cells, (dx, dy, dz))]

    # cx, cy, cz, num of candids
    # where num of candids = 27 * max cell capacity
    cell_nbr_candidates = torch.concatenate(cell_nbr_candidates, axis=-1)
    candid_ids = cell_nbr_candidates[cell_inds[0],cell_inds[1],cell_inds[2], :]
    # mask out the self interactions
    candid_ids = self_mask(candid_ids)

    return candid_ids


@torch.compile(dynamic=True)
def create_sparse_neighbor_list(coords: Tensor, lattice_lengths: Tensor, candid_ids: Tensor, cutoff: float):
    '''
    Create COO based sparse neighbor list
    coords: 3xN Tensor
    lattice_lengths: [3,] Tensor
    candid_ids: NxK Tensor, K is # candidates
    '''
    mask = calculate_mask(coords, candid_ids, lattice_lengths, cutoff)
    cumsum = torch.cumsum(mask, dim=1)
    max_occupancy = torch.max(cumsum[:, -1])

    index = torch.argwhere(mask)
    source, target = index[:,0], candid_ids[index[:,0], index[:,1]]
    return torch.stack((source, target))

@torch.compile(dynamic=True)
def create_dense_neighbor_list(coords: Tensor, lattice_lengths: Tensor, candid_ids: Tensor, cutoff: float):
    '''
    Create ELLPACK based dense neighbor list
    coords: 3xN Tensor
    lattice_lengths: [3,] Tensor
    candid_ids: NxK Tensor, K is # candidates

    '''
    mask = calculate_mask(coords, candid_ids, lattice_lengths, cutoff)
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

@torch.compile(dynamic=True)
def create_dense_neighbor_list_triton(candid_ids: Tensor, nbr_counts: Tensor):
    '''
    Create ELLPACK based dense neighbor list
    coords: 3xN Tensor
    lattice_lengths: [3,] Tensor
    candid_ids: NxK Tensor, K is # candidates

    '''
    max_occupancy = torch.max(nbr_counts)
    return candid_ids[:, :max_occupancy]

@torch.compile(dynamic=True)
def create_sparse_neighbor_list_triton(candid_ids: Tensor, nbr_counts: Tensor):
    '''
    Create COO based sparse neighbor list
    coords: 3xN Tensor
    lattice_lengths: [3,] Tensor
    candid_ids: NxK Tensor, K is # candidates
    '''
    candid_ids = create_dense_neighbor_list_triton(candid_ids, nbr_counts)
    mask = candid_ids != -1
    index = torch.argwhere(mask)
    source, target = index[:,0], candid_ids[index[:,0], index[:,1]]
    return torch.stack((source, target))

def generate_neighbor_list(coords: Tensor, lattice_vectors: Union[Tensor, None], cutoff: float,
                           is_dense: bool = True,
                           use_triton: bool = False):
    """
    Generates a neighbor list for a given set of coordinates, considering periodic or non-periodic boundary conditions.

    This function constructs a neighbor list by dividing the simulation space into cells and identifying neighboring 
    particles within a specified cutoff distance. It supports both dense and sparse neighbor list formats and provides 
    an optional Triton-based implementation for performance optimization.

    Args:
        coords (torch.Tensor): Atomic coordinates. Shape: `(3, N)`, where:
            - `N` is the number of atoms.
            - `3` represents the x, y, and z coordinates.
        lattice_vectors (torch.Tensor or None): Lattice vectors defining periodic boundary conditions. Shape: `(3, 3)`, 
            where each row represents a lattice basis vector. If `None`, the system is treated as non-periodic.
        cutoff (float): Cutoff distance for neighbor search (scalar).
        is_dense (bool, optional): If `True`, returns a dense neighbor list; otherwise, a sparse representation is used. 
            Defaults to `True`.
        use_triton (bool, optional): If `True`, uses a Triton-based implementation for neighbor list computation. 
            Defaults to `False`.

    Returns:
        - **If `is_dense=True`**:
            - `torch.Tensor` (shape `(N, K)`) - Dense neighbor list, where:
                - `N` is the number of atoms.
                - `K` is the maximum number of neighbors per atom.
        - **If `is_dense=False`**:
            - `torch.Tensor` (shape `(2, M)`)
                - `M` is number of pairs in COO format.
    TODO:
        - Improve handling of non-periodic small systems.
        - max_nbr_limit needs to be calculated dynamically.
    """
    N = coords[1]
    # shift the coords towards [0,0,0]
    # TODO: I think this may not be necessary since we have map to central for position changes
    coords = coords - coords.min(dim=1, keepdim=True)[0]
    if lattice_vectors != None:
        lattice_lengths = torch.linalg.norm(lattice_vectors, dim=1)
        is_periodic = True
    else:
        lattice_lengths = coords.max(dim=1)[0] + 1.0 # add some buffer room
        is_periodic = False

    cell_size_per_dim, cells_per_side, cell_count = calculate_cell_dimensions(lattice_lengths, cutoff)
    cell_inds = (coords / cell_size_per_dim[:, None]).to(torch.int32)
    cells_per_side_np = cells_per_side.cpu().numpy()

    cell_sizes = count_flattened_cell_sizes(cell_inds, lattice_lengths, cutoff)
    max_cell_capacity = torch.max(cell_sizes)
    #multiple_of = 8
    #max_cell_capacity = math.ceil(max_cell_capacity / multiple_of) * multiple_of

    cells = populate_cells(cell_inds, cells_per_side, cell_count, max_cell_capacity)
    if use_triton == False:
        all_shifts = []
        for i in range(-1, min(cells_per_side_np[0]-1, 2)):
            for j in range(-1, min(cells_per_side_np[1]-1, 2)):
                for k in range(-1, min(cells_per_side_np[2]-1, 2)):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    all_shifts.append((i,j,k))
        #all_shifts = torch.tensor(all_shifts)
        candid_ids = generate_candidates_v2(cell_inds, cells, all_shifts)
        # trick to make the the nonperitodic work
        if is_periodic == False:
            lattice_lengths = lattice_lengths + (cutoff + 1.0)
        if is_dense:
            return create_dense_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)
        else:
            return create_sparse_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)
    else:
        max_nbr_limit = int(0.1 * cutoff**3 * 4/3 * torch.pi * 1.5)
        candid_ids, nbr_counts = compute_neighbor_list_triton(coords, lattice_lengths, cells, cutoff, 
                                                              cell_size_per_dim,  max_nbr_limit, only_count=False)
        if is_dense:
            return create_dense_neighbor_list_triton(candid_ids, nbr_counts)
        else:
            return create_sparse_neighbor_list_triton(candid_ids, nbr_counts)
