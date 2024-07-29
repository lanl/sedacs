import torch
from torch import Tensor
import numpy as np
from util import generate_system, generate_nbr_list
#from matscipy.neighbours import neighbour_list
import time
import math 

DUMMY_IND = -1
'''
def matscipy_nbr_list(coords, box, cutoff):
    pbc = np.array([True, True, True])
    sender, receiver = neighbour_list(quantities="ij",
                                          pbc=pbc,
                                          cell=box,
                                          positions=coords,
                                          cutoff=cutoff)
    return sender, receiver
'''
def fractional_cell_size(lattice_vecs, cutoff):
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

def coords_cart_to_frac(cart_coords, lattice_vecs):
    '''
    Convert cart. coordinates to fractional ones
    '''
    A_transpose = lattice_vecs
    A_transpose_inv = torch.linalg.inv(A_transpose)
    frac_coords = torch.matmul(cart_coords, A_transpose_inv)
    return frac_coords

def coords_frac_to_cart(frac_coords, lattice_vecs):
    '''
    Convert frac. coordinates to cart. ones
    '''
    A_transpose = lattice_vecs
    cart_coords = torch.matmul(frac_coords, A_transpose)
    return cart_coords

def self_mask(idx):
    '''
    Mask the self interactions
    '''
    self_mask = idx == torch.reshape(torch.arange(idx.shape[0], dtype=torch.int32, device=idx.device),
                                   (idx.shape[0], 1))
    return torch.where(self_mask, DUMMY_IND, idx)



def unflatten_cell_buffer(arr,
                           cells_per_side,
                           dim: int):  
    cells_per_side = tuple([int(x) for x in torch.flip(cells_per_side,dims=(0,))])

    return torch.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])


def calculate_cell_dimensions(lattice_lengths, min_cell_size):
    '''
    We need to modify the cell size to have a balanced clean split
    '''
    cells_per_side = torch.floor(lattice_lengths / min_cell_size).int()
    cell_size_per_dim = lattice_lengths / cells_per_side
    cell_count = torch.prod(cells_per_side)
    
    return cell_size_per_dim, cells_per_side, cell_count
    

def calculate_flattened_cell_offset(cells_per_side):
    '''
    First increment is 1, second one is size of prev. dim
    last one is the mult. of prev 2 dims
    '''
    offsets = torch.ones_like(cells_per_side)
    offsets[1] = cells_per_side[0]
    offsets[2] = cells_per_side[0] * cells_per_side[1]
    return offsets.to(torch.int32)

def count_flattened_cell_sizes(coords, lattice_vectors, cell_size):
    '''
    Count # atoms per cell in a flattened fashion
    '''
    lattice_lengths = torch.linalg.norm(lattice_vectors, axis=1)
    
    [cell_size_per_dim, 
     cells_per_side, 
     cell_count] = calculate_cell_dimensions(lattice_lengths, cell_size)
    # count the atom size in each box
    cell_inds = (coords / cell_size_per_dim[None, :]).to(torch.int32)
    # to be able to use index add in one go, flattened to cells and indices
    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # calculate the flat. cell ind. for each atom
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals, dtype=torch.int32, dim=1)
    flat_cell_sizes = torch.zeros(cell_count, dtype=torch.int32, device=coords.device)
    # reduce the counts
    flat_cell_sizes = flat_cell_sizes.index_add_(0, particle_flat_cell_inds, 
                                                 torch.ones_like(particle_flat_cell_inds))    
    return flat_cell_sizes

@torch.compile(dynamic=True)
def populate_cells(coords, cell_size_per_dim, cells_per_side, cell_count, max_cell_capacity):
    '''
    Assign atoms to their cells, each cell stores the indices of the atoms it holds
    '''
    N = coords.shape[0]
    device=coords.device
    atom_ids = torch.arange(N, device=device, dtype=torch.int32)
    
    cell_inds = (coords / cell_size_per_dim).to(torch.int32)
    offset_vals = calculate_flattened_cell_offset(cells_per_side)
    # atom to flat cell id
    particle_flat_cell_inds = torch.sum(cell_inds * offset_vals, dtype=torch.int32, dim=1)
    # sort to group the atoms which belong to the same cell together
    sorted_flat_cell_ids, sorted_flat_cell_id_map = torch.sort(particle_flat_cell_inds)
    # empty ones are DUMMY_IND, flat version of the cells
    cells = DUMMY_IND + torch.zeros((cell_count * max_cell_capacity, 1), dtype=torch.int32,
                             device=device)
    
    sorted_atom_ids = atom_ids[sorted_flat_cell_id_map]
    sorted_atom_ids = torch.reshape(sorted_atom_ids, (N, 1))
    # find the exact spot for each atom in the cell index
    # Here we get the column indices using mod, it is collision free as we know no
    # cell has more than the max capacity.
    sorted_cell_ids = atom_ids % max_cell_capacity 
    # for a matrix with [N,K]:
    # to go from 2d index (i, j) to flat index: i * K + j
    sorted_cell_ids = sorted_flat_cell_ids * max_cell_capacity + sorted_cell_ids
    
    cells[sorted_cell_ids] = sorted_atom_ids
    cells = unflatten_cell_buffer(cells, cells_per_side, 3)
    
    return cells

@torch.compile(dynamic=True)
def shift_array(arr, dindex):
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
def generate_candidates(cells, N):
    '''
    Generate the candidate neighbors for each atom
    '''
    # go through 27 neighbors for each cell and concat. the neighboring cells together
    def neighboring_cells():
      for dindex in np.ndindex(3,3,3):
        yield torch.tensor(dindex, dtype=torch.int32, device=cells.device) - 1
    idx = cells
    cell_idx = [idx,]
    for dindex in neighboring_cells():
        if torch.all(dindex == 0):
            continue
        cell_idx += [shift_array(idx, dindex)]
    cell_idx = torch.concatenate(cell_idx, axis=-2)
    
    cell_idx = cell_idx[..., None, :, :]
    cell_idx = torch.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])
    
    def copy_values_from_cell(value, cell_value, cell_id):
      scatter_indices = torch.reshape(cell_id, (-1,))
      cell_value = torch.reshape(cell_value, (-1,) + cell_value.shape[-2:])
      value[scatter_indices] = cell_value
      return value
    
    neighbor_idx = DUMMY_IND + torch.zeros((N + 1,) + cell_idx.shape[-2:], dtype=torch.int32, device=device)
    neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
    return neighbor_idx[:-1, :, 0]

@torch.compile(dynamic=True)
def create_sparse_neighbor_list(coords, lattice_lengths, candid_ids, cutoff: float):
    N = coords.shape[0]
    lattice_lengths = lattice_lengths[None,:]
    neigh_position = coords[candid_ids]
    disp = coords[:, None, :] - neigh_position
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    dists = torch.linalg.norm(disp, dim=2)
    mask = (dists < cutoff) & (candid_ids != -1)
    cumsum = torch.cumsum(mask, dim=1)
    max_occupancy = torch.max(cumsum[:, -1])
    
    index = torch.argwhere(mask)
    source, target = index[:,0], candid_ids[index[:,0], index[:,1]]
    return source, target

@torch.compile(dynamic=True)
def create_dense_neighbor_list(coords, lattice_lengths, candid_ids, cutoff: float):
    N = coords.shape[0]
    lattice_lengths = lattice_lengths[None,:]
    neigh_position = coords[candid_ids]
    disp = coords[:, None, :] - neigh_position
    disp = ((disp + 0.5 * lattice_lengths) % lattice_lengths) - 0.5 * lattice_lengths
    dists = torch.linalg.norm(disp, dim=2)
    mask = (dists < cutoff) & (candid_ids != -1)
    
    cumsum = torch.cumsum(mask, dim=1)
    max_occupancy = torch.max(cumsum[:, -1])
    DUMMY_IND = -1
    
    out_idx = DUMMY_IND + torch.zeros(candid_ids.shape, dtype=torch.int32, device=coords.device)
    index = torch.where(mask, cumsum - 1, candid_ids.shape[1] - 1)
    p_index = torch.arange(candid_ids.shape[0])[:, None]
    out_idx[p_index, index] = candid_ids
    
    return out_idx[:, :max_occupancy]


    

density = 0.1
cutoff = 10.0
cell_size = 10.0
device="cuda"
np_dtype = np.float32
torch_dtype = torch.float32

for i in range(10):
    N = 100000 + i
    coords_orig, lattice_vectors_orig = generate_system(N, density, dtype=np_dtype)
    lattice_vectors_orig = lattice_vectors_orig
    use_fraq=False
    lattice_vectors_orig = torch.from_numpy(lattice_vectors_orig).to(device)
    coords_orig = torch.from_numpy(coords_orig).to(device)

    start = time.perf_counter()

    lattice_vectors = lattice_vectors_orig
    coords = coords_orig

        
    lattice_lengths = torch.linalg.norm(lattice_vectors, dim=1)
    cell_size_per_dim, cells_per_side, cell_count = calculate_cell_dimensions(lattice_lengths, cell_size)

    cell_sizes = count_flattened_cell_sizes(coords, lattice_vectors, cell_size)
    max_cell_capacity = torch.max(cell_sizes)

    cells = populate_cells(coords, cell_size_per_dim, cells_per_side, cell_count, max_cell_capacity)

    candid_ids = generate_candidates(cells, N)
    candid_ids = self_mask(candid_ids)

    N = coords.shape[0]
    s,t = create_sparse_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)
    #new_nbr_2d = create_dense_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)
    torch.cuda.synchronize()
    end = time.perf_counter()
    #new_nbr_2d = create_dense_neighbor_list(coords, lattice_lengths, candid_ids, cutoff)
    print(end-start)
    '''
    nbr_list_2d, id1, id2 = generate_nbr_list(coords.cpu().numpy(), lattice_vectors.cpu().numpy(), cutoff,device=device)
    nbr_list_2d = torch.sort(nbr_list_2d, dim=1)[0]
    new_nbr_2d = torch.sort(new_nbr_2d, dim=1)[0]

    print(torch.all(nbr_list_2d == new_nbr_2d))
    '''