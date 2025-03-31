import functools

import numpy as np
import warnings
import torch

from hippynn.graphs import find_relatives, find_unique_relative, get_subgraph, copy_subgraph, replace_node, GraphModule
from hippynn.graphs.gops import check_link_consistency

from hippynn.graphs.nodes.base.node_functions import NodeOperationError
from hippynn.graphs.nodes.base import InputNode
from hippynn.graphs.nodes.tags import Encoder, AtomIndexer, PairIndexer
from hippynn.graphs.nodes.pairs import ExternalNeighborIndexer
from hippynn.graphs.nodes.physics import GradientNode
from hippynn.graphs.nodes.pairs import PairFilter

from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from hippynn.experiment.serialization import load_checkpoint
from ase import units
from sedacs.integrator import SystemState
from ase import Atoms

import time

def dublicate_state(state, dx, dy, dz):
    masses = torch.hstack([state.masses,] * (dx * dy * dz))
    new_cell = state.cell.clone()
    new_cell[0,0] *= dx
    new_cell[1,1] *= dy
    new_cell[2,2] *= dz
    types = torch.hstack([state.types,] * (dx * dy * dz))
    prev_pos = state.positions
    prev_cell = state.cell
    for ind, d in zip(range(3), [dx, dy, dz]):
        new_pos = []   
        for i in range(d):
            upd_pos = prev_pos.clone()
            upd_pos[:, ind] += prev_cell[ind,ind] * i
            new_pos.append(upd_pos)
        prev_pos = torch.vstack(new_pos)
        prev_cell = prev_cell.clone()
        prev_cell[ind, ind] *= d
    new_state = SystemState(prev_pos, types, masses, new_cell, use_shadow=state.use_shadow)
    new_state.velocities = torch.vstack([state.velocities,] * (dx * dy * dz))
    new_state.forces = torch.vstack([state.forces,] * (dx * dy * dz))
    new_state.charges = torch.hstack([state.charges,] * (dx * dy * dz))
    new_state.energy = state.energy
    if state.use_shadow:
        new_state.p_history = torch.hstack([state.p_history,] * (dx * dy * dz))
        new_state.delta_p = torch.hstack([state.delta_p,] * (dx * dy * dz))

    return new_state

def dublicate_ase_atoms(atoms, dx=1, dy=1, dz=1):
    orig_pos = atoms.get_positions(wrap=True).copy()
    orig_cell = atoms.cell.array.copy()
    orig_species = atoms.get_atomic_numbers()

    new_species = [orig_species, ] * (dx * dy * dz)
    new_species = np.hstack(new_species)

    new_cell = orig_cell.copy()
    new_cell[0,0] *= dx
    new_cell[1,1] *= dy
    new_cell[2,2] *= dz
    
    prev_pos = orig_pos
    prev_cell = orig_cell
    for ind, d in zip(range(3), [dx, dy, dz]):
        new_pos = []   
        for i in range(d):
            upd_pos = prev_pos.copy()
            upd_pos[:,ind] += prev_cell[ind,ind] * i
            new_pos.append(upd_pos)
        prev_pos = np.vstack(new_pos)
        prev_cell = prev_cell.copy()
        prev_cell[ind, ind] *= d
    new_atoms = Atoms(positions=prev_pos, numbers=new_species, cell=new_cell, pbc=(True, True, True))
    #new_atoms.pbc = atoms.pbc
    return new_atoms


def setup_graph(energy, charges=None, extra_properties=None):

    if charges is None:
        required_nodes = [energy]
    else:
        required_nodes = [energy, charges]

    if extra_properties is not None:
        extra_names = list(extra_properties.keys())
        for ename in extra_names:
            if not ename.isidentifier():
                raise ValueError("ASE properties must be a valid python identifier. (Got '{}')".format(ename))
        del ename
        required_nodes = required_nodes + list(extra_properties.values())

    why = "Generating ASE Calculator interface"
    subgraph = get_subgraph(required_nodes)

    search_fn = lambda targ, sg: lambda n: n in sg and isinstance(n, targ)

    try:
        pair_indexers = find_relatives(required_nodes, search_fn(PairIndexer, subgraph), why_desc=why)
    except NodeOperationError as ee:
        raise ValueError(
            "No Pair indexers found. Why build an ASE interface with no need for neighboring atoms?"
        ) from ee

    # The required nodes passed back are copies of the ones passed in.
    # We use assume_inputed to avoid grabbing pieces of the graph
    # that are only prerequisites for the pair indexer.
    new_required, new_subgraph = copy_subgraph(required_nodes, assume_inputed=pair_indexers)
    # We now need access to the copied indexers, rather than the originals
    pair_indexers = find_relatives(new_required, search_fn(PairIndexer, new_subgraph), why_desc=why)

    species = find_unique_relative(new_required, search_fn(SpeciesNode, new_subgraph), why_desc=why)
    positions = find_unique_relative(new_required, search_fn(PositionsNode, new_subgraph), why_desc=why)

    # TODO: is .clone necessary? Or good? Or torch.as_tensor instead?
    encoder = find_unique_relative(species, search_fn(Encoder, new_subgraph), why_desc=why)
    species_set = torch.as_tensor(encoder.species_set).to(torch.int64)  # works with lists or tensors
    indexer = find_unique_relative(species, search_fn(AtomIndexer, new_subgraph), why_desc=why)
    min_radius = max(p.dist_hard_max for p in pair_indexers)
    ###############################################################

    ###############################################################
    # Set up graph to accept external pair indices and shifts

    in_shift = InputNode("shift_vector")
    in_cell = CellNode("cell")
    in_pair_first = InputNode("pair_first")
    in_pair_second = InputNode("pair_second")
    external_pairs = ExternalNeighborIndexer(
        "external_neighbors",
        (positions, indexer.real_atoms, in_shift, in_cell, in_pair_first, in_pair_second),
        hard_dist_cutoff=min_radius,
    )
    new_inputs = [species, positions, in_cell, in_pair_first, in_pair_second, in_shift]

    # Construct Filters
    # Replace the existing pair indexers with the corresponding new (filtered) node
    # that accepts external pairs of atoms:
    # (This is the primary reason we needed to copy the subgraph --)
    #  we don't want to break the original computation, and `replace_node` mutates graph connectivity
    for pi in pair_indexers:
        if pi.dist_hard_max == min_radius:
            mapped_node = external_pairs
        else:
            mapped_node = PairFilter(
                "DistanceFilter_external_neighbors",
                (external_pairs),
                dist_hard_max=pi.dist_hard_max, 
            )
        replace_node(pi, mapped_node, disconnect_old=True)

    energy, *new_required = new_required
    '''
    cellscaleinducer = StrainInducer("Strain_inducer", (positions, in_cell))
    strain = cellscaleinducer.strain
    derivatives = StressForceNode("StressForceCalculator", (energy, strain, positions, in_cell))

    replace_node(positions, cellscaleinducer.strained_coordinates)
    replace_node(in_cell, cellscaleinducer.strained_cell)

    pbc_handler = PBCHandle(derivatives)
    '''
    forces = GradientNode("forces", (energy, positions), sign=-1)
    implemented_nodes = energy.main_output, forces.main_output
    implemented_properties = ["potential_energy", "forces"]
    #### Add other properties here:
    if extra_properties is not None:
        implemented_nodes = *implemented_nodes, *new_required
        implemented_properties = implemented_properties + extra_names

    ###############################################################

    # Finally, assemble the graph!
    check_link_consistency((*new_inputs, *implemented_nodes))
    mod = GraphModule(new_inputs, implemented_nodes)
    mod.eval()

    return mod, min_radius

def create_hippynn_model(parent, device, dtype):
    structure = load_checkpoint(f"{parent}/experiment_structure.pt", f"{parent}/best_checkpoint.pt", weights_only=False)
    training_modules = structure["training_modules"]
    model, loss, evaluator = training_modules 
    model = model.to(device)
    model = model.double()
    energy_node = model.node_from_name("energy")
    model, cutoff = setup_graph(energy_node, None, None)
    return model.to(dtype).to(device), cutoff

def calculate_hippynn_energy_forces(species, coords, cell, nbr_state, model):
    pair_shiftvecs = nbr_state.calculate_shift(coords).T
    pair_first = nbr_state.nbr_inds[0]
    pair_second = nbr_state.nbr_inds[1]
    inputs = species.unsqueeze(0), coords.T.unsqueeze(0), cell, pair_first, pair_second, pair_shiftvecs
    results = model(*inputs)

    energy = results[0][0] * (units.kcal / units.mol)
    forces = results[1][0] * (units.kcal / units.mol)

    return energy.detach(), forces.detach()