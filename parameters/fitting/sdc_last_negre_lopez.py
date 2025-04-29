#!/usr/bin/env python3
"""
SDC Main Grove: Tight-binding and force-field optimization framework.

This script optimizes bond integrals, pseudopotentials, electron configurations, and dispersion potentials
to match target reference forces and energies (typically from SIESTA calculations).

Developed for atomistic machine learning and tight-binding modeling.

Author: Christian Negre and Alejandro Lopez
"""

# Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

import logging
import sys
from sedacs.periodic_table import PeriodicTable
# from sdc_system import *
# from sdc_ptable import ptable
import ctypes as ct
from sedacs.system import *
from gpmd import *
import numpy as np
from sedacs.system import *
import os
import scipy.linalg as sp
import sedacs.file_io as fileio
from scipy.optimize import minimize
from scipy import optimize
from gpmd import *
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import dual_annealing


def get_energy(coords_in, lattice_vectors, atom_types, symbols):
    """
    Calculate the total energy for a given molecular configuration.

    Args:
        coords_in (np.ndarray): Flattened atomic coordinates.
        lattice_vectors (np.ndarray): Lattice vectors defining the simulation box.
        atom_types (list): List of atomic types.
        symbols (list): List of atomic symbols.

    Returns:
        float: Total energy of the configuration.
    """
    field = np.zeros(3)
    coords = coords_in.reshape(len(atom_types), 3)
    err, charges_out, forces_saved, dipole_out, energy = gpmd(
        lattice_vectors, symbols, atom_types, coords, field, verb
    )

    # Write trajectory to file
    with open("traj.xyz", 'a+') as traj_file:
        traj_file.write(f"{len(atom_types)}\n")
        traj_file.write("t=\n")
        for i, atom in enumerate(atom_types):
            traj_file.write(f"{symbols[atom]} {coords[i, 0]} {coords[i, 1]} {coords[i, 2]}\n")
        traj_file.flush()
    # Write energy to file
    with open("energ.dat", 'a+') as energy_file:
        energy_file.write(f"{energy}\n")
        energy_file.flush()

    return energy


def optimize_coordinates(lattice_vectors, atom_types, symbols, coords, method='CG', iterations=10000, tol=0.01):
    """
    Optimize atomic coordinates to minimize energy.

    Args:
        lattice_vectors (np.ndarray): Lattice vectors defining the simulation box.
        atom_types (list): List of atomic types.
        symbols (list): List of atomic symbols.
        coords (np.ndarray): Initial atomic coordinates.
        method (str): Optimization method (default is 'CG').
        iterations (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        np.ndarray: Optimized atomic coordinates.
    """
    coordsv = coords.reshape(-1)
    result = minimize(
        get_energy, coordsv, args=(lattice_vectors, atom_types, symbols),
        method=method, tol=tol, options={'maxiter': iterations}
    )
    return result.x


def read_file_to_dict(filename):
    """
    Parse a structured file into a list of dictionaries.

    Args:
        filename (str): File path to read.

    Returns:
        list[dict]: List of dictionaries containing parsed data.
    """
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()

    # Extract headers from the file
    headers = lines[0].split()
    num_headers = 3
    first_set = headers[num_headers:num_headers + 8]
    second_set = [f"{header}_2" for header in first_set]
    all_headers = headers[:num_headers] + first_set + second_set

    # Parse lines into dictionaries
    dict_list = []
    for line in lines[1:]:
        values = line.split()
        data_dict = {all_headers[i]: (values[i] if i < num_headers else float(values[i]))
                     for i in range(len(all_headers))}
        dict_list.append(data_dict)

    return dict_list


def get_dipole(coords, charges):
    """
    Calculate the dipole moment.

    Args:
        coords (np.ndarray): Atomic coordinates.
        charges (np.ndarray): Atomic charges.

    Returns:
        np.ndarray: Dipole moment vector.
    """
    dipole = np.zeros(3)
    for i, charge in enumerate(charges):
        dipole += charge * coords[i]
    return dipole


def get_field_born_charges(lattice_vectors, symbols, atom_types, coords, verb):
    """
    Compute Born effective charges using the field method.

    Args:
        lattice_vectors (np.ndarray): Lattice vectors defining the simulation box.
        symbols (list): List of atomic symbols.
        atom_types (list): List of atomic types.
        coords (np.ndarray): Atomic coordinates.
        verb (int): Verbosity level.

    Returns:
        np.ndarray: Born effective charges.
    """
    # Initialize variables
    field = np.zeros(3)
    err, charges_out, forces_saved, dipole_out, energy = gpmd(
        lattice_vectors, symbols, atom_types, coords, field, verb
    )
    nats = len(atom_types)
    born_field = np.zeros((nats, 3, 3))

    # Compute forces with perturbed fields
    for i in range(3):
        for delta in [-1e-4, 1e-4]:
            field[i] = delta
            err, charges_out, forces_out, dipole_out, energy = gpmd(
                lattice_vectors, symbols, atom_types, coords, field, verb
            )
            born_field[:, :, i] += (forces_out - forces_saved) / np.linalg.norm(field)

    # Calculate average Born charges
    born_charges = [sum(sp.eigh(born_field[i])[0]) / 3 for i in range(nats)]
    return np.array(born_charges)


def read_ppotfile_to_dict(filename):
    """
    Parse a pseudopotential file into a list of dictionaries.

    Args:
        filename (str): File path to read.

    Returns:
        list[dict]: List of dictionaries containing pseudopotential data.
    """
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()

    # Extract the headers (column names)
    headers = lines[0].split()

    # Find the index where the second set of repeated headers starts (after the third column)
    all_headers = headers
    # Initialize a list to store all the dictionaries
    dict_list = []

    # Process the remaining lines to create dictionaries
    for line in lines[1:]:
        values = line.split()

        # Create a dictionary using headers as keys
        # Convert numerical values (from the 4th column onward) to floats
        data_dict = {all_headers[i]: (values[i] if i < 2 else float(values[i]))
                     for i in range(len(all_headers))}

        # Add the dictionary to the list
        dict_list.append(data_dict)

    return dict_list


def read_dpotfile_to_dict(filename):
    """
    Parse a dispersion potential file into a list of dictionaries.

    Args:
        filename (str): File path to read.

    Returns:
        list[dict]: List of dictionaries containing pseudopotential data.
    """
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()

    # Extract the headers (column names)
    headers = lines[0].split()

    # Find the index where the second set of repeated headers starts (after the third column)
    all_headers = headers
    # Initialize a list to store all the dictionaries
    dict_list = []

    # Process the remaining lines to create dictionaries
    for line in lines[1:]:
        values = line.split()

        # Create a dictionary using headers as keys
        # Convert numerical values (from the 4th column onward) to floats
        data_dict = {all_headers[i]: (values[i] if i < 2 else float(values[i]))
                     for i in range(len(all_headers))}

        # Add the dictionary to the list
        dict_list.append(data_dict)

    return dict_list




def get_displ_born_charges(lattice_vectors, symbols, atom_types, coords_in, verb):
    """
    Compute Born effective charges using the displacement method.

    Args:
        lattice_vectors (np.ndarray): Lattice vectors defining the simulation box.
        symbols (list): List of atomic symbols.
        atom_types (list): List of atomic types.
        coords_in (np.ndarray): Initial atomic coordinates.
        verb (int): Verbosity level.

    Returns:
        tuple: (Born effective charges, displacement matrix).
    """
    nats = len(atom_types)
    born_displ = np.zeros((nats, 3, 3))
    dspl = 1.0E-5
    field = np.zeros(3)
    coords = np.copy(coords_in)

    # Calculate dipole for initial configuration
    _, _, _, dipole_saved, _ = gpmd(lattice_vectors, symbols, atom_types, coords, field, verb)

    # Perturb coordinates and compute dipole differences
    for dim in range(3):  # Iterate over X, Y, Z
        for j in range(nats):
            for delta in [-dspl, dspl]:
                coords[j, dim] += delta
                _, _, _, dipole_out, _ = gpmd(lattice_vectors, symbols, atom_types, coords, field, verb)
                sign = 1 if delta > 0 else -1
                born_displ[j, :, dim] += sign * dipole_out
                coords[j, dim] -= delta

    # Average Born charges
    born_charges = [sum(sp.eigh(born_displ[i])[0]) / 3 for i in range(nats)]
    return np.array(born_charges), born_displ


def get_forces_siesta(filename):
    """
    Read atomic forces from a SIESTA output file.

    Args:
        filename (str): Path to the forces file.

    Returns:
        np.ndarray: Array of atomic forces.
    """
    with open(filename, 'r') as file:
        num_lines = int(file.readline().strip())
        forces = [list(map(float, file.readline().split()[1:])) for _ in range(num_lines)]
    return np.array(forces)


def cost_function2(
        all_values, int_names, ppot_names, elect_names, dpot_names,
        all_integrals, all_ppots, all_elect, all_dpots,
        forces_siesta_list, energies_siesta_list,
        coords_list, lattice_vectors_list, symbols_list, atom_types_list, executor
):
    """
    Calculate the cost function for optimization across multiple structures.

    Returns:
        float: Total cost combining forces and energies.
    """
    try:
        # Unpack all parameters
        n_int = len(all_integrals) * len(int_names)
        n_pp = len(all_ppots) * len(ppot_names)
        n_el = len(all_elect) * len(elect_names)
        n_dp = len(all_dpots) * len(dpot_names)

        int_values = all_values[:n_int]
        ppot_values = all_values[n_int:n_int + n_pp]
        elect_values = all_values[n_int + n_pp:n_int + n_pp + n_el]
        dpot_values = all_values[n_int + n_pp + n_el:]

        # Update system
        modif_CC_all_integrals(int_values, int_names, all_integrals)
        modif_ppots(ppot_values, ppot_names, all_ppots)
        modif_elect(elect_values, elect_names, all_elect)
        modif_dpots(dpot_values, dpot_names, all_dpots)

        n_structures = len(coords_list)
        total_force_diffs = []
        energies_gpmd = np.zeros(n_structures)

        results = list(executor.map(
            gpmd,
            lattice_vectors_list,
            symbols_list,
            atom_types_list,
            coords_list,
            [np.zeros(3)] * n_structures,
            [verb] * n_structures
        ))

        forces_gpmd_all = []
        forces_siesta_all = []

        for i, (err, charges, gpmd_forces, dipole, energy) in enumerate(results):
            forces_gpmd_all.append(gpmd_forces)
            forces_siesta_all.append(forces_siesta_list[i])
            energies_gpmd[i] = energy
            total_force_diffs.extend(np.linalg.norm(gpmd_forces - forces_siesta_list[i], axis=1))

        force_cost = np.linalg.norm(total_force_diffs) ** 2
        energy_cost = 0.0
        total_cost = force_cost + energy_cost

        if np.isnan(total_cost) or np.isinf(total_cost):
            return 1e7

        total_points = sum(len(atom_types) * 3 for atom_types in atom_types_list)

        # Save correl.dat
        with open("correl.dat", 'w') as correl_file:
            for forces_siesta, forces_gpmd in zip(forces_siesta_all, forces_gpmd_all):
                for j in range(len(forces_siesta)):
                    correl_file.write(f"{forces_siesta[j][0]} {forces_gpmd[j][0]}\n")
                    correl_file.write(f"{forces_siesta[j][1]} {forces_gpmd[j][1]}\n")
                    correl_file.write(f"{forces_siesta[j][2]} {forces_gpmd[j][2]}\n")

        # Save correlP.dat and compute phosphorus-focused cost
        pcost = 0.0
        cost = 0.0
        nppoints = 0
        ntpoints = 0

        with open("correlP.dat", 'w') as correlp_file:
            for i, (forces_siesta, forces_gpmd, symbols, atom_types) in enumerate(
                    zip(forces_siesta_all, forces_gpmd_all, symbols_list, atom_types_list)
            ):
                for j in range(len(atom_types)):
                    symbol = symbols[atom_types[j]]
                    fs = forces_siesta[j]
                    fg = forces_gpmd[j]
                    if symbol == "P":
                        for k in range(3):
                            f_weight = np.exp(-0.1 * fs[k] ** 2)
                            pcost += f_weight * (fg[k] - fs[k]) ** 2
                            nppoints += 1
                        correlp_file.write(f"{fs[0]} {fg[0]}\n")
                        correlp_file.write(f"{fs[1]} {fg[1]}\n")
                        correlp_file.write(f"{fs[2]} {fg[2]}\n")

                    for k in range(3):
                        cost += (fg[k] - fs[k]) ** 2
                        ntpoints += 1

        pcost = np.sqrt(pcost / nppoints) if nppoints else 0.0
        cost = np.sqrt(cost / ntpoints) if ntpoints else 0.0
        cost_per_atom = np.sqrt((force_cost + energy_cost) / total_points)

        print_cost(cost, pcost)

        # Save TBparams if very good
        if cost_per_atom < 0.1:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            os.system(f"cp -r ./TBparams ./TBparams_{timestamp}")

        return cost if cost else 1e7

    except Exception as e:
        logger.exception("Exception in cost_function2")
        return 1e7



def modif_CC_all_integrals(params_values, params_names, all_integrals):
    """
    Modify bond integrals based on the provided parameters.

    Args:
        params_values (list): List of parameter values.
        params_names (list): Names of the parameters.
        all_integrals (list): List of integrals data.
    """
    name_pos = {'H0': 3, 'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7, 'B5': 8, 'R1': 9, 'Rcut': 10, 'H0_2': 11, 'B1_2': 12,
                'B2_2': 13, 'B3_2': 14, 'B4_2': 15, 'B5_2': 16, 'R1_2': 17, 'Rcut_2': 18}

    fname_bondints = "TBparams/bondints.nonortho_reference"
    all_lines = []
    with open(fname_bondints, 'r') as file:
        num_lines = int(file.readline().split()[1])
        header = file.readline()
        all_lines.append(f"Noints= {num_lines}\n")
        all_lines.append(header)

        for _ in range(num_lines):
            line_spl = file.readline().split()
            element1, element2, k_ind = line_spl[:3]
            for idx, d in enumerate(all_integrals):
                if d.get('Element1') == element1 and d.get('Element2') == element2 and d.get('Kind') == k_ind:
                    for p_idx, name in enumerate(params_names):
                        line_spl[name_pos[name]] = f"{params_values[idx * len(params_names) + p_idx]:.8f}"
                    break
            all_lines.append(" ".join(line_spl) + "\n")

    # Write modified bond integrals to a new file
    with open("TBparams/bondints.nonortho", 'w') as output_file:
        output_file.writelines(all_lines)


def extract_and_create_bounds(integrals, params_names):
    """
    Generate bounds for parameters based on their values in the integrals data.

    Args:
        integrals (dict): Dictionary of integral data.
        params_names (list): List of parameter names.

    Returns:
        list[tuple]: Bounds for each parameter as (lower_bound, upper_bound).
    """
    result = []
    for key in params_names:
        if key in integrals:
            value = integrals[key]
            lower_bound = value * 0.7
            upper_bound = value * 1.9
            result.append((lower_bound, upper_bound))
    return result


def extract_and_create_guess(integrals, params_names):
    """
    Generate initial guesses for parameters based on their values in the integrals data.

    Args:
        integrals (dict): Dictionary of integral data.
        params_names (list): List of parameter names.

    Returns:
        list[float]: Initial guesses for each parameter.
    """
    return [integrals[key] for key in params_names if key in integrals]


def print_cost(total, costP):
    """
    Save the cost metrics to output files.

    Args:
        total (float): Total cost.
        index (int): Index of the maximum cost contributor.
        maxval (float): Maximum value of the cost component.
        mean_force (float): Mean force magnitude.
    """
    with open("cost.dat", 'a+') as cost_file:
        cost_file.write(f"{total} {costP}\n")


def read_electfile_to_dict(filename):
    """
    Parse an electron configuration file into a list of dictionaries.

    Args:
        filename (str): Path to the electron configuration file.

    Returns:
        list[dict]: List of dictionaries containing electron configuration data.
    """
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()

    headers = lines[0].split()
    # Find the index where the second set of repeated headers starts (after the third column)
    all_headers = headers
    # Initialize a list to store all the dictionaries
    dict_list = []

    # Process the remaining lines to create dictionaries
    for line in lines[1:]:
        values = line.split()
        # Create a dictionary using headers as keys
        # Convert numerical values (from the 4th column onward) to floats
        data_dict = {all_headers[i]: (values[i] if i < 2 else float(values[i])) for i in range(len(all_headers))}
        # Add the dictionary to the list
        dict_list.append(data_dict)

    return dict_list




def modif_ppots(params_values, params_names, all_ppots):
    """
    Modify pseudopotential parameters based on the provided values.

    Args:
        params_values (list): List of pseudopotential parameter values.
        params_names (list): List of pseudopotential parameter names.
        all_ppots (list): List of pseudopotential data.
    """
    name_pos = {'A0': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'A5': 7, 'A6': 8, 'C': 9, 'R1': 10, 'Rcut': 11}
    fname_ppot = "TBparams/ppots.nonortho_reference"

    all_lines = []
    with open(fname_ppot, 'r') as file:
        num_lines = int(file.readline().split()[1])
        header = file.readline()
        all_lines.append(f"Nopps= {num_lines}\n")
        all_lines.append(header)

        for _ in range(num_lines):
            line_spl = file.readline().split()
            element1, element2 = line_spl[:2]
            for index, ppot in enumerate(all_ppots):
                if ppot['Ele1'] == element1 and ppot['Ele2'] == element2:
                    for p_idx, name in enumerate(params_names):
                        line_spl[name_pos[name]] = f"{params_values[index * len(params_names) + p_idx]:.8f}"
                    break
            all_lines.append(" ".join(line_spl) + "\n")

    with open("TBparams/ppots.nonortho", 'w') as file:
        file.writelines(all_lines)



def modif_dpots(params_values, params_names, all_dpots):
    """
    Modify pseudopotential parameters based on the provided values.

    Args:
        params_values (list): List of pseudopotential parameter values.
        params_names (list): List of pseudopotential parameter names.
        all_ppots (list): List of pseudopotential data.
    """
    name_pos = {'A0': 2, 'A1': 3, 'A2': 4}
    fname_dpot = "TBparams/disppot.nonortho_reference"

    all_lines = []
    with open(fname_dpot, 'r') as file:
        num_lines = int(file.readline().split()[1])
        header = file.readline()
        all_lines.append(f"Npairs= {num_lines}\n")
        all_lines.append(header)

        for _ in range(num_lines):
            line_spl = file.readline().split()
            element1, element2 = line_spl[:2]
            for index, dpot in enumerate(all_dpots):
                if dpot['Ele1'] == element1 and dpot['Ele2'] == element2:
                    for p_idx, name in enumerate(params_names):
                        line_spl[name_pos[name]] = f"{params_values[index * len(params_names) + p_idx]:.8f}"
                    break
            all_lines.append(" ".join(line_spl) + "\n")

    with open("TBparams/disppot.nonortho", 'w') as file:
        file.writelines(all_lines)



def modif_elect(params_values, params_names, all_elect):
    """
    Modify electron configuration parameters based on the provided values.

    Args:
        params_values (list): List of electron parameter values.
        params_names (list): List of electron parameter names.
        all_elect (list): List of electron configuration data.
    """
    name_pos = {'Es': 3, 'Ep': 4, 'Ed': 5, 'Ef': 6, 'HubbardU': 8, 'Wss': 9, 'Wpp': 10, 'Wdd': 11, 'Wff': 12}
    fname_elect = "TBparams/electrons_reference.dat"

    all_lines = []
    with open(fname_elect, 'r') as file:
        num_lines = int(file.readline().split()[1])
        header = file.readline()
        all_lines.append(f"Noelem= {num_lines}\n")
        all_lines.append(header)

        for _ in range(num_lines):
            line_spl = file.readline().split()
            element = line_spl[0]
            for index, elect in enumerate(all_elect):
                if elect['Element'] == element:
                    for p_idx, name in enumerate(params_names):
                        line_spl[name_pos[name]] = f"{params_values[index * len(params_names) + p_idx]:.8f}"
                    break
            all_lines.append(" ".join(line_spl) + "\n")

    with open("TBparams/electrons.dat", 'w') as file:
        file.writelines(all_lines)


def get_atom_atoms_kind(atom1, atom2, kind):
    for d0 in all_dicts:
        if d0['Element1'] == atom1 and d0['Element2'] == atom2 and d0['Kind'] == kind:
            return d0


def get_atom_atoms_for_ppot(atom1, atom2, all_dicts_ppot):
    for d0 in all_dicts_ppot:
        if d0['Ele1'] == atom1 and d0['Ele2'] == atom2:
            return d0

def get_atom_atoms_for_dpot(atom1, atom2, all_dicts_dpot):
    for d0 in all_dicts_dpot:
        if d0['Ele1'] == atom1 and d0['Ele2'] == atom2:
            return d0

def get_atom_for_elect(atom, all_dicts_elect):
    for d0 in all_dicts_elect:
        if d0['Element'] == atom:
            return d0

def list_files_ending_with(folder_path, ending):
    """
    Lists all PDB files in the given folder.

    Args:
        folder_path (str): Path to the folder containing PDB files.

    Returns:
        list: A list of full file paths to all PDB files in the folder.
        :param ending:
    """
    files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(ending):  # Check if the file has a .pdb extension
            full_path = os.path.join(folder_path, file_name)
            files.append(full_path)
    return files


###############################
## Main starts here
###############################
from scipy.optimize import differential_evolution

# Read all the parameters files
#all_dicts = read_file_to_dict('TBparams/bondints.nonortho_reference-original')
#all_dicts_ppot = read_ppotfile_to_dict('TBparams/ppots.nonortho_reference-original')
#all_dicts_elect = read_electfile_to_dict('TBparams/electrons_reference.dat-original')
all_dicts = read_file_to_dict('TBparams/bondints.nonortho_reference')
all_dicts_ppot = read_ppotfile_to_dict('TBparams/ppots.nonortho_reference')
all_dicts_elect = read_electfile_to_dict('TBparams/electrons_reference.dat')
all_dicts_dpot = read_dpotfile_to_dict('TBparams/disppot.nonortho_reference')

field = [0, 0, 0]
verb = 0


forces_siesta_list, energies_siesta_list,coords_list, latticeVectors_list,symbols_list,atomTypes_list= [],[],[],[],[],[]

folder_path='/projects/shared/alopezb/negre_molec/siesta_runs_finished/'
finished = np.genfromtxt(folder_path+'/finished_folders.dat', dtype='str')

all_coord_files , forces_list =[],[]
for folder in finished:
    all_coord_files.append('siesta_runs_finished/'+folder+"/molecule.xyz")
    forces_list.append('siesta_runs_finished/'+folder+"/molecule.FA")


# all_coord_files = sorted(list_files_ending_with( folder_path, ".xyz"))
# forces_list = sorted(list_files_ending_with(folder_path, ".FA"))
# all_energ_files = ["fullerenes/energy_" + prefix + ".dat" for prefix in prefix_list]

# print(len(all_coord_files))
# print(len(forces_list))
# sys.exit()
number_of_structures = len(forces_list)
for i in range(number_of_structures):
    forces_siesta = get_forces_siesta(forces_list[i])
    _, symbols, atomTypes, coords0 = fileio.read_xyz_file(all_coord_files[i], lib="None", verb=verb)

    latticeVectors = np.diag([30.00, 30.00, 30.00])
    forces_siesta_list.append(forces_siesta)
    coords_list.append(coords0)
    latticeVectors_list.append(latticeVectors)
    symbols_list.append(symbols)
    atomTypes_list.append(atomTypes)

atom_atom_kind = {'PP': ('P', 'P', ('sss', 'sps', 'pps', 'ppp')),
                  'CC': ('C', 'C', ('sss', 'sps', 'pps', 'ppp')),
                  'OO': ('O', 'O', ('sss', 'sps', 'pps', 'ppp')),
                  'NN': ('N', 'N', ('sss', 'sps', 'pps', 'ppp')),
                  'PO': ('P', 'O', ('sss', 'sps', 'pps', 'ppp')),
                  'OP': ('O', 'P', ['sps']),
                  'PN': ('P', 'N', ('sss', 'sps', 'pps', 'ppp')),
                  'NP': ('N', 'P', ['sps']),
                  'PC': ('P', 'C', ('sss', 'sps', 'pps', 'ppp')),
                  'CP': ('C', 'P', ['sps']),
                  'PS': ('P', 'S', ('sss', 'sps', 'pps', 'ppp')),
                  'SP': ('S', 'P', ['sps']),
                  'CN': ('C', 'N', ('sss', 'sps', 'pps', 'ppp')),
                  'CO': ('C', 'O', ('sss', 'sps', 'pps', 'ppp')),
                  'OC': ('O', 'C', ['sps']),
                  'NC': ('N', 'C', ['sps']),
                  'HP': ('H', 'P', ('sss', 'sps')),
                  'HO': ('H', 'O', ('sss', 'sps')),
                  'HH': ('H', 'H', ['sss']),
                  'HC': ('H', 'C', ('sss', 'sps'))}

wished = ['PP' ,'PC','PN','HP','PO','OP','NP','HP', 'CO', 'OC', 'OO', 'HC', 'HO', 'HH']#,'CP']#]
all_integrals_SS = []
for elms in wished:
    val = atom_atom_kind[elms]
    for v in val[2]:
        all_integrals_SS.append(get_atom_atoms_kind(val[0], val[1], v))

wished_dpots = [('P', 'H'), ('C', 'H'), ('O', 'H'), ('S', 'H'), ('N', 'H'), ('H', 'H')]
wished_ppots = [('P', 'P'), ('P', 'C'), ('P', 'O'), ('S','S'), ('S', 'H'), ('S', 'C'), ('S', 'O'), ('S', 'N'),
                ('P', 'N'), ('P', 'S'), ('P', 'H'), ('H', 'H'), ('O', 'O'), ('O', 'H'), ('N', 'O'), ('C', 'N'),
                ('N', 'N'), ('C', 'O'), ('C', 'C'), ('N', 'H'), ('C', 'H')]


all_dpots = [get_atom_atoms_for_dpot(elms[0], elms[1], all_dicts_dpot) for elms in wished_dpots]
all_ppots = [get_atom_atoms_for_ppot(elms[0], elms[1], all_dicts_ppot) for elms in wished_ppots]

unique_elements=list(set( [ s for symb in symbols_list for s in symb] ) )
wished_elect = ['P']
all_elect = [get_atom_for_elect(element, all_dicts_elect)  for element in wished_elect]

params_names = ['H0', 'B1', 'B2','B5','H0_2', 'H0_2', 'B1_2', 'B2_2', 'B3_2', 'B4_2']
ppotparams_names = ['A0','A1','A2','A5']
dpotparams_names = ['A0','A1','A2']
electparams_names = [ 'Es','Ep','HubbardU']

bounds = [v for integrals in all_integrals_SS for v in extract_and_create_bounds(integrals, params_names)]
guess = [v for integrals in all_integrals_SS for v in extract_and_create_guess(integrals, params_names)]
pguess = [v for ppots in all_ppots for v in extract_and_create_guess(ppots, ppotparams_names)]
eguess = [v for elect in all_elect for v in extract_and_create_guess(elect, electparams_names)]
dguess = [v for dpots in all_dpots for v in extract_and_create_guess(dpots, dpotparams_names)]

all_guess = np.array(guess + pguess + eguess + dguess)  # Merging lists

all_guess_v = np.array(all_guess, dtype=float)
bounds = np.column_stack((all_guess , all_guess ))
bounds[:, 0] = 0.0
bounds[:, 1] = 100.0

optimize = False
if (optimize):
    latticeVectors = latticeVectors_list[0]
    symbols = symbols_list[0]
    atomTypes = atomTypes_list[0]
    coords0 = coords_list[0]
    coords_new = optimize_coordinates(latticeVectors, atomTypes, symbols, coords0, method='Nelder-Mead',
                                      iterations=0, tol=0.01)
    write_xyz_coordinates("opt_str.xyz", coords_new, atomTypes, symbols)
    exit(0)

method = "Else"
if method == "Anneal":
    with ProcessPoolExecutor(max_workers=min(number_of_structures,96)) as executor:
        result = dual_annealing(cost_function2, bounds, args=(
            params_names, ppotparams_names,
            electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list, energies_siesta_list,
            coords_list,
            latticeVectors_list,
            symbols_list,
            atomTypes_list, executor
        ), maxiter=1000, initial_temp=1.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, seed=None,
                                no_local_search=False, callback=None, x0=None)
elif method == "Diff":
    with ProcessPoolExecutor(max_workers=min(number_of_structures,96)) as executor:
        result = differential_evolution(cost_function2, bounds=bounds, args=(
            params_names, ppotparams_names,
            electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list, energies_siesta_list,
            coords_list,
            latticeVectors_list,
            symbols_list,
            atomTypes_list, executor
        ),strategy='best1bin', maxiter=3, recombination=0.1, mutation=0.1, tol=0.01, popsize=15, disp=True)
elif method == "Else":
    with ProcessPoolExecutor(max_workers=min(number_of_structures,96)) as executor:
        result = minimize(cost_function2, all_guess, args=(params_names, ppotparams_names,
                                                           electparams_names,dpotparams_names, all_integrals_SS, all_ppots, all_elect,
                                                           all_dpots,
                                                           forces_siesta_list, energies_siesta_list,
                                                           coords_list,
                                                           latticeVectors_list,
                                                           symbols_list,
                                                           atomTypes_list, executor),
                          method='Nelder-Mead',
                          #method='BFGS',
                          jac=None, hess=None,
                          hessp=None, bounds=bounds, constraints=(), tol=0.0001, callback=None, options={'maxiter': 1000000})

exit(0)
