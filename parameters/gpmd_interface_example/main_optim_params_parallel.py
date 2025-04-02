#!/usr/bin/env python3
import sys

import numpy as np
from sedacs.system import *
from sedacs.periodic_table import PeriodicTable
# from sdc_system import *
# from sdc_ptable import ptable
import ctypes as ct
import os
import scipy.linalg as sp
import sedacs.file_io as fileio
from scipy.optimize import minimize
from scipy import optimize
from gpmd import *
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# !/usr/bin/env python3

import sys
import numpy as np
from sedacs.system import *
from sedacs.periodic_table import PeriodicTable
import ctypes as ct
import os
import scipy.linalg as sp
from scipy.optimize import minimize
from scipy import optimize
from gpmd import *
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor


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

    # Write energy to file
    with open("energ.dat", 'a+') as energy_file:
        energy_file.write(f"{energy}\n")

    return energy


def optimize_coordinates(lattice_vectors, atom_types, symbols, coords, method='CG', iterations=100, tol=0.01):
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


def cost_function(params_values, params_names, forces_siesta, all_integrals):
    """
    Calculate the cost function for optimization by comparing GPMD and SIESTA forces.

    Args:
        params_values (list): List of parameter values to modify.
        params_names (list): Names of the parameters.
        forces_siesta (np.ndarray): Reference forces from SIESTA.
        all_integrals (list): List of integrals data for modification.

    Returns:
        float: Total cost as the sum of squared differences between forces.
    """
    modif_CC_all_integrals(params_values, params_names, all_integrals)
    _, _, gpmd_forces, _, _ = gpmd(latticeVectors, symbols, atomTypes, coords0, field, verb)

    # Compute sum of squared differences
    total_cost = sum(np.linalg.norm(gp - si) ** 2 for gp, si in zip(gpmd_forces, forces_siesta))
    print(f"Total Cost: {total_cost}")
    return total_cost


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


def print_cost(total, index, maxval, mean_force):
    """
    Save the cost metrics to output files.

    Args:
        total (float): Total cost.
        index (int): Index of the maximum cost contributor.
        maxval (float): Maximum value of the cost component.
        mean_force (float): Mean force magnitude.
    """
    with open("cost.dat", 'a+') as cost_file:
        cost_file.write(f"{total}\n")
    with open("index.dat", 'a+') as index_file:
        index_file.write(f"{index} {maxval} {mean_force}\n")


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

    # Extract the headers (column names)
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


def cost_function2_old(all_values, int_names, ppot_names, elect_names, all_integrals, all_ppots, all_elect,
                       forces_siesta_list, energies_siesta_list, coords_list, lattice_vectors_list, symbols_list,
                       atom_types_list):
    """
    Calculate the cost function for optimization across multiple structures.

    Args:
        all_values (list): Flattened list of all parameter values (integrals, pseudopotentials, electrons).
        ...

    Returns:
        float: Total cost combining forces and energies.
    """
    nstr = len(coords_list)
    nint = len(all_integrals) * len(int_names)
    npp = len(all_ppots) * len(ppot_names)
    nel = len(all_elect) * len(elect_names)

    int_values = [] * nint
    ppot_values = [] * npp
    elect_values = [] * nel
    int_values[:] = all_values[0:nint]
    ppot_values[:] = all_values[nint:nint + npp]
    elect_values[:] = all_values[nint + npp:nint + npp + nel]

    modif_CC_all_integrals(int_values, params_names, all_integrals)
    modif_ppots(ppot_values, ppot_names, all_ppots)
    modif_elect(elect_values, elect_names, all_elect)

    nats = len(atomTypes_list[0])
    buffg = np.zeros(3 * nats)
    buffs = np.zeros(3 * nats)
    diffs_total = np.zeros(3 * nstr * nats)
    forcess = np.zeros(3 * nstr * nats)
    forcesg = np.zeros(3 * nstr * nats)

    energies_gpmd = np.zeros(nstr)

    ################### Start parallel execution   ###################

    futures_lst = []
    with ProcessPoolExecutor(max_workers=7) as exe:
        for istr in range(nstr):
            nats = len(atomTypes_list[istr])
            latticeVectors = latticeVectors_list[istr]
            symbols = symbols_list[istr]
            atomTypes = atomTypes_list[istr]
            coords0 = coords_list[istr]
            forces_siesta = forces_siesta_list[istr]

            futures_lst.append(exe.submit(gpmd, latticeVectors, symbols, atomTypes, coords0, field, verb))

        for future in futures_lst:
            err, charges, gpmd_forces, dipole, energy = future.result()

            for i in range(nats):
                for i in range(nats):
                    # assign all three components at once using slicing
                    buffs[i * 3:(i + 1) * 3] = gpmd_forces[i]
                    buffg[i * 3:(i + 1) * 3] = forces_siesta[i]

            forcess[istr * nats * 3:(istr + 1) * nats * 3] = buffs[:]
            forcesg[istr * nats * 3:(istr + 1) * nats * 3] = buffg[:]
            print("ENERGY", istr, energy)

            energies_gpmd[istr] = energy

    ################### End parallel execution  ###################

    diffs_total[:] = abs(forcess[:] - forcesg[:])

    energies_gpmd[:] = energies_gpmd[:] - energies_gpmd[0]
    diffs_energ = np.zeros((nstr))
    diffs_energ[:] = abs(energies_siesta_list[:] - energies_gpmd[:])
    cost_energ = (np.linalg.norm(diffs_energ)) ** 2

    cost = (np.linalg.norm(diffs_total)) ** 2
    cost = cost + 100 * cost_energ
    if (np.isnan(cost)): cost = 10000000.0
    # print("gpmd",gpmd_forces)
    # print("siesta",forces_siesta)
    # print("reldif",(forces_siesta-gpmd_forces)/forces_siesta)
    # exit(0)

    f = open("correl.dat", 'w')
    for i in range(len(forcess)):
        print(forcess[i], forcesg[i], file=f)
    f.close()
    f = open("correlP.dat", 'w')
    cont = 0
    for i in range(nstr):
        for j in range(nats):
            for k in range(3):
                if (symbols_list[0][atomTypes_list[0][j]] == "P"):
                    print(forcess[cont], forcesg[cont], file=f)
                cont = cont + 1
    f.close()

    maxval = np.max(diffs_total)
    maxindex = np.argmax(diffs_total)
    cost_per_atom = np.sqrt(cost / (nats * nstr))
    mean_force = np.sqrt(sum(abs(forcess)) / (nats * nstr))
    print_cost(cost_per_atom, maxindex, maxval, mean_force)

    # Save the result if there is a very good min
    if (cost_per_atom < 0.1):
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        cmd = "cp -r ./TBparams " + "TBparams" + str(ts)
        os.system(cmd)
        exit(0)

    return cost


def cost_function2(all_values, int_names, ppot_names, elect_names, all_integrals, all_ppots, all_elect,
                   forces_siesta_list, energies_siesta_list, coords_list, lattice_vectors_list, symbols_list,
                   atom_types_list):
    """
    Calculate the cost function for optimization across multiple structures.

    Args:
        all_values (list): Flattened list of all parameter values (integrals, pseudopotentials, electrons).
	...

    Returns:
        float: Total cost combining forces and energies.
    """
    # Extract parameter groups
    n_int = len(all_integrals) * len(int_names)
    n_pp = len(all_ppots) * len(ppot_names)
    n_el = len(all_elect) * len(elect_names)

    int_values = all_values[:n_int]
    ppot_values = all_values[n_int:n_int + n_pp]
    elect_values = all_values[n_int + n_pp:n_int + n_pp + n_el]

    # Update parameters
    modif_CC_all_integrals(int_values, int_names, all_integrals)
    modif_ppots(ppot_values, ppot_names, all_ppots)
    modif_elect(elect_values, elect_names, all_elect)

    n_structures = len(coords_list)
    total_force_diffs = []
    energies_gpmd = np.zeros(n_structures)

    # Buffers to store forces and diffs
    forces_gpmd_all = np.zeros((n_structures, len(forces_siesta_list[0]), 3))
    forces_siesta_all = np.zeros_like(forces_gpmd_all)

    # Parallel evaluation of structures
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(len(coords_list)):
            futures.append(
                executor.submit(
                    gpmd,
                    lattice_vectors_list[i],
                    symbols_list[i],
                    atom_types_list[i],
                    coords_list[i],
                    np.zeros(3),  # Field is zero by default
                    verb,
                )
            )

        for i, future in enumerate(futures):
            err, charges, gpmd_forces, dipole, energy = future.result()

            # Store forces for correlation file
            forces_gpmd_all[i] = gpmd_forces
            forces_siesta_all[i] = forces_siesta_list[i]

            # Compute energy differences
            energies_gpmd[i] = energy

            # Compute force differences for cost
            total_force_diffs.extend(np.linalg.norm(gpmd_forces - forces_siesta_list[i], axis=1))

    # Compute energy and force costs
    diffs_energ = np.abs(energies_siesta_list - energies_gpmd)
    energy_cost = np.linalg.norm(diffs_energ) ** 2

    force_cost = np.linalg.norm(total_force_diffs) ** 2

    # Weight energy cost more heavily
    total_cost = force_cost + 100 * energy_cost

    # Handle NaN or infinity in the cost
    if np.isnan(total_cost) or np.isinf(total_cost):
        total_cost = 1e7

    # Update correlation files
    with open("correl.dat", 'w') as correl_file:
        for i in range(n_structures):
            for j in range(len(forces_siesta_list[i])):
                correl_file.write(f"{forces_gpmd_all[i, j, 0]} {forces_siesta_all[i, j, 0]}\n")

    with open("correlP.dat", 'w') as correlp_file:
        for i in range(n_structures):
            for j in range(len(atom_types_list[i])):
                if symbols_list[0][atom_types_list[0][j]] == "P":
                    correlp_file.write(f"{forces_gpmd_all[i, j, 0]} {forces_siesta_all[i, j, 0]}\n")

    # Print useful metrics
    max_diff = np.max(total_force_diffs)
    max_diff_index = np.argmax(total_force_diffs)
    cost_per_atom = np.sqrt(force_cost / (n_structures * len(atom_types_list[0])))
    mean_force = np.mean(np.linalg.norm(forces_gpmd_all, axis=2))

    print_cost(cost_per_atom, max_diff_index, max_diff, mean_force)

    # Save good results for potential future use
    if cost_per_atom < 0.1:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        os.system(f"cp -r ./TBparams ./TBparams_{timestamp}")

    return total_cost


def modif_ppots(params_values, params_names, all_ppots):
    """
    Modify pseudopotential parameters based on the provided values.

    Args:
        params_values (list): List of pseudopotential parameter values.
        params_names (list): List of pseudopotential parameter names.
        all_ppots (list): List of pseudopotential data.
    """
    name_pos = {'A0': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'A5': 7, 'C': 9, 'R1': 10, 'Rcut': 11}
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


def get_atom_for_elect(atom, all_dicts_elect):
    for d0 in all_dicts_elect:
        if d0['Element'] == atom:
            return d0


###############################
## Main starts here
###############################
from scipy.optimize import differential_evolution

# Read all the parameters files
all_dicts = read_file_to_dict('TBparams/bondints.nonortho_reference')
all_dicts_ppot = read_ppotfile_to_dict('TBparams/ppots.nonortho_reference')
all_dicts_elect = read_electfile_to_dict('TBparams/electrons_reference.dat')

number_of_structures = 31
field = [0, 0, 0]
verb = 0

forces_siesta_list = []
energies_siesta_list = np.zeros(number_of_structures)
coords_list = []
latticeVectors_list = []
symbols_list = []
atomTypes_list = []

# Read the structures into lists
for i in range(number_of_structures):

    # For water, crashes
    #fname = "forces_positions_h2o/h2o_" + str(i) + ".FA"
    #forces_siesta = get_forces_siesta(fname)
    #coord_files = "forces_positions_h2o/coordinates_" + str(i) + ".xyz"
    #latticeVectors, symbols, atomTypes, coords0 = fileio.read_xyz_file(coord_files, lib="None", verb=verb)
    #energ_files = "forces_positions_h2o/energy_" + str(i) + ".dat"

    # Fullerenes
    prefix_list=["C44-D2-85_E_0.01x", "C44-C1-62_E_0.01x", "C44-Cs-11_E_0.01x", "C44-C2-81_E_0.01x", "C44-C1-64_E_0.01x", "C44-C2-87_E_0.01x", "C44-D2-2_E_0.01x", "C44-D3d-38_E_0.01x", "C44-Cs-28_E_0.01x", "C44-C1-59_E_0.01x", "C44-Cs-70_E_0.01x", "C44-D2-75_E_0.01x", "C44-C2-44_E_0.01x", "C44-Cs-84_E_0.01x", "C44-C2-34_E_0.01x", "C44-D3-35_E_0.01x", "C44-C1-60_E_0.01x", "C44-D2-24_E_0.01x", "C44-C1-58_E_0.01x", "C44-D3-80_E_0.01x", "C44-Cs-71_E_0.01x", "C44-C1-47_E_0.01x", "C44-C1-65_E_0.01x", "C44-C2-79_E_0.01x", "C44-C2-76_E_0.01x", "C44-C1-67_E_0.01x", "C44-C1-63_E_0.01x", "C44-Cs-54_E_0.01x", "C44-C2-74_E_0.01x", "C44-D3d-3_E_0.01x", "C44-Cs-33_E_0.01x"]
    fnames = ["fullerenes/"+ prefix + ".FA" for prefix in prefix_list]
    forces_siesta = get_forces_siesta(fnames[i])
    
    all_coord_files = ["fullerenes/"+ prefix + ".xyz" for prefix in prefix_list]
    latticeVectors, symbols, atomTypes, coords0 = fileio.read_xyz_file(all_coord_files[i], lib="None", verb=verb)
    all_energ_files = ["fullerenes/energy_"+ prefix + ".dat" for prefix in prefix_list]
    energ_files=all_energ_files[i] 
    
    #fname = "forces_positions/forces_0" + str(i) + ".FA"
    #forces_siesta = get_forces_siesta(fname)
    #coord_files = "forces_positions/coordinates_0" + str(i) + ".xyz"
    #latticeVectors, symbols, atomTypes, coords0 = fileio.read_xyz_file(coord_files, lib="None", verb=verb)
    #energ_files = "forces_positions/energies_0" + str(i) + ".EN"

    with open(energ_files, 'r') as f:
        energies_siesta_list = [float(lines.split()[0]) for lines in f]
    
    latticeVectors = np.diag([20.00, 20.00, 20.00])
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
                  'CP': ('C', 'P', ('sss', 'sps', 'pps', 'ppp')),
                  'CN': ('C', 'N', ('sss', 'sps', 'pps', 'ppp')),
                  'CO': ('C', 'O', ('sss', 'sps', 'pps', 'ppp')),
                  'OC': ('O', 'C', ['sps']),
                  'NC': ('N', 'C', ['sps']),
                  'HP': ('H', 'P', ('sss', 'sps')),
                  'HO': ('H', 'O', ('sss', 'sps')),
                  'HH': ('H', 'H', ['sss']),
                  'HC': ('H', 'C', ('sss', 'sps'))}

wished = ['CC']# , 'CO', 'OC', 'OO', 'HC', 'HO', 'HH']
all_integrals_SS = []
for elms in wished:
    val = atom_atom_kind[elms]
    for v in val[2]:
        all_integrals_SS.append(get_atom_atoms_kind(val[0], val[1], v))

wished_ppots = [('C', 'C')]#, ('C', 'C'), ('H', 'H'), ('C', 'O'), ('C', 'H'), ('O', 'H')]
all_ppots = [get_atom_atoms_for_ppot(elms[0], elms[1], all_dicts_ppot) for elms in wished_ppots]
dic_elect_H = get_atom_for_elect('H', all_dicts_elect)
dic_elect_C = get_atom_for_elect('C', all_dicts_elect)
dic_elect_O = get_atom_for_elect('O', all_dicts_elect)

all_elect = [dic_elect_O, dic_elect_C, dic_elect_H]

params_names = ['H0', 'B1', 'B2', 'H0_2', 'H0_2', 'B1_2', 'B2_2', 'B3_2', 'B4_2']
ppotparams_names = ['A0', 'A1', 'A2']
electparams_names = ['Es', 'Ep', 'HubbardU']

# params_names = ['H0', 'B1', 'B2']#'B1_2', "B2_2"]#, "B3_2", "B4_2", "B5_2"]
bounds = [v for integrals in all_integrals_SS for v in extract_and_create_bounds(integrals, params_names)]
guess = [v for integrals in all_integrals_SS for v in extract_and_create_guess(integrals, params_names)]
pguess = [v for ppots in all_ppots for v in extract_and_create_guess(ppots, ppotparams_names)]
eguess = [v for elect in all_elect for v in extract_and_create_guess(elect, electparams_names)]

all_guess = guess + pguess + eguess  # Merging lists

all_guess_v = np.zeros(len(all_guess))
all_guess_v[:] = all_guess[:]

bounds = np.zeros((len(all_guess), 2))
bounds[:, 0] = all_guess[:]
bounds[:, 1] = all_guess[:]
bounds[:, 0] = bounds[:, 0] - 0.5 * abs(bounds[:, 0])
bounds[:, 1] = bounds[:, 1] + 0.5 * abs(bounds[:, 1])

# guess = [-8.0, -1, -0.5  , -0.5 , -0.5, -0.5, -0.5, -0.5, -0.5]
cost_ref = cost_function2(all_guess_v, params_names, ppotparams_names, electparams_names, all_integrals_SS, all_ppots,
                          all_elect, forces_siesta_list, energies_siesta_list,
                          coords_list,
                          latticeVectors_list,
                          symbols_list,
                          atomTypes_list
                          )

optimize = False
if (optimize):
    latticeVectors = latticeVectors_list[0]
    symbols = symbols_list[0]
    atomTypes = atomTypes_list[0]
    coords0 = coords_list[0]
    coords_new = optimize_coordinates(latticeVectors, atomTypes, symbols, coords0, method='BFGS', parampath='/tmp',
                                      iterations=100, tol=0.01)
    write_xyz_coordinates("opt_str.xyz", coords_new, atoTypes, symbols)

method = "Else"
if method == "Anneal":
    result = dual_annealing(cost_function2, bounds, args=(
        params_names, ppotparams_names,
        electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list,
        coords_list,
        latticeVectors_list,
        symbols_list,
        atomTypes_list
    ), maxiter=1000, initial_temp=10.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, seed=None,
                            no_local_search=False, callback=None, x0=None)
elif method == "Diff":
    result = differential_evolution(cost_function2, bounds=bounds, args=(
        params_names, ppotparams_names,
        electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list, energies_siesta_list,
        coords_list,
        latticeVectors_list,
        symbols_list,
        atomTypes_list
    ), maxiter=3, recombination=0.1, mutation=0.1, tol=0.01, popsize=15, disp=True)
else:
    result = minimize(cost_function2, all_guess, args=(params_names, ppotparams_names,
                                                       electparams_names, all_integrals_SS, all_ppots, all_elect,
                                                       forces_siesta_list, energies_siesta_list,
                                                       coords_list,
                                                       latticeVectors_list,
                                                       symbols_list,
                                                       atomTypes_list), method='Nelder-Mead', jac=None, hess=None,
                      hessp=None,
                      bounds=bounds, constraints=(), tol=0.0001, callback=None, options={'maxiter': 600000})

exit(0)

