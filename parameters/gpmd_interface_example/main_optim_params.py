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


#Run GPMD and get the total energy.
def get_energy(coords_in,lattiveVectors,atomTypes,symbols):
    field = np.zeros((3))
    coords = coords_in.reshape(len(atomTypes),3)
    err, charges_out, forces_saved, dipole_out, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field, verb)        
    with open("traj.xyz", 'a+') as line_file:
        line_file.write((str(len(atomTypes))+ "\n"))
        line_file.write(("t=")+"\n")
        for i in range(len(atomTypes)):
            line_file.write((symbols[atomTypes[i]] + " " + str(coords[i,0]) + " " + str(coords[i,1]) + " " + str(coords[i,2]) +  "\n"))

    with open("energ.dat", 'a+') as line_file:
        line_file.write(str(energy) + "\n")

    return energy 

def optimize_coordinates(lattiveVectors,types,symbols,coords,method='CG',parampath='/tmp',iterations=100,tol=0.01):
  
    coordsv = coords.reshape(-1)
    result = minimize(get_energy, coordsv, args=(lattiveVectors,atomTypes,symbols),
            method=method, jac=None, hess=None, hessp=None,
            bounds=bounds, constraints=(), tol=0.001, callback=None, options={'maxiter': 100})

#    bounds = np.zeros((len(coordsv), 2))
#    bounds[:, 0] = coordsv[:]
#    bounds[:, 1] = coordsv[:]
#    bounds[:, 0] = bounds[:, 0] - 5.1
#    bounds[:, 1] = bounds[:, 1] + 5.1
    
#    dual_annealing(get_energy,bounds, args=(lattiveVectors,atomTypes,symbols), maxiter=1000, initial_temp=10.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, seed=None, no_local_search=False, callback=None, x0=None)

#    result = optimize.anneal(get_energy, coordsv, args=(lattiveVectors,atomTypes,symbols), schedule='boltzmann',
#                          full_output=True, maxiter=500, lower=-10,
#                          upper=10, dwell=250, disp=True)

    return result.x

def read_electfile_to_dict(filename):
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


def read_ppotfile_to_dict(filename):
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


def read_file_to_dict(filename):
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()

    # Extract the headers (column names)
    headers = lines[0].split()

    # Find the index where the second set of repeated headers starts (after the third column)
    num_first_headers = 3
    first_set = headers[num_first_headers:num_first_headers + 8]  # First set of H0, B1, etc.
    second_set = [header + "_2" for header in first_set]  # Rename second set

    # Combine the first 3 headers, first set of H0, B1, ..., and second set
    all_headers = headers[:num_first_headers] + first_set + second_set

    # Initialize a list to store all the dictionaries
    dict_list = []

    # Process the remaining lines to create dictionaries
    for line in lines[1:]:
        values = line.split()

        # Create a dictionary using headers as keys
        # Convert numerical values (from the 4th column onward) to floats
        data_dict = {all_headers[i]: (values[i] if i < num_first_headers else float(values[i]))
                     for i in range(len(all_headers))}

        # Add the dictionary to the list
        dict_list.append(data_dict)

    return dict_list


def get_dipole(coords, charges):
    dipole = np.zeros(3)

    for i in range(len(charges)):
        dipole[0] = dipole[0] + charges[i] * coords[i, 0]
        dipole[1] = dipole[1] + charges[i] * coords[i, 1]
        dipole[2] = dipole[2] + charges[i] * coords[i, 2]


def get_field_born_charges(latticeVectors, symbols, atomTypes, coords, verb):
    field = np.zeros(3)
    err, charges_out, forces_saved, dipole_out, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field, verb)
    nats = len(atomTypes)
    bornField = np.zeros((nats, 3, 3))
    for i in range(3):
        field = np.zeros(3)
        field[i] = -1.0E-4
        err, charges_out, forces_out, dipole_out, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field, verb)
        bornField[:, :, i] = (forces_out - forces_saved) / np.linalg.norm(field)

        field = np.zeros(3)
        field[i] = 1.0E-4
        err, charges_out, forces_out, dipole_out, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field, verb)
        bornField[:, :, i] = (forces_out - forces_saved) / np.linalg.norm(field)

    bchMat = np.zeros((3, 3))
    bornCharges = np.zeros((nats))
    for i in range(len(bornField[:, 0, 0])):
        bchMat[:, :] = bornField[i, :, :]
        E, Q = sp.eigh(bchMat)
        bornCharges[i] = sum(E) / 3.0

    return bornCharges


def get_displ_born_charges(latticeVectors, symbols, atomTypes, coordsIn, verb):
    nats = len(atomTypes)
    bornDispl = np.zeros((nats, 3, 3))
    dspl = 1.0E-5
    field = np.zeros(3)
    coords = np.copy(coordsIn)
    err, charges_out, forces_out, dipole_saved, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field, verb)

    # X-coords
    for j in range(nats):
        coords[j, 0] = coordsIn[j, 0] + dspl
        err, charges_out, forces_out, dipole_out_p, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        coords[j, 0] = coordsIn[j, 0] - dspl
        err, charges_out, forces_out, dipole_out_m, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        bornDispl[j, :, 0] = (dipole_out_p - dipole_out_m) / (2 * dspl)

    # Y-coords
    for j in range(nats):
        coords[j, 1] = coordsIn[j, 1] + dspl
        err, charges_out, forces_out, dipole_out_p, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        coords[j, 1] = coordsIn[j, 1] - dspl
        err, charges_out, forces_out, dipole_out_m, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        bornDispl[j, :, 1] = (dipole_out_p - dipole_out_m) / (2 * dspl)

    # Z-coords
    for j in range(nats):
        coords[j, 2] = coordsIn[j, 2] + dspl
        err, charges_out, forces_out, dipole_out_p, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        coords[j, 2] = coordsIn[j, 2] - dspl
        err, charges_out, forces_out, dipole_out_m, energy = gpmd(latticeVectors, symbols, atomTypes, coords, field,
                                                                  verb)
        bornDispl[j, :, 2] = (dipole_out_p - dipole_out_m) / (2 * dspl)

    bchMat = np.zeros((3, 3))
    bornCharges = np.zeros((nats))
    for i in range(len(bornDispl[:, 0, 0])):
        bchMat[:, :] = bornDispl[i, :, :]
        E, Q = sp.eigh(bchMat)
        bornCharges[i] = sum(E) / 3.0

    return bornCharges, bornDispl


def get_forces_siesta(fname):
    forces_siesta = []
    with open(fname) as f:
        numlines = int(f.readline())
        for i in range(numlines):
            l = f.readline().split()
            forces_siesta.append([float(l[1]), float(l[2]), float(l[3])])
    forces_siesta = np.array(forces_siesta)
    return forces_siesta

#params_names, ppotparams_names,forces_siesta, all_integrals_SS,all_ppots
#params_values, params_names, forces_siesta, all_integrals

def cost_function(params_names, ppotparams_names,forces_siesta, all_integrals_SS,all_ppots):
    modif_CC_all_integrals(params_values, params_names, all_integrals)
    modif_ppots(ppot_values, ppot_names, all_ppots)
    modif_elect(elect_values, elect_names, all_elect)
    err, charges, gpmd_forces, dipole, energy = gpmd(latticeVectors, symbols, atomTypes, coords0, field, verb)
    sum_sq = []
    for gp, si in zip(forces_siesta, gpmd_forces):
        sum_sq.append(np.linalg.norm(gp - si))
    total=sum( v*v for v in sum_sq)
    print("total_cost", total)
    # sys.exit()
    return total


def cost_function2(all_values, int_names, ppot_names, elect_names, all_integrals, all_ppots, all_elect,
                   forces_siesta_list,
                   coords_list,
                   latticeVectors_list,
                   symbols_list,
                   atomTypes_list
                   ):
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
    buffg = np.zeros((3 * nats))
    buffs = np.zeros((3 * nats))
    diffs_total = np.zeros((3 * nstr * nats))
    forcess = np.zeros((3 * nstr * nats))
    forcesg = np.zeros((3 * nstr * nats))

    for istr in range(nstr):
        nats = len(atomTypes_list[istr])
        latticeVectors = latticeVectors_list[istr]
        symbols = symbols_list[istr]
        atomTypes = atomTypes_list[istr]
        coords0 = coords_list[istr]
        forces_siesta = forces_siesta_list[istr]

        err, charges, gpmd_forces, dipole, energy = gpmd(latticeVectors, symbols, atomTypes, coords0, field, verb)

        for i in range(nats):
            buffs[i * 3 + 0] = gpmd_forces[i, 0]
            buffs[i * 3 + 1] = gpmd_forces[i, 1]
            buffs[i * 3 + 2] = gpmd_forces[i, 2]

            buffg[i * 3 + 0] = forces_siesta[i, 0]
            buffg[i * 3 + 1] = forces_siesta[i, 1]
            buffg[i * 3 + 2] = forces_siesta[i, 2]

        forcess[istr * nats * 3:(istr + 1) * nats * 3] = buffs[:]
        forcesg[istr * nats * 3:(istr + 1) * nats * 3] = buffg[:]

    diffs_total[:] = abs(forcess[:] - forcesg[:])

    cost = (np.linalg.norm(diffs_total)) ** 2
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


    #Save the result if there is a very good min
    if(cost_per_atom < 0.9):
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        cmd = "cp -r ./TBparams " + "TBparams"+ts
        os.system(cmd) 
        exit(0)
    return cost


def print_cost(total, index, maxval, mean_force):
    with open("cost.dat", 'a+') as line_file:
        line_file.write(str(total) + "\n")
    with open("index.dat", 'a+') as line_file:
        line_file.write(str(index) + " " + str(maxval) + " " + str(mean_force) + "\n")


def modif_CC_all_integrals(params_values, params_names, all_integrals):
    name_pos = {'H0': 3, 'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7, 'B5': 8, 'R1': 9, 'Rcut': 10, 'H0_2': 11, 'B1_2': 12,
                'B2_2': 13, 'B3_2': 14, 'B4_2': 15, 'B5_2': 16, 'R1_2': 17, 'Rcut_2': 18}

    fname_bondints = "TBparams/bondints.nonortho_reference"
    all_lines = []
    with open(fname_bondints, 'r') as fi:
        numlines = int(fi.readline().split(" ")[1])
        header = fi.readline()
        all_lines.append(f"Noints= {numlines}\n")
        all_lines.append(header)
        for _ in range(numlines):
            line_spl = fi.readline().split()
            element1, element2, k_ind = line_spl[0], line_spl[1], line_spl[2]
            for index, d in enumerate(all_integrals):
                if d.get('Element1') == element1 and d.get('Element2') == element2 and d.get('Kind') == k_ind:
                    for indx, name in enumerate(params_names):
                        line_spl[name_pos[name]] = str(params_values[index * len(params_names) + indx])[:18]
                    with open("lines.dat", 'a+') as line_file:
                        line_file.write((" ".join(map(str, line_spl)) + "\n"))
                    break
            all_lines.append(" ".join(map(str, line_spl)) + "\n")

    with open("TBparams/bondints.nonortho", 'w') as fo:
        fo.writelines(all_lines)
    # sys.exit()


def modif_ppots(params_values, params_names, all_ppots):
    name_pos = {'A0': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'A5': 7, 'A6': 8, 'C': 9, 'R1': 10, 'Rcut': 11}
    fname_ppot = "TBparams/ppots.nonortho_reference"
    all_lines = []
    with open(fname_ppot, 'r') as fi:
        numlines = int(fi.readline().split(" ")[1])
        header = fi.readline()
        all_lines.append(f"Nopps= {numlines}\n")
        all_lines.append(header)
        for _ in range(numlines):
            line_spl = fi.readline().split()
            element1, element2 = line_spl[0], line_spl[1]
            for index, d in enumerate(all_ppots):
                print(d.get('Ele1'), element1)
                if d.get('Ele1') == element1 and d.get('Ele2') == element2:
                    for indx, name in enumerate(params_names):
                        print(line_spl)
                        print(index, indx, len(params_names))
                        print(params_values)
                        line_spl[name_pos[name]] = str(params_values[index * len(params_names) + indx])[:12]
                    with open("linesp.dat", 'a+') as line_file:
                        line_file.write((" ".join(map(str, line_spl)) + "\n"))
                    break
            all_lines.append(" ".join(map(str, line_spl)) + "\n")

    with open("TBparams/ppots.nonortho", 'w') as fo:
        fo.writelines(all_lines)


def modif_elect(params_values, params_names, all_elect):
    name_pos = {'Es': 3, 'Ep': 4, 'Ed': 5, 'Ef': 6, 'HubbardU': 8, 'Wss': 9, 'Wpp': 10, 'Wdd': 11, 'Wff': 12}
    fname_elect = "TBparams/electrons_reference.dat"
    all_lines = []
    with open(fname_elect, 'r') as fi:
        numlines = int(fi.readline().split()[1])
        header = fi.readline()
        all_lines.append(f"Noelem= {numlines}\n")
        all_lines.append(header)
        for _ in range(numlines):
            line_spl = fi.readline().split()
            element1 = line_spl[0]
            for index, d in enumerate(all_elect):
                if d.get('Element') == element1:
                    for indx, name in enumerate(params_names):
                        print(line_spl)
                        print(index, indx, len(params_names))
                        print(params_values)
                        line_spl[name_pos[name]] = str(params_values[index * len(params_names) + indx])[:12]
                    with open("linese.dat", 'a+') as line_file:
                        line_file.write((" ".join(map(str, line_spl)) + "\n"))
                    break
            all_lines.append(" ".join(map(str, line_spl)) + "\n")

    with open("TBparams/electrons.dat", 'w') as fo:
        fo.writelines(all_lines)


def extract_and_create_bounds(integrals, params_names):
    result = []
    for key in params_names:
        if key in integrals:
            value = integrals[key]
            lower_bound = value * 0.7
            upper_bound = value * 1.9
            result.append((min(lower_bound, upper_bound), max(lower_bound, upper_bound)))
    return result


def extract_and_create_guess(integrals, params_names):
    result = []
    for key in params_names:
        if key in integrals:
            value = integrals[key]
            result.append(value)
    return result


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

number_of_structures = 1
field = [0, 0, 0]
verb = 0

forces_siesta_list = []
coords_list = []
latticeVectors_list = []
symbols_list = []
atomTypes_list = []

# Read the structures into lists
for i in range(number_of_structures):
    fname = "phosphorus_forces_positions/forces_0" + str(i) + "_phosph.FA"
    forces_siesta = get_forces_siesta(fname)
    coord_files = "phosphorus_forces_positions/coordinates_0" + str(i) + "_phosph.xyz"
    latticeVectors, symbols, atomTypes, coords0 = fileio.read_xyz_file(coord_files, lib="None", verb=verb)
    latticeVectors[:] = 0.0
    latticeVectors[0,0] = 28.00
    latticeVectors[1,1] = 29.00
    latticeVectors[2,2] = 35.00
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
                  'NC': ('N', 'C', ['sps']),
                  'HP': ('H', 'P', ('sss', 'sps')),
                  'HC': ('H', 'C', ('sss', 'sps'))}

wished = ['PP', 'PO', 'OP', 'PC', 'PN', 'NP', 'HP']
all_integrals_SS = []
for elms in wished:
    val = atom_atom_kind[elms]
    for v in val[2]:
        all_integrals_SS.append(get_atom_atoms_kind(val[0], val[1], v))

wished_ppots = [('P', 'P'), ('P', 'O'), ('P', 'C'), ('P', 'N'), ('P', 'H')]
all_ppots = [get_atom_atoms_for_ppot(elms[0], elms[1], all_dicts_ppot) for elms in wished_ppots]
dic_elect_P = get_atom_for_elect('P', all_dicts_elect)

all_elect = [dic_elect_P]

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
bounds[:, 0] = bounds[:, 0] - 0.5*abs(bounds[:, 0])
bounds[:, 1] = bounds[:, 1] + 0.5*abs(bounds[:, 1])

# guess = [-8.0, -1, -0.5  , -0.5 , -0.5, -0.5, -0.5, -0.5, -0.5]
cost_ref = cost_function2(all_guess_v, params_names, ppotparams_names, electparams_names, all_integrals_SS, all_ppots,
                         all_elect, forces_siesta_list,
                         coords_list,
                         latticeVectors_list,
                         symbols_list,
                         atomTypes_list
                         )


optimize = False
if(optimize):
    latticeVectors = latticeVectors_list[0]
    symbols = symbols_list[0]
    atomTypes = atomTypes_list[0]
    coords0 = coords_list[0]
    coords_new = optimize_coordinates(latticeVectors,atomTypes,symbols,coords0,method='BFGS',parampath='/tmp',iterations=100,tol=0.01)

    write_xyz_coordinates("opt_str.xyz", coords_new, atoTypes, symbols)

method = "Diff"
if(method == "Anneal"):
    result = dual_annealing(cost_function2,bounds, args=(
        params_names, ppotparams_names,
        electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list,
        coords_list,
        latticeVectors_list,
        symbols_list,
        atomTypes_list
        ), maxiter=1000, initial_temp=10.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000000.0, seed=None, no_local_search=False, callback=None, x0=None)
elif(method == "Diff"):
    result = differential_evolution(cost_function2, bounds=bounds, args=(
        params_names, ppotparams_names,
        electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list,
        coords_list,
        latticeVectors_list,
        symbols_list,
        atomTypes_list
        ), maxiter=3, recombination=0.1, mutation=0.1, tol=0.01, popsize=15, disp=True)
else:
    result = minimize(cost_function2, all_guess, args=(params_names, ppotparams_names, 
         electparams_names, all_integrals_SS, all_ppots, all_elect, forces_siesta_list,
        coords_list,
        latticeVectors_list,
        symbols_list,
        atomTypes_list), method='Nelder-Mead', jac=None, hess=None, hessp=None, 
        bounds=bounds, constraints=(), tol=0.001, callback=None, options={'maxiter': 6000})


exit(0)

