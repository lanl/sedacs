import numpy as np
import io, sys, os

atomic_numbers = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95,
    "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100
}

atomic_masses = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.6, "I": 126.9, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.5, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209, "At": 210, "Rn": 222,
    "Fr": 223, "Ra": 226, "Ac": 227, "Th": 232.04, "Pa": 231.04, "U": 238.03,
    "Np": 237, "Pu": 244, "Am": 243, "Cm": 247, "Bk": 247, "Cf": 251, "Es": 252,
    "Fm": 257
}

UNIT_LENGTH = 'Ang'
UNIT_ENERGY = 'Ry'


def list_xyz_files(folder_path):
    """
    Lists all xyz files in the given folder.

    Args:
        folder_path (str): Path to the folder containing xyz files.

    Returns:
        list: A list of full file paths to all xyz files in the folder.
    """
    xyz_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xyz"):  # Check if the file has a .xyz extension
            full_path = os.path.join(folder_path, file_name)
            xyz_files.append(full_path)
    return xyz_files


def list_pdb_files(folder_path):
    """
    Lists all PDB files in the given folder.

    Args:
        folder_path (str): Path to the folder containing PDB files.

    Returns:
        list: A list of full file paths to all PDB files in the folder.
    """
    pdb_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdb"):  # Check if the file has a .pdb extension
            full_path = os.path.join(folder_path, file_name)
            pdb_files.append(full_path)
    return pdb_files
        


def pdb_to_xyz(pdb_file, xyz_file="", write_file=0):
    """
    Converts a PDB file to an XYZ file.

    Args:
        pdb_file (str): Path to the input PDB file.
        xyz_file (str): Path to the output XYZ file.
    """
    atom_data = []

    # Read the PDB file
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_type = line[76:78].strip()  # Atom type
                x = float(line[30:38].strip())  # X-coordinate
                y = float(line[38:46].strip())  # Y-coordinate
                z = float(line[46:54].strip())  # Z-coordinate
                atom_data.append((atom_type, x, y, z))

    # Write to the XYZ file
    if write_file == 1:
        if xyz_file=="":xyz_file=input_file[:-3]+"xyz"
        with open(xyz_file, 'w') as file:
            file.write(f"{len(atom_data)}\n")  # Number of atoms
            file.write("Converted from PDB file\n")  # Comment line
            for atom in atom_data:
                file.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
    return atom_data



def read_xyz_file(input_file):
    species, coord = [], []

    # Open input file for reading
    with open(input_file, 'r') as infile:
        # Read number of atoms
        num_atoms = int(infile.readline().strip())
        # Skip blank line
        infile.readline()

        # Process each line containing atom coordinates
        for i in range(int(num_atoms)):
            # Split line into atom species, x, y, z coordinates
            line = infile.readline()
            l = line.split()
            atom_specie, x, y, z = l[0], l[1], l[2], l[3]
            species.append(atom_specie)
            coord.append([float(x), float(y), float(z)])
        return species, coord


def get_siesta_default_params():
    params = {
        'SystemName': 'molecule',
        'SystemLabel': 'molecule',
        'NumberOfAtoms': '8',
        'NumberOfSpecies': '3',
        'ChemicalSpeciesLabel': [],
        'LatticeConstant': '1.0 Ang',
        'LatticeVectors': [],
        'XC.functional': 'GGA',
        'XC.authors': 'PBE',
        'MeshCutoff': '300 Ry',
        'MaxSCFIterations': '180',
        'DM.NumberPulay': '5',
        'DM.MixingWeight': '0.05',
        'DM.Tolerance': '1.d-4',
        'DM.UseSaveDM': 'T',
        'UseSaveData': 'T',
        'SolutionMethod': 'diagon',
        'ElectronicTemperature': '15 K',
        'SaveHS': 'F',
        'LongOutput': 'T',
        'WriteCoorStep': 'T',
        'WriteEigenvalues': 'T',
        'WriteCoorXmol': 'T',
        'WriteMullikenPop': '1',
        'WriteForces': 'T',
        'AtomicCoordinatesFormat': 'Ang',
        # 'MD.TypeOfRun': 'FC',
        # 'BornCharge': 'T',
        # 'MD.FCDispl': '0.01 bohr',
        # 'PolarizationGrids': [],
        'MD.NumCGsteps': '0',
        'MD.MaxForceTol': '0.01 eV/Ang',
        'MD.VariableCell': 'F',
        'MD.MaxStressTol': '1 GPa',
        'WriteMDXmol': 'T',
        'kgrid_Monkhorst_Pack': [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0]],
        'BandLinesScale': 'ReciprocalLatticeVectors',
        'Eigenvectors': 'T',
        'Diag.ParallelOverK': 'F',
        'WriteKbands': 'F',
        'BandLines': [['1', '0.0', '0.0', '0.0']],
        '%block AtomicCoordinatesAndAtomicSpecies   <': 'siesta.dat',
        'Slab.DipoleCorrection': 'False'
    }
    return params


def write_block(name, block):
    """
    Generates a siesta block and returns text.

    """
    lines = [f'\n%block {name}']
    for row in block:
        data = ' '.join(str(r) + '\t' for r in row)
        lines.append(f'  {data}')
    lines.append(f'%endblock {name}' + '\n')
    return '\n'.join(lines) + '\n'


def generate_text(param, filename='input.fdf', folder='./'):
    """
    Generates a text file from the given parameters.

    :param param: Dictionary containing SIESTA parameters.
    :param filename: Name of the file to write the parameters to (default is 'input.fdf').
    :param folder: Folder path where the file will be saved (default is current directory).
    """
    buffer = io.StringIO()
    for key, value in param.items():
        if isinstance(value, list):
            buffer.write(write_block(key, value))
        else:
            buffer.write(f"{key}\t{value}\n")
    with open(folder + filename, 'w') as f:
        f.write(buffer.getvalue())
    buffer.close()


def write_coordinates(output_file_name, species, coordinates, folder='./'):
    with open(folder + output_file_name, 'w') as outfile:
        for specie, coords in zip(species, coordinates):
            outfile.write(f"{coords[0]:>8}\t{coords[1]:>8}\t{coords[2]:>8}\t"
                          f"{species_list.index(specie) + 1}\t{atomic_masses[specie]:>8}\n")


def compose_basic_params_and_blocks(input_file):
    """
    Reads an XYZ file to extract atomic species and coordinates, computes lattice vectors,
    and updates SIESTA parameters accordingly.

    :param input_file: Path to the XYZ file.
    :return: Tuple containing species, coordinates, and species list.
    """
    try:
        # Extract species and coordinates from the XYZ file
        species, coordinates = read_xyz_file(input_file)

        # Create a list of unique species
        species_list = list(set(species))

        block_ChemicalSpeciesLabel = [[str(i + 1), str(atomic_numbers[sp]), sp] for i, sp in enumerate(species_list)]
        min_max = [round(max(np.array(coordinates).T[i]) - min(np.array(coordinates).T[i])) for i in range(3)]
        block_latticeVectors = [
            [min_max[0] + 20, 0.0, 0.0],
            [0.0, min_max[1] + 20, 0.0],
            [0.0, 0.0, min_max[2] + 20]
        ]

        # Update SIESTA parameters
        siesta_params['NumberOfAtoms'] = len(coordinates)
        siesta_params['NumberOfSpecies'] = len(species_list)
        siesta_params['ChemicalSpeciesLabel'] = block_ChemicalSpeciesLabel
        siesta_params['LatticeVectors'] = block_latticeVectors

        return species, coordinates, species_list

    except Exception as e:
        print(f"Error processing  xyz file, {e}")
        sys.exit(0)


def composed_block_ExternalElectricField(efield_coord, efield_strenght):
    """
    Composes the ExternalElectricField block for SIESTA input files based on the specified coordinate and strength.

    :param efield_coord: Coordinate axis ('x', 'y', or 'z') along which the electric field is applied.
    :param efield_strength: Strength of the electric field in V/Angstrom.
    :return: List representing the ExternalElectricField block for the specified axis and strength.
    """
    if efield_coord == 'x':
        return [[efield_strenght, 0.0, 0.0, 'V/Ang']]
    elif efield_coord == 'y':
        return [[0.0, efield_strenght, 0.0, 'V/Ang']]
    elif efield_coord == 'z':
        return [[0.0, 0.0, efield_strenght, 'V/Ang']]
    else:
        print("Problem composing block_ExternalElectricField")
        print("Exiting")
        sys.exit(0)


def copy_pseudopotentials_from_directory(elements, subfolder_name_orig, subfolder_name_dest):
    import shutil
    # Check if the subfolder_name_orig exits and is a valid path
    if subfolder_name_orig is None:
        print(subfolder_name_orig + " folder is not set.")
        exit(1)
    elif not os.path.exists(subfolder_name_orig):
        print(f"Path '{subfolder_name_orig}' specified in subfolder_name_orig does not exist.")
        exit(1)

    # Check if the subfolder_name_dest exits and is a valid path
    if subfolder_name_dest is None:
        print(subfolder_name_dest + " folder is not set.")
        exit(1)
    elif not os.path.exists(subfolder_name_orig):
        print(f"Path '{subfolder_name_dest}' specified in subfolder_name_dest does not exist.")
        exit(1)

    try:
        # Copy the file from source_path to destination_path
        for elem in elements:
            shutil.copy(subfolder_name_orig + '/' + elem + '.psf', subfolder_name_dest)
        # print("File copied successfully!")
    except Exception as e:
        print("An error occurred while copying the file:", e)


# def copy_pseudopotentials_from_environmental_variable(elements, subfolder_name):
#     import shutil
#     # Get the path from the environmental variable
#     source_path = os.getenv('SIESTA_PSEUDOPOTENTIALS')
#
#     # Check if the environmental variable is set and contains a valid path
#     if source_path is None:
#         print("Environmental variable SOURCE_PATH is not set.")
#         exit(1)
#     elif not os.path.exists(source_path):
#         print(f"Path '{source_path}' specified in SOURCE_PATH does not exist.")
#         exit(1)
#
#     try:
#         # Copy the file from source_path to destination_path
#         for elem in elements:
#             shutil.copy(source_path + '/' + elem + '.psf', subfolder_name)
#         # print("File copied successfully!")
#     except Exception as e:
#         print("An error occurred while copying the file:", e)


def create_efield_folders(efields, species, subfolder_name_orig):
    """
    Creates directories for different electric field configurations and generates necessary SIESTA input files.

    :param efields: List of electric field strengths to be applied.
    :param species: List of atomic species involved in the simulation.
    :param subfolder_name_orig: folder from where pseudopotential will be copied
    :return: List of created folder names.
    """
    folder_list = []

    for efield_strength in efields:
        for efield_coord in ('x', 'y', 'z'):
            # Construct the subfolder name
            subfolder_name = base_folder + '/' + input_file.split('/')[-1][:-4] + '_E_' + str(
                efield_strength) + efield_coord + '/'
            print("Creating", subfolder_name)

            # Add the subfolder name to the list
            folder_list.append(subfolder_name)

            # Create the directory
            os.makedirs(subfolder_name, exist_ok=True)

            # Update SIESTA parameters
            siesta_params['Slab.DipoleCorrection'] = 'True'
            siesta_params['ExternalElectricField'] = composed_block_ExternalElectricField(efield_coord, efield_strength)

            # Write SIESTA input file and coordinates file
            generate_text(siesta_params, folder=subfolder_name)
            write_coordinates('siesta.dat', species, coordinates, folder=subfolder_name)

            # Copy pseudopotential files for each species
            for specie in species:
                copy_pseudopotentials_from_directory(specie, subfolder_name_orig=subfolder_name_orig,
                                                     subfolder_name_dest=subfolder_name)

    return folder_list


def create_single_energy_point_folder(species, subfolder_name_orig):
    """
    Creates directories for different electric field configurations and generates necessary SIESTA input files.

    :param species: List of atomic species involved in the simulation.
    :param subfolder_name_orig: folder from where pseudopotential will be copied
     """
    folder_list = []

    # Construct the subfolder name
    subfolder_name = base_folder + '/' + input_file.split('/')[-1][:-4] + '_singleE' + '/'
    print("Creating", subfolder_name)

    # Create the directory
    os.makedirs(subfolder_name, exist_ok=True)

    # Update SIESTA parameters
    siesta_params['MeshCutoff'] = '300 Ry'

    # Write SIESTA input file and coordinates file
    generate_text(siesta_params, folder=subfolder_name)
    write_coordinates('siesta.dat', species, coordinates, folder=subfolder_name)

    # Copy pseudopotential files for each species
    for specie in species:
        copy_pseudopotentials_from_directory(specie, subfolder_name_orig=subfolder_name_orig,
                                             subfolder_name_dest=subfolder_name)
                                             

def create_second_input(species, param, value, filename='input.fdf'):
    subfolder_name = base_folder + '/' + input_file.split('/')[-1][:-4] + '_singleE' + '/'
    print("Creating second", subfolder_name)
    # Update SIESTA parameters
    siesta_params[param] = value
    generate_text(siesta_params, folder=subfolder_name, filename=filename)
    
    
    
# pseudopot_dir = "/projects/shared/alopezb/grove/pseudopotentials/"
pseudopot_dir = '/projects/shared/pseudopotentials/'
path_to_files="/projects/shared/alopezb/rattle/ntc-rattled-split/"
xyz_files = list_xyz_files(path_to_files)
input_files=xyz_files
# Define SIESTA parameters
siesta_params = get_siesta_default_params()

base_folder = "siesta_runs" # pdb_files[0].split('/')[-1][:-4]
os.makedirs(base_folder, exist_ok=True)

for input_file in input_files:
    species, coordinates, species_list = compose_basic_params_and_blocks(input_file)
# efields = [0.01, 0.02, -0.01, -0.02]
# create_efield_folders(efields, species, pseudopot_dir)
#    create_single_energy_point_folder(species, pseudopot_dir)
    create_second_input(siesta_params, 'ElectronicTemperature', '100 K', filename='input2.fdf')
    create_second_input(siesta_params, 'ElectronicTemperature', '50 K' , filename='input3.fdf')
    
    
    
    
# FILE_LIST=all.dat
# while IFS= read -r file_name; do python  /home/alopezb/lanl/materials/DWave/bethe/bethe.py/sdc_siesta.py molecules_c40_xyz/$file_name ; done < "$FILE_LIST"

# for v in {1..9}; do python3 files/sdc_siesta.py files/configuration_0$v.xyz;  cp run-bebop.pbs configuration_0$v ; done
# for v in {1..9}; do cd configuration_0$v; qsub run-bebop.pbs ; cd ..; done


# Generate the Bash for loop
# bash_for_loop = ""
# for folder_name in folder_list:
#     bash_for_loop += f"cd {folder_name} && command_to_execute && cd ..\n"
#
# print(bash_for_loop)
# block_PolarizationGrids = [[20, 4, 4, 'yes'], [4, 20, 4, 'yes'], [4, 4, 20, 'yes']]
# siesta_params['PolarizationGrids'] = block_PolarizationGrids
