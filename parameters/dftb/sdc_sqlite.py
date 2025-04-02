import sqlite3, os
import sys

import numpy as np


# Function to execute an SQL query and return results avoiding duplicating code.
def execute_query(db_name, query, params=None):
    """
    Execute a SQL query on the database.

    Args:
        db_name (str): The name of the SQLite database.
        query (str): The SQL query to execute.
        params (tuple, optional): Parameters to pass to the query.

    Returns:
        list: The result of the query.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        if params:
            c.execute(query, params)
        else:
            c.execute(query)

        result = c.fetchall() if c.description else None
        return result
    except sqlite3.Error as e:
        print("Error executing query:", e)
        return None
    finally:
        if conn:
            conn.close()


def get_column_names(table_name, db_name):
    query = f"PRAGMA table_info({table_name})"
    columns = execute_query(db_name, query)

    # Extract column names from the fetched data
    column_names = [column[1] for column in columns]
    return column_names


# Function to get column names for all tables in the database
def get_all_table_column_names(db_name):
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Get table names in the database
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = c.fetchall()

        table_column_names = {}

        # Iterate over table names and get column names for each table
        for table in table_names:
            table_name = table[0]
            # column_names = get_column_names(table_name, db_name) #instead of opening a new conn

            c.execute(f"PRAGMA table_info({table_name})")
            columns = c.fetchall()
            column_names = [column[1] for column in columns]  # Extract column names from the fetched data
            table_column_names[table_name] = column_names

        conn.close()

        return table_column_names
    except sqlite3.Error as e:
        print("Error get_all_table_column_names function", e)
        return None
    finally:
        if conn:
            conn.close()


# Function to extract dipole (in Debye) from output.txt using grep
def extract_dipole(file_path):
    import subprocess
    output = subprocess.getoutput(' grep dipol ' + file_path + ' |head -1')
    return tuple(map(float, output.split()[-3:]))


# Function to extract dipole moments
def get_all_dipole_moments(db_name):
    """
    Retrieve all dipole moments from the database.

    Returns:
        numpy.ndarray: An array containing all dipole moments stored in the database.
    """
    query = "SELECT dipole_x, dipole_y, dipole_z FROM Molecule"
    dipole_data = execute_query(db_name, query)
    if dipole_data:
        return np.array(dipole_data)
    else:
        return np.array([])


# Function to read atomic positions and species from molecule.xyz
def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0])
        atomic_data = [line.split() for line in lines[2:]]
        species = [data[0] for data in atomic_data]
        atomic_positions = [list(map(float, data[1:])) for data in atomic_data]
    return num_atoms, species, atomic_positions


# Function to read Born charges from molecule.BC file
def read_born_charges(file_path):
    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line
            bq = [list(map(float, line.split())) for line in file]
            born_charges = [bq[i:i + 3] for i in range(0, len(bq), 3)]
            return born_charges
    except Exception as e:
        print("Error reading born charges:", e)
        return None


# Function to read forces from molecule.FA file
def read_forces(file_path):
    with open(file_path, 'r') as file:
        num_atoms = int(next(file))  # Read the number of atoms
        forces = [tuple(map(float, line.split()[1:])) for line in file]
        return forces


# Function to create a SQLite3 database and insert data
def create_database(db_name, folders):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS Molecule (
                 id INTEGER PRIMARY KEY,
                 name TEXT,
                 num_atoms INTEGER,
                 dipole_x REAL,
                 dipole_y REAL,
                 dipole_z REAL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS Atom (
                 id INTEGER PRIMARY KEY,
                 molecule_id INTEGER,
                 atom_number INTEGER,
                 species TEXT,
                 x REAL,
                 y REAL,
                 z REAL,
                 force_x REAL,
                 force_y REAL,
                 force_z REAL,
                 born_charge_xx REAL,
                 born_charge_xy REAL,
                 born_charge_xz REAL,
                 born_charge_yx REAL,
                 born_charge_yy REAL,
                 born_charge_yz REAL,
                 born_charge_zx REAL,
                 born_charge_zy REAL,
                 born_charge_zz REAL,                 
                 FOREIGN KEY(molecule_id) REFERENCES Molecule(id))''')

    # Insert data
    for folder in folders:
        molecule_name = folder.split('_')[1]
        print(molecule_name)
        xyz_file = os.path.join(folder, 'molecule.xyz')
        born_charges_file = os.path.join(folder, 'molecule.BC')
        forces_file = os.path.join(folder, 'molecule.FA')
        output_file = os.path.join(folder, 'output.txt')
        try:

            # Extract data
            num_atoms, species, atomic_positions = read_xyz(xyz_file)
            # born_charges = np.reshape(read_born_charges(born_charges_file), (num_atoms, 9))
            born_charges = read_born_charges(born_charges_file)
            forces = read_forces(forces_file)
            dipole = extract_dipole(output_file)

            # Insert into Molecule table
            c.execute("INSERT INTO Molecule (name, num_atoms, dipole_x, dipole_y, dipole_z) VALUES (?, ?, ?, ?, ?)",
                      (molecule_name, num_atoms, dipole[0], dipole[1], dipole[2]))
            molecule_id = c.lastrowid

            for i, (species_info, atom_position, born_charge, force) in enumerate(
                    zip(species, atomic_positions, born_charges, forces), 1):
                c.execute(
                    "INSERT INTO Atom (molecule_id, atom_number, species, x, y, z, force_x, force_y, force_z, "
                    "born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,born_charge_yz,"
                    "born_charge_zx,born_charge_zy,born_charge_zz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?)",
                    (molecule_id, i, species_info, atom_position[0], atom_position[1], atom_position[2],
                     force[0], force[1], force[2], born_charge[0][0], born_charge[0][1], born_charge[0][2],
                     born_charge[1][0], born_charge[1][1], born_charge[1][2], born_charge[2][0], born_charge[2][1],
                     born_charge[2][2]))

            # print(atomic_positions)
            # Insert into Atom table
            # for i, (species_info, atom_position, born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,born_charge_yz, born_charge_zx,born_charge_zy,born_charge_zz, force) in enumerate(
            #         zip(species, atomic_positions,  born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,born_charge_yz, born_charge_zx,born_charge_zy,born_charge_zz, forces), 1):
            #     c.execute(
            #         "INSERT INTO Atom (molecule_id, atom_number, species, x, y, z, force_x, force_y, force_z, "
            #         "born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,born_charge_yz,"
            #         "born_charge_zx,born_charge_zy,born_charge_zz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            #         "?, ?, ?, ?)",
            #         (molecule_id, i, species_info, atom_position[0], atom_position[1], atom_position[2],
            #          force[0], force[1], force[2], born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,born_charge_yz,born_charge_zx,born_charge_zy,born_charge_zz))
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

    conn.commit()
    conn.close()


# Generic function to extract data from a specific molecule
def extract_data_from_molecule(db_name, molecule_name, query, *args):
    """
    Extract data related to a specific molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule.
        query (str): The SQL query to execute.
        *args: Additional arguments to pass to the query.

    Returns:
        tuple or list: The fetched data from the database.
    """
    try:
        data = execute_query(db_name, query, (molecule_name,) + args)
        return data
    except sqlite3.Error as e:
        print("Error accessing database:", e)


def extract_atomic_positions_of_molecule(db_name, molecule_name):
    """
    Extract atomic positions of all atoms in a specific molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule.

    Returns:
        list: A list containing atomic positions for all atoms in the molecule.
    """
    positions_query = ("SELECT x, y, z FROM Atom "
                       "INNER JOIN Molecule ON Atom.molecule_id = Molecule.id WHERE Molecule.name=?")

    positions_data = execute_query(db_name, positions_query, (molecule_name,))
    if positions_data:
        return positions_data
    else:
        return []


# Function to extract born charges for all atoms in a specific molecule
def extract_born_charges_of_molecule1(db_name, molecule_name):
    """
    Extract born charges for all atoms in a specific molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule.

    Returns:
        list: A list containing born charges for all atoms in the molecule.
    """

    # Get molecule ID
    q1 = "SELECT id FROM Molecule WHERE name=?", (molecule_name,)
    molecule_id = execute_query(db_name, q1, (molecule_name,))
    q1 = "SELECT num_atoms FROM Molecule WHERE id=?", (molecule_id,)
    num_atoms = execute_query(db_name, q1, (molecule_name,))

    born_charges_data = execute_query(db_name, born_charges_query, (molecule_name,))
    if born_charges_data:
        return born_charges_data
    else:
        return []


def extract_born_charges_of_molecule(db_name, molecule_name):
    """
    Extract born charges for all atoms in a specific molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule.

    Returns:
        list: A list containing born charges for all atoms in the molecule.
    """
    born_charges_query = ("SELECT born_charge_xx,born_charge_xy,born_charge_xz,born_charge_yx,born_charge_yy,"
                          "born_charge_yz, born_charge_zx,born_charge_zy,born_charge_zz FROM Atom "
                          "INNER JOIN Molecule ON Atom.molecule_id = Molecule.id WHERE Molecule.name=?")

    born_charges_data = execute_query(db_name, born_charges_query, (molecule_name,))
    if born_charges_data:
        return born_charges_data
    else:
        return []


# regression_testing function
def regression_testing():
    db_name = 'molecules.db'
    molecule_name = '5'

    # Example usage: Retrieve dipole moment from a specific molecule
    dipole_query = "SELECT dipole_x, dipole_y, dipole_z FROM Molecule WHERE name=?"
    dipole_data = extract_data_from_molecule(db_name, molecule_name, dipole_query)
    print(f"Dipole moment of molecule {molecule_name}: {dipole_data}")

    # Example usage: Retrieve forces from a specific atom in a specific molecule
    atom_number = 1
    forces_query = ("SELECT force_x, force_y, force_z FROM Atom INNER JOIN Molecule ON Atom.molecule_id = Molecule.id "
                    "WHERE Molecule.name=? AND Atom.atom_number=?")
    forces_data = extract_data_from_molecule(db_name, molecule_name, forces_query, atom_number)
    print(f"Forces on atom {atom_number} in molecule {molecule_name}: {forces_data}")

    # Retrieve atomic position of a specific atom in a specific molecule
    position_query = ("SELECT x, y, z FROM Atom INNER JOIN Molecule ON Atom.molecule_id = Molecule.id "
                      "WHERE Molecule.name=? AND Atom.atom_number=?")
    position_data = extract_data_from_molecule(db_name, molecule_name, position_query, atom_number)
    print(f"Atomic position of atom {atom_number} in molecule {molecule_name}: {position_data}")

    # Retrieve all atomic positions in a specific molecule
    positions_data = extract_atomic_positions_of_molecule(db_name, molecule_name)
    print(f"Atomic positions of all atoms in molecule {molecule_name}: {positions_data}")

    # Retrieve all born_charges in a specific molecule
    born_charges_data = extract_born_charges_of_molecule(db_name, molecule_name)
    print(f"Born_charges of all atoms in molecule {molecule_name}: {born_charges_data}")

    # retriev bond lengths for a given molecule
    bond_lengths_matrix = calculate_bond_lengths_for_molecule(db_name, "2")
    print("Bond Lengths Matrix:")
    for v in bond_lengths_matrix: print(v)

    # Calculate statistics of all mag. mom. aligned along z
    dipole_moments = get_all_dipole_moments(db_name)
    rotated_dipoles = rotate_dipoles_to_z_axis(dipole_moments)
    analyze_dipole_moments(rotated_dipoles, db_name)

    # Get info about the column name of a table (Atom)
    d = get_column_names("Atom", db_name)
    for v in d: print(v)

    # Get info about a molecule
    d = extract_molecule_data(db_name, "2")
    for v in d: print(v, d[v])


def analyze_dipole_moments(dipoles, db_name):
    dipole_array = get_all_dipole_moments(db_name)

    # Calculate statistics
    dipole_mean = np.mean(dipole_array, axis=0)
    dipole_median = np.median(dipole_array, axis=0)
    dipole_std = np.std(dipole_array, axis=0)
    # dipole_mean, dipole_median, dipole_std = analyze_dipole_moments(db_name)
    print("Dipole moment statistics:")
    print("Mean:", dipole_mean)
    print("Median:", dipole_median)
    print("Standard Deviation:", dipole_std)


# Function to calculate bond lengths between pairs of atoms
def calculate_bond_lengths(positions):
    num_atoms = len(positions)
    bond_lengths = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            bond_lengths.append((i, j, distance))

    return bond_lengths


# Function to extract positions of atoms for each molecule from the database
def extract_positions_for_molecule(db_name, molecule_name):
    """
    Extract atomic positions for a specific molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule.

    Returns:
        numpy.ndarray: Array containing atomic positions for the molecule.
    """
    query = ("SELECT x, y, z FROM Atom "
             "INNER JOIN Molecule ON Atom.molecule_id = Molecule.id "
             "WHERE Molecule.name=?")
    positions = execute_query(db_name, query, (molecule_name,))
    return np.array(positions)


# Function to calculate bond lengths for each molecule
def calculate_bond_lengths_for_molecule(db_name, molecule_name):
    # Extract positions of atoms for the molecule
    positions = extract_positions_for_molecule(db_name, molecule_name)
    # Calculate bond lengths
    return calculate_bond_lengths(positions)


# Function to calculate bond angles between sets of three atoms
def calculate_bond_angles(positions):
    num_atoms = len(positions)
    bond_angles = []

    for i in range(num_atoms):
        for j in range(num_atoms):
            for k in range(num_atoms):
                if i != j and i != k and j != k:
                    vec_ij = positions[i] - positions[j]
                    vec_kj = positions[k] - positions[j]
                    angle = np.arccos(np.dot(vec_ij, vec_kj) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_kj)))
                    bond_angles.append((i, j, k, np.degrees(angle)))

    return bond_angles


# Function to calculate dihedral angles between sets of four atoms
def calculate_dihedral_angles(positions):
    num_atoms = len(positions)
    dihedral_angles = []

    for i in range(num_atoms):
        for j in range(num_atoms):
            for k in range(num_atoms):
                for l in range(num_atoms):
                    if len({i, j, k, l}) == 4:  # Ensure all atoms are distinct
                        vec_ij = positions[i] - positions[j]
                        vec_jk = positions[j] - positions[k]
                        vec_kl = positions[k] - positions[l]

                        normal_ijk = np.cross(vec_ij, vec_jk)
                        normal_jkl = np.cross(vec_jk, vec_kl)

                        angle = np.arccos(np.dot(normal_ijk, normal_jkl) /
                                          (np.linalg.norm(normal_ijk) * np.linalg.norm(normal_jkl)))

                        # Determine sign of the angle based on the sign of the dot product
                        sign = np.sign(np.dot(normal_ijk, vec_kl))

                        dihedral_angle = np.degrees(angle) * sign
                        dihedral_angles.append((i, j, k, l, dihedral_angle))

    return dihedral_angles


# Function to extract data from a molecule folder
def extract_molecule_data(db_name, molecule_name):
    """
    Extract data for a molecule from the database.

    Args:
        db_name (str): The name of the SQLite database.
        molecule_name (str): The name of the molecule to extract data for.

    Returns:
        dict: A dictionary containing information about the molecule, including its name, atomic positions,
              Born charges, and dipole moment.
    """
    # Extract molecule name from folder name
    molecule_data = {'name': molecule_name}

    # Define SQL queries
    atom_query = ("SELECT species, x, y, z FROM Atom "
                  "INNER JOIN Molecule ON Atom.molecule_id = Molecule.id "
                  "WHERE Molecule.name=?")
    born_charge_query = ("SELECT born_charge_x, born_charge_y, born_charge_z FROM Atom "
                         "INNER JOIN Molecule ON Atom.molecule_id = Molecule.id "
                         "WHERE Molecule.name=?")
    dipole_query = "SELECT dipole_x, dipole_y, dipole_z FROM Molecule WHERE name=?"

    # Execute SQL queries
    atom_records = execute_query(db_name, atom_query, (molecule_name,))
    born_charge_records = execute_query(db_name, born_charge_query, (molecule_name,))
    dipole_record = execute_query(db_name, dipole_query, (molecule_name,))

    # Process fetched data
    atoms = [{'species': record[0], 'x': record[1], 'y': record[2], 'z': record[3]} for record in atom_records]
    dipole = {'x': dipole_record[0][0], 'y': dipole_record[0][1], 'z': dipole_record[0][2]} if dipole_record else None
    born_charges = [{'x': record[0], 'y': record[1], 'z': record[2]} for record in born_charge_records]

    # Populate molecule_data dictionary
    molecule_data['atoms'] = atoms
    molecule_data['born_charges'] = born_charges
    molecule_data['dipole'] = dipole

    return molecule_data


def insert_new_molecule_data(db_name, molecule_name, num_atoms, dipole, atomic_positions, species, born_charges,
                             forces):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    try:
        # Insert into Molecule table
        c.execute("INSERT INTO Molecule (name, num_atoms, dipole_x, dipole_y, dipole_z) VALUES (?, ?, ?, ?, ?)",
                  (molecule_name, num_atoms, dipole[0], dipole[1], dipole[2]))
        molecule_id = c.lastrowid

        # Insert into Atom table
        for i, (species_info, atom_position, born_charge, force) in enumerate(
                zip(species, atomic_positions, born_charges, forces), 1):
            c.execute(
                "INSERT INTO Atom (molecule_id, atom_number, species, x, y, z, force_x, force_y, force_z, "
                "born_charge_xx, born_charge_xy, born_charge_xz, born_charge_yx, born_charge_yy, born_charge_yz, "
                "born_charge_zx, born_charge_zy, born_charge_zz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (molecule_id, i, species_info, atom_position[0], atom_position[1], atom_position[2],
                 force[0], force[1], force[2], born_charge[0][0], born_charge[0][1], born_charge[0][2],
                 born_charge[1][0], born_charge[1][1], born_charge[1][2], born_charge[2][0], born_charge[2][1],
                 born_charge[2][2]))

        conn.commit()
        print(f"Data for molecule '{molecule_name}' inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data for molecule '{molecule_name}': {e}")
    finally:
        conn.close()


# Function to check if a molecule exists in the database
def molecule_exists_in_database(db_name, molecule_name):
    """
   Check if a molecule exists in the database.

   Args:
       db_name (str): The name of the SQLite database.
       molecule_name (str): The name of the molecule to check.

   Returns:
       bool: True if the molecule exists in the database, False otherwise.
   """
    query = "SELECT COUNT(*) FROM Molecule WHERE name=?"
    result = execute_query(db_name, query, (molecule_name,))
    count = result[0][0] if result else 0
    return count > 0


def print_table_column_names():
    table_column_names = get_all_table_column_names(db_name)
    for table_name, column_names in table_column_names.items():
        print(f"Table: {table_name}")
        print("Columns:", column_names)


def rotate_dipoles_to_z_axis(dipole_moments):
    rotated_dipoles = []

    for dipole in dipole_moments:
        # Compute magnitude of dipole moment
        magnitude = np.linalg.norm(dipole)

        # Find largest component and corresponding index
        max_index = np.argmax(np.abs(dipole))

        # Compute rotation angle
        if max_index == 0:  # x-component is largest
            angle = np.arctan2(dipole[1], dipole[0])
        elif max_index == 1:  # y-component is largest
            angle = np.arctan2(dipole[0], dipole[1]) - np.pi / 2
        else:  # z-component is largest, no rotation needed
            angle = 0

        # Create rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])

        # Rotate dipole moment
        rotated_dipole = np.dot(rotation_matrix, dipole / magnitude) * magnitude
        rotated_dipoles.append(rotated_dipole)

    return rotated_dipoles


def compute_born_charges_from_FA_files(prefix, field, force_filename, print_charges=0):
    # Define directories
    dirsp = [prefix + str(field) + 'x', prefix + str(field) + 'y', prefix + str(field) + 'z']
    dirsm = [prefix + str(field * -1) + 'x', prefix + str(field * -1) + 'y', prefix + str(field * -1) + 'z']
    # Read forces from files
    forces_pos = [read_forces(v + '/' + force_filename) for v in dirsp]
    forces_minus = [read_forces(v + "/" + force_filename) for v in dirsm]
    charges = []
    for j in range(len(forces_minus[0])):
        charges_atomk = []
        # Compute charges for each atom
        for k in range(3):
            charges_atomk.append([(forces_pos[i][j][k] - forces_minus[i][j][k]) / (2 * field) for i in range(3)])
            if print_charges: print(
                '{:,d}\t {:,.3f}\t {:,.3f}\t {:,.3f} '.format(j+1, charges_atomk[k][0], charges_atomk[k][1],
                                                              charges_atomk[k][2]))
        if print_charges: print('')
        charges.append(charges_atomk)
    return charges


def compute_born_charges_from_FA_files_highOrder(prefix, field1, field2, force_filename, print_charges=0):
    # Define directories
    dirsp1 = [prefix + str(field1) + 'x', prefix + str(field1) + 'y', prefix + str(field1) + 'z']
    dirsm1 = [prefix + str(field1 * -1) + 'x', prefix + str(field1 * -1) + 'y', prefix + str(field1 * -1) + 'z']
    dirsp2 = [prefix + str(field2) + 'x', prefix + str(field2) + 'y', prefix + str(field2) + 'z']
    dirsm2 = [prefix + str(field2 * -1) + 'x', prefix + str(field2 * -1) + 'y', prefix + str(field2 * -1) + 'z']

    # Read forces from files
    forces_pos1 = [read_forces(v + '/' + force_filename) for v in dirsp1]
    forces_minus1 = [read_forces(v + "/" + force_filename) for v in dirsm1]
    forces_pos2 = [read_forces(v + '/' + force_filename) for v in dirsp2]
    forces_minus2 = [read_forces(v + "/" + force_filename) for v in dirsm2]

    charges = []
    for j in range(len(forces_minus1[0])):
        charges_atomk = []
        # Compute charges for each atom
        for k in range(3):
            charges_atomk.append([(-1.0 / 12 * forces_pos2[i][j][k] + 2.0 / 3 * forces_pos1[i][j][k] - 2.0 / 3 *
                                   forces_minus1[i][j][k] + 1.0 / 12 * forces_minus2[i][j][k]) / (1 * field1) for i in
                                  range(3)])
            if print_charges: print(
                '{:,d}\t {:,.3f}\t {:,.3f}\t {:,.3f} '.format(j+1, charges_atomk[k][0], charges_atomk[k][1],
                                                              charges_atomk[k][2]))
        if print_charges: print('')
        charges.append(charges_atomk)
    return charges


if __name__ == "__main__":
    db_name = 'molecules2.db'
    # molecule_folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith('molecule_')]
    # molecule_folders = ["molecule_149"]
    # create_database(db_name, molecule_folders)
    # regression_testing()
    # d = extract_born_charges_of_molecule(db_name, "149")
    # print(d)
    # print(extract_molecule_data(db_name,"2"))
    # print_table_column_names()

    # TEST:
    # insert_new_molecule_data(db_name, molecule_name, num_atoms, dipole, atomic_positions, species, born_charges, forces)
    # insert_new_molecule_data(
    # update_database_with_new_molecules(db_name, molecule_folders)
    # bond_angles = calculate_bond_angles(positions)
    # dihedral_angles = calculate_dihedral_angles(positions)
    # prefix = "C100-C1-10_E_"
    # prefix = "C40-C1-4_E_"
    prefix = "molecule_E_"
    field1, field2 = 0.01, 0.02
    force_file_name = 'molecule.FA'

    charges = compute_born_charges_from_FA_files(prefix, field2, force_file_name, print_charges=1)
    # charges = compute_born_charges_from_FA_files_highOrder(prefix, field1, field2, force_file_name, print_charges=1)
