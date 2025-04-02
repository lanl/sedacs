import sys
import pathlib

from sedacs.types import ArrayLike

import numpy as np

from sedacs.system import parameters_to_vectors

__all__ = [
    "read_coords_file",
    "read_xyz_file",
    "write_xyz_coordinates",
    "read_xyz_trajectory",
    "read_pdb_file",
    "write_pdb_coordinates",
    "are_files_equivalent",
]


## Coordinates main reader
# @brief This will read the coodinates of a chemical system (so far only xyz and pdb
# are available).
#
def read_coords_file(fileName: str,
                     lib: str ="None",
                     verb=True) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Reads in atomic structure files, including lattice information.

    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be xyz, or pdb).
    lib: str
        If using an external library to parse the structure (such as ASE, PMG, etc.)
        *Currently not used.
    verb: bool
        Verbosity of the parsing procedure.

    Returns
    -------
    latticeVectors: ArrayLike (3, 3)
        The lattice vectors for the system of interest. Default behavior is for this to be 
        returned in the full format. Care should be taken if the ASE parser is used on 
        orthorhombic cells, as the return shape may be (3,) instead of (3,3)
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    """
    
    if lib not in ["ase", "None"] and lib is not None:
        raise NotImplementedError("Parsing lib options: ase, 'None', or None")

    ext = pathlib.Path(fileName).suffix
    if ext == ".xyz":
        read_fn = read_xyz_file
    elif ext == ".pdb":
        read_fn = read_pdb_file
    else:
        raise ValueError(f"Extension '{ext}' not recognized")

    latticeVectors, symbols, types, coords = read_fn(fileName, lib=lib, verb=False)

    return latticeVectors, symbols, types, coords


## xyz file parser
#  Reads in an xyz file with lattice informations.
#
#     Example xyz file format as follows:
#
# \verbatim
#        3
#        Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"
#        O  0.0 0.0 0.0
#        H  1.0 0.0 1.0
#        H -1.0 0.0 1.0
# \endverbatim
#
# @param fileName File name of the xyz file. Example: "coords.xyz"
# @param lib If using a particular library. Default is "None"
# @param verb Verbosity. If set to True will output relevant information.
# @return latticeVectors Lattice vectors. z-coordinate of the first vector = latticeVectors[0,2]
# @return symbols Symbol for each atom type. Symbol for first atom type = symbols[0]
# @return types Index type for each atom in the system. Type for first atom = type[0]
# @return coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
#
# @code{.unparsed}
# NumberOfAtomTypes = len(symbols)
# NumberOfAtoms = len(coordinates[:,0])
# @endcode
#
def read_xyz_file(fileName: str,
                  lib: str ="None",
                  verb=True) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Reads in atomic structure files in XYZ format, including lattice information.

    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be xyz only).
    lib: str
        If using an external library to parse the structure (such as ASE, PMG, etc.)
        *Currently not used.
    verb: bool
        Verbosity of the parsing procedure.

    Returns
    -------
    latticeVectors: ArrayLike (3, 3)
        The lattice vectors for the system of interest. Default behavior is for this to be 
        returned in the full format. Care should be taken if the ASE parser is used on 
        orthorhombic cells, as the return shape may be (3,) instead of (3,3)
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    """

    if lib is None or lib == "None":
        latticeVectors, symbols, types, coords = read_xyz_file_nolib(fileName)

    if lib == "ase":
        latticeVectors, symbols, types, coords = read_xyz_file_ase(fileName)

    if verb:
        print("latticeVectors", latticeVectors)
        print("symbols", symbols)
        print("coords", coords)

    return latticeVectors, symbols, types, coords

def read_xyz_file_nolib(fileName: str) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Reads in atomic structure files in XYZ format without the use of external libraries.

    ***Lattice information must be in the comment line of the xyz file, prefixed by 
    'Lattice:'***

    If no lattice information is provided, the returned latticeVectors will simply be
    determined by the max, min Cartesian positions with 5 Angstrom of padding.



    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be xyz only).

    Returns
    -------
    latticeVectors: ArrayLike (3, 3)
        The lattice vectors for the system of interest. Default behavior is for this to be 
        returned in the full format. Care should be taken if the ASE parser is used on 
        orthorhombic cells, as the return shape may be (3,) instead of (3,3)
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    """

    fileIn = open(fileName, "r")
    count = -1
    latticeVectors = np.zeros((3, 3))
    symbols = []  # Symbols for each atom type
    noBox = False
    typesIndex = -1
    for lines in fileIn:
        linesSplit = lines.split()

        # Adding an exception in case there is a blank second line
        if (len(linesSplit) == 0) and (count == 0):
            noBox = True
            count = count + 1

        if len(linesSplit) != 0:
            count = count + 1
            if count == 0:
                nats = int(linesSplit[0])
                coords = np.zeros((nats, 3))
                types = np.zeros((nats), dtype=int)
            if count == 1:
                latticeKey = linesSplit[0][0:8]
                if (latticeKey == "Lattice=") or (latticeKey == "Lattice"):
                    linesSplit = lines.split('"')
                    if linesSplit[0] == "Lattice":
                        boxInfoList = linesSplit[2].split()
                    else:
                        boxInfoList = linesSplit[1].split()
                    # Reading the lattice vectors
                    latticeVectors[0, 0] = float(boxInfoList[0])
                    latticeVectors[0, 1] = float(boxInfoList[1])
                    latticeVectors[0, 2] = float(boxInfoList[2])

                    latticeVectors[1, 0] = float(boxInfoList[3])
                    latticeVectors[1, 1] = float(boxInfoList[4])
                    latticeVectors[1, 2] = float(boxInfoList[5])

                    latticeVectors[2, 0] = float(boxInfoList[6])
                    latticeVectors[2, 1] = float(boxInfoList[7])
                    latticeVectors[2, 2] = float(boxInfoList[8])

                else:
                    noBox = True
            if (count >= 2) and (count <= nats + 2):
                # Reading the coordinates
                coords[count - 2, 0] = float(linesSplit[1])
                coords[count - 2, 1] = float(linesSplit[2])
                coords[count - 2, 2] = float(linesSplit[3])
                newSymbol = linesSplit[0]
                if not (newSymbol in symbols):
                    symbols.append(newSymbol)
                    typesIndex = typesIndex + 1
                    types[count - 2] = typesIndex
                else:
                    types[count - 2] = symbols.index(newSymbol)
    fileIn.close()
    if noBox:
        # If there is no box we create one by taking the coordinate
        # limits given by the positions of the atoms
        latticeVectors[0, 0] = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 5.0
        latticeVectors[1, 1] = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 5.0
        latticeVectors[2, 2] = np.max(coords[:, 2]) - np.min(coords[:, 2]) + 5.0

    return latticeVectors, symbols, types, coords

def read_xyz_file_ase(fileName: str) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Reads in atomic structure files in XYZ format without the ASE library.
    Currently this does *not* support lattice vector parsing. This will be added
    in the future (see TODO below).

    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be xyz only).

    Returns
    -------
    latticeVectors: ArrayLike (3, 3)
        The lattice vectors for the system of interest. Default behavior is for this to be 
        returned in the full format. Care should be taken if the ASE parser is used on 
        orthorhombic cells, as the return shape may be (3,) instead of (3,3)
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    """


    import ase
    system = ase.io.read(fileName)
    coords = system.get_positions()
    symbols = []  # Symbols for each atom type
    latticeVectors = np.zeros((3, 3))

    # TODO, ASE parses cell info via JSON-style lattice information in the comment line
    # so we can update this accordingly at some point.
    noBox = True  # Ace xyz reader does not read lattice vectors (system.cell = 0)
    symbolsForEachAtom = system.get_chemical_symbols()
    types = np.zeros(len(symbolsForEachAtom), dtype=int)
    typesIndex = -1
    count = -1
    for symb in symbolsForEachAtom:
        count = count + 1
        if not (symb in symbols):
            symbols.append(symb)
            typesIndex = typesIndex + 1
            types[count] = typesIndex
        else:
            types[count] = symbols.index(symb)
    return latticeVectors, symbols, types, coords

def write_xyz_coordinates(fileName: str,
                          coords,
                          types,
                          symbols) -> None:
    """
    Writes structural information to xyz files.
    TODO, add support for Lattice information.

    Parameters
    ----------
    fileName: str
        The name of the file to be written (should be xyz only).
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    symbols: ArrayLike
        The unique chemical elements in the structure.

    Returns
    -------
    None
    """

    nats = len(coords[:, 1])
    myFileOut = open(fileName, "w")
    print(nats, file=myFileOut)
    print("xyz format", file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print(symb, coords[i, 0], coords[i, 1], coords[i, 2], file=myFileOut)

    myFileOut.close()


## xyz trajectory parser
#  Reads in an xyz file trajectory.
#
#     Example xyz file format as follows:
#
# \verbatim
# 8
# frame 0
# Bl 0.0 0.0 0.0 0.0
# H 1.0 2.0 3.0 0.09
# He 2.0 4.0 6.0 0.18
# Li 3.0 6.0 9.0 0.27
# Be 4.0 8.0 12.0 0.36
# B 5.0 10.0 15.0 0.44999999999999996
# C 6.0 12.0 18.0 0.54
# N 7.0 14.0 21.0 0.63
# 8
# frame 1
# Bl 0.1 0.1 0.1 0.0
# H 1.1 2.1 3.1 0.08
# He 2.1 4.1 6.1 0.16
# Li 3.1 6.1 9.1 0.24
# Be 4.1 8.1 12.1 0.32
# B 5.1 10.1 15.1 0.4
# C 6.1 12.1 18.1 0.48
# N 7.1 14.1 21.1 0.56
# \endverbatim

#
# @param fileName File name of the xyz trajectorey. Example: "traj.xyz"
# @param lib If using a particular library. Default is "None"
# @param verb Verbosity. If set to True will output relevant information.
# @return elems Symbol for each atom type. Symbol for first atom type = symbols[0]
# @return coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @return values Index type for each atom in the system. Values (e.g. charges) for atoms
#
def read_xyz_trajectory(fileName: str,
                        lib: str = "None",
                        verb: bool = True) -> tuple[ArrayLike, ArrayLike, ArrayLike:
    """
    Reads in atomic structure files in XYZ format without the ASE library.
    TODO: This function could probably use an example for the format in the docs.

    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be xyz only).
    lib: str
        If using an external library to parse the structure (such as ASE, PMG, etc.)
    verb: bool
        Verbosity of the parsing procedure.

    Returns
    -------
    elems: ArrayLike
        Element info.
    coords: ArrayLike
        Positional information along the trajectory.
    values: ArrayLike
        Chemical information in the fourth column of the XYZ file.
    """

    with open(fileName) as f:
        ext = pathlib.Path(fileName).suffix
        lines = np.array(f.readlines())
        nats = int(lines[0])
        mask = np.ones(len(lines), dtype=bool)
        mask[np.arange(0, len(lines), nats + 2)] = False
        mask[np.arange(1, len(lines), nats + 2)] = False
        lines = lines[mask]
        lines = lines.tolist()
        xyzc = np.loadtxt(lines, usecols=range(1, 5)).astype(float)
        nframes = int(len(xyzc) / nats)
        elems = np.loadtxt(lines, usecols=0, dtype="U2")
        xyzc = np.reshape(xyzc, (nframes, nats, 4))
        coords = xyzc[:, :, 0:3]
        values = xyzc[:, :, 3]

        return elems[:nats], coords, values


## Read a pdb file
#  Reads in an pdb file with lattice informations.
#
#     Example pdb file format as follows:
#
# \verbatim
#    TITLE coords.pdb
#    CRYST1   11.598   17.395   17.591  90.00  90.00  90.00 P 1           1
#    MODEL                   1
#    ATOM      1  O   MOL     1       0.000   0.805  -0.230  0.00  0.00          O
#    ATOM      2  H   MOL     1      -0.101   0.855   2.111  0.00  0.00          H
#    ATOM      3  H   MOL     1       0.827  -0.475   3.907  0.00  0.00          H
#    TER
#    END
# \endverbatim
#
#
# @code{.unparsed}
# NumberOfAtomTypes = len(symbols)
# NumberOfAtoms = len(coordinates[:,0])
# @endcode
# @todo Add test!


def read_pdb_file(fileName: str,
                  lib: str = "None",
                  resInfo: bool = False,
                  verb: bool = False):
    """
    Reads in atomic structure files in PDB format, including lattice information.

    Parameters
    ----------
    fileName: str
        The name of the file to be parsed (should be pdb only).
    lib: str
        If using an external library to parse the structure (such as ASE, PMG, etc.)
        *Currently not used.
    resinfo: bool
        Defaults to false. Whether to change the return signature of the function to include
        information about the residues.
    verb: bool
        Verbosity of the parsing procedure.

    Returns
    -------
    latticeVectors: ArrayLike (3, 3)
        The lattice vectors for the system of interest. Default behavior is for this to be 
        returned in the full format. Care should be taken if the ASE parser is used on 
        orthorhombic cells, as the return shape may be (3,) instead of (3,3)
    symbols: ArrayLike
        The unique chemical elements in the structure.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.

    If resinfo is True:

    resIds: ArrayLike
    resNames: ArrayLike
        Residue ID and names in the PDB structure.
    """

    if verb:
        print("\nIn read_pdb_file...\n")
    if (lib == "None") or (lib is None):
        fileIn = open(fileName, "r")
        count = 0
        latticeVectors = np.zeros((3, 3))
        symbols = []  # Symbols for each atom type
        noBox = False
        boxFlag = False
        typesList = []
        coordsxList = []
        coordsyList = []
        coordszList = []
        resNames = []
        resIds = []
        typesIndex = -1
        for lines in fileIn:
            linesSplit = lines.split()
            if len(linesSplit) != 0:
                if linesSplit[0] == "CRYST1":
                    boxFlag = True
                    paramA = float(linesSplit[1])
                    paramB = float(linesSplit[2])
                    paramC = float(linesSplit[3])
                    paramAlpha = float(linesSplit[4])
                    paramBeta = float(linesSplit[5])
                    paramGamma = float(linesSplit[6])
                    latticeVectors = parameters_to_vectors(
                        paramA, paramB, paramC, paramAlpha, paramBeta, paramGamma, latticeVectors
                    )
                if (linesSplit[0] == "ATOM") or (linesSplit[0] == "HETATM"):
                    count = count + 1
                    if len(linesSplit) == 11:
                        newSymbol = linesSplit[10]
                    else:
                        newSymbol = linesSplit[2]
                        if(newSymbol[0:2] == "OW"):
                            newSymbol = "O"
                        elif(newSymbol[0:2] == "HW"):
                            newSymbol = "H"
                    if not (newSymbol in symbols):
                        symbols.append(newSymbol)
                        typesIndex = typesIndex + 1
                        typesList.append(typesIndex)
                    else:
                        typesList.append(symbols.index(newSymbol))
                    coordsxList.append(float(linesSplit[5]))
                    coordsyList.append(float(linesSplit[6]))
                    coordszList.append(float(linesSplit[7]))
                    if(resInfo):
                        resNames.append(linesSplit[3])
                        resIds.append(int(linesSplit[4]))
        fileIn.close()
        if not boxFlag:
            noBox = True

    coords = np.zeros((count, 3))
    for i in range(count):
        coords[i, 0] = coordsxList[i]
        coords[i, 1] = coordsyList[i]
        coords[i, 2] = coordszList[i]
    types = np.array(typesList, dtype=int)

    if noBox:
        latticeVectors[0, 0] = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 5.0
        latticeVectors[1, 1] = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 5.0
        latticeVectors[2, 2] = np.max(coords[:, 2]) - np.min(coords[:, 2]) + 5.0

    if(resInfo):
        return latticeVectors, symbols, types, coords, resIds, resNames
    else:
        return latticeVectors, symbols, types, coords


def write_pdb_coordinates(fileName, coords, types, symbols, molIds=np.zeros((0), dtype=int)):
    """
    Writes structural information to xyz files.
    TODO, add support for Lattice information.

    Parameters
    ----------
    fileName: str
        The name of the file to be written (should be xyz only).
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates for all atoms in the system.
    types: ArrayLike (Natoms, )
        The element type of each atom in the system.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    molIds: np.ndarray
        Identifies for the molecules in the structure.

    Returns
    -------
    None
    """

    nats = len(coords[:, 1])

    if len(molIds) == 0:
        molIds = np.zeros((nats), dtype=int)
        molIds[:] = 1

    myFileOut = open(fileName, "w")
    print("TITLE ", "PDB written by SEDACS", file=myFileOut)
    print(
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1", file=myFileOut
    )  # $$$ ??? Maybe adaptive vectors ???
    print("MODEL", file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print(
            "ATOM",
            "{:6d}".format(i + 1),
            " " + symb,
            "  MOL",
            "{:5d}".format(molIds[i]),
            "    ",
            "{:05.3f}".format(coords[i, 0]),
            "",
            "{:05.3f}".format(coords[i, 1]),
            "",
            "{:05.3f}".format(coords[i, 2]),
            " 1.00  0.00          ",
            symb,
            file=myFileOut,
        )
    print("TER", file=myFileOut)
    print("END", file=myFileOut)
    myFileOut.close()


def are_files_equivalent(file1: str,
                         file2: str) -> bool:
    """
    Checks if the two files contain the same information

    Parameters
    ----------
    file1: str
        Name of the first file.
    file2: str
        Name of the second file.

    Returns
    -------
    True or False if files are/are not equivalent.
    """
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            if line1.strip() != line2.strip():
                return False
    return True
