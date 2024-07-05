import numpy as np
from latte import *
from sedacs.system import *
from sedacs.periodic_table import *


## General LATTE dm API call
# This function will take coordinates and atom type and
# retreive the density matrix.
# @param box Lattice vectors. box[0,:] = first lattice vectors
# @param symbols List of elements symbols for each atom type
# @param types A list of types for every atom in the system
# @param coords Positions for every atom in the system. coords[0,2] z-coordinate of atom 0
# @return dm Density matrix
#
def get_latte_dm(box, symbols, types, coords):
    nats = len(types)  # Number of atoms
    atele = []
    for i in range(nats):
        atele.append(symbols[types[i]])  # Element for every atom in the system

    # Read bond integrals and atomic info
    (
        noelem,
        ele,
        ele1,
        ele2,
        basis,
        hes,
        hep,
        hed,
        hef,
        hcut,
        scut,
        noint,
        btype,
        tabh,
        tabs,
        sspl,
        hspl,
        lentabint,
        tabr,
        atocc,
        mass,
        hubbardu,
    ) = read_bondints("electrons.dat", "bondints.table", myVerb)

    # Get element pointer
    elempointer = get_elempointer(atele, ele, myVerb)

    # Get the dimension of Hamiltonian
    hdim = get_hdim(nats, elempointer, basis, myVerb)

    # et cutoff list
    cutoffList = get_cutoffList(nats, atele, ele1, ele2, hcut, scut, noint, myVerb)

    # Get integral map
    iglMap = build_integralMap(noint, btype, atele, ele1, ele2, nats, myVerb)

    # Construct the Hamiltonian and Overlap
    smat, ham = build_HS(
        nats,
        coords,
        box,
        elempointer,
        basis,
        hes,
        hep,
        hed,
        hef,
        hdim,
        cutoffList,
        iglMap,
        tabh,
        tabs,
        sspl,
        hspl,
        lentabint,
        tabr,
        hcut,
        scut,
        myVerb,
    )

    # Get the inverse overlap factors
    zmat = genX(smat, method="Diag", verbose=True)

    # Initializing a periodic table
    pt = PeriodicTable()

    # Getting number of electrons
    numel = 0
    for i in range(nats):
        atnum = pt.get_atomic_number(symbols[types[i]])
        numel = numel + pt.numel[atnum]

    # Getting the number of occupied states
    nocc = int(numel / 2.0)

    # Getting the density matrix
    T = 100.0
    dm, focc, C, e, mu0, entropy = fermi_exp(ham, zmat, nocc, T)

    return dm


if __name__ == "__main__":
    myVerb = True
    # Read coordinates from pdb file
    box, symbols, types, coords = read_xyz_file("coords.xyz", lib="Ase", verb=myVerb)

    # Call latte to get the density matrix
    dm = get_latte_dm(box, symbols, types, coords)
    print("Density matrix =")
    print(dm)
