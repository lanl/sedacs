"""ffields
Some functions to create force fields/potentials

So far: Harmonic ffields.
"""

import numpy as np

from sedacs.message import error_at, status_at

__all__ = ["get_classical_forces", "harmonic_potential"]


def get_classical_forces(coords, types, symbols, latticeVectors, nl, nlTrX, nlTrY, nlTrZ, fFieldName, verb=False):
    if verb:
        status_at("get_sdc_classical_force")
    if fFieldName == "HarmonicAll":
        return harmonic_potential(coords, nl, nlTrX, nlTrY, nlTrZ, verb=False)

    error_at("get_sdc_classical_force", message="No valid force filed name")


def harmonic_potential(coords, nl, nlTrX, nlTrY, nlTrZ, verb=False):
    if verb:
        status_at("harmonic_potential")

    # Get the forces on each atom
    nats = len(coords[:, 0])
    forces = np.zeros((nats, 3))
    dVect = np.zeros(3)
    forcesIJ = np.zeros(3)
    for i in range(nats):
        for j in range(1, nl[i, 0] + 1):
            jj = nl[i, j]
            jjX = coords[jj, 0] + nlTrX[j]
            jjY = coords[jj, 1] + nlTrY[j]
            jjZ = coords[jj, 2] + nlTrZ[j]

            dVect[0] = coords[i, 0] - jjX
            dVect[1] = coords[i, 1] - jjY
            dVect[2] = coords[i, 2] - jjZ

            distance2 = dVect[0] ** 2 + dVect[1] ** 2 + dVect[2] ** 2
            distance = np.sqrt(distance2)
            if distance != 0.0:
                dVectVersor = dVect / distance

                forcesIJ[:] = -2 * (distance - 1.0) * dVectVersor[:]
                forces[i] = forces[i] + forcesIJ
            # else:
            # print("WARNING")

    return forces
