import numpy as np

from proxies.python.first_level import get_random_coordinates
from sedacs.force_fields import get_classical_forces
from sedacs.integrate import do_timestep
from sedacs.periodic_table import PeriodicTable

__all__ = ["do_MD"]


def do_MD(init_coords, types, symbols, latticeVectors, nl, nlTrX, nlTrY, nlTrZ, init_vels, dt, num_steps, gamma=0.1):
    nats = len(init_coords[:, 0])
    init_coords = get_random_coordinates(nats)
    for i in range(2):
        symbols[i] = "C"
    p = PeriodicTable()
    atomicnumbers = np.zeros(nats, dtype=int)
    for i in range(nats):
        atomicnumbers[i] = p.get_atomic_number(symbols[types[i]])
    masses = p.mass[atomicnumbers[:]]

    masses3d = np.zeros((nats, 3))
    masses3d[:, 0] = masses[:]
    masses3d[:, 1] = masses[:]
    masses3d[:, 2] = masses[:]

    masses3d[:, :] = 40.0

    print("doing MD...")
    coords = init_coords
    vels = init_vels
    gamma = 0.1

    myFile = open("traj.xyz", "w")
    for i in range(0, num_steps):
        forces = get_classical_forces(
            coords, types, symbols, latticeVectors, nl, nlTrX, nlTrY, nlTrZ, "HarmonicAll", verb=False
        )

        coords, vels = do_timestep(coords, vels, forces, masses3d, gamma, dt)
        if i % 100 == 0:
            print(nats, file=myFile)
            print("frame", i, file=myFile)
            for j in range(nats):
                print(symbols[types[j]], coords[j, 0], coords[j, 1], coords[j, 2], file=myFile)

            print(coords[0, :])
