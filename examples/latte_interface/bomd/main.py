"""
main.py
====================================
This script runs a Born-Oppenheimer molecular dynamics (BOMD) simulation using the LATTE interface through SEDACS package.
User can enalbe shadow molecular dynamics simulation by setting the shadow_md flag to 1.

"""

import sys
import argparse
import math
import torch
import numpy as np
import gc
from copy import deepcopy

torch.set_default_dtype(torch.float64)

from sedacs.driver.init import init, available_device
from sedacs.graph_partition import get_coreHaloIndices, graph_partition
from sedacs.driver.graph_kernel_byparts import get_kernel_byParts, apply_kernel_byParts, rankN_update_byParts
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_adaptive_sp_energy_forces
from sedacs.file_io import read_latte_tbparams
from sedacs.periodic_table import PeriodicTable
from mpi4py import MPI
from sedacs.system import build_nlist
from sedacs.graph import get_initial_graph

####
# Global Constants
# Coversion factor from mass*velocity^2 to kinetic energy
MVV2KE = 166.0538782 / 1.602176487
# Conversion factor from kinetic energy to temperature
KE2T = 1.0 / 0.000086173435
# Conversion factor from force to velocity
F2V = 0.01602176487 / 1.660548782


def main(args):
    """! main program"""
    # Set random seed
    torch.manual_seed(137)
    np.random.seed(137)
    # Set numpy printing threshold
    np.set_printoptions(threshold=sys.maxsize)
    # Initialize sedacs parameters
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(
        args
    )
    if rank == 0:
        # Open files to write down information during MD simulation
        MD_xyz = open("MD.xyz", "w")
        Energy_dat = open("Energy.dat", "w")
    # Set verbosity
    sdc.verb = False
    # Chemical potential
    mu = args.mu
    # Number of timesteps
    MD_Iter = args.md_iter
    # Size of the timestep
    dt = args.dt
    # System temperature
    Temperature = args.temp
    # If we want to run shadow md
    shadow_md = args.shadow_md
    # If we want to use kernel
    use_kernel = args.use_kernel
    # Initialize periodic table
    pt = PeriodicTable()
    # Get the atomic symbols for each atom in the system
    element_type = np.array(sy.symbols)[sy.types]
    # Load the LATTE tight-binding parameters
    latte_tbparams = read_latte_tbparams(
        "../../../parameters/latte/TBparam/electrons.dat"
    )
    # Get the Hubbard U values for each atom in the system
    Hubbard_U = [latte_tbparams[symbol]["HubbardU"] for symbol in sy.symbols]
    Hubbard_U = np.array(Hubbard_U)[sy.types]
    sy.hubbard_u = Hubbard_U 
    # Get the atomic masses for each atom in the system
    Mnuc = [pt.mass[pt.get_atomic_number(symbol)] for symbol in sy.symbols]
    Mnuc = np.array(Mnuc)[sy.types]
    # Convert the hubbard u and atomic masses to a tensor
    Hubbard_U = torch.tensor(Hubbard_U)
    Mnuc = torch.tensor(Mnuc)
    # Read the box size as a tensor
    LBox = torch.tensor(
        [sy.latticeVectors[0][0], sy.latticeVectors[1][1], sy.latticeVectors[2][2]]
    )
    # Read the coordinates as tensors
    coords = torch.tensor(sy.coords)
    # Perform a graph-adaptive calculation of the charges with SCF cycles
    graphDH, sy.charges, mu, parts, partsCoreHalo, subSysOnRank = get_adaptiveSCFDM(
        sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu
    )
    #breakpoint()
    # Perform a single-point graph-adaptive calculation of the energy and forces
    graphDH, charges, EPOT, FTOT, mu, parts, partsCoreHalo, subSysOnRank = (
        get_adaptive_sp_energy_forces(
            sdc, eng, comm, rank, numranks, sy, parts, partsCoreHalo, hindex, graphDH, mu
        )
    )
    # Convert the charges to a tensor
    q = torch.tensor(sy.charges)
    # Read the number of atoms
    Nr_atoms = sy.nats
    # Convert the potential energy and forces to tensors
    EPOT = torch.tensor(EPOT)
    FTOT = torch.tensor(FTOT)

    # Initial BC for n, using the net Mulliken occupation per atom as extended electronic degrees of freedom
    n = q
    n_0 = q
    n_1 = q
    n_2 = q
    n_3 = q
    n_4 = q
    n_5 = q
    n_6 = q  # Set all "old" n-vectors to the same at t = t0
    # Coefficients for modified Verlet integration
    C0 = -14
    C1 = 36
    C2 = -27
    C3 = -2
    C4 = 12
    C5 = -6
    C6 = 1
    kappa = 1.84
    alpha = 0.0055

    # Initialize velocities
    V = torch.sqrt(Temperature / KE2T / MVV2KE / Mnuc).unsqueeze(1) * torch.randn_like(
        coords
    )
    # Compute and remvoe center of mass velocity
    COM_V = torch.sum(V.T * Mnuc, axis=1) / Mnuc.sum()
    V = V - COM_V

    # Record unwrapped coordsinates
    unwrap_coords = coords.clone().detach().double()

    renew = 0
    # MAIN MD LOOP {dR2(0)/dt2: V(0)->V(1/2); dn2(0)/dt2: n(0)->n(1); V(1/2): R(0)->R(1); dR2(1)/dt2: V(1/2)->V(1)}
    for MD_step in range(MD_Iter):
        # Calculate kinetic energy from particle velocities
        EKIN = torch.sum(0.5 * MVV2KE * torch.dot(Mnuc, torch.sum(V**2, dim=1)))
        # Calculate temperature from kinetic energy
        Temperature = (2.0 / 3.0) * KE2T * EKIN / Nr_atoms
        # Calculate the total energy from kinetic and potential energy
        ETOT = EKIN.item() + EPOT
        # Current time
        Time = (MD_step) * dt
        print(
            f"Time = {Time:<16.8f} Etotal = {ETOT:<16.8f} Temperature = {Temperature:<16.8f}"
        )

        # dR2(0)/dt2: V(0)->V(1/2)
        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1)  # - 0.2 * V
        if rank == 0:
            # Here we record the time, temperature, and charges. Note that the last term, q, would be constant if not solving exact charges during MD
            with torch.no_grad():
                Energy_dat.write(
                    f"{Time/1000:<16.8f} {ETOT:<16.16f} {Temperature:<16.8f} {EKIN.item():<16.16f} {EPOT.item():<16.16f} {torch.sum(q).item():<16.16f} {torch.sum(n_0).item():<16.16f} {mu:<16.16f}\n"
                )

            # Here we dump the MD trajectory
            if (Time % 50) == 49:
                # MD_xyz.write(f'## MD_step= {MD_step}\n')
                MD_xyz.write(f"{Nr_atoms}\n\n")
                for I in range(Nr_atoms):
                    MD_xyz.write(
                        f"{element_type[I].item()} {sy.coords[I, 0].item()} {sy.coords[I, 1].item()} {sy.coords[I, 2].item()} {sy.charges[I]}\n"
                    )
                MD_xyz.flush()
                Energy_dat.flush()

        if numranks > 1 and (Time % 10) == 9:
            coords_sum = torch.zeros_like(coords)
            comm.Allreduce(coords, coords_sum, op=MPI.SUM)
            coords = coords_sum / numranks

        # Caculate the residual between q[n] and n 
        Res = q - n_0 # Note that n_0 is n from the previous step
        # if use_kernel:
        if use_kernel:
            if MD_step == 0 or renew == 1:
                get_kernel_byParts(sdc, rank, numranks, parts, partsCoreHalo, sy, mu) 
                syk = deepcopy(sy)
                syk.subSy_list = deepcopy(sy.subSy_list)
                for i, subSy in enumerate(syk.subSy_list):
                    subSy.ker = deepcopy(sy.subSy_list[i].ker)
                partsk = deepcopy(parts)
                partsCoreHalok = deepcopy(partsCoreHalo)
                #breakpoint()
                #KK0 = torch.tensor(sy.subSy_list[0].ker)
                #KK0 = torch.tensor(collect_kernel_byParts(
                #    q, n_0, sdc, rank, numranks, comm, parts, partsCoreHalo, sy
                #))
            #dn2dt2 = -torch.matmul(KK0, Res)   
                dn2dt2 = 0
            #    ker = sy.subSy_list[0].ker
            #else:
            #    sy.subSy_list[0].ker = ker
                renew = 0
            if MD_step > 0:
                for i, subSy in enumerate(sy.subSy_list):
                    subSy.ker = deepcopy(syk.subSy_list[i].ker)
                dn2dt2 = -rankN_update_byParts(
                        q, n_0, 6, sdc, rank, numranks, comm, parts, partsCoreHalo, sy, mu=mu
                        )
                #dn2dt2 = -rankN_update_byParts(
                #        q, n_0, 6, sdc, eng, rank, numranks, comm, partsk, partsCoreHalok, syk, hindex, mu=mu
                #        )
                #dn2dt2 = -torch.tensor(apply_kernel_byParts(
                #     q, n_0, sdc, rank, numranks, comm, partsk, syk
                #))
            #breakpoint()
        else:
            dn2dt2 = 0.8 * Res
        # Propagating charge vector n for a better initial guess
        # Or Propagating charge vector for shadow MD
        n = (
            2 * n_0
            - n_1
            + kappa * dn2dt2 
            + alpha
            * (
                C0 * n_0
                + C1 * n_1
                + C2 * n_2
                + C3 * n_3
                + C4 * n_4
                + C5 * n_5
                + C6 * n_6
            )
        )
        #        breakpoint()
        n_6 = n_5
        n_5 = n_4
        n_4 = n_3
        n_3 = n_2
        n_2 = n_1
        n_1 = n_0
        n_0 = n
        sy.charges = n.numpy()

        # Update positions with full Verlet step for R
        disp = dt * V
        coords = coords + disp
        with torch.no_grad():
            unwrap_coords = unwrap_coords + disp
        # Reset coordinates within the periodic box
        coords = coords - LBox * torch.floor(coords / LBox)
        # Update sy.coords in the system object
        sy.coords = coords.numpy()
        # Update neighbor list
        #nl, nlTrX, nlTrY, nlTrZ = build_nlist(
        #   sy.coords,
        #   sy.latticeVectors,
        #   sdc.rcut,
        #   api="old",
        #   rank=rank,
        #   numranks=numranks,
        #   verb=False,
        #)
        #comm.Barrier()
        # Create initial graph based on distances
        #if rank == 0:
        #   graphNL = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg)
        #graphNL = comm.bcast(graphNL, root=0)
        if not shadow_md:
            # Perform a graph-adaptive calculation of the charges with SCF cycles
            graphDH, sy.charges, mu, parts, subSysOnRank = get_adaptiveSCFDM(
                sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu
            )
        #else:
        #    graphDH = graphNL

        if MD_step % 100 == 99:
            # Partition the graph
            #parts = graph_partition(
            #    sdc, eng, graphDH, sdc.partitionType, sdc.nparts, sy.coords, True
            #)

            renew = 1
            
        njumps = 1
        partsCoreHalo = []
        numCores = []
        for i in range(sdc.nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], graphDH, njumps)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)
            print("MD_step, core,halo size:", MD_step, i, "=", nc, nh)
        # Perform a single-point graph-adaptive calculation of the energy and forces
        graphDH, sy.charges, EPOT, FTOT, mu, parts, partsCoreHalo, subSysOnRank = (
            get_adaptive_sp_energy_forces(
                sdc,
                eng,
                comm,
                rank,
                numranks,
                sy,
                parts,
                partsCoreHalo,
                hindex,
                graphDH,
                mu,
                shadow_md=shadow_md,
            )
        )
        q = torch.tensor(sy.charges)
        # Constant shift in charges to maintain exact charge neutrality
        #q = q - (torch.sum(q)/len(q))
        # Convert the energy and forces to tensors
        EPOT = torch.tensor(EPOT)
        FTOT = torch.tensor(FTOT)

        # dR2(1)/dt2: V(1/2)->V(1)
        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1)
    
    if rank == 0:
        MD_xyz.close()
        Energy_dat.close()


if __name__ == "__main__":
    # Pass arguments from command line
    parser = argparse.ArgumentParser(
        description="Regular Born-Oppenheimer MD with sedacs"
    )
    parser.add_argument(
        "--use-torch", help="Use pytorch", required=False, action="store_true"
    )
    parser.add_argument(
        "--input-file",
        help="Specify input file",
        required=False,
        type=str,
        default="input.in",
    )
    parser.add_argument(
        "--md_iter", type=int, default=10000, help="Number of timesteps"
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep size (fs)")
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Initial system temperature (K)"
    )
    parser.add_argument(
        "--mu", type=float, default=0.0, help="Initial Chemical potential (eV)"
    )
    parser.add_argument(
        "--shadow_md",
        type=int,
        default=1,
        help="Set to 1/0 to enable/disable shadow MD",
    )
    parser.add_argument(
        "--use_kernel",
        type=int,
        default=1,
        help="Set to 1/0 to enable/disable kernel calculation",
    )
    args = parser.parse_args()
    if args.use_torch:
        args.device = available_device()

    print("Start running MD......")
    main(args)
