import sys
import argparse
import math
import torch
import numpy as np
import gc
#import os
torch.set_default_dtype(torch.float64)

####
from sedacs.driver.init import init, get_args 
from sedacs.driver.graph_adaptive_scf import get_adaptiveSCFDM
from sedacs.driver.graph_adaptive_sp_energy_forces import get_sp_energy_forces
####
# Global Constants
## @brief number of timesteps
MD_Iter = 100000 
## @brief coversion factor from mass*velocity^2 to kinetic energy
MVV2KE = 166.0538782 / 1.602176487
## @brief size of the timestep
dt = 1.00
## @brief conversion factor from kinetic energy to temperature
KE2T = 1.0 / 0.000086173435
## @brief conversion factor from force to velocity
F2V = 0.01602176487 / 1.660548782

def main():
    """! main program"""
    # Set random seed
    torch.manual_seed(137)
    np.random.seed(137)
    # Open files to write down information during MD simulation
    MD_xyz = open("MD.xyz", "w")
    Energy_dat = open("Energy.dat", "w")
    # Pass arguments from comand line
    args = get_args()
    # Initialize sedacs
    np.set_printoptions(threshold=sys.maxsize)
    # Initialize sdc parameters
    sdc, eng, comm, rank, numranks, sy, hindex, graphNL, nl, nlTrX, nlTrY, nlTrZ = init(args)
    sdc.verb = False#True
    mu = 0.0
    
    element_type = torch.tensor(sy.types)
    symbols = np.array(sy.symbols)[sy.types]
    Hubbard_U = np.where(symbols == 'H', 12.054683, 0.0) + np.where(symbols == 'O', 11.876141, 0.0)
    Mnuc = np.where(symbols == 'H', 1.0, 0.0) + np.where(symbols == 'O', 16.0, 0.0)
    Hubbard_U = torch.tensor(Hubbard_U)
    Mnuc = torch.tensor(Mnuc)
    # Perform a graph-adaptive calculation of the density matrix
    graphDH,sy.charges,mu,parts,subSysOnRank = get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphNL, mu)
    graphDH,sy.charges,EPOT,FTOT,mu,parts,subSysOnRank = get_sp_energy_forces(sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu)
    LBox = torch.tensor([sy.latticeVectors[0][0], sy.latticeVectors[1][1], sy.latticeVectors[2][2]])
    coords = torch.tensor(sy.coords)
#   coords = sy.coords + 55.0
    q = torch.tensor(sy.charges)
    Nr_atoms = sy.nats
    EPOT = torch.tensor(EPOT)
    FTOT = torch.tensor(FTOT)
   
    # Initialize velocities
    V = 0 * coords.clone().detach()
    #V = torch.sqrt(300 / KE2T / MVV2KE / Mnuc).unsqueeze(1) * torch.randn_like(coords)  
#    COM_V = torch.sum(V.T * Mnuc, axis=1) / Mnuc.sum() 
#    V = V - COM_V
#    with torch.no_grad():
#        total_angular_momentum = torch.cross(coords, V * Mnuc.unsqueeze(1), axis=1).sum(axis=0)
#        COM = torch.sum(coords * Mnuc.unsqueeze(1), axis=0) / Mnuc.sum()
#        r = coords - COM
#        V = V - torch.cross(total_angular_momentum.unsqueeze(0) / Mnuc.sum(), r, axis=1) / (r * r).sum(axis=1).unsqueeze(1) 
    # Record unwrapped coordsinates
    unwrap_coords = coords.clone().detach().double()

    # MAIN MD LOOP {dR2(0)/dt2: V(0)->V(1/2); dn2(0)/dt2: n(0)->n(1); V(1/2): R(0)->R(1); dR2(1)/dt2: V(1/2)->V(1)}
    for MD_step in range(MD_Iter):
        # Calculate kinetic energy from particle velocities
        EKIN = torch.sum(0.5 * MVV2KE * torch.dot(Mnuc, torch.sum(V**2, dim=1)))
        # Calculate temperature from kinetic energy
        Temperature = (2.0 / 3.0) * KE2T * EKIN / Nr_atoms
        ETOT = EKIN.item() + EPOT
        # Current time
        Time = (MD_step) * dt
        print(
            f"Time = {Time:<16.8f} Etotal = {ETOT:<16.8f} Temperature = {Temperature:<16.8f}")

        # dR2(0)/dt2: V(0)->V(1/2)
        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1) #- 0.2 * V 
        # Here we record the time, temperature, and charges. Note that the last term, q, would be constant if not solving exact charges during MD
        with torch.no_grad():
             Energy_dat.write(
				f"{Time/1000:<16.8f} {ETOT:<16.16f} {Temperature:<16.8f} {EKIN.item():<16.16f} {EPOT.item():<16.16f}\n"
			)

        # Here we dump the MD trajectory
        if (Time % 50) == 49:
            #MD_xyz.write(f'## MD_step= {MD_step}\n')
            MD_xyz.write(f"{Nr_atoms}\n\n")
            for I in range(Nr_atoms):
                MD_xyz.write(f"{element_type[I].item()} {sy.coords[I, 0].item()} {sy.coords[I, 1].item()} {sy.coords[I, 2].item()} {n[I].item()} {qx[I].item()} {qqx[I].item()} {qqqx[I].item()} {q[I].item()}\n")
            MD_xyz.flush()
            Energy_dat.flush()
            gc.collect()
            torch.cuda.empty_cache()

        # Update positions with full Verlet step for R
        disp = dt * V
        #disp = FTOT / torch.max(FTOT) * 0.001
        coords = coords + disp  # + dt * V
        with torch.no_grad():
            unwrap_coords = unwrap_coords + disp
        # Reset sy.coordsinates within the periodic box
        coords = coords - LBox * torch.floor(coords / LBox)

        sy.coords = coords.numpy()
        # Perform a graph-adaptive calculation of the density matrix
        graphDH,sy.charges,mu,parts,subSysOnRank = get_adaptiveSCFDM(sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu)
        graphDH,sy.charges,EPOT,FTOT,mu,parts,subSysOnRank = get_sp_energy_forces(sdc, eng, comm, rank, numranks, sy, hindex, graphDH, mu)
        sy.coords = torch.tensor(sy.coords)
        EPOT = torch.tensor(EPOT)
        FTOT = torch.tensor(FTOT)

        V = V + 0.5 * dt * F2V * FTOT / Mnuc.unsqueeze(1) #- 0.01 * V
       
    MD_xyz.close()
    Energy_dat.close()


if __name__ == "__main__":
    print('start')
    main()
