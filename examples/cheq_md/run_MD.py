import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sedacs.ewald import ewald_energy
from sedacs.ewald import calculate_num_kvecs_ch_indep, init_PME_data, calculate_alpha_and_num_grids
from sedacs.cheq import QEQ_solver
from sedacs.cheq import (shadow_QEQ_solver, calculate_jacob_vec, 
                        linear_ChEQ_solver, calculate_fixed_parts_linear_ChEQ_solver)
from sedacs.neighbor_list import NeighborState, calculate_displacement
from sedacs.ewald import construct_kspace, calculate_PME_ewald
from ase import Atoms
from sedacs.ewald.util import mixed_precision_sum
from ase.io import read
from ase import units
from util import create_hippynn_model, calculate_hippynn_energy_forces
from collections import defaultdict
import time
from sedacs.integrator import SystemState, NVEState, create_NVE_integrator, create_NVT_integrator

from util import dublicate_ase_atoms, dublicate_state
torch.manual_seed(0)

@torch.compile(dynamic=True)
def calculate_dist_dips(pos_T, long_nbr_state):
    nbr_inds = long_nbr_state.nbr_inds
    disps = calculate_displacement(pos_T.to(torch.float64), nbr_inds,
                                long_nbr_state.lattice_lengths.to(torch.float64))
    dists = torch.linalg.norm(disps, dim=0)
    nbr_inds = torch.where((dists > cutoff) | (dists == 0.0), -1, nbr_inds)
    dists = torch.where(dists == 0, 1, dists)
    return disps.to(pos_T.dtype), dists.to(pos_T.dtype), nbr_inds

@torch.compile
def self_energy(charges, elect, hub):
    return mixed_precision_sum(elect * charges
                            + hub * torch.square(charges) / 2.0)

def ewald_vec_create(R, cell, nbr_inds, nbr_disps, nbr_dists, kvecs, I, alpha, cutoff):
    ewald_vec = lambda x: ewald_energy(R, cell, 
                 nbr_inds, nbr_disps, nbr_dists, 
                 x, kvecs, I, 
                 alpha,
                 cutoff, 
                 calculate_forces = 0,
                 calculate_dq = 1)[2]
    return ewald_vec

def PME_ewald_vec_create(R, cell, nbr_inds, nbr_disps, nbr_dists, PME_data, alpha, cutoff, dtype=torch.float64):
    def ewald_vec(x):
        return calculate_PME_ewald(R, x.to(dtype), cell, 
                                nbr_inds, nbr_disps, nbr_dists, 
                                alpha,
                                cutoff,
                                nbr_dists, calculate_forces = 0, calculate_dq = 1)[2].to(dtype)
    return ewald_vec

def calculate_energy_and_forces(system, NN_params, cheq_params, init=False, ewald_method="Ewald", update_shadow_history=True):
    global benchmark_flags
    global fixed_parts, time_data
    global all_iters, A_inv, q_tot
    model,short_nbr_state = NN_params
    all_start = time.perf_counter()
    start = time.perf_counter()
    pos_T = system.positions.T.contiguous()
    short_nbr_state.update(pos_T)
    if benchmark_flags['sync']:
        torch.cuda.synchronize()
    end = time.perf_counter()
    time_data["short_nbr"] += (end-start) * benchmark_flags['timing']
    start = time.perf_counter()
    h_en, h_f = calculate_hippynn_energy_forces(system.types, pos_T.to(torch.float64), 
                                                system.cell.to(torch.float64), short_nbr_state, model)
    h_f = h_f.to(pos_T.dtype)
    if benchmark_flags['sync']:
        torch.cuda.synchronize()
    end = time.perf_counter()
    time_data["hippynn"] += (end-start) * benchmark_flags['timing']

    if cheq_params != None:
        if ewald_method == "Ewald":
            (kvecs, I, alpha, hu_np, b_vec, 
            chi, hu, long_nbr_state, rtol, cutoff)  = cheq_params
        else:
            (PME_data, alpha, hu_np, b_vec, 
            chi, hu, long_nbr_state, rtol, cutoff) = cheq_params
        start = time.perf_counter()
        long_nbr_state.update(pos_T)
        if benchmark_flags['sync']:
            torch.cuda.synchronize()
        end = time.perf_counter()
        time_data["long_nbr_update"] += (end-start) * benchmark_flags['timing']

        start = time.perf_counter()
        disps, dists, nbr_inds = calculate_dist_dips(pos_T, long_nbr_state)
        disps = disps.to(pos_T.dtype)
        dists = dists.to(pos_T.dtype)
        if benchmark_flags['sync']:
            torch.cuda.synchronize()
        end = time.perf_counter()
        time_data["disp_dist"] += (end-start) * benchmark_flags['timing']
        start = time.perf_counter()
        if ewald_method == "Ewald":
            ewald_vec = ewald_vec_create(pos_T, system.cell, nbr_inds, disps, dists, kvecs, I, alpha, long_nbr_state.cutoff)
        else:
            ewald_vec = PME_ewald_vec_create(system.positions, system.cell, nbr_inds, disps, dists, PME_data, 
                                             alpha, long_nbr_state.cutoff, dtype=system.positions.dtype)
        if init == True or system.use_shadow == False:
            rtol = 1e-6 if system.use_shadow else rtol
            
            x, iter_cnt = QEQ_solver(pos_T, 
                                ewald_vec,
                                b_vec, 
                                hu_np,
                                init_charges=None,
                                A_inv=A_inv,
                                rtol=rtol, maxiter=1000)
            time_data["Num MatVec"] += iter_cnt * benchmark_flags['timing']
            all_iters.append(iter_cnt)
            q = x[:-1]
            system.charges = q
            if ewald_method == "Ewald":
                ewald_e, forces1, dq = ewald_energy(pos_T, system.cell, 
                                nbr_inds, disps, dists, 
                                q, kvecs, I, 
                                alpha,
                                long_nbr_state.cutoff, 
                                calculate_forces = 1,
                                calculate_dq = 0)
            else:
                ewald_e, forces1, dq =  calculate_PME_ewald(system.positions, q, system.cell, 
                                        nbr_inds, disps, dists, 
                                        alpha,
                                        cutoff,
                                        PME_data, calculate_forces = 1, calculate_dq = 0)
            if system.use_shadow and update_shadow_history:
                system.p_history[:] = q
                system.delta_p = system.delta_p * 0


            cou_en = (ewald_e) + self_energy(q, chi, hu)
            cou_f = forces1 
        
        elif init == False and system.use_shadow == True:
            sub_start = time.perf_counter()
            # Integrate and shift partial charges
            p = 2*system.p_history[0] - system.p_history[1] - system.kappa*system.delta_p + system.alpha * mixed_precision_sum(system.coefficients[:, None] * system.p_history, dim=0)
            #p = 2*p0 - p1 - kappa_md*delta_p + alpha_md*(c0*p0+c1*p1+c2*p2+c3*p3+c4*p4+c5*p5+c6*p6)
            if update_shadow_history:
                system.p_history = torch.roll(system.p_history, 1, dims=0)
                system.p_history[0,:] = p

            q_p, mu = linear_ChEQ_solver(ewald_vec(p), chi, hu, q_tot, fixed_parts=fixed_parts)
            low_rank_jacob_vec = lambda x: calculate_jacob_vec(q_p, ewald_vec(x), chi, hu, q_tot, x, fixed_parts=fixed_parts)
            if benchmark_flags['sync']:
                torch.cuda.synchronize()  
            sub_end = time.perf_counter()
            time_data["  Shadow setup"] += (sub_end-sub_start) * benchmark_flags['timing']
            sub_start = time.perf_counter()
            delta_p, iters = shadow_QEQ_solver(low_rank_jacob_vec, (q_p - p), precond=None, init=None, rtol=rtol) 
            all_iters.append(iters)
            if benchmark_flags['sync']:
                torch.cuda.synchronize()    
            sub_end = time.perf_counter()
            time_data["  Shadow Solver"] += (sub_end-sub_start) * benchmark_flags['timing']
            time_data["Num MatVec"] += iters * benchmark_flags['timing']
            #print("iters", iters)
            sub_start = time.perf_counter()
            # Updated p for 1st-level shadow potential
            p1 = p - delta_p # p1 = p + dp2dt2/w^2
            if ewald_method == "Ewald":
                ewald_e1, forces1, dq_p1 = ewald_energy(pos_T, system.cell, 
                                nbr_inds, disps, dists, 
                                p1, kvecs, I, 
                                alpha,
                                long_nbr_state.cutoff, 
                                calculate_forces = 1,
                                calculate_dq = 1)
            else:
                ewald_e1, forces1, dq_p1 =  calculate_PME_ewald(system.positions.to(torch.float64), p1.to(torch.float64), system.cell.to(torch.float64), 
                                        nbr_inds, disps.to(torch.float64), dists.to(torch.float64), 
                                        alpha,
                                        cutoff,
                                        PME_data, calculate_forces = 1, calculate_dq = 1)

            q_p, mu = linear_ChEQ_solver(dq_p1, chi, hu, q_tot, fixed_parts=fixed_parts)
            system.charges = q_p
            cou_f = forces1 * ((2*q_p/p1 - 1.0)).reshape(1,-1)
            cou_en = torch.sum(dq_p1 * (2*q_p - p1)) * 0.5 + self_energy(p1, chi, hu) - self_energy((q_p - p1), chi, hu)
            if update_shadow_history:
                system.delta_p = delta_p
            if benchmark_flags['sync']:
                torch.cuda.synchronize()   
            sub_end = time.perf_counter()
            time_data["  Shadow End"] += (sub_end-sub_start) * benchmark_flags['timing']

        total_f = h_f + cou_f.T
        total_en = h_en + cou_en
        if benchmark_flags['sync']:
            torch.cuda.synchronize()   
        end = time.perf_counter()
        time_data["ChEQ + Ewald"] += (end-start) * benchmark_flags['timing']

    else:
        total_f = h_f
        total_en = h_en  
    if benchmark_flags['sync']:
        torch.cuda.synchronize()  
    all_end = time.perf_counter()
    time_data["Energy + Force"] += (all_end-all_start) * benchmark_flags['timing']
    system.forces = total_f
    system.energy = float(total_en.item())


parser = argparse.ArgumentParser()

parser.add_argument("--model_folder", type=str, default="model_data", help="Model folder.")
parser.add_argument("--dt", type=float, default=0.4, help="Timestep size (fs).")
parser.add_argument("--N", type=int, default=6540, choices=[6540, 10008, 25050, 52320],
                    help="System size.")
parser.add_argument("--use_ewald", type=int, default=1, help="Flag for using long-range computation.")
parser.add_argument("--use_shadow", type=int, default=1, help="Flag for using shadow MD.")
parser.add_argument("--rtol", type=float, default=1e-1, help="Solver tolerance.")
parser.add_argument("--prefix", type=str, default="MD", help="Prefix for output files.")
parser.add_argument("--sim_length", type=int, default=200, help="Simulation length (fs).")
parser.add_argument("--is_nvt", type=int, default=0, help="Flag for NVT (1) or NVE (0).")
parser.add_argument("--restart", type=int, default=0, help="Flag to start from a restart file.")
parser.add_argument("--cutoff", type=float, default=10.0, help="Long-range cutoff distance (Å).")
parser.add_argument("--buffer", type=float, default=1.0, help="Neighbor list buffer length (Å).")
parser.add_argument("--ewald_method", choices=['Ewald', 'PME'], default="PME", help="Ewald method.")
parser.add_argument("--ewald_err", type=float, default=5e-4, help="Ewald force accuracy.")
parser.add_argument("--precision", choices=['FP32', 'FP64'], default="FP64", help="Precision.")
parser.add_argument("--use_jacobi_precond", type=int, default=0,
                    help="Flag for using Jacobi preconditioner in the QEQ solver.")
parser.add_argument("--dx", type=int, default=1,
                    help="System replication in x dimension (1 means no replication).")
parser.add_argument("--dy", type=int, default=1,
                    help="System replication in y dimension (1 means no replication).")
parser.add_argument("--dz", type=int, default=1,
                    help="System replication in z dimension (1 means no replication).")
parser.add_argument("--plot_results", type=int, default=1, help="Flag for plotting.")

args = parser.parse_args()

dx = args.dx
dy = args.dy
dz = args.dz

if dx != 1 or dy != 1 or dz != 1:
    N = args.N * dx * dy * dz
    print("Replicated N", N)
else:
    N = args.N

print(args)

prefix = args.prefix
t_err = args.ewald_err
PME_order = 6
plot_results = args.plot_results

prefix = f"{prefix}_{N}_{args.precision}_dt_{args.dt}_{args.ewald_method}_ew_err_{t_err}_is_shadow_{args.use_shadow}_rtol_{args.rtol}_is_nvt_{args.is_nvt}"
print("prefix:", prefix)
device = "cuda"
parent = args.model_folder
dtype = torch.float64
np_dtype = np.float64
N = args.N
geo_file = f"geo_data/water_{N}.pdb"
if args.precision == "FP32":
    np_dtype = np.float32
    dtype = torch.float32
cutoff = args.cutoff
buffer = args.buffer
rtol = args.rtol
use_shadow = args.use_shadow == 1
dt_mult = args.dt
dt = dt_mult * units.fs

total_sim_length = args.sim_length # fs



model, nn_cutoff = create_hippynn_model(parent, device, torch.float64)

atoms = read(geo_file)
if dx != 1 or dy != 1 or dz != 1:
    atoms = dublicate_ase_atoms(atoms, dx, dy, dz)
N = len(atoms)
lattice_vecs_np = np.array(atoms.cell.array, dtype=np_dtype) 
types_np = np.array(atoms.numbers)
coords_np = np.array(atoms.get_positions(wrap=True), dtype=np_dtype)
lattice_vecs = torch.from_numpy(lattice_vecs_np).to(device).to(dtype)
coords = torch.from_numpy(coords_np).to(device).to(dtype)
types = torch.from_numpy(types_np).to(device).long()
mass = torch.from_numpy(atoms.get_masses()).to(device).to(dtype) 
coords_T = coords.T.contiguous()

system = SystemState(coords, types, mass, lattice_vecs, use_shadow=use_shadow)

mu0 = 0.0
q_tot = 0.0

lattice_lengths = torch.norm(lattice_vecs, dim=1)
nbr_state = NeighborState(coords_T, lattice_vecs, None, cutoff, is_dense=True, buffer=buffer, use_triton=True)
nbr_state_NN = NeighborState(coords_T, lattice_vecs, None, nn_cutoff, is_dense=False, buffer=buffer, use_triton=True)

chi = [0.0] * 10
hu = [0.0] * 10
# trained QEQ parameters for water
chi[1], chi[8]  = 29.4295, 143.3731
hu[1], hu[8]  = 151.8699, 115.5745
chi = np.array(chi, dtype=np_dtype)
hu = np.array(hu, dtype=np_dtype)
my_chi_np = chi[atoms.numbers]
my_hu_np = hu[atoms.numbers]
my_hu = torch.from_numpy(my_hu_np).to(device).to(dtype)
my_chi = torch.from_numpy(my_chi_np).to(device).to(dtype)

b_vec = np.zeros(N+1)
b_vec[:N] = -my_chi_np


NN_params = model, nbr_state_NN
if args.ewald_method == "Ewald":
    kcounts, alpha = calculate_num_kvecs_ch_indep(t_err, cutoff, lattice_vecs_np)
    kxmax = 2 * np.pi / lattice_vecs_np[0, 0] * kcounts[0]
    kymax = 2 * np.pi / lattice_vecs_np[1, 1] * kcounts[1]
    kzmax = 2 * np.pi / lattice_vecs_np[2, 2] * kcounts[2]

    kvec_cutoff = max(kxmax, kymax, kzmax)
    I, kvecs = construct_kspace(lattice_vecs, kcounts, kvec_cutoff, alpha)
    cheq_params = (kvecs, I, alpha, my_hu_np, b_vec, 
    my_chi, my_hu, nbr_state, rtol, cutoff)
else:
    alpha, grid_dimensions = calculate_alpha_and_num_grids(lattice_vecs_np, cutoff, t_err)
    PME_data = init_PME_data(grid_dimensions, lattice_vecs.to(torch.float64), alpha, PME_order)
    cheq_params = (PME_data, alpha, my_hu_np, b_vec, 
        my_chi, my_hu, nbr_state, rtol, cutoff)
    
# need to incorperate 
fixed_parts = calculate_fixed_parts_linear_ChEQ_solver(my_chi, my_hu)

A_inv = None  
if args.use_jacobi_precond == 1: 
    A_inv = np.ones(N+1)
    A_inv[:-1] = 1/my_hu_np


if args.use_ewald == 0:
    cheq_params = None

energy_and_forces = lambda system, init: calculate_energy_and_forces(system, NN_params, cheq_params, init, args.ewald_method, update_shadow_history=True)
energy_and_forces_wo_update = lambda system, init: calculate_energy_and_forces(system, NN_params, cheq_params, init, args.ewald_method, update_shadow_history=False)

unit_ps = units.fs * 1000.0
if args.is_nvt:
    init_fn, step_fn = create_NVT_integrator(energy_and_forces, dt, friction= 10.0 / unit_ps, target_temp_in_K=300.0)
else:
    init_fn, step_fn = create_NVE_integrator(energy_and_forces, dt)

all_tot_energy = []
all_time = []
all_temp = []
all_pot = []
all_charges = []
all_positions = []
all_iters = []
time_data = defaultdict(float)
benchmark_flags = {"sync":1, "timing":1}
if args.is_nvt == 1:
    state = init_fn(system, temp_in_K=0.0)
else:
    state = init_fn(system, temp_in_K=0.0)
if args.restart:
    sys_state = SystemState.load(f"geo_data/NVT_300K_water_{args.N}.pt", device=device, dtype=dtype)
    if dx != 1 or dy != 1 or dz != 1:
        sys_state = dublicate_state(sys_state, dx, dy, dz)
    state.system = sys_state
    state.system.use_shadow = use_shadow
    print("restarting...")
energy_and_forces_wo_update(state.system, init=False) 
all_tot_energy.append(state.system.get_total_energy())
all_temp.append(state.system.get_temperature())
all_pot.append(state.system.get_potential_energy())
all_time.append(0.0)
all_charges.append(state.system.charges.detach().cpu().numpy())
all_positions.append(state.system.positions.detach().cpu().numpy())
timed_iter_count = 0
all_iter_count = 1
for i in tqdm(range(int(total_sim_length / dt_mult))):
    if i == 10:
        benchmark_flags['timing'] = 1
    timed_iter_count += benchmark_flags['timing']
    all_iter_count += 1
    state = step_fn(state)
    if plot_results == False:
        all_tot_energy.append(state.system.get_total_energy())
        all_temp.append(state.system.get_temperature())
        all_time.append((i+1) * dt_mult)
        all_pot.append(state.system.get_potential_energy())

    if i % 500 == 0 and plot_results == False:
        all_charges.append(state.system.charges.detach().cpu().numpy())
        all_positions.append(state.system.positions.detach().cpu().numpy())
    if i % 50 == 0 and plot_results == False:
        print(all_temp[-1], all_pot[-1], all_tot_energy[-1], all_iters[-1], flush=True)



if plot_results == False:
    final_total = all_tot_energy[-1]
    final_kinetic = state.system.get_kinetic_energy()
    final_potential = state.system.get_potential_energy()

    fig1, ax1 = plt.subplots( nrows=1, ncols=1 )
    fig2, ax2 = plt.subplots( nrows=1, ncols=1 )
    all_time = np.array(all_time)
    all_tot_energy = np.array(all_tot_energy)
    all_temp = np.array(all_temp)
    all_pot = np.array(all_pot)
    all_charges = np.array(all_charges)
    all_positions = np.array(all_positions)
    all_iters = np.array(all_iters)


    plt.figure()
    ax1.plot(all_time[:], all_pot[:], label=f"dt={dt_mult}")
    ax1.set_xlabel("Time (fs)")
    ax1.set_ylabel("Potential Energy (eV)")
    ax1.set_title(f"Avg. temp: {np.mean(all_temp):.2f}")
    ax1.legend()
    fig1.savefig(f"{prefix}_potential_energy.pdf")

    np.savetxt(f"{prefix}_potential_energy.txt", all_pot)
    np.savetxt(f"{prefix}_temp.txt", all_temp)
    np.savetxt(f"{prefix}_total_energy.txt", all_tot_energy)
    np.savetxt(f"{prefix}_all_iters.txt", all_iters)

    np.save(f"{prefix}_all_charges.npy", all_charges)
    np.save(f"{prefix}_all_positions.npy", all_positions)

    ax2.plot(all_time[:], all_temp[:], label=f"dt={dt_mult}")
    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("Temp")
    ax2.legend()
    fig2.savefig(f"{prefix}_temp.pdf")

    atoms = read(geo_file)
    if dx != 1 or dy != 1 or dz != 1:
        atoms = dublicate_ase_atoms(atoms, dx, dy, dz)

    atoms.set_positions(state.system.positions.detach().cpu().numpy())
    atoms.write(prefix + "_final.pdb")

    state.system.save(prefix + "_system.pt")


with open(f"{prefix}_timing.txt", 'w') as fptr:
    for k,v in time_data.items():
        if k != "Num MatVec":
            v = v / timed_iter_count 
            v = v * 1000.0
            print(f"{k}: {v:.2} ms", file = fptr)
    
    avg_iter = time_data["Num MatVec"] / (timed_iter_count)
    print(f"Avg. Num. MatVec: {avg_iter:.2f}", file = fptr)
    print(all_iter_count)
    print(nbr_state_NN.reneighbor_cnt)
    print(nbr_state.reneighbor_cnt)
    avg_short = nbr_state_NN.reneighbor_cnt / all_iter_count
    avg_long = nbr_state.reneighbor_cnt / all_iter_count
    print(f"Avg. Num. Short Reneighbor: {avg_short:.2f}", file = fptr)
    print(f"Avg. Num. Long Reneighbor: {avg_long:.2f}", file = fptr)

if plot_results == False:
    plt.figure()
    plt.plot(all_time[:], all_tot_energy[:], label=f"dt={dt_mult}")
    plt.xlabel("Time (fs)")
    plt.ylabel("Total Energy (eV)")
    plt.title(f"Avg. temp: {np.mean(all_temp):.2f}")
    plt.legend()
    plt.savefig(f"{prefix}_total_energy.pdf")
