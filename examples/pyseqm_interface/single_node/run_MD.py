# ruff: noqa: INP001, PLR0915, SIM115
"""Run molecular dynamics with the SEDACS PySEQM interface."""

import argparse
import time

import numpy as np
import torch
from ase import units

from sedacs.driver.graph_adaptive import get_adaptiveDM
from sedacs.driver.init import init
from sedacs.graph import get_initial_graph
from sedacs.integrator import SystemState, create_NVE_integrator, create_NVT_integrator
from sedacs.neighbor_list import calculate_dist_dips
from sedacs.periodic_table import PeriodicTable


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MD with SEDACS + PySEQM.")
    parser.add_argument("--input-file", type=str, default="input.in", help="SEDACS input file.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for SEDACS setup.")
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep in fs.")
    parser.add_argument("--sim-length", type=float, default=20.0, help="Simulation length in fs.")
    parser.add_argument("--ensemble", choices=["NVE", "NVT"], default="NVE", help="Thermodynamic ensemble.")
    parser.add_argument("--temp", type=float, default=300.0, help="Initial temperature in K.")
    parser.add_argument("--friction", type=float, default=10.0, help="Langevin friction in 1/ps for NVT.")
    parser.add_argument("--seed", type=int, default=137, help="Random seed.")
    parser.add_argument("--prefix", type=str, default="pyseqm_md", help="Output filename prefix.")
    parser.add_argument("--print-interval", type=int, default=10, help="Print every N MD steps.")
    parser.add_argument("--save-interval", type=int, default=10, help="Write trajectory frame every N MD steps.")
    parser.add_argument(
        "--graph-refresh-interval",
        type=int,
        default=20,
        help="Rebuild geometric graph every N force calls (0 disables geometric refresh).",
    )
    parser.add_argument(
        "--save-driver-files",
        action="store_true",
        help="Write per-step driver files (forces.npy and potential_energy.npy).",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    """Resolve device selection from CLI."""
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_graph_from_coords(sdc, sy, device):
    """Rebuild initial graph from current coordinates."""
    coords_t = torch.from_numpy(sy.coords).to(device).T.contiguous()
    sy.nbr_state.update(coords_t)
    sy.nl_disps, sy.nl_dists, sy.nl = calculate_dist_dips(coords_t, sy.nbr_state)
    sy.nl = sy.nl.cpu()
    sy.nl_disps = sy.nl_disps.cpu()
    sy.nl_dists = sy.nl_dists.cpu()

    if sdc.rcut < sdc.coulcut:
        nl = torch.where((sy.nl_dists > sdc.rcut) | (sy.nl_dists == 0.0), -1, sy.nl)
        nl = nl.sort(dim=1, descending=True)[0]
        max_nbrs = int(torch.max(torch.sum(nl != -1, dim=1)).item())
        nl = nl[:, :max_nbrs]
    elif sdc.rcut == sdc.coulcut:
        nl = sy.nl
    else:
        raise ValueError("Rcut cannot be larger than Coulcut.")

    num_neighbors = torch.sum(nl != -1, dim=1)
    nl = torch.cat((num_neighbors.unsqueeze(1), nl.sort(dim=1, descending=True)[0]), dim=1).cpu().numpy()
    box_lengths = np.diag(sy.latticeVectors)
    graph, _ = get_initial_graph(sy.coords, nl, sdc.rcut, sdc.maxDeg, box_lengths, graphweights=False, verb=False)
    return graph


def write_xyz_frame(handle, atom_symbols, coords):
    """Append one XYZ trajectory frame."""
    handle.write(f"{len(coords)}\n")
    handle.write("frame\n")
    for symb, xyz in zip(atom_symbols, coords, strict=False):
        handle.write(f"{symb:>2s} {xyz[0]:>20.10f} {xyz[1]:>20.10f} {xyz[2]:>20.10f}\n")


def main():
    """Run MD."""
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_args = argparse.Namespace(input_file=args.input_file, use_torch=False, device=resolve_device(args.device))
    sdc, eng, comm, rank, numranks, sy, hindex, graph_nl, _ = init(init_args)

    if rank == 0:
        print(f"Using {numranks} rank(s).")
        print(f"Device: {init_args.device}")
        if args.graph_refresh_interval > 0:
            print(f"Graph refresh interval: every {args.graph_refresh_interval} force calls")
        else:
            print("Graph refresh interval: disabled (reuse adaptive graph from previous step)")

    atom_types = np.asarray(sy.types, dtype=np.int64)
    atom_symbols = np.asarray(sy.symbols)[atom_types]
    pt = PeriodicTable()
    masses_by_type = np.array([pt.mass[pt.get_atomic_number(symb)] for symb in sy.symbols], dtype=np.float64)
    masses = masses_by_type[atom_types]

    device = init_args.device
    positions_t = torch.from_numpy(sy.coords).to(device=device, dtype=torch.float64)
    types_t = torch.from_numpy(atom_types).to(device=device, dtype=torch.long)
    masses_t = torch.from_numpy(masses).to(device=device, dtype=torch.float64)
    cell_t = torch.from_numpy(sy.latticeVectors).to(device=device, dtype=torch.float64)
    system = SystemState(positions_t, types_t, masses_t, cell_t, use_shadow=False)

    graph_state = {"graph": graph_nl, "force_calls": 0}

    def calculate_energy_and_forces(system_state, init=False):
        del init
        sy.coords = system_state.positions.detach().cpu().numpy()
        force_calls = graph_state["force_calls"]
        should_refresh_graph = (
            args.graph_refresh_interval > 0 and force_calls > 0 and (force_calls % args.graph_refresh_interval == 0)
        )
        if should_refresh_graph:
            graph_state["graph"] = build_graph_from_coords(sdc, sy, device)

        md_result, updated_graph = get_adaptiveDM(
            sdc,
            eng,
            comm,
            rank,
            numranks,
            sy,
            hindex,
            graph_state["graph"],
            save_output_files=args.save_driver_files,
            return_graph=True,
        )
        graph_state["graph"] = updated_graph
        graph_state["force_calls"] = force_calls + 1

        force_t = torch.from_numpy(np.asarray(md_result["forces"])).to(
            device=system_state.positions.device, dtype=system_state.positions.dtype
        )
        system_state.forces = force_t
        system_state.energy = float(md_result["potential_energy"])

    dt = args.dt * units.fs
    if args.ensemble == "NVT":
        friction = args.friction / (1000.0 * units.fs)
        init_fn, step_fn = create_NVT_integrator(
            calculate_energy_and_forces,
            dt,
            friction=friction,
            target_temp_in_K=args.temp,
        )
    else:
        init_fn, step_fn = create_NVE_integrator(calculate_energy_and_forces, dt)

    state = init_fn(system, temp_in_K=args.temp)
    nsteps = round(args.sim_length / args.dt)
    if nsteps < 1:
        raise ValueError("Simulation length is too short: increase --sim-length or decrease --dt.")

    all_time = []
    all_pot = []
    all_temp = []
    all_tot = []

    traj_handle = None
    energy_handle = None
    if rank == 0:
        traj_handle = open(f"{args.prefix}_traj.xyz", "w", encoding="utf-8")
        energy_handle = open(f"{args.prefix}_energy.dat", "w", encoding="utf-8")
        energy_handle.write("# time_fs potential_eV kinetic_eV total_eV temperature_K\n")

    def record(step_idx):
        time_fs = step_idx * args.dt
        potential = state.system.get_potential_energy()
        kinetic = state.system.get_kinetic_energy()
        total = potential + kinetic
        temp = state.system.get_temperature()
        all_time.append(time_fs)
        all_pot.append(potential)
        all_temp.append(temp)
        all_tot.append(total)

        if rank == 0:
            energy_handle.write(f"{time_fs:.6f} {potential:.12f} {kinetic:.12f} {total:.12f} {temp:.6f}\n")
            if step_idx % args.save_interval == 0:
                coords = state.system.positions.detach().cpu().numpy()
                write_xyz_frame(traj_handle, atom_symbols, coords)
            if step_idx % args.print_interval == 0:
                print(
                    f"step={step_idx:6d} t_fs={time_fs:10.4f} "
                    f"Epot={potential:14.8f} Etot={total:14.8f} T={temp:10.4f}"
                )

    tic = time.perf_counter()
    record(0)
    for step in range(1, nsteps + 1):
        state = step_fn(state)
        record(step)

    if rank == 0:
        wall = time.perf_counter() - tic
        if traj_handle is not None:
            traj_handle.close()
        if energy_handle is not None:
            energy_handle.close()

        np.savetxt(f"{args.prefix}_time_fs.txt", np.array(all_time))
        np.savetxt(f"{args.prefix}_potential_energy.txt", np.array(all_pot))
        np.savetxt(f"{args.prefix}_temperature.txt", np.array(all_temp))
        np.savetxt(f"{args.prefix}_total_energy.txt", np.array(all_tot))
        np.save(f"{args.prefix}_final_positions.npy", state.system.positions.detach().cpu().numpy())
        state.system.save(f"{args.prefix}_system.pt")

        print(f"MD finished in {wall:.2f} s")


if __name__ == "__main__":
    main()
