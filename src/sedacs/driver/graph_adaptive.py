"""Graph adaptive self-consistenf charge solver."""
# ruff: noqa: D205, D401, E741, N802, N803, N806, N816, PLR0912, PLR0913, PLR0915, PLR2004, RUF059, SIM108, W291

import itertools
import os
import pickle
import socket
import time
import warnings

import numpy as np
import torch
from numba import njit
from seqm.seqm_functions.pack import pack

from sedacs.chemical_potential import get_mu
from sedacs.density_matrix_renorm import get_density_matrix_renorm
from sedacs.energy import get_eNuc, get_eTot
from sedacs.evals import get_eVals
from sedacs.file_io import write_pdb_coordinates, write_xyz_coordinates
from sedacs.graph import (
    add_mult_graphs,
    collect_graph_from_rho_PYSEQM,
    get_ch_graph,
    get_maskd,
    update_dm_contraction,
)
from sedacs.graph_partition import (
    get_coreHaloIndices,
    get_coreHaloIndicesPYSEQM,
    graph_partition,
)
from sedacs.hamiltonian import get_hamiltonian
from sedacs.interface.pyseqm import (
    get_coreHalo_ham_inds,
    get_diag_guess_pyseqm,
    get_molecule_pyseqm,
    pyseqmObjects,
)
from sedacs.system import System, extract_subsystem

warnings.simplefilter("ignore", FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"

try:
    from mpi4py import MPI

    is_mpi_available = True
except ModuleNotFoundError:
    is_mpi_available = False
__all__ = ["get_adaptiveSCFDM", "get_occ_singlePoint"]


def _write_subsystems(rank, part_index, sub_sy, sub_sy_core):
    """Write core+halo and core geometries for a partition."""
    sub_file = f"subSy{rank}_{part_index}"
    write_pdb_coordinates(f"{sub_file}.pdb", sub_sy.coords, sub_sy.types, sub_sy.symbols)
    write_xyz_coordinates(f"{sub_file}.xyz", sub_sy.coords, sub_sy.types, sub_sy.symbols)

    core_file = f"CoreSubSy{rank}_{part_index}"
    write_pdb_coordinates(f"{core_file}.pdb", sub_sy_core.coords, sub_sy_core.types, sub_sy_core.symbols)
    write_xyz_coordinates(f"{core_file}.xyz", sub_sy_core.coords, sub_sy_core.types, sub_sy_core.symbols)


def _concat_rank_chunks(chunks, uhf):
    """Concatenate per-partition arrays while handling empty-rank cases."""
    if not chunks:
        return np.empty((2, 0)) if uhf else np.array([])
    return np.concatenate(chunks, axis=1 if uhf else 0)


def _collect_parts_core_halo(parts, full_graph, sdc, eng, sy, rank, use_pyseqm=False):
    """Build core+halo partitions with shared logging."""
    parts_core_halo = []
    if rank == 0:
        print("Core and halos indices for every part:")

    for i, part in enumerate(parts):
        if use_pyseqm:
            core_halo, _ = get_coreHaloIndicesPYSEQM(eng, part, full_graph, sdc.numJumps, sdc, sy)
        else:
            core_halo, _, _ = get_coreHaloIndices(part, full_graph, sdc.numJumps, eng=eng)

        parts_core_halo.append(core_halo)
        if sdc.verb and (use_pyseqm or rank == 0):
            print("coreHalo for part", i, "=", core_halo)
        if rank == 0:
            print(f"  N atoms in core/coreHalo {i:>6d} : {len(part):>6d} {len(core_halo):>6d}")

    return parts_core_halo


def _update_partition_graph_data(sdc, sy, full_graph, parts, parts_core_halo, P_contr, graph_for_pairs, device, rank):
    """Refresh CH graph metadata and DM contraction."""
    tic = time.perf_counter()
    new_graph_for_pairs = get_ch_graph(sdc, sy, full_graph, parts, parts_core_halo)
    if rank == 0:
        print(f"Time to updt DM and mod graphs {time.perf_counter() - tic:>7.2f} (s)")

    tic = time.perf_counter()
    update_dm_contraction(sdc, sy, P_contr, graph_for_pairs, new_graph_for_pairs, device)
    graph_for_pairs = new_graph_for_pairs
    if rank == 0:
        print(f"Time to updt DM and mod graphs {time.perf_counter() - tic:>7.2f} (s)")

    tic = time.perf_counter()
    graph_maskd = get_maskd(sdc, sy, graph_for_pairs)
    if rank == 0:
        print(f"Time to updt DM and mod graphs {time.perf_counter() - tic:>7.2f} (s)")

    return graph_for_pairs, graph_maskd


def _configure_driver_dtypes(sdc, eng):
    """Set shared numpy/torch dtypes based on torch default precision."""
    if torch.get_default_dtype() == torch.float32:
        eng.torch_dt = torch.float32
        eng.torch_int_dt = torch.int32
        eng.np_dt = np.float32
        eng.np_int_dt = np.int32
    else:
        eng.torch_dt = torch.float64
        eng.torch_int_dt = torch.int64
        eng.np_dt = np.float64
        eng.np_int_dt = np.int64

    sdc.torch_dt = eng.torch_dt
    sdc.torch_int_dt = eng.torch_int_dt
    sdc.np_dt = eng.np_dt
    sdc.np_int_dt = eng.np_int_dt


def _get_uhf_nocc_targets(sy):
    """Return alpha/beta target occupations for UHF mu updates."""
    try:
        nocc = sy.nocc
    except AttributeError:
        nocc = None
    if isinstance(nocc, (list, tuple, np.ndarray)) and len(nocc) == 2:
        return float(nocc[0]), float(nocc[1])

    try:
        total = float(sy.numel)
    except AttributeError:
        total = 0.0
    return total / 2.0, total / 2.0


def _use_backprop_forces(sdc, eng):
    """Return True when force evaluation must use autograd/backprop."""
    if not sdc.doForces:
        return False
    return (not sdc.analyticForces) or eng.use_pyseqm_lt


def _init_density_diis_state(sdc):
    """Initialize DM-DIIS state/config from optional driver attributes."""
    damp = float(getattr(sdc, "diisDamp", 1.0))
    damp = min(max(damp, 0.0), 1.0)
    return {
        "enabled": bool(getattr(sdc, "diis", True)),
        "history": max(2, int(getattr(sdc, "diisHistory", 6))),
        "start": max(1, int(getattr(sdc, "diisStart", 2))),
        "reg": float(getattr(sdc, "diisReg", 1.0e-12)),
        "max_coeff": float(getattr(sdc, "diisMaxCoeff", 20.0)),
        "damp": damp,
        "entries": [],
    }


def _reset_density_diis_history(diis_state):
    """Clear DM-DIIS history buffers."""
    if diis_state is None:
        return
    diis_state["entries"].clear()


def _build_dm_sparse_entry(P_contr, P_contr_new, graph_for_pairs, uhf, alpha):
    """Build sparse DM vectors in graph-invariant global keys.

    Keys are directed pairs `(spin, j, i)` encoded into int64.
    Stores:
      y = x + alpha * r  (linearly mixed map at this iteration)
      r = F(x) - x       (Pulay residual)
    """
    if torch.is_tensor(graph_for_pairs):
        graph_np = graph_for_pairs.detach().cpu().numpy()
    else:
        graph_np = np.asarray(graph_for_pairs)

    graph_np = np.ascontiguousarray(graph_np, dtype=np.int64)
    nats = int(graph_np.shape[0])
    total_pairs = int(np.sum(graph_np[:, 0], dtype=np.int64))
    spin_mult = 2 if uhf else 1
    total_entries = spin_mult * total_pairs

    keys = np.empty(total_entries, dtype=np.int64)
    y_vals = np.empty((total_entries, 16), dtype=np.float64)
    r_vals = np.empty((total_entries, 16), dtype=np.float64)
    targets = np.empty((total_entries, 3 if uhf else 2), dtype=np.int64)
    p_in_np = np.ascontiguousarray(P_contr.detach().numpy())
    p_out_np = np.ascontiguousarray(P_contr_new.detach().numpy())

    if uhf:
        cursor = _build_dm_sparse_entry_uhf_numba(
            graph_np,
            p_in_np,
            p_out_np,
            float(alpha),
            keys,
            y_vals,
            r_vals,
            targets,
        )
    else:
        cursor = _build_dm_sparse_entry_rhf_numba(
            graph_np,
            p_in_np,
            p_out_np,
            float(alpha),
            keys,
            y_vals,
            r_vals,
            targets,
        )

    keys = np.ascontiguousarray(keys[:cursor], dtype=np.int64)
    y_vals = np.ascontiguousarray(y_vals[:cursor], dtype=np.float64)
    r_vals = np.ascontiguousarray(r_vals[:cursor], dtype=np.float64)
    targets = np.ascontiguousarray(targets[:cursor], dtype=np.int64)

    if keys.size == 0:
        return {"keys": keys, "y": y_vals, "r": r_vals, "targets": targets, "dup_keys": False}

    order = np.argsort(keys, kind="mergesort")
    keys = np.ascontiguousarray(keys[order], dtype=np.int64)
    y_vals = np.ascontiguousarray(y_vals[order], dtype=np.float64)
    r_vals = np.ascontiguousarray(r_vals[order], dtype=np.float64)
    targets = np.ascontiguousarray(targets[order], dtype=np.int64)
    dup_keys = bool(_has_adjacent_duplicates_numba(keys))
    return {"keys": keys, "y": y_vals, "r": r_vals, "targets": targets, "dup_keys": dup_keys}


def _sparse_dot(keys_a, vals_a, keys_b, vals_b):
    """Dot product over sparse vectors with sorted unique global pair keys."""
    return float(_sparse_dot_numba(keys_a, vals_a, keys_b, vals_b))


@njit(cache=True)
def _sparse_dot_numba(keys_a, vals_a, keys_b, vals_b):
    """Numba-accelerated sparse key merge-dot."""
    na = keys_a.shape[0]
    nb = keys_b.shape[0]
    if na == 0 or nb == 0:
        return 0.0

    i = 0
    j = 0
    total = 0.0

    while i < na and j < nb:
        ka = keys_a[i]
        kb = keys_b[j]
        if ka == kb:
            s = 0.0
            for k in range(16):
                s += vals_a[i, k] * vals_b[j, k]
            total += s
            i += 1
            j += 1
        elif ka < kb:
            i += 1
        else:
            j += 1

    return total


def _accumulate_projected(accum, coeff, src_keys, src_vals, tgt_keys):
    """accum += coeff * src_vals projected onto tgt_keys (missing -> zero)."""
    if src_keys.size == 0 or tgt_keys.size == 0 or coeff == 0.0:
        return
    _accumulate_projected_numba(accum, float(coeff), src_keys, src_vals, tgt_keys)


@njit(cache=True)
def _build_dm_sparse_entry_rhf_numba(graph_np, p_in_np, p_out_np, alpha, keys, y_vals, r_vals, targets):
    """Build sparse RHF DM entry buffers."""
    nats = graph_np.shape[0]
    cursor = 0
    for j in range(nats):
        n_j = graph_np[j, 0]
        for slot in range(n_j):
            neigh = graph_np[j, 1 + slot]
            keys[cursor] = j * nats + neigh
            targets[cursor, 0] = slot
            targets[cursor, 1] = j

            k = 0
            for a in range(4):
                for b in range(4):
                    pin = p_in_np[slot, j, a, b]
                    rout = p_out_np[slot, j, a, b] - pin
                    y_vals[cursor, k] = pin + alpha * rout
                    r_vals[cursor, k] = rout
                    k += 1
            cursor += 1
    return cursor


@njit(cache=True)
def _build_dm_sparse_entry_uhf_numba(graph_np, p_in_np, p_out_np, alpha, keys, y_vals, r_vals, targets):
    """Build sparse UHF DM entry buffers."""
    nats = graph_np.shape[0]
    pair_base = nats * nats
    cursor = 0
    for j in range(nats):
        n_j = graph_np[j, 0]
        for slot in range(n_j):
            neigh = graph_np[j, 1 + slot]
            for spin in range(2):
                keys[cursor] = spin * pair_base + j * nats + neigh
                targets[cursor, 0] = spin
                targets[cursor, 1] = slot
                targets[cursor, 2] = j

                k = 0
                for a in range(4):
                    for b in range(4):
                        pin = p_in_np[spin, slot, j, a, b]
                        rout = p_out_np[spin, slot, j, a, b] - pin
                        y_vals[cursor, k] = pin + alpha * rout
                        r_vals[cursor, k] = rout
                        k += 1
                cursor += 1
    return cursor


@njit(cache=True)
def _has_adjacent_duplicates_numba(keys):
    """Check if a sorted key vector has duplicates."""
    n = keys.shape[0]
    for i in range(1, n):
        if keys[i] == keys[i - 1]:
            return True
    return False


@njit(cache=True)
def _accumulate_projected_numba(accum, coeff, src_keys, src_vals, tgt_keys):
    """accum += coeff * src_vals projected onto sorted tgt_keys."""
    i = 0
    j = 0
    ns = src_keys.shape[0]
    nt = tgt_keys.shape[0]

    while i < ns and j < nt:
        ka = src_keys[i]
        kb = tgt_keys[j]
        if ka == kb:
            for k in range(16):
                accum[j, k] += coeff * src_vals[i, k]
            i += 1
            j += 1
        elif ka < kb:
            i += 1
        else:
            j += 1


@njit(cache=True)
def _sorted_subset_positions_numba(sorted_superset, sorted_subset):
    """Return positions of sorted_subset items inside sorted_superset."""
    out = np.empty(sorted_subset.shape[0], dtype=np.int64)
    i = 0
    j = 0
    k = 0
    ns = sorted_superset.shape[0]
    nt = sorted_subset.shape[0]

    while i < ns and j < nt:
        a = sorted_superset[i]
        b = sorted_subset[j]
        if a == b:
            out[k] = i
            k += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1

    return out[:k]


def _core_positions_in_core_halo(core_halo, core, out_dtype):
    """Get core atom positions inside core+halo list."""
    core_halo = np.asarray(core_halo, dtype=np.int64)
    core = np.asarray(core, dtype=np.int64)
    if core.size == 0:
        return np.empty(0, dtype=out_dtype)

    core_halo_sorted = (core_halo.size < 2) or np.all(core_halo[1:] >= core_halo[:-1])
    core_sorted = (core.size < 2) or np.all(core[1:] >= core[:-1])
    if core_halo_sorted and core_sorted:
        pos = _sorted_subset_positions_numba(core_halo, core)
        if pos.shape[0] == core.shape[0]:
            return pos.astype(out_dtype, copy=False)

    # Defensive fallback if ordering/subset assumptions do not hold.
    return np.nonzero(np.isin(core_halo, core))[0].astype(out_dtype, copy=False)


@njit(cache=True)
def _update_p_contr_new_rhf_numba(p_in, p_out, tmp2, part_atoms, max_len):
    """Write RHF SCF map blocks and accumulate max/sum DM differences."""
    max_dif = 0.0
    sum_dif = 0.0
    n_core = part_atoms.shape[0]
    for c in range(n_core):
        atom = part_atoms[c]
        for row in range(max_len):
            for a in range(4):
                for b in range(4):
                    new_val = tmp2[row, c, a, b]
                    old_val = p_in[row, atom, a, b]
                    diff = old_val - new_val
                    if diff < 0.0:
                        diff = -diff
                    if diff > max_dif:
                        max_dif = diff
                    sum_dif += diff
                    p_out[row, atom, a, b] = new_val
    return max_dif, sum_dif


@njit(cache=True)
def _update_p_contr_new_uhf_numba(p_in, p_out, tmp2, part_atoms, max_len):
    """Write UHF SCF map blocks and accumulate max/sum DM differences."""
    max_dif = 0.0
    sum_dif = 0.0
    n_core = part_atoms.shape[0]
    for spin in range(2):
        for c in range(n_core):
            atom = part_atoms[c]
            for row in range(max_len):
                for a in range(4):
                    for b in range(4):
                        new_val = tmp2[spin, row, c, a, b]
                        old_val = p_in[spin, row, atom, a, b]
                        diff = old_val - new_val
                        if diff < 0.0:
                            diff = -diff
                        if diff > max_dif:
                            max_dif = diff
                        sum_dif += diff
                        p_out[spin, row, atom, a, b] = new_val
    return max_dif, sum_dif


def _scatter_sparse_dm_values(P_contr, values, targets, uhf):
    """Scatter sparse block values into the current contracted-DM slot layout."""
    if values.size == 0:
        return
    blocks = values.reshape(-1, 4, 4)
    p_np = P_contr.detach().numpy()
    if uhf:
        p_np[targets[:, 0], targets[:, 1], targets[:, 2], :, :] = blocks
    else:
        p_np[targets[:, 0], targets[:, 1], :, :] = blocks


def _mix_density_with_diis(sdc, gsc, P_contr, P_contr_new, graph_for_pairs, diis_state):
    """Mix contracted DM using key-based residual-DIIS with linear fallback.

    Parameters
    ----------
    P_contr:
        Current SCF input density (updated in place).
    P_contr_new:
        SCF output map for the current iteration.
    """
    alpha = float(sdc.alpha)
    current = _build_dm_sparse_entry(P_contr, P_contr_new, graph_for_pairs, sdc.UHF, alpha)
    y_linear = current["y"]

    def _fallback(mode):
        _scatter_sparse_dm_values(P_contr, y_linear, current["targets"], sdc.UHF)
        return mode, None

    if current["dup_keys"]:
        # Defensive fallback for malformed graph rows with repeated neighbors.
        _reset_density_diis_history(diis_state)
        return _fallback("linear_dupkey")

    if (diis_state is None) or (not diis_state["enabled"]):
        return _fallback("linear")

    entries = diis_state["entries"]
    entries.append({"keys": current["keys"], "y": current["y"], "r": current["r"]})
    if len(entries) > diis_state["history"]:
        entries.pop(0)

    m = len(entries)
    if (gsc < diis_state["start"]) or (m < 2):
        return _fallback("linear")

    B = np.full((m + 1, m + 1), -1.0, dtype=np.float64)
    B[m, m] = 0.0
    rhs = np.zeros(m + 1, dtype=np.float64)
    rhs[m] = -1.0

    for i in range(m):
        for j in range(i, m):
            val = _sparse_dot(entries[i]["keys"], entries[i]["r"], entries[j]["keys"], entries[j]["r"])
            B[i, j] = val
            B[j, i] = val

    reg = diis_state["reg"]
    if reg > 0.0:
        diag = np.diag(B[:m, :m])
        scale = float(np.mean(diag)) if diag.size else 1.0
        B[np.arange(m), np.arange(m)] += reg * max(scale, 1.0)

    coeffs = None
    try:
        coeffs = np.linalg.solve(B, rhs)[:m]
    except np.linalg.LinAlgError:
        coeffs = None

    if coeffs is None or (not np.all(np.isfinite(coeffs))):
        return _fallback("linear_fallback")

    max_coeff = diis_state["max_coeff"]
    if (max_coeff > 0.0) and (np.max(np.abs(coeffs)) > max_coeff):
        return _fallback("linear_clipped")

    y_diis = np.zeros_like(current["y"])
    for c, entry in zip(coeffs, entries):
        c = float(c)
        _accumulate_projected(y_diis, c, entry["keys"], entry["y"], current["keys"])

    damp = float(diis_state["damp"])
    mixed_vals = (1.0 - damp) * y_linear + damp * y_diis
    _scatter_sparse_dm_values(P_contr, mixed_vals, current["targets"], sdc.UHF)

    return "diis", coeffs


def _collect_single_point_data(
    sdc,
    eng,
    partsPerGPU,
    partsPerNode,
    node_id,
    node_rank,
    rank,
    gpu_comm,
    parts,
    partsCoreHalo,
    sy,
    hindex,
    mu0,
    molecule_whole,
    P_contr,
    graph_for_pairs,
    graph_maskd,
):
    """Common implementation for the duplicated single-point routines."""
    del hindex
    partIndex1 = node_rank * partsPerGPU + node_id * partsPerNode
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id * partsPerNode

    eValOnRank_chunks = []
    dValOnRank_chunks = []
    eValOnRank_list = []
    Q_list = []
    Nocc_list = []
    core_indices_in_sub_expanded_list = []
    NH_Nh_Hs_list = []
    EELEC = 0.0

    if sdc.scfDevice == "cuda":
        device = f"cuda:{node_rank}"
        P_contr_device = P_contr.to(device)
        molecule_whole_device = get_molecule_pyseqm(
            sdc,
            sy.coords,
            sy.symbols,
            sy.types,
            do_large_tensors=sdc.use_pyseqm_lt,
            device=device,
            requires_grad=False,
        )[0]
    else:
        device = "cpu"
        P_contr_device = P_contr
        molecule_whole_device = molecule_whole
    if torch.is_tensor(graph_for_pairs):
        graph_for_pairs_device = graph_for_pairs.to(device=device, dtype=sdc.torch_int_dt)
    else:
        graph_for_pairs_device = torch.from_numpy(graph_for_pairs).to(device=device, dtype=sdc.torch_int_dt)
    graph_maskd_device = torch.as_tensor(graph_maskd, device=device, dtype=torch.long)

    for partIndex in range(partIndex1, partIndex2):
        ticHam = time.perf_counter()

        subSy = System(len(partsCoreHalo[partIndex]))
        subSy.symbols = sy.symbols
        subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])

        subSyCore = System(len(parts[partIndex]))
        subSyCore.symbols = sy.symbols
        subSyCore.coords, subSyCore.types = extract_subsystem(sy.coords, sy.types, sy.symbols, parts[partIndex])

        if sdc.writeGeom:
            _write_subsystems(rank, partIndex, subSy, subSyCore)

        core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub = get_coreHalo_ham_inds(
            parts[partIndex],
            partsCoreHalo[partIndex],
            sdc,
            sy,
            subSy,
            device=device,
        )

        ham_timing = {}
        ham, eElec = get_hamiltonian(
            sdc,
            eng,
            subSy.coords,
            subSy.types,
            subSy.symbols,
            parts[partIndex],
            partsCoreHalo[partIndex],
            molecule_whole_device,
            P_contr_device,
            graph_for_pairs_device,
            graph_maskd_device,
            core_indices_in_sub_expanded,
            ham_timing,
            verbose=False,
        )

        EELEC += eElec

        tic = time.perf_counter()
        occ = subSy.nats // 2  # Kept for compatibility with callers.
        coreSize = len(parts[partIndex])
        eVals, dVals, Q, NH_Nh_Hs = get_eVals(
            eng,
            sdc,
            sy,
            ham,
            subSy.coords,
            subSy.symbols,
            subSy.types,
            sdc.Tel,
            mu0,
            core_indices_in_sub,
            core_indices_in_sub_expanded,
            hindex_sub,
            coreSize,
            subSy,
            subSyCore,
            parts[partIndex],
            partsCoreHalo[partIndex],
            verbose=False,
        )
        del ham

        if torch.is_tensor(dVals):
            dVals_np = dVals.detach().cpu().numpy()
        else:
            dVals_np = np.asarray(dVals)

        eValOnRank_chunks.append(eVals.detach().cpu().numpy())
        dValOnRank_chunks.append(dVals_np)
        eValOnRank_list.append(eVals.cpu())

        Q_list.append(Q.cpu())
        core_indices_in_sub_expanded_list.append(core_indices_in_sub_expanded)
        NH_Nh_Hs_list.append(NH_Nh_Hs)
        Nocc_list.append(occ)

        ham_timing["eVals/dVals"] = time.perf_counter() - tic
        ham_timing["TOT"] = time.perf_counter() - ticHam
        formatted_string = " | ".join(f"{key} {value:8.3f}" for key, value in ham_timing.items())
        print("Rank", rank, "part", partIndex, ":", formatted_string)

    eValOnRank = _concat_rank_chunks(eValOnRank_chunks, sdc.UHF)
    dValOnRank = _concat_rank_chunks(dValOnRank_chunks, sdc.UHF)

    tic = time.perf_counter()
    full_dVals = None
    full_eVals = None
    eValOnRank_size = np.array(eValOnRank.shape[-1], dtype=int)
    eValOnRank_SIZES = None
    recvcounts = None
    if rank == 0:
        eValOnRank_SIZES = np.empty(gpu_comm.Get_size(), dtype=int)

    gpu_comm.Gather(eValOnRank_size, eValOnRank_SIZES, root=0)

    if rank == 0:
        if sdc.UHF:
            full_dVals = np.empty((2, np.sum(eValOnRank_SIZES)), dtype=eValOnRank.dtype)
            full_eVals = np.empty((2, np.sum(eValOnRank_SIZES)), dtype=eValOnRank.dtype)
            recvcounts = [2 * size for size in eValOnRank_SIZES]
        else:
            full_dVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)
            full_eVals = np.empty(np.sum(eValOnRank_SIZES), dtype=eValOnRank.dtype)

    if sdc.UHF:
        gpu_comm.Gatherv(dValOnRank.flatten(), recvbuf=(full_dVals, recvcounts), root=0)
        gpu_comm.Gatherv(eValOnRank.flatten(), recvbuf=(full_eVals, recvcounts), root=0)
    else:
        gpu_comm.Gatherv(dValOnRank, [full_dVals, eValOnRank_SIZES], root=0)
        gpu_comm.Gatherv(eValOnRank, [full_eVals, eValOnRank_SIZES], root=0)

    eVal_LIST = gpu_comm.gather(eValOnRank_list, root=0)
    Q_LIST = gpu_comm.gather(Q_list, root=0)
    NH_Nh_Hs_LIST = gpu_comm.gather(NH_Nh_Hs_list, root=0)
    core_indices_in_sub_expanded_LIST = gpu_comm.gather(core_indices_in_sub_expanded_list, root=0)
    Nocc_LIST = gpu_comm.gather(Nocc_list, root=0)

    if rank == 0:
        eVal_LIST = list(itertools.chain(*eVal_LIST))
        Q_LIST = list(itertools.chain(*Q_LIST))
        NH_Nh_Hs_LIST = list(itertools.chain(*NH_Nh_Hs_LIST))
        core_indices_in_sub_expanded_LIST = list(itertools.chain(*core_indices_in_sub_expanded_LIST))
        Nocc_LIST = list(itertools.chain(*Nocc_LIST))
    else:
        Q_LIST = None

    if node_rank == 0:
        print(f"| t commLists {time.perf_counter() - tic:>9.4f} (s)", rank)

    if rank == 0:
        tic = time.perf_counter()
        if sdc.UHF:
            nocc_alpha, nocc_beta = _get_uhf_nocc_targets(sy)
            mu0 = np.asarray(mu0, dtype=float)
            if mu0.shape == ():
                mu0 = np.array([float(mu0), float(mu0)])
            mu0 = np.array(
                [
                    get_mu(mu0[0], full_eVals[0], sdc.Tel, nocc_alpha, dvals=full_dVals[0]),
                    get_mu(mu0[1], full_eVals[1], sdc.Tel, nocc_beta, dvals=full_dVals[1]),
                ]
            )
        else:
            mu0 = get_mu(mu0, full_eVals, sdc.Tel, sy.numel / 2, dvals=full_dVals)
        print(f"Time mu0 {time.perf_counter() - tic:>9.4f} (s)")

    if sdc.scfDevice == "cuda":
        del molecule_whole_device

    return (
        EELEC,
        eVal_LIST,
        Q_LIST,
        NH_Nh_Hs_LIST,
        core_indices_in_sub_expanded_LIST,
        Nocc_LIST,
        mu0,
    )


def get_singlePoint(
    sdc,
    eng,
    partsPerGPU,
    partsPerNode,
    node_id,
    node_rank,
    rank,
    gpu_comm,
    parts,
    partsCoreHalo,
    sy,
    hindex,
    mu0,
    molecule_whole,
    P_contr,
    graph_for_pairs,
    graph_maskd,
):
    return _collect_single_point_data(
        sdc,
        eng,
        partsPerGPU,
        partsPerNode,
        node_id,
        node_rank,
        rank,
        gpu_comm,
        parts,
        partsCoreHalo,
        sy,
        hindex,
        mu0,
        molecule_whole,
        P_contr,
        graph_for_pairs,
        graph_maskd,
    )


def get_singlePoint_charges(
    sdc,
    eng,
    partsPerGPU,
    partsPerNode,
    node_id,
    node_rank,
    rank,
    gpu_comm,
    parts,
    partsCoreHalo,
    sy,
    hindex,
    gscf,
    mu0,
    molecule_whole,
    P_contr,
    graph_for_pairs,
    graph_maskd,
):
    del gscf
    return _collect_single_point_data(
        sdc,
        eng,
        partsPerGPU,
        partsPerNode,
        node_id,
        node_rank,
        rank,
        gpu_comm,
        parts,
        partsCoreHalo,
        sy,
        hindex,
        mu0,
        molecule_whole,
        P_contr,
        graph_for_pairs,
        graph_maskd,
    )


def get_singlePointForces(
    sdc,
    eng,
    partsPerGPU,
    partsPerNode,
    node_id,
    node_rank,
    rank,
    parts,
    partsCoreHalo,
    sy,
    hindex,
    forces,
    molecule_whole,
    P,
    P_contr,
    graph_for_pairs,
    graph_maskd,
):
    """Accumulate electronic forces for rank-local CH partitions.

    The function updates ``forces`` in place and returns the summed electronic
    energy contribution from partitions handled by the current rank.
    Analytical forces are used by default; backprop is used only when required.
    """
    del hindex, P
    partIndex1 = (node_rank) * partsPerGPU + node_id * partsPerNode
    partIndex2 = (node_rank + 1) * partsPerGPU + node_id * partsPerNode
    EELEC = 0.0
    use_backprop_forces = _use_backprop_forces(sdc, eng)
    force_device = molecule_whole.coordinates.device
    reset_requires_grad = use_backprop_forces and (not molecule_whole.coordinates.requires_grad)
    if reset_requires_grad:
        molecule_whole.coordinates.requires_grad_(True)

    try:
        # Loop over the partitions allocated to the current rank.
        for partIndex in range(partIndex1, partIndex2):
            # Extract the subsystem information. (CORE + HALO).
            subSy = System(len(partsCoreHalo[partIndex]))
            subSy.symbols = sy.symbols
            subSy.coords, subSy.types = extract_subsystem(sy.coords, sy.types, sy.symbols, partsCoreHalo[partIndex])

            # Extract indices needed for generating the Hamiltonian.
            _, core_indices_in_sub_expanded, _ = get_coreHalo_ham_inds(
                parts[partIndex], partsCoreHalo[partIndex], sdc, sy, subSy, device=force_device
            )

            tic = time.perf_counter()
            ham_timing = {}

            # Get hamiltonian with forces.
            f, eElec = get_hamiltonian(
                sdc,
                eng,
                subSy.coords,
                subSy.types,
                subSy.symbols,
                parts[partIndex],
                partsCoreHalo[partIndex],
                molecule_whole,
                P_contr,
                graph_for_pairs,
                graph_maskd,
                core_indices_in_sub_expanded,
                ham_timing,
                doForces=True,
                verbose=False,
            )

            # Modify forces in-place.
            forces += f

            # Sum electronic energy.
            EELEC += eElec

            # Print timing.
            ham_timing["TOT"] = time.perf_counter() - tic
            formatted_string = " | ".join(f"{key} {value:8.3f}" for key, value in ham_timing.items())
            print(
                "Rank",
                rank,
                "part",
                partIndex,
                ":",
                formatted_string,
                f"|| EelecCH {eElec.item():>7.3f} eV ||",
            )
            del eElec, subSy, f
    finally:
        if reset_requires_grad:
            molecule_whole.coordinates.requires_grad_(False)
            if molecule_whole.coordinates.grad is not None:
                molecule_whole.coordinates.grad.zero_()

    return EELEC


def get_singlePointDM(
    sdc,
    eng,
    rank,
    node_numranks,
    node_comm,
    parts,
    partsCoreHalo,
    sy,
    hindex,
    mu0,
    P_contr,
    P_contr_new,
    graph_for_pairs,
    eValOnRank_list,
    Q_list,
    NH_Nh_Hs_list,
    core_indices_in_sub_expanded_list,
):
    """Build contracted SCF output map and graph from core columns of CH DM.

    This is done in parallel on ALL local ranks of node 0 on CPU. 
    TODO: Improve the efficiency.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng: Engine
        The SEDACS engine.
    rank:
        Current global rank.
    node_numranks:
        Number of ranks on a node (local ranks).
    node_comm:
        Local comminicator on a node.
    parts:
        List of core indices.
    partsCoreHalo:
        List of core+halo indices.
    sy:
        System object.
    hindex:
        Atom->orbtial index mapping.
    mu0:
        The chemical potential.
    P_contr:
        Contracted density matrix. Shape: (sy.nats, sdc.maxDeg, 4,4)
    P_contr_new:
        SCF output map for contracted density matrix (same shape as ``P_contr``).
    graph_for_pairs:
        Graph of communities. E.g. graph_for_pairs[i] is a whole CH community
        in which atom i is a core atom, including itself. graph_for_pairs[i][0]
        is a community size.
    eValOnRank_list:
        Eigenvalues of CHs. Here, for all CHs.
    Q_list:
        Eigenvectors of CHs. Here, only those used by this rank are present.
    NH_Nh_Hs_list:
        List of [number_of_heavy_atoms, number_of_hydrogens,
        dim_of_coreHalo_ham]. Here, for all CHs.
    core_indices_in_sub_expanded_list:
        Indices of core columns in CH. E.g., CH[i] contains atoms [0,1,2,3],
        core atoms are [1,3], 4 AOs per atom. Then,
        core_indices_in_sub_expanded_list[i] is [4,5,6,7, 12,13,14,15].

    Returns
    -------
    graphOnRank:
        Connectivity graph for the density matrix contribution on this rank.
    scfErrorOnRank:
        Maximum absolute DM element change over partitions processed by this rank.

    """
    if rank == 0:
        print(f"eElec:   {sdc.EelecNew:>10.8f} | \u0394E| {abs(sdc.EelecNew - sdc.EelecOld):>10.8f}")

    sdc.EelecOld = sdc.EelecNew

    # Parititon per rank and determine the partitions the current rank is
    # responsible for.
    partsPerRank = int(sdc.nparts / node_numranks)
    partIndex1 = rank * partsPerRank
    partIndex2 = (rank + 1) * partsPerRank
    graphOnRank = None
    P_contr_maxDifList = []
    P_contr_sumDifTot = 0

    local_parts = list(range(partIndex1, partIndex2))
    core_indices_lookup = [
        _core_positions_in_core_halo(partsCoreHalo[p], parts[p], eng.np_int_dt) for p in local_parts
    ]
    p_in_np = P_contr.detach().numpy()
    p_out_np = P_contr_new.detach().numpy()

    for i, partIndex in enumerate(local_parts):
        # This will calculate the DM in subsys and update the whole DM
        # rho_ren is a dm contructed with electronic temperature. It's shaped
        # into 4x4 blocks, even for hydrogen atoms, as required by PYSEQM.
        rho_ren, maxDif, sumDif = get_density_matrix_renorm(
            sdc,
            eng,
            sdc.Tel,
            mu0,
            P_contr,
            graph_for_pairs,
            eValOnRank_list[partIndex],
            Q_list[i],
            NH_Nh_Hs_list[partIndex],
            core_indices_in_sub_expanded_list[partIndex],
        )

        # Core column blocks in CH DM (assuming its shaped as:
        # [n_atoms, n_atoms, 4, 4])
        core_indices_in_sub = core_indices_lookup[i]
        n_core_halo = NH_Nh_Hs_list[partIndex][0] + NH_Nh_Hs_list[partIndex][1]
        P_contr_maxDif = 0.0
        P_contr_sumDif = 0
        if sdc.UHF:  # Open shell.
            # Vectorized. Faster for larger cores.
            max_len = graph_for_pairs[parts[partIndex][0]][0]
            part_atoms = np.asarray(parts[partIndex], dtype=np.int64)

            # Get part of P_contr that corresponds to cores of current CH
            TMP2 = (
                rho_ren.reshape(
                    (
                        1,
                        2,
                        n_core_halo,
                        4,
                        n_core_halo,
                        4,
                    )
                )
                .transpose(3, 4)
                .reshape(2, n_core_halo, n_core_halo, 4, 4)
                .transpose(3, 4)
                .transpose(1, 2)[:, :, core_indices_in_sub]
            )
            tmp2_np = TMP2.detach().numpy()
            P_contr_maxDif, P_contr_sumDif = _update_p_contr_new_uhf_numba(
                p_in_np,
                p_out_np,
                tmp2_np,
                part_atoms,
                int(max_len),
            )

            # Packing rho_ren from 4x4 blocks into normal form based on number
            # of AOs per atom.
            rho_ren = pack(rho_ren[0] + rho_ren[1], NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])

        else:  # Closed shell. See documentation in open-shell.
            max_len = graph_for_pairs[parts[partIndex][0]][0]
            part_atoms = np.asarray(parts[partIndex], dtype=np.int64)
            TMP2 = (
                rho_ren.reshape(
                    (
                        1,
                        n_core_halo,
                        4,
                        n_core_halo,
                        4,
                    )
                )
                .transpose(2, 3)
                .reshape(n_core_halo, n_core_halo, 4, 4)
                .transpose(2, 3)
                .transpose(0, 1)[:, core_indices_in_sub]
            )
            tmp2_np = TMP2.detach().numpy()
            P_contr_maxDif, P_contr_sumDif = _update_p_contr_new_rhf_numba(
                p_in_np,
                p_out_np,
                tmp2_np,
                part_atoms,
                int(max_len),
            )

            # Packing rho_ren from 4x4 blocks into normal form based on number
            # of AOs per atom.
            rho_ren = pack(rho_ren, NH_Nh_Hs_list[partIndex][0], NH_Nh_Hs_list[partIndex][1])

        # Store max differences.
        P_contr_maxDifList.append(P_contr_maxDif)

        # Total of the summed differences.
        P_contr_sumDifTot += P_contr_sumDif

        # Get connectivity graph for the dm of current CH
        graphOnRank = collect_graph_from_rho_PYSEQM(
            graphOnRank, rho_ren, sdc.gthresh, sy.nats, sdc.maxDeg, partsCoreHalo[partIndex], hindex, verb=False
        )
        del rho_ren

    if P_contr_maxDifList:
        scfErrorOnRank = float(max(P_contr_maxDifList))
        print(
            f" MAX |\u0394DM_ij|: {scfErrorOnRank:>10.7f} at SubSy {int(np.argmax(P_contr_maxDifList)):>5d}"
        )
    else:
        scfErrorOnRank = 0.0
        print(f" MAX |\u0394DM_ij|: {0.0:>10.7f} at SubSy {-1:>5d}")
    print(f" \u03a3   |\u0394DM_ij|: {P_contr_sumDifTot:>10.7f}")

    return graphOnRank, scfErrorOnRank


def get_adaptiveDM_PYSEQM(
    sdc,
    eng,
    comm,
    rank,
    numranks,
    sy,
    hindex,
    graphNL,
    save_output_files=True,
    return_graph=False,
):
    """The main driver function. It initializes supplementary comms, dm, graphs,
    performs scf cycle with graph and dm updates, and then computes forces.

    sdc:
        SEDACS driver.
    eng:
        SEDACS Engine.
    comm:
        Master MPI communicator.
    rank:
        The global rank.
    numranks:
        The number of global ranks.
    sy:
        The full SEDACS System.
    hindex:
        Orbital indices for each atom in the system.
    graphNL:
        Initial connectivity graph
    """
    # SCF Initialization time.
    md_result = None
    t_INIT = time.perf_counter()
    tic = time.perf_counter()
    sdc.EelecOld = 0.0

    # Whether or not to use large tensors. Prohibitively large for big systems.
    eng.use_pyseqm_lt = sdc.use_pyseqm_lt

    if sdc.doForces and (not eng.use_pyseqm_lt) and (not sdc.analyticForces):
        raise NotImplementedError(
            "Backpropagation-based electronic forces are disabled when use_pyseqm_lt=False. "
            "Use analyticForces=True, or set use_pyseqm_lt=True for backprop fallback."
        )

    # Reconstructs the full density matrix for debugging purpose.
    eng.reconstruct_dm = False
    sdc.reconstruct_dm = eng.reconstruct_dm

    # //Set up the MPI communicators.

    # Local communicator for ranks on a given node.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    # The local rank on a given node.
    node_rank = node_comm.Get_rank()

    # The size of a communicator on a given node.
    node_numranks = node_comm.Get_size()

    # Get relevant information specific to the nodes the job has been
    # allocated.
    node_name = socket.gethostname()
    node_names = comm.allgather(node_name)

    unique_nodes = list(set(node_names))
    num_nodes = len(unique_nodes)
    node_id = int(rank / node_numranks)

    # Get primary ranks on each node.
    # E.g. when running 16 ranks on two nodes, these are ranks 0 and 8.
    # [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]
    primary_rank = None
    if node_rank == 0:
        primary_rank = rank  # Global rank of the primary rank on each node

    # Gather the primary ranks from each node
    primary_ranks = comm.allgather(primary_rank)

    # Filter out Nones.
    primary_ranks = [r for r in primary_ranks if r is not None]
    color = 0 if rank in primary_ranks else MPI.UNDEFINED

    # Communicator for primary ranks.
    primary_comm = comm.Split(color=color, key=rank)

    device = "cpu"
    if sdc.scfDevice == "cuda":
        if sdc.numGPU == -1:
            # Use the minimum count across nodes so heterogeneous nodes stay consistent.
            local_num_gpus = torch.cuda.device_count()
            num_gpus = comm.allreduce(local_num_gpus, op=MPI.MIN)
        else:
            num_gpus = sdc.numGPU
    else:
        num_gpus = node_numranks

    num_gpus = int(min(num_gpus, node_numranks))
    if num_gpus < 1:
        raise ValueError("No compute devices available for SCF execution.")

    local_partition_errors = []
    if sdc.nparts % num_nodes != 0:
        local_partition_errors.append(
            f"NumParts ({sdc.nparts}) must be divisible by number of nodes ({num_nodes})."
        )
    if sdc.nparts % (num_gpus * num_nodes) != 0:
        local_partition_errors.append(
            f"NumParts ({sdc.nparts}) must be divisible by "
            f"num_gpus_per_node * num_nodes ({num_gpus} * {num_nodes})."
        )
    if sdc.nparts % node_numranks != 0:
        local_partition_errors.append(
            f"NumParts ({sdc.nparts}) must be divisible by ranks per node ({node_numranks})."
        )

    gathered_partition_errors = comm.allgather(local_partition_errors)
    if any(errors for errors in gathered_partition_errors):
        first_errors = next(errors for errors in gathered_partition_errors if errors)
        raise ValueError("Invalid distributed partitioning setup: " + " ".join(first_errors))

    color = 0 if node_rank < num_gpus else MPI.UNDEFINED

    # Global communicator for ranks with GPU.
    # Identical to comm if running on CPU.
    gpu_comm = comm.Split(color=color, key=rank)

    # Assume all nodes have same number of GPUs!
    partsPerGPU = sdc.nparts // (num_gpus * num_nodes)

    # How many CH are processed by each node.
    partsPerNode = sdc.nparts // num_nodes

    # //Communicator setup finished.

    # Some data type info for numpy and torch.
    # Double precision is necessary for pyseqm.
    _configure_driver_dtypes(sdc, eng)

    fullGraph = graphNL.copy()

    # Get PYSEQM Molecule object.
    tic = time.perf_counter()
    molecule_whole = get_molecule_pyseqm(
        sdc,
        sy.coords,
        sy.symbols,
        sy.types,
        do_large_tensors=sdc.use_pyseqm_lt,
        device=device,
        requires_grad=False,
    )[0]

    if rank == 0:
        print(f"Time to init molSysData {time.perf_counter() - tic:>7.2f} (s)", rank)

    # //Initialization for SCF

    # Things that are calculated on each primary rank of each node.
    if node_rank == 0:
        tic = time.perf_counter()
        if rank == 0:
            print("Computing cores.")

        # Core partitions.
        parts = graph_partition(sdc, eng, fullGraph, sdc.partitionType, sdc.nparts, sy.coords, sdc.verb)
        if rank == 0:
            print(f"Time to compute cores {time.perf_counter() - tic:>7.2f} (s)", rank)

        tic = time.perf_counter()
        if rank == 0:
            print("Loading the molecule and parameters.")

        if rank == 0:
            print("\n|||| Adaptive iter:", 0, "||||")
        partsCoreHalo = _collect_parts_core_halo(
            parts, fullGraph, sdc, eng, sy, rank, use_pyseqm=True
        )

        print(f"Time to compute halos {time.perf_counter() - tic:>7.2f} (s)", rank)

        tic = time.perf_counter()

        # graph_for_pairs[i] is the CH for atom i.
        graph_for_pairs = get_ch_graph(sdc, sy, fullGraph, parts, partsCoreHalo)

        # Mask for diagonal block in contracted density matrix.
        graph_maskd = get_maskd(sdc, sy, graph_for_pairs)

        print(f"Time to init mod graphs {time.perf_counter() - tic:>7.2f} (s)", rank)

        tic = time.perf_counter()
        if sdc.UHF:
            # Contracted density matrix.
            P_contr = torch.zeros(2, sy.nats * sdc.maxDeg, 4, 4, dtype=eng.torch_dt, device=device)

            # Diagonal initial guess.
            P_contr[:, graph_maskd] = 0.5 * get_diag_guess_pyseqm(molecule_whole, sy)

            # Shape is: (2, n_atoms, max_deg, 4, 4).
            # A rectangle of 4x4 square blocks.
            P_contr = P_contr.reshape(2, sy.nats, sdc.maxDeg, 4, 4).transpose(1, 2)
            P_contr_size = P_contr.size()
            P_contr_nbytes = P_contr.numel() * P_contr.element_size()

        else:
            # Contracted density matrix.
            P_contr = torch.zeros(sy.nats * sdc.maxDeg, 4, 4, dtype=eng.torch_dt, device=device)

            # Diagonal initial guess.
            P_contr[graph_maskd] = get_diag_guess_pyseqm(molecule_whole, sy)

            # Shape is: (n_atoms, max_deg, 4, 4).
            # A rectangle of 4x4 square blocks.
            P_contr = P_contr.reshape(sy.nats, sdc.maxDeg, 4, 4).transpose(0, 1)

            P_contr_size = P_contr.size()
            P_contr_nbytes = P_contr.numel() * P_contr.element_size()

        print(f"Time to init DM {time.perf_counter() - tic:>7.2f} (s)", rank)

        del graphNL

    else:
        parts = None
        sdc.nparts = None

        fullGraph = None
        partsCoreHalo = None

        graph_for_pairs = None
        graph_maskd = None

        P_contr = None
        P_contr_size = None
        P_contr_nbytes = 0

    tic = time.perf_counter()
    parts = node_comm.bcast(parts, root=0)
    sdc.nparts = node_comm.bcast(sdc.nparts, root=0)

    if rank == 0:
        print(f"BCST1 {time.perf_counter() - tic:>7.2f} (s)", rank)

    tic = time.perf_counter()
    # P_contr is in shared memory between ranks on one node
    # but each node has its own copy.
    P_contr_size = node_comm.bcast(P_contr_size, root=0)

    # 8 is the size of torch.float64
    P_contr_win = MPI.Win.Allocate_shared(
        P_contr_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm
    )

    P_contr_buf, P_contr_itemsize = P_contr_win.Shared_query(0)

    P_contr_ary = np.ndarray(buffer=P_contr_buf, dtype=eng.np_dt, shape=(P_contr_size))

    if node_rank == 0:
        P_contr_ary[:] = P_contr.cpu().numpy()

    comm.Barrier()

    del P_contr

    P_contr = torch.from_numpy(P_contr_ary).to(device)

    # Shared buffer for SCF output map (P_out) used by DM-DIIS mixing.
    P_contr_new_win = MPI.Win.Allocate_shared(
        P_contr_nbytes, torch.tensor(0, dtype=eng.torch_dt).element_size(), comm=node_comm
    )
    P_contr_new_buf, _ = P_contr_new_win.Shared_query(0)
    P_contr_new_ary = np.ndarray(buffer=P_contr_new_buf, dtype=eng.np_dt, shape=(P_contr_size))
    if node_rank == 0:
        P_contr_new_ary[:] = P_contr_ary[:]
    comm.Barrier()
    P_contr_new = torch.from_numpy(P_contr_new_ary).to(device)

    if rank == 0:
        print(f"BCST2 {time.perf_counter() - tic:>7.2f} (s)", rank)

    tic = time.perf_counter()
    shared_state = (fullGraph, partsCoreHalo, graph_maskd, graph_for_pairs) if node_rank == 0 else None
    fullGraph, partsCoreHalo, graph_maskd, graph_for_pairs = node_comm.bcast(shared_state, root=0)
    if rank == 0:
        print(f"BCST3 {time.perf_counter() - tic:>7.2f} (s)", rank)

    print(f"Time to init bcast and share DM {time.perf_counter() - tic:>7.2f} (s)", rank)

    if rank == 0:
        print(f"Time INIT {time.perf_counter() - t_INIT:>7.2f} (s)")

    # //Initilziation for SCF finished.

    # //Begin SCF Cycle.

    # Initial chemical potential guess. TODO: This should probably not be
    # hard-coded.
    mu0 = -4.7
    if sdc.UHF:
        mu0 = np.array([mu0 + 0.1, mu0 - 0.1])
        mu0 = np.array([-1.3, -5.5])

    diis_state = _init_density_diis_state(sdc) if rank == 0 else None

    # Iteration loop.
    for gsc in range(sdc.numAdaptIter):
        if rank == 0:
            print("\n\n|||| Adaptive iter:", gsc, "||||")
        TIC_iter = time.perf_counter()
        tic = time.perf_counter()

        # Broadcasts dm from root rank (assuming its rank 0 on node 0) to
        # primary ranks on other nodes. E.g. for ranks arranged as
        # {node0:[0,1,2,3] node1:[4,5,6,7]}, dm is broadcates from 0 to 4.
        # One of the major bottlenecks.
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)

        if rank == 0:
            print(f"Time to  bcast DM_cpu_np {time.perf_counter() - tic:>7.2f} (s)", rank)

        tic = time.perf_counter()

        # Lots of things have been done during initialization,
        # so after iteration 0 we can proceed right to get_singlePoint.
        if gsc > 0:
            # Halos, dm contraction, and graphs are performed on primary ranks
            # of each node and then broadcasted locally to other ranks.
            if node_rank == 0:
                # //Begin HALOS
                tic = time.perf_counter()
                partsCoreHalo = _collect_parts_core_halo(parts, fullGraph, sdc, eng, sy, rank)

                if rank == 0:
                    print(f"Time to compute halos {time.perf_counter() - tic:>7.2f} (s)")
                # //End HALOS

                graph_for_pairs, graph_maskd = _update_partition_graph_data(
                    sdc, sy, fullGraph, parts, partsCoreHalo, P_contr, graph_for_pairs, device, rank
                )

            else:
                partsCoreHalo = None
                graph_for_pairs = None
                graph_maskd = None

            tic = time.perf_counter()
            graph_state = (partsCoreHalo, graph_for_pairs, graph_maskd) if node_rank == 0 else None
            partsCoreHalo, graph_for_pairs, graph_maskd = node_comm.bcast(graph_state, root=0)
            if node_rank == 0:
                print(f"Time to bcast DM and mod graphs {time.perf_counter() - tic:>7.2f} (s)", rank)

        tic = time.perf_counter()

        # Single point part. For efficiency, the PySEQM density matrix
        # needs to be reshaped in 4x4 blocks.

        # This will sum electronic energy from CHs on ranks, giving total Eelec
        global_Eelec = np.zeros(1, dtype=np.float64)

        # TODO: All of this is so PYSEQM specific already, this should probably
        # be removed.
        if eng.interface == "PySEQM":
            with torch.no_grad():
                # This condition is for GPU jobs only because sometimes there
                # are fewer GPUs per node than ranks per nodes.

                if node_rank < num_gpus:
                    # We want more ranks per node because dm update always
                    # happens on CPU, on node 0, in parallel.
                    (
                        eElec,
                        eValOnRank_list,
                        Q_list,
                        NH_Nh_Hs_list,
                        core_indices_in_sub_expanded_list,
                        Nocc_list,
                        mu0,
                    ) = get_singlePoint(
                        sdc,
                        eng,
                        partsPerGPU,
                        partsPerNode,
                        node_id,
                        node_rank,
                        rank,
                        gpu_comm,
                        parts,
                        partsCoreHalo,
                        sy,
                        hindex,
                        mu0,
                        molecule_whole,
                        P_contr,
                        graph_for_pairs,
                        graph_maskd,
                    )

                    # if gsc > 2: exit(0)
                    gpu_comm.Allreduce(eElec, global_Eelec, op=MPI.SUM)

                else:
                    (
                        eElec,
                        eValOnRank_list,
                        Q_list,
                        NH_Nh_Hs_list,
                        core_indices_in_sub_expanded_list,
                        Nocc_list,
                        mu0,
                    ) = 0, None, None, None, None, None, None

            comm.Barrier()

        else:
            raise ValueError(
                f"ERROR!!!: Interface type not recognized: '{eng.interface}'. "
                + "Use any of the following: Module,File,Socket,MDI"
            )

        if gsc == 0 and sdc.UHF:
            print("sym break")

            for I in range(len(Q_list)):
                orb_idx = NH_Nh_Hs_list[I][3][0]
                Q_list[I][0, :, orb_idx] = 0.9 * Q_list[I][0, :, orb_idx - 1] + 0.1 * Q_list[I][0, :, orb_idx]

        sdc.EelecNew = global_Eelec[0]

        if rank == 0:
            print(f"Time to get_singlePoint {time.perf_counter() - tic:>7.2f} (s)")

        # If True, these files will be read and used instead as default initial guess.
        if sdc.restartLoad:
            sdc.restartLoad = False
            if node_rank == 0:
                P_contr[:] = torch.load("P_contr.pt")
            with open("parts.pkl", "rb") as f:
                parts = pickle.load(f)
            with open("partsCoreHalo.pkl", "rb") as f:
                partsCoreHalo = pickle.load(f)
            with open("fullGraph.pkl", "rb") as f:
                fullGraph = pickle.load(f)

            mu0 = np.load("mu0.npy")
            graph_for_pairs = np.load("graph_for_pairs.npy")
            graph_maskd = np.load("graph_maskd.npy")

            if rank == 0:
                eValOnRank_list = torch.load("eValOnRank_list.pt")
                Q_list = torch.load("Q_list.pt")
                NH_Nh_Hs_list = torch.load("NH_Nh_Hs_list.pt")
                core_indices_in_sub_expanded_list = torch.load("core_indices_in_sub_expanded_list.pt")
                Nocc_list = torch.load("Nocc_list.pt")
                _reset_density_diis_history(diis_state)

        # Save for future restart. Slows things down significantly.
        if rank == 0 and sdc.restartSave:
            torch.save(eValOnRank_list, "eValOnRank_list.pt")
            torch.save(Q_list, "Q_list.pt")
            torch.save(NH_Nh_Hs_list, "NH_Nh_Hs_list.pt")
            torch.save(core_indices_in_sub_expanded_list, "core_indices_in_sub_expanded_list.pt")
            torch.save(Nocc_list, "Nocc_list.pt")
            torch.save(P_contr, "P_contr.pt")
            with open("parts.pkl", "wb") as f:
                pickle.dump(parts, f)
            with open("partsCoreHalo.pkl", "wb") as f:
                pickle.dump(partsCoreHalo, f)
            with open("fullGraph.pkl", "wb") as f:
                pickle.dump(fullGraph, f)
            np.save("mu0", mu0)
            np.save("graph_for_pairs", graph_for_pairs)
            np.save("graph_maskd", graph_maskd)

        # This defines what part of density matrix will be
        # updated by each rank on node 0.
        scfError = None
        if rank < node_numranks:
            tic = time.perf_counter()
            partsPerRank = int(sdc.nparts / node_numranks)
            q_slices = None
            if rank == 0:
                q_slices = [Q_list[r * partsPerRank : (r + 1) * partsPerRank] for r in range(node_numranks)]
            Q_list_on_rank = node_comm.scatter(q_slices, root=0)
            if rank == 0:
                print(f"Time send Q_list slice {time.perf_counter() - tic:>7.2f} (s)")

        if rank < node_numranks:
            tic = time.perf_counter()

            # Broadcast data across ranks on node 0.
            dm_state = (
                eValOnRank_list,
                NH_Nh_Hs_list,
                core_indices_in_sub_expanded_list,
                mu0,
            ) if rank == 0 else None
            eValOnRank_list, NH_Nh_Hs_list, core_indices_in_sub_expanded_list, mu0 = node_comm.bcast(
                dm_state, root=0
            )
            if rank == 0:
                P_contr_new[:] = P_contr
            node_comm.Barrier()

            # Density matrix update and the graph from the DM.
            with torch.no_grad():
                fullGraphRho, scfErrorOnRank = get_singlePointDM(
                    sdc,
                    eng,
                    rank,
                    node_numranks,
                    node_comm,
                    parts,
                    partsCoreHalo,
                    sy,
                    hindex,
                    mu0,
                    P_contr,
                    P_contr_new,
                    graph_for_pairs,
                    eValOnRank_list,
                    Q_list_on_rank,
                    NH_Nh_Hs_list,
                    core_indices_in_sub_expanded_list,
                )

            if rank == 0:
                print(f"Time to updt DM {time.perf_counter() - tic:>7.2f} (s)")

            node_comm.Barrier()

            tic = time.perf_counter()

            # Get graph derived from the density matrix (on each rank.)
            fullGraphRho_LIST = node_comm.gather(fullGraphRho, root=0)
            scfErrorOnRank_LIST = node_comm.gather(scfErrorOnRank, root=0)
            if rank == 0:
                # Adds the graph we got on the previous iteration. NOTE:, when
                # doing SCF, the graph keeps growing, no nodes from previous
                # iterations are removed.
                fullGraphRho_LIST.append(fullGraph)

                # Combines graphs from the Python List.
                fullGraph = add_mult_graphs(fullGraphRho_LIST)
                scfError = float(max(scfErrorOnRank_LIST))
                print("SCF ERR =", scfError)

                mix_mode, diis_coeffs = _mix_density_with_diis(
                    sdc, gsc, P_contr, P_contr_new, graph_for_pairs, diis_state
                )
                print(
                    "DM MIX =",
                    mix_mode,
                    "| hist =",
                    len(diis_state["entries"]) if diis_state is not None else 0,
                )
                if diis_coeffs is not None and sdc.verb:
                    print("DIIS COEFF =", np.array2string(diis_coeffs, precision=4))

                print(f"Time to add graphs {time.perf_counter() - tic:>7.2f} (s)")

            # Ensure node-0 workers don't read stale DM state before the next phase.
            node_comm.Barrier()

            del fullGraphRho

            if rank == 0:
                tic = time.perf_counter()
                if sdc.UHF:
                    trace = torch.sum(
                        P_contr.transpose(1, 2)
                        .reshape(2, molecule_whole.molsize * (len(graph_for_pairs[0]) - 1), 4, 4)[:, graph_maskd]
                        .diagonal(dim1=-2, dim2=-1),
                        dim=(1, 2),
                    )

                    print(f"DM TRACE: {trace[0]:>10.8f}, {trace[1]:>10.8f}")

                else:
                    trace = torch.sum(
                        P_contr.transpose(0, 1)
                        .reshape(molecule_whole.molsize * (len(graph_for_pairs[0]) - 1), 4, 4)[graph_maskd]
                        .diagonal(dim1=-2, dim2=-1)
                    )

                    print(f"DM TRACE: {trace:>10.7f}")

                print(f"Time to get trace {time.perf_counter() - tic:>7.2f} (s)")

        else:
            fullGraph = None

        tic = time.perf_counter()

        # Broadcast the new graph across ALL ranks.
        fullGraph = comm.bcast(fullGraph, root=0)

        if rank == 0:
            print(f"Time to bcast fullGraph {time.perf_counter() - tic:>7.2f} (s)")

        if rank == 0:
            tol = float(sdc.scfTol)
            converged = (tol > 0.0) and (scfError is not None) and (scfError < tol)
            if converged:
                print(f"SCF converged with SCF error = {scfError}")
        else:
            converged = None
        converged = comm.bcast(converged, root=0)

        del eValOnRank_list, Q_list, NH_Nh_Hs_list

        if sdc.scfDevice == "cuda":
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"t Iter {time.perf_counter() - TIC_iter:>8.2f} (s)")
        if converged:
            break

    # //SCF Cycle complete.

    # //Forces begin
    tic_F_INIT = time.perf_counter()

    if node_rank < num_gpus:
        # Broadcast density matrix from root rank (assuming rank=node=0.)
        # to primary ranks on other nodes.
        if node_rank == 0:
            primary_comm.Bcast([P_contr.cpu().numpy(), MPI.DOUBLE], root=0)
            forces = np.zeros(sy.coords.shape)
            partsCoreHalo = _collect_parts_core_halo(parts, fullGraph, sdc, eng, sy, rank)
            graph_for_pairs, graph_maskd = _update_partition_graph_data(
                sdc, sy, fullGraph, parts, partsCoreHalo, P_contr, graph_for_pairs, device, rank
            )

        else:
            forces = None
            partsCoreHalo = None
            graph_for_pairs = None
            graph_maskd = None

        if sdc.scfDevice == "cuda":
            device = f"cuda:{node_rank}"
        else:
            device = "cpu"

        molecule_whole = get_molecule_pyseqm(
            sdc,
            sy.coords,
            sy.symbols,
            sy.types,
            do_large_tensors=sdc.use_pyseqm_lt,
            device=device,
            requires_grad=_use_backprop_forces(sdc, eng),
        )[0]

        force_state = (forces, partsCoreHalo, graph_for_pairs, graph_maskd) if node_rank == 0 else None
        forces, partsCoreHalo, graph_for_pairs, graph_maskd = gpu_comm.bcast(force_state, root=0)
        if rank == 0:
            forces[:] = 0.0
        gpu_comm.Barrier()

        if rank == 0:
            print(f"Time init forces {time.perf_counter() - tic_F_INIT:>8.2f} (s)")

        tic = time.perf_counter()
        if eng.interface == "PySEQM":
            P_contr_device = P_contr.to(device)
            eElec = get_singlePointForces(
                sdc,
                eng,
                partsPerGPU,
                partsPerNode,
                node_id,
                node_rank,
                rank,
                parts,
                partsCoreHalo,
                sy,
                hindex,
                forces,
                molecule_whole,
                None,
                P_contr_device,
                graph_for_pairs,
                graph_maskd,
            )

            global_Eelec = np.zeros(1, dtype=np.float64)
            gpu_comm.Barrier()
            gpu_comm.Allreduce(MPI.IN_PLACE, forces, op=MPI.SUM)

            # Primary communicator.
            gpu_comm.Allreduce(eElec, global_Eelec, op=MPI.SUM)

        if rank == 0:
            print(f"Time to get electron forces {time.perf_counter() - tic:>8.2f} (s)")

            print(
                f"eElec:   {global_Eelec[0]:>10.12f}",
            )

        # Nuclear energy and forces. For now, done on one cpu/gpu,
        # for the whole system at once (pyseqm style). Hence, do_large_tensors
        # = True. Needs to be fixed.
        if rank == 0:
            # Object with whatever initial parameters and tensors.
            molSysData = pyseqmObjects(sdc, sy.coords, sy.symbols, sy.types, do_large_tensors=True, device=device)
            # Nuclear forces are computed via autograd here.
            if not molSysData.molecule_whole.coordinates.requires_grad:
                molSysData.molecule_whole.coordinates.requires_grad_(True)

            tic = time.perf_counter()
            eNucAB = get_eNuc(eng, molSysData)
            eTot, eNuc = get_eTot(eng, molSysData, eNucAB, 0)
            print(
                f"Enuc:   {eNuc:>10.12f}",
            )
            L = eNuc.sum()
            L.backward()
            forceNuc = -molSysData.molecule_whole.coordinates.grad.detach()
            molSysData.molecule_whole.coordinates.grad.zero_()
            print(f"Time to get nuclear forces {time.perf_counter() - tic:>8.2f} (s)")
            total_forces = forces + forceNuc.cpu().numpy()[0]
            eNuc_scalar = eNuc.detach().cpu().item() if torch.is_tensor(eNuc) else float(eNuc)
            potential_energy = float(global_Eelec[0] + eNuc_scalar)
            if save_output_files:
                np.save(
                    "forces",
                    total_forces,
                )
                np.save("potential_energy", np.array([potential_energy], dtype=np.float64))
            md_result = {"forces": np.array(total_forces, copy=True), "potential_energy": potential_energy}

    # //Forces finished.
    result = comm.bcast(md_result, root=0)
    if return_graph:
        return result, fullGraph
    return result


def get_adaptiveDM(
    sdc,
    eng,
    comm,
    rank,
    numranks,
    sy,
    hindex,
    graphNL,
    save_output_files=True,
    return_graph=False,
):
    """Get the adaptive denisty matrix at the current iteration.

    Parameters
    ----------
    sdc:
        The SEDACS driver.
    eng:
        The SEDACS Engine.
    comm:
        The MPI communicator.
    rank:
        The MPI rank.
    numranks:
        Number of global MPI ranks.
    sy:
        The full system SEDACS System object.
    hindex:
        Atom-wise orbital indices.
    graphNL:
        The thresholded graph neighborlist.
    save_output_files:
        If True, write force/energy helper files (`forces.npy`, `potential_energy.npy`) on rank 0.
    return_graph:
        If True, return the updated adaptive graph along with force/energy data.

    Returns
    -------
    fullGraph:
        The full graph for the entire system.
    charges:
        Charges for the atoms in the system.
    parts:
        The core partitions.
    subSysOnRank:
        Rank indices for a given subsystem.

    """
    if eng.interface != "PySEQM":
        raise NotImplementedError(
            "graph_adaptive currently supports only the PySEQM interface. "
            "Use graph_adaptive_scf.py or another interface-specific driver."
        )

    return get_adaptiveDM_PYSEQM(
        sdc,
        eng,
        comm,
        rank,
        numranks,
        sy,
        hindex,
        graphNL,
        save_output_files=save_output_files,
        return_graph=return_graph,
    )


get_occ_singlePoint = get_singlePoint
get_adaptiveSCFDM = get_adaptiveDM
