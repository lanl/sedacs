import numpy as np
import sys 
import scipy.linalg as sp
from sedacs.system import get_hindex
import sys

from sedacs.types import ArrayLike
from typing import Union

from sedacs.types import ArrayLike
from sedacs.parser import Input
import torch

try:
  import seqm; PYSEQM = True
  from seqm.seqm_functions.constants import Constants
  from seqm.seqm_functions.make_dm_guess import make_dm_guess
  from seqm.Molecule import Molecule
  from seqm.ElectronicStructure import Electronic_Structure
  import numpy as np
  from seqm.seqm_functions.read_xyz import read_xyz
  from seqm.MolecularDynamics import Geometry_Optimization_SD
  from seqm.seqm_functions.fock import fock
  from seqm.seqm_functions.fock_u_batch import fock_u_batch
  from seqm.seqm_functions.hcore import hcore
  from seqm.seqm_functions.diag import sym_eig_trunc
  from seqm.seqm_functions.pack import *
  from seqm.seqm_functions.energy import *
  seqm.seqm_functions.scf_loop.debug=False
  import time

except: PYSEQM = False

class pyseqmObjects(torch.nn.Module):
    '''
    Container for pyseqm objects
    '''
    def __init__(self,
                 sdc: Input,
                 coords: ArrayLike,
                 symbols: ArrayLike,
                 atomTypes: ArrayLike,
                 do_large_tensors: bool = True,
                 device: str = 'cpu'):
        """
        Constructor for the pyseqmObjects class.

        Parameters
        ----------
        sdc: Input
            The sedacs driver.
        coords: ArrayLike (Natoms, 3)
            The Cartesian coordinates in the system of interest.
        symbols: ArrayLike
            The unique chemical elements in the structure.
        atomTypes: ArrayLike (Natoms, )
            The element type of each atom in the system.
        do_large_tensors: bool
            If False, PySEQM won't calculate large tensors like idxi, idxj, rij, xij, mask.
        device: str
            PyTorch device.

        Returns
        -------
        None
        """
        super().__init__()
       
        self.M_whole, self.w_whole = None, None

        # Get the full PYSEQM Molecule object from the system information.
        self.molecule_whole = get_molecule_pyseqm(sdc, coords, symbols, atomTypes, do_large_tensors=do_large_tensors, device=device)[0].to(device)


        # Grab relevant information if the full tensor is to be used.
        if do_large_tensors: ### some tensors for calculating nuclear forces
          self.w_ssss = torch.zeros_like(self.molecule_whole.idxi)

          ev = 27.21
          rho_0 = 0.5*ev/self.molecule_whole.parameters['g_ss']
          self.rho0xi_whole = rho_0[self.molecule_whole.idxi].clone()
          self.rho0xj_whole = rho_0[self.molecule_whole.idxj].clone()
          A = (self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxi] != 0.000)
          B = (self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxj] != 0.000)
          self.rho0xi_whole[A] =self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxi][A]
          self.rho0xj_whole[B] =self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxj][B]

def get_coreHalo_ham_inds(partIndex: ArrayLike,
                          partCoreHaloIndex: ArrayLike,
                          sdc: Input,
                          sy,
                          subSy,
                          device='cpu') -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    '''
    Gets core-halo indices with respect to the Hamiltonian used for the SEQM calculation.

    Parameters
    ----------
    partIndex: ArrayLike
        Core indices.
    partCoreHaloIndex:
        Core + halo indices
    sdc: Input
        The sedacs driver.
    sy:
    subSy:

    Returns 
    -------
    core_indices_in_sub: ArrayLike
        Core indices of atoms in the core+halo.
    core_indices_in_sub_expanded: ArrayLike
        Core indices of core+halo hamiltonian in 4x4 blocks form (pyseqm format).
        PYSEQM format pads the H-atoms as if they had 3 p-orbitals present.
    hindex_sub: ArrayLike
        Orbital index for each atom in the CH (in CH numbering)
    '''
     # Generate local indexing of core+halo atoms
    indices_in_sub = torch.linspace(0, len(partCoreHaloIndex) - 1, len(partCoreHaloIndex), dtype=sdc.torch_int_dt, device=device)

    core_indices_in_sub = indices_in_sub[torch.isin(torch.tensor(partCoreHaloIndex, device=device), torch.tensor(partIndex, device=device))] # $$$ torch.searchsorted might be better
    block_size = 4

    # Generate the expanded indices for each block
    base_indices = torch.arange(block_size, dtype=sdc.torch_int_dt, device=device)  # Create a base index tensor of size block_size

    core_indices_in_sub_expanded = core_indices_in_sub.unsqueeze(1) * block_size + base_indices
    core_indices_in_sub_expanded = core_indices_in_sub_expanded.flatten()

    norbs, norbs_for_every_type, hindex_sub, numel, numel_for_every_type = get_hindex(sdc.orbs,
                                                                                      sy.symbols,
                                                                                      subSy.types,
                                                                                      valency=sdc.valency)

    hindex_sub = torch.from_numpy(hindex_sub).to(device, dtype=sdc.torch_int_dt)
    return core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub

def get_nucAB_energy_pyseqm(Z,
                            const,
                            nmol,
                            ni,
                            nj,
                            idxi,
                            idxj,
                            rij,
                            rho0xi,
                            rho0xj,
                            alp,
                            chi,
                            gam,
                            method,
                            parnuc):
    """
    Wrapper function for the pyseqm nuclear repulsion routines. see pyseqm documentation.
    Not intended for use outside the curated PYSEQM routines.
    """
    return pair_nuclear_energy(Z, const, nmol, ni, nj, idxi, idxj, rij, \
                                     rho0xi,rho0xj,alp, chi, gam=gam, method=method, parameters=parnuc)

def get_total_energy_pyseqm(nmol,
                            pair_molid,
                            EnucAB,
                            Eelec):
    """
    Wrapper function for the pyseqm total energy routine. see pyseqm documentation.
    Not intended for use outside the curated PYSEQM routines.
    """

    return total_energy(nmol, pair_molid, EnucAB, Eelec)
   
def get_full_fock_pyseqm(nmol,
                         molsize,
                         P,
                         M,
                         maskd,
                         mask,
                         idxi,
                         idxj,
                         w,
                         W,
                         gss,
                         gpp,
                         gsp,
                         gp2,
                         hsp,
                         themethod,
                         zetas,
                         zetap,
                         zetad,
                         Z,
                         F0SD,
                         G2SD):

    """
    Wrapper function for the PYSEQM Fock routine. See PYSEQM documentation.
    Not intended for use outside the curated PYSEQM routines.
    """

    return fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,
         themethod, zetas, zetap, zetad, Z, F0SD, G2SD)

def get_fock_pyseqm(P: ArrayLike,
                    P_sub: ArrayLike,
                    M: ArrayLike,
                    w_2: ArrayLike,
                    block_indices: ArrayLike,
                    nmol: int,
                    idxi: ArrayLike,
                    idxj: ArrayLike,
                    rij: ArrayLike,
                    parameters: ArrayLike,
                    maskd_sub: ArrayLike,
                    mask_sub: ArrayLike) -> ArrayLike:
    '''
    Returns the Fock matrix in the PYSEQM format, with padding on the H-atoms, s.t.
    they are in the 4x4 block format.

    Parameters
    ----------
    P: ArrayLike
        Diagonal density matrix blocks of the whole system.
    P_sub: ArrayLike
        Subsystem density matrices.
    M: ArrayLike
        One electron Hamiltonian of the subsystem.
    w_2: ArrayLike
        Two-center, two-electron integrals.
        These are compute for:
            subsystem-subsystem
            subsystem-outer
            NOT outer-outer
    block_indices: ArrayLike
        Subsystem atom numbers.
    nmol: int
        Number of molecules in a batch. Always 1 in SEDACS.
    idxi: ArrayLike
        Unique pairs between atoms in subsystem or between an atom in subsystem and in the outer system.
    idxj: ArrayLike
        Unique pairs between atoms in subsystem or between an atom in subsystem and in the outer system.
    rij: ArrayLike
        Distances for unique pairs (corresponding to those specified in idxi, and idxj above.
    parameters:
        Seqm atomic parameters.
    maskd_sub:
        Indices of diagonal blocks in the subsystem.
    mask_sub:
        Indices of off-diagonal blocks in the subsystem.

    Returns
    -------
    F0 : ArrayLike
        The Fock matrix.
    '''

    idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
    max_key = max(idx_to_idx_mapping.keys())


    lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long, device = P_sub.device)

    # Populate the lookup tensor
    for key, value in idx_to_idx_mapping.items():
        lookup_tensor[key] = value


    max_i = idxi.max()
    max_j = idxj.max()
    atom_max = max(max_i,max_j)

    in_block_mask = torch.zeros(atom_max+1,dtype=torch.bool, device = P_sub.device)
    in_block_mask[block_indices]=True

    isini = in_block_mask[idxi]#.to(torch.bool)
    where_isini = torch.nonzero(isini).squeeze()
    isinj = in_block_mask[idxj]#.to(torch.bool)
    where_isinj = torch.nonzero(isinj).squeeze()

    loc_i = idxi[isini]
    loc_j = idxj[isinj]

    ### first doing idxi because its sorted
    # idxi_sub_ovrlp_with_rest = torch.isin(idxi, block_indices) # <- instead of this
    # Searchsorted gives you the indices where the elements should be placed to maintain order. Works with idxi (sorted) but not with idxj (not sorted)
    pos = torch.searchsorted(block_indices, idxi)
    # Ensure the indices are within bounds
    pos = torch.clamp(pos, max=len(block_indices) - 1)
    # Check if the positions are valid and match
    idxi_sub_ovrlp_with_rest = (pos < len(block_indices)) & (block_indices[pos] == idxi)

    ### second, doing indx i because its a sequence of sorted maxtrix triangle rows
    #     idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices) # <- instead of this
    # start_ind = 0
    # end_ind = len(P) - 1
    # idxj_sub_ovrlp_with_rest = torch.zeros(int((len(P)*(len(P)-1)/2)), dtype=torch.bool, device=P.device)
    # tmp_j = idxj[start_ind:end_ind]
    # pos = torch.searchsorted(block_indices, tmp_j)
    # pos = torch.clamp(pos, max=len(block_indices) - 1)
    # valid_top_row = (pos < len(block_indices)) & (block_indices[pos] == tmp_j)
    # del tmp_j, pos
    # for i in range(0,len(P)): ### $$$ needs vecorization
    #     idxj_sub_ovrlp_with_rest[start_ind:end_ind] = valid_top_row[i:]
    #     start_ind = end_ind
    #     end_ind = end_ind + len(P) - i - 2
    idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices)
    
    ### Populate diagonal 1c ###
    F = M.clone()
    Pptot = P_sub[...,1,1]+P_sub[...,2,2]+P_sub[...,3,3]
    TMP = torch.zeros_like(M)
    TMP[maskd_sub,0,0] = 0.5*P_sub[maskd_sub,0,0]*parameters['g_ss'][block_indices] + Pptot[maskd_sub]*(parameters['g_sp'][block_indices]-0.5*parameters['h_sp'][block_indices])
    for i in range(1,4):
        #(p,p)
        TMP[maskd_sub,i,i] = P_sub[maskd_sub,0,0]*(parameters['g_sp'][block_indices]-0.5*parameters['h_sp'][block_indices]) + 0.5*P_sub[maskd_sub,i,i]*parameters['g_pp'][block_indices] \
                        + (Pptot[maskd_sub] - P_sub[maskd_sub,i,i]) * (1.25*parameters['g_p2'][block_indices]-0.25*parameters['g_pp'][block_indices])
        #(s,p) = (p,s) upper triangle
        TMP[maskd_sub,0,i] = P_sub[maskd_sub,0,i]*(1.5*parameters['h_sp'][block_indices] - 0.5*parameters['g_sp'][block_indices])
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        TMP[maskd_sub,i,j] = P_sub[maskd_sub,i,j]* (0.75*parameters['g_pp'][block_indices] - 1.25*parameters['g_p2'][block_indices])

    F.add_(TMP)
    del TMP, Pptot

    ##############################################
    ### Populate diagonal 2c ###
    dtype = P.dtype
    device = P.device
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    PA = (P[idxi[idxj_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
    PB = (P[idxj[idxi_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))

    #w_2_inj=w_2[where_isinj]
    suma = torch.einsum('ijk,ijk->ik',PA,w_2[where_isinj])
    sumA = torch.zeros(torch.sum(isinj),4,4,dtype=dtype, device=device)
    sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
    jjj = lookup_tensor[loc_j]
    indj_of_new_diag_in_old = maskd_sub[jjj]
    F.index_add_(0,indj_of_new_diag_in_old, sumA)
    del PA, where_isinj, suma, jjj, loc_j, indj_of_new_diag_in_old, sumA

    #w_2_ini=w_2[where_isini]
    sumb = torch.einsum('ijk,ijk->ij',PB,w_2[where_isini])
    sumB = torch.zeros(torch.sum(isini),4,4,dtype=dtype, device=device)
    sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
    iii=lookup_tensor[loc_i]
    indi_of_new_diag_in_old = maskd_sub[iii]
    F.index_add_(0,indi_of_new_diag_in_old, sumB)
    del PB, where_isini, sumb, iii, loc_i, indi_of_new_diag_in_old, sumB

    ####################################################
    ### Populate off-diagonal ###
    sub_inds = idxi_sub_ovrlp_with_rest * idxj_sub_ovrlp_with_rest

    summ = torch.zeros(w_2[sub_inds].shape[0],4,4,dtype=dtype, device=device)
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)

    # Pp =P[mask], P_{mu \in A, lambda \in B}
    Pp = -0.5*P_sub[mask_sub] #* (rij.unsqueeze(-1).unsqueeze(-1) < 2.5) #*(rij > 2.0)
    w2_sub_inds=w_2[sub_inds]

    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            a1=w2_sub_inds[...,ind[i],:][...,:,ind[j]]
            summ[...,i,j] = torch.einsum('ijk,ijk->i',Pp,a1)#torch.sum(Pp*a1,dim=(1,2))
    del Pp
    F.index_add_(0,mask_sub,summ)
    del summ

    F0 = F.reshape(nmol,len(block_indices),len(block_indices),4,4).transpose(2,3) \
                     .reshape(nmol, 4*len(block_indices), 4*len(block_indices))
    F0.add_(F0.triu(1).transpose(1,2));       

    return F0

def get_fock_pyseqm_u(P: ArrayLike,
                      P_sub: ArrayLike,
                      M: ArrayLike,
                      w_2: ArrayLike,
                      block_indices: ArrayLike,
                      nmol: int,
                      idxi: ArrayLike,
                      idxj: ArrayLike,
                      rij: ArrayLike,
                      parameters: ArrayLike,
                      maskd_sub: ArrayLike,
                      mask_sub: ArrayLike):
    '''

    Returns the Fock matrix in the PYSEQM format (unrestricted case), with padding on the
    H-atoms, s.t. they are in the 4x4 block format.

    Parameters
    ----------
    P: ArrayLike
        Diagonal density matrix blocks of the whole system.
    P_sub: ArrayLike
        Subsystem density matrices.
    M: ArrayLike
        One electron Hamiltonian of the subsystem.
    w_2: ArrayLike
        Two-center, two-electron integrals.
        These are compute for:
            subsystem-subsystem
            subsystem-outer
            NOT outer-outer
    block_indices: ArrayLike
        Subsystem atom numbers.
    nmol: int
        Number of molecules in a batch. Always 1 in SEDACS.
    idxi: ArrayLike
        Unique pairs between atoms in subsystem or between an atom in subsystem and in the outer system.
    idxj: ArrayLike
        Unique pairs between atoms in subsystem or between an atom in subsystem and in the outer system.
    rij: ArrayLike
        Distances for unique pairs (corresponding to those specified in idxi, and idxj above.
    parameters:
        Seqm atomic parameters.
    maskd_sub:
        Indices of diagonal blocks in the subsystem.
    mask_sub:
        Indices of off-diagonal blocks in the subsystem.

    Returns
    -------
    F0 : ArrayLike
        The Fock matrix.
    '''

    idx_to_idx_mapping = {value: idx for idx, value in enumerate(block_indices)}
    max_key = max(idx_to_idx_mapping.keys())
    lookup_tensor = torch.zeros(max_key + 1, dtype=torch.long, device = P_sub.device)
    # Populate the lookup tensor
    for key, value in idx_to_idx_mapping.items():
        lookup_tensor[key] = value
    max_i = idxi.max()
    max_j = idxj.max()
    atom_max = max(max_i,max_j)
    in_block_mask = torch.zeros(atom_max+1,dtype=torch.bool, device = P_sub.device)
    in_block_mask[block_indices]=True

    isini = in_block_mask[idxi]#.to(torch.bool)
    where_isini = torch.nonzero(isini).squeeze()

    isinj = in_block_mask[idxj]#.to(torch.bool)
    where_isinj = torch.nonzero(isinj).squeeze()

    loc_i = idxi[isini]
    loc_j = idxj[isinj]

    ### first doing idxi because its sorted
    #     idxi_sub_ovrlp_with_rest = torch.isin(idxi, block_indices) # <- insted of this
    # Searchsorted gives you the indices where the elements should be placed to maintain order. Works with idxi (sorted) but not with idxj (not sorted)
    pos = torch.searchsorted(block_indices, idxi)
    # Ensure the indices are within bounds
    pos = torch.clamp(pos, max=len(block_indices) - 1)
    # Check if the positions are valid and match
    idxi_sub_ovrlp_with_rest = (pos < len(block_indices)) & (block_indices[pos] == idxi)

    ### second, doing indx i because its a sequence of sorted maxtrix triangle rows
    #     idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices) # <- instead of this
    # start_ind = 0
    # end_ind = len(P) - 1
    # idxj_sub_ovrlp_with_rest = torch.zeros(int((len(P)*(len(P)-1)/2)), dtype=torch.bool, device=P.device)
    # tmp_j = idxj[start_ind:end_ind]
    # pos = torch.searchsorted(block_indices, tmp_j)
    # pos = torch.clamp(pos, max=len(block_indices) - 1)
    # valid_top_row = (pos < len(block_indices)) & (block_indices[pos] == tmp_j)
    # del tmp_j, pos
    # for i in range(0,len(P)): ### $$$ needs vecorization
    #     idxj_sub_ovrlp_with_rest[start_ind:end_ind] = valid_top_row[i:]
    #     start_ind = end_ind
    #     end_ind = end_ind + len(P) - i - 2
    idxj_sub_ovrlp_with_rest = torch.isin(idxj, block_indices)

    P_sum_ab = P_sub[0]+P_sub[1]

    PAlpha_ = P_sub

    
    ### Populate diagonal 1c ###
    F = M.expand(2,-1,-1,-1).clone()
    Pptot = P_sum_ab[...,1,1]+P_sum_ab[...,2,2]+P_sum_ab[...,3,3]
    PAlpha_ptot_ = PAlpha_[...,1,1]+PAlpha_[...,2,2]+PAlpha_[...,3,3]
    #(s,s)
    TMP = torch.zeros_like(F)
    TMP[:,maskd_sub,0,0] = PAlpha_[[1,0]][:,maskd_sub,0,0]*parameters['g_ss'][block_indices] + Pptot[maskd_sub]*parameters['g_sp'][block_indices] - PAlpha_ptot_[:,maskd_sub]*parameters['h_sp'][block_indices]
    for i in range(1,4):
        #(p,p)
        TMP[:,maskd_sub,i,i] = P_sum_ab[maskd_sub,0,0]*parameters['g_sp'][block_indices]-PAlpha_[:,maskd_sub,0,0]*parameters['h_sp'][block_indices] + PAlpha_[[1,0]][:,maskd_sub,i,i]*parameters['g_pp'][block_indices] \
                        +(Pptot[maskd_sub]-P_sum_ab[maskd_sub,i,i])*parameters['g_p2'][block_indices] - 0.5*(PAlpha_ptot_[:,maskd_sub]-PAlpha_[:,maskd_sub,i,i])*(parameters['g_pp'][block_indices]-parameters['g_p2'][block_indices])

        #(s,p) = (p,s) upper triangle
        TMP[:,maskd_sub,0,i] = 2*P_sum_ab[maskd_sub,0,i]*parameters['h_sp'][block_indices] - PAlpha_[:,maskd_sub,0,i]*(parameters['h_sp'][block_indices]+parameters['g_sp'][block_indices])
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        TMP[:,maskd_sub,i,j] = P_sum_ab[maskd_sub,i,j] * (parameters['g_pp'][block_indices] - parameters['g_p2'][block_indices]) - 0.5*PAlpha_[:,maskd_sub,i,j]*(parameters['g_pp'][block_indices] + parameters['g_p2'][block_indices])

    F.add_(TMP)
    del TMP, Pptot, PAlpha_ptot_

    ##############################################
    ### Populate diagonal 2c ###
    dtype = P.dtype
    device = P.device
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    PA = ((P[0]+P[1])[idxi[idxj_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
    PB = ((P[0]+P[1])[idxj[idxi_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))

    #w_2_inj=w_2[where_isinj]
    suma = torch.sum(PA*w_2[where_isinj],dim=1)
    sumA = torch.zeros(torch.sum(isinj),4,4,dtype=dtype, device=device)
    sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
    jjj = lookup_tensor[loc_j]
    indj_of_new_diag_in_old = maskd_sub[jjj]
    F[0].index_add_(0,indj_of_new_diag_in_old, sumA)
    F[1].index_add_(0,indj_of_new_diag_in_old, sumA)
    del PA, where_isinj, suma, jjj, loc_j, indj_of_new_diag_in_old, sumA

    #w_2_ini=w_2[where_isini]
    sumb = torch.einsum('ijk,ijk->ij',PB,w_2[where_isini])
    sumB = torch.zeros(torch.sum(isini),4,4,dtype=dtype, device=device)
    sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
    iii=lookup_tensor[loc_i]
    indi_of_new_diag_in_old = maskd_sub[iii]
    F[0].index_add_(0,indi_of_new_diag_in_old, sumB)
    F[1].index_add_(0,indi_of_new_diag_in_old, sumB)
    del PB, where_isini, sumb, iii, loc_i, indi_of_new_diag_in_old, sumB

    ####################################################
    ### Populate off-diagonal ###
    sub_inds = idxi_sub_ovrlp_with_rest * idxj_sub_ovrlp_with_rest

    summ = torch.zeros(2, w_2[sub_inds].shape[0],4,4,dtype=dtype, device=device)
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)

    # Pp =P[mask], P_{mu \in A, lambda \in B}
    w2_sub_inds=w_2[sub_inds]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            # a1=w2_sub_inds[...,ind[i],:][...,:,ind[j]]
            # summ[...,i,j] = torch.einsum('ijk,ijk->i',-PAlpha_,a1)#torch.sum(Pp*a1,dim=(1,2))
            summ[...,i,j] = torch.sum(-PAlpha_[:,mask_sub]*w2_sub_inds[...,ind[i],:][...,:,ind[j]],dim=(2,3))

    F.index_add_(1,mask_sub,summ)
    del summ

    F0 = F.reshape(2, nmol, len(block_indices), len(block_indices), 4, 4) \
                 .transpose(3,4) \
                 .reshape(2, nmol, 4*len(block_indices), 4*len(block_indices)).transpose(0,1)
    
    F0.add_(F0.triu(1).transpose(2,3))
    return F0


def get_hcore_pyseqm(coords,
                     symbols,
                     atomTypes,
                     device='cpu',
                     verb=False) -> tuple[ArrayLike, ArrayLike, Molecule, ArrayLike, ArrayLike]:
  """
  Get the core Hamiltonian from PYSEQM. Interall calls the PYSEQM hcore routine.
  TODO: Add support for user-defined PYSEQM parameters.

  Parameters
  ----------
  coords: ArrayLike (Natoms, 3)
      The Cartesian coordinates for all atoms in the system.
  symbols: ArrayLike
      The unique chemical elements in the structure.
  atomTypes: ArrayLike (Natoms, )
      The element type of each atom in the system.
  device: str
      PyTorch device.
  verb: bool
      Flag for verbose output.

  Returns
  -------
  M: ArrayLike
    One-electron hamiltonian
  w: ArrayLike
    Two-center, two-electron integrals.
  molecule: Molecule
    PYSEQM Molecule Object
  rho0xi: ArrayLike
  rho0xj: ArrayLike
  """

  print('Creating Hcore.')
  if(PYSEQM == False):
    print("ERROR: No PySEQM installed")
  symbols_internal = np.array([ "Bl" ,                               
      "H" ,                                     "He",        
      "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",          \
      "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar",
      ], dtype=str)
  numel_internal = np.zeros(len(symbols_internal),dtype=int)
  numel_internal[:] = 0,   \
      1 ,                  2,   \
      1 ,2 ,3 ,4 ,5 ,6 ,7, 8,   \
      1 ,2 ,3 ,4 ,5 ,6 ,7, 8,

  bas_per_atom = np.zeros(len(symbols_internal),dtype=int)
  bas_per_atom[:] =   0,   \
      1 ,1 ,\
      4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
      4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
  
  # Map symbols to indices in symbols_internal
  symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}

  # Translate `symbols` to `symbols_internal` indices
  mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])

  # Convert atomTypes to `symbols_internal` indices
  atom_internal_indices = mapped_indices[atomTypes]

  species = torch.as_tensor(np.array([atom_internal_indices,]), dtype=torch.int64, device=device)
  coordinates = torch.tensor(np.array([coords,]), device=device, dtype=torch.float64)
  const = Constants().to(device)
  elements = [0]+sorted(set(species.reshape(-1).tolist()))
  seqm_parameters = {
                    'method' : 'PM6_SP',  # AM1, MNDO, PM3, PM6, PM6_SP. PM6_SP is PM6 without d-orbitals. Effectively, PM6 for the first two rows of periodic table
                    'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                    'scf_converger' : [0,0.8,0.93,30], # converger used for scf loop
                                          # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                          # [1], adaptive mixing
                                          # [2], adaptive mixing, then pulay
                    'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                              #[True, eps] or [False], eps for SP2 conve criteria
                    'elements' : elements, #[0,1,6,8],
                    'learned' : [], # learned parameters name list, e.g ['U_ss']
                    #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                    'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                    'eig' : True, # store orbital energies
                    }
  molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
  molecule.coordinates.requires_grad_(True)

  torch.cuda.empty_cache()
  M, w, rho0xi, rho0xj = hcore(molecule)

  return M, w, molecule, rho0xi, rho0xj

def get_molecule_pyseqm(sdc: Input,
                        coords: ArrayLike,
                        symbols: ArrayLike,
                        atomTypes: ArrayLike,
                        do_large_tensors: bool = True,
                        device: str = 'cpu',
                        verb: bool = False) -> tuple[Molecule, int]:
  '''
  Function returns pyseqm molecule object for SEDACS

  Parameters
  ----------
  sdc: Input
      The SEDACS driver.
  coords: ArrayLike (Natoms, 3)
      The Cartesian coordinates for all atoms in the system.
  symbols: ArrayLike
      The unique chemical elements in the structure.
  atomTypes: ArrayLike (Natoms, )
      The element type of each atom in the system.
  do_large_tensors: bool
      If False, PySEQM won't calculate large tensors like idxi, idxj, rij, xij, mask.
  device: str
      PyTorch device.
  verb: bool
      Flag for verbose output.

  Returns
  -------

  molecule: Molecule
    The PYSEQM Molecule object.
  molecule.nocc: int
    The number of occupied states in the corresponding Molecule.
  '''

  # move to a sep file $$$
  torch.cuda.empty_cache()
  if(PYSEQM == False):
    print("ERROR: No PYSEQM installed")

  symbols_internal = np.array([ "Bl" ,                               
      "H" ,                                     "He",        
      "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",          \
      "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar",
      ], dtype=str)
  numel_internal = np.zeros(len(symbols_internal),dtype=int)
  numel_internal[:] = 0,   \
      1 ,                  2,   \
      1 ,2 ,3 ,4 ,5 ,6 ,7, 8,   \
      1 ,2 ,3 ,4 ,5 ,6 ,7, 8,

  bas_per_atom = np.zeros(len(symbols_internal),dtype=int)
  bas_per_atom[:] =   0,   \
      1 ,1 ,\
      4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
      4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
  
  # Map symbols to indices in symbols_internal
  symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}
  # Translate `symbols` to `symbols_internal` indices
  mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])
  # Convert atomTypes to `symbols_internal` indices
  atom_internal_indices = mapped_indices[atomTypes]
  if sdc.torch_dt == torch.float64:
    dtype_int = torch.int64
  else:
    dtype_int = torch.int32
  species = torch.as_tensor(np.array([atom_internal_indices,]), dtype=dtype_int, device=device)
  
  if torch.is_tensor(coords):
    coordinates = coords
  else:
    coordinates = torch.tensor(np.array([coords]), device=device, dtype=sdc.torch_dt)
 
  const = Constants().to(device)
  elements = [0]+sorted(set(species.reshape(-1).tolist()))
  seqm_parameters = {
                    'method' : 'PM6_SP',  # AM1, MNDO, PM3, PM6, PM6_SP. PM6_SP is PM6 without d-orbitals. Effectively, PM6 for the first two rows of periodic table
                    'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                    'scf_converger' : [0,0.8,0.93,30], # converger used for scf loop
                                          # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                          # [1], adaptive mixing
                                          # [2], adaptive mixing, then pulay
                    'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                              #[True, eps] or [False], eps for SP2 conve criteria
                    'elements' : elements, #[0,1,6,8],
                    'learned' : [], # learned parameters name list, e.g ['U_ss']
                    #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                    'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                    'eig' : True, # store orbital energies
                    'UHF': sdc.UHF,
                    }

  # Charge does not matter. Assigned to avoid internal pyseqm error.
  if torch.sum(species)%2 == 0:
     charges = 0
  else:
     charges = -1
     
  molecule = Molecule(const, seqm_parameters, coordinates, species, charges=charges, do_large_tensors=do_large_tensors).to(device)  
  return molecule, molecule.nocc


def get_eVals_pyseqm(sdc: Input,
                     H: ArrayLike,
                     Nocc: int,
                     core_indices_in_sub_expanded_packed: ArrayLike,
                     molecule: Molecule,
                     verb: bool = False,
                     calcD: bool = False) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
  '''
  Function returns eigenvalues, dVals, eigenvectors, list of [number_of_heavy_atoms, number_of_hydrogens, dim_of_coreHalo_ham].


  Parameters
  ----------
  H: ArrayLike
    Hamiltonian in PYSEQM 4x4 block format.
  Nocc: int
    Number of occupied states.
  core_indices_in_sub_expanded_packed: ArrayLike
    Core indices of core+halo hamiltonian in normal form corresponding to the number of AOs per atom

  Returns
  -------
  E_val: ArrayLike
    Eigenvalues.
  dVals: ArrayLike
    Norm over the core parts of the eigenvectors, Q.
  Q: ArrayLike
    Eigenvectors.
  '''
  if(verb): print("Computing eVals/dVals")

  E_val, Q = sym_eig_trunc( H, molecule.nHeavy, molecule.nHydro, Nocc, eig_only=True)
  Q = Q[0]

  if sdc.UHF: # open shell
    N = Q.shape[-1]
    dVals = torch.sum(Q[:, core_indices_in_sub_expanded_packed, :] ** 2, dim=1)
    return E_val[0,:,:N], dVals.cpu().numpy(), Q
  else: # closed shell
    N = len(Q)
    E_val = E_val[0,:N]

    #homoIndex = Nocc - 1
    #lumoIndex = Nocc
    #mu_test = 0.5*(E_val[homoIndex] + E_val[lumoIndex]) #don't need it 
    #print(' SubSys HOMO/LUMO:', np.round(E_val[homoIndex].item(),4), np.round(E_val[lumoIndex].item(),4), end=" ")

    # rho = Q@f_vector@Q.T
    # or
    # rho_ij = SUM_k Q_ik * f_kk * Q_jk
    dVals = torch.sum(Q[core_indices_in_sub_expanded_packed, :] ** 2, dim=0)
    return E_val, dVals.cpu().numpy(), Q


def get_densityMatrix_renormalized_pyseqm(sdc: Input,
                                          E_val: ArrayLike,
                                          Q: ArrayLike,
                                          Tel: Union[float, torch.Tensor],
                                          mu0: Union[float, torch.Tensor],
                                          NH_Nh_Hs: list) -> ArrayLike:
  '''
  Returns the density matrix corrected by fermi occupancies.

  Parameters
  ----------
  E_val: ArrayLike
    Eigenvalues
  Q: ArrayLike
    Eigenvectors
  Tel: float or single element torch.Tensor
    Electronic temperature (in Kelvin).
  mu0: float or single element torch.Tensor
    Chemical potential (in eV).
  NH_Nh_Hs: list
    [number_of_heavy_atoms, number_of_hydrogens, dim_of_coreHalo_ham]
    Needed for handling the special indexing of hydrogens in PYSEQM.
  '''
  
  kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
  beta = 1./(kB*Tel)

  
  # two lines below are vectorization of this: D = 2*sum(torch.outer(Q[:, i],Q[:, i]*f[i]) for i in range(Nocc))
  if sdc.UHF:
    f = (1/(torch.exp(beta*(E_val - np.expand_dims(mu0, axis=1))) + 1)).unsqueeze(1)
    Q_weighted = Q * f  # Broadcasting multiplication
    D = Q @ Q_weighted.transpose(-2,-1)
    D = unpack(D, NH_Nh_Hs[0].repeat_interleave(2), NH_Nh_Hs[1].repeat_interleave(2), NH_Nh_Hs[2])
  else:
    f = 1/(torch.exp(beta*(E_val - mu0)) + 1)
    Q_weighted = Q * f  # Broadcasting multiplication
    D = 2 * Q @ Q_weighted.T
    D = unpack(D, NH_Nh_Hs[0], NH_Nh_Hs[1], NH_Nh_Hs[2])

  return D


def get_overlap_pyseqm(coords: ArrayLike,
                       symbols: ArrayLike,
                       atomTypes: ArrayLike,
                       hindex: ArrayLike,
                       verb=False) -> ArrayLike:
    '''

    Gets the overlap matrix from PYSEQM. Returned in a packed form (NOT 4x4 blocks pyseqm format.)

    Parameters
    ----------
    coords: ArrayLike (Natoms, 3)
        The Cartesian coordinates in the system of interest.
    symbols: ArrayLike
        The unique chemical elements in the structure.
    atomTypes: ArrayLike (Natoms, )
        The element type of each atom in the system.
    hindex_sub: ArrayLike
        Orbital index for each atom in the CH (in CH numbering)
    verb: bool
        Boolean flag for verbose output.

    Returns
    -------
    di_full[0]: ArrayLike
       The packed (e.g. not 4x4 even for Hydrogens) overlap matrix.
    '''

    # COHO
    # symbols: (C, O, H)
    # atomsTypes (0,1,2,1)
    # construc dict: 
    if(PYSEQM == False):
        print("ERROR: No PySEQM installed")
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    import numpy as np
    from seqm.seqm_functions.pack import pack
    from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
    from seqm.seqm_functions.constants import overlap_cutoff
    seqm.seqm_functions.scf_loop.debug=False

    device = torch.device('cpu')
    symbols_internal = np.array([ "Bl" ,                               
        "H" ,                                     "He",        
        "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",          \
        "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar",
        ], dtype=str)
    numel_internal = np.zeros(len(symbols_internal),dtype=int)
    numel_internal[:] = 0,   \
        1 ,                  2,   \
        1 ,2 ,3 ,4 ,5 ,6 ,7, 8,   \
        1 ,2 ,3 ,4 ,5 ,6 ,7, 8,

    bas_per_atom = np.zeros(len(symbols_internal),dtype=int)
    bas_per_atom[:] =   0,   \
        1 ,1 ,\
        4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \
        4 ,4 ,4 ,4 ,4 ,4 ,4 , 4,  \

    # Map symbols to indices in symbols_internal
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbols_internal)}
    # Translate `symbols` to `symbols_internal` indices
    mapped_indices = np.array([symbol_to_index[symbol] for symbol in symbols])
    # Convert atomTypes to `symbols_internal` indices
    atom_internal_indices = mapped_indices[atomTypes]
    # Vectorized approach to combine the arrays
    combined_array = np.column_stack((atom_internal_indices[:, np.newaxis], coords)).tolist()
    # Convert to the desired format
    molecule_elem_coord = [[int(item[0]), tuple(item[1:])] for item in combined_array]


    species = torch.as_tensor(np.array([np.array(atom_internal_indices)]), dtype=torch.int64, device=device)
    coordinates = torch.tensor(np.array([coords]), device=device, dtype=torch.float64)
    const = Constants().to(device)
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    seqm_parameters = {
                    'method' : 'PM6_SP',  # AM1, MNDO, PM3, PM6, PM6_SP. PM6_SP is PM6 without d-orbitals. Effectively, PM6 for the first two rows of periodic table
                    'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                    'scf_converger' : [0,0.8,0.93,30], # converger used for scf loop
                                            # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                            # [1], adaptive mixing
                                            # [2], adaptive mixing, then pulay
                    'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                                #[True, eps] or [False], eps for SP2 conve criteria
                    'elements' : elements, #[0,1,6,8],
                    'learned' : [], # learned parameters name list, e.g ['U_ss']
                    #'parameter_file_dir' : '../seqm/params/', # file directory for other required parameters
                    'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                    'eig' : True # store orbital energies
                    }
    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    dtype = molecule.xij.dtype
    device = molecule.xij.device
    qn_int = molecule.const.qn_int
    qnD_int = molecule.const.qnD_int

    if(molecule.method == 'PM6'):
        zeta = torch.cat((molecule.parameters['zeta_s'].unsqueeze(1), molecule.parameters['zeta_p'].unsqueeze(1), molecule.parameters['zeta_d'].unsqueeze(1)),dim=1)
    else:
        zeta = torch.cat((molecule.parameters['zeta_s'].unsqueeze(1), molecule.parameters['zeta_p'].unsqueeze(1)),dim=1)
    overlap_pairs = molecule.rij<=overlap_cutoff

    if molecule.method == 'PM6_SP':
        di = torch.zeros((molecule.xij.shape[0], 4, 4),dtype=dtype, device=device)
        di[overlap_pairs] = diatom_overlap_matrix_PM6_SP(molecule.ni[overlap_pairs],
                                molecule.nj[overlap_pairs],
                                molecule.xij[overlap_pairs],
                                molecule.rij[overlap_pairs],
                                zeta[molecule.idxi][overlap_pairs],
                                zeta[molecule.idxj][overlap_pairs],
                                qn_int)
    
    di_full = torch.zeros((molecule.nmol*molecule.molsize*molecule.molsize, 4, 4),dtype=dtype, device=device)
    mask_H = molecule.Z==1
    mask_heavy = molecule.Z>1
    H_self_ovr = torch.zeros((4,4), dtype=dtype, device=device)
    H_self_ovr[0,0] = 1.0

    di_full[molecule.maskd[mask_H]] = H_self_ovr
    di_full[molecule.maskd[mask_heavy]] = torch.eye(4, dtype=dtype, device=device)
    di_full[molecule.mask] = di
    di_full[molecule.mask_l] = di.transpose(1,2)
    di_full = di_full.reshape(molecule.nmol,molecule.molsize,molecule.molsize,4,4).transpose(2,3) \
                 .reshape(molecule.nmol, 4*molecule.molsize, 4*molecule.molsize)
    di_full = pack(di_full, molecule.nHeavy, molecule.nHydro)
    return di_full[0]

def get_diag_guess_pyseqm(molecule: Molecule,
                          sy,
                          verb: bool = False) -> ArrayLike:
    '''
    Initial diagonal guess for the density matrix.

    Parameters
    ----------
    molecule: Molecule
        PYSEQM Molecule object.
    sy: 
    verb: bool
        Controls verbosity of output.
    '''
    tore = molecule.const.tore
    method = 'PM6_SP'
    if method == 'PM6':
        P0 = torch.zeros(sy.nats,9,9,dtype=molecule.coordinates.dtype, device=tore.device)  # density matrix
        P0[molecule.Z>1,0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[:,1,1] = P0[:,0,0]
        P0[:,2,2] = P0[:,0,0]
        P0[:,3,3] = P0[:,0,0]
        P0[molecule.Z==1,0,0] = 1.0
    else:
        P0 = torch.zeros(sy.nats,4,4,dtype=molecule.coordinates.dtype, device=tore.device)  # density matrix
        P0[molecule.Z>1,0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[:,1,1] = P0[:,0,0]
        P0[:,2,2] = P0[:,0,0]
        P0[:,3,3] = P0[:,0,0]
        P0[molecule.Z==1,0,0] = 1.0        

    return P0


    
