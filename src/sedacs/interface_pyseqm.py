import numpy as np
import sys 
import scipy.linalg as sp
from sedacs.system import get_hindex
import sys

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
  import torch
  import time

except: PYSEQM = False


import scipy
from scipy.linalg import fractional_matrix_power

class pyseqmObjects(torch.nn.Module):
    def __init__(self, sdc, coords, symbols,atomTypes, do_large_tensors=True, device='cpu'):
        """
        Constructor
        """
        super().__init__()
        # self.M_whole, self.w_whole, self.molecule_whole, self.rho0xi_whole, self.rho0xj_whole = \
        #     get_hcore_pyseqm(coords, symbols, atomTypes)
        
        self.M_whole, self.w_whole = None, None
        self.molecule_whole = get_molecule_pyseqm(sdc, coords, symbols, atomTypes, do_large_tensors=do_large_tensors, device=device)[0].to(device)
        if do_large_tensors:
          self.w_ssss = torch.zeros_like(self.molecule_whole.idxi)
          #print('Creating DM guess.')
          #make_dm_guess(self.molecule_whole, self.molecule_whole.seqm_parameters, mix_homo_lumo=False, mix_coeff=0.3, overwrite_existing_dm=True);

          ev = 27.21
          rho_0 = 0.5*ev/self.molecule_whole.parameters['g_ss']
          self.rho0xi_whole = rho_0[self.molecule_whole.idxi].clone()
          self.rho0xj_whole = rho_0[self.molecule_whole.idxj].clone()
          A = (self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxi] != 0.000)
          B = (self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxj] != 0.000)
          self.rho0xi_whole[A] =self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxi][A]
          self.rho0xj_whole[B] =self.molecule_whole.parameters['rho_core'][self.molecule_whole.idxj][B]




def get_coreHalo_ham_inds(partIndex, partCoreHaloIndex, sdc, sy, subSy, device='cpu'):
    
    #indices_in_sub = np.linspace(0,len(partCoreHaloIndex)-1, len(partCoreHaloIndex), dtype = sdc.torch_int_dt)
    indices_in_sub = torch.linspace(0, len(partCoreHaloIndex) - 1, len(partCoreHaloIndex), dtype=sdc.torch_int_dt, device=device)

    #core_indices_in_sub = indices_in_sub[np.isin(partCoreHaloIndex, partIndex)]
    core_indices_in_sub = indices_in_sub[torch.isin(torch.tensor(partCoreHaloIndex, device=device), torch.tensor(partIndex, device=device))] # $$$ torch.searchsorted might be better

    block_size = 4
    # Generate the expanded indices for each block
    #base_indices = np.arange(block_size)  # Create a base index tensor of size block_size
    base_indices = torch.arange(block_size, dtype=sdc.torch_int_dt, device=device)  # Create a base index tensor of size block_size

    #core_indices_in_sub_expanded = np.expand_dims(core_indices_in_sub, axis=1) * block_size + base_indices  # Broadcast and add
    core_indices_in_sub_expanded = core_indices_in_sub.unsqueeze(1) * block_size + base_indices

    #core_indices_in_sub_expanded = core_indices_in_sub_expanded.flatten()  # Flatten the result
    core_indices_in_sub_expanded = core_indices_in_sub_expanded.flatten()

    #core_indices_in_sub_expanded = torch.from_numpy(core_indices_in_sub_expanded)
    norbs, norbs_for_every_type, hindex_sub, numel = get_hindex(sdc.orbs, sdc.valency, sy.symbols, subSy.types)
    hindex_sub = torch.from_numpy(hindex_sub).to(device, dtype=sdc.torch_int_dt)

    
    # Given tensor of block indices and block size
    coreHalo_rows_in_whole = torch.tensor(partCoreHaloIndex, dtype=sdc.torch_int_dt)
    block_size = 4
    # Generate the expanded indices for each block
    base_indices = torch.arange(block_size)  # Create a base index tensor of size block_size
    coreHalo_rows_in_whole_expanded = coreHalo_rows_in_whole.unsqueeze(1) * block_size + base_indices  # Broadcast and add
    coreHalo_rows_in_whole_expanded = coreHalo_rows_in_whole_expanded.flatten()  # Flatten the result

    # Given tensor of block indices and block size
    core_cols_in_whole = torch.tensor(partIndex, dtype=sdc.torch_int_dt)
    # Generate the expanded indices for each block
    base_indices = torch.arange(block_size)  # Create a base index tensor of size block_size
    core_cols_in_whole_expanded = core_cols_in_whole.unsqueeze(1) * block_size + base_indices  # Broadcast and add
    core_cols_in_whole_expanded = core_cols_in_whole_expanded.flatten()  # Flatten the result
    
    
    #I_core = torch.meshgrid(core_cols_in_whole_expanded, core_cols_in_whole_expanded, indexing='ij')
    if sdc.reconstruct_dm:
        I = torch.meshgrid(coreHalo_rows_in_whole_expanded, core_cols_in_whole_expanded, indexing='ij', device=device)
        I_halo = torch.meshgrid(coreHalo_rows_in_whole_expanded, coreHalo_rows_in_whole_expanded, indexing='ij', device=device)
    else:
        I = None
        I_halo = None
    return core_indices_in_sub, core_indices_in_sub_expanded, hindex_sub, I, I_halo

def get_elec_energy_pyseqm(P,F,Hcore, doTriu=True):
   return elec_energy(P,F,Hcore, doTriu=doTriu)

def get_nucAB_energy_pyseqm(Z, const, nmol, ni, nj, idxi, idxj, rij, \
                                     rho0xi,rho0xj,alp, chi, gam, method, parnuc):
   
   return pair_nuclear_energy(Z, const, nmol, ni, nj, idxi, idxj, rij, \
                                     rho0xi,rho0xj,alp, chi, gam=gam, method=method, parameters=parnuc)

def get_total_energy_pyseqm(nmol, pair_molid, EnucAB, Eelec):
   return total_energy(nmol, pair_molid, EnucAB, Eelec)
   

def get_full_fock_pyseqm(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,
         themethod, zetas, zetap, zetad, Z, F0SD, G2SD):
    
    return fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp,
         themethod, zetas, zetap, zetad, Z, F0SD, G2SD)

def get_fock_pyseqm_2(P, P_sub, M, w_2, block_indices, nmol, idxi, idxj, rij, parameters, maskd_sub, mask_sub):
    ### optimized version by Nick. 3x faster ###
    # P: diagonal dm blocks of the whole system
    # P_sub: subsystem dm
    # M: 1elec hamiltonian of subsystem
    # w_2: 2c2e ints. subsystem-subsystem and subsystem-outer. no outer-outer
    # block_indices: subsystem atom numbers
    # nmol: number of molecules in a batch. Always 1 in SEDACS.
    # idxi, idxj: unique pairs between atoms in subsystem or between an atom in subsystem and in the outer system.
    # rij: distances for unique pairs
    # parameters: seqm atomic params
    # maskd_sub: indices of diagonal blocks in subsystem
    # mask_sub: indices of off-diagonal blocks in subsystem

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

    dtype = P.dtype
    device = P.device
    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))

    PA_test = (P[idxi[idxj_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
    PB_test = (P[idxj[idxi_sub_ovrlp_with_rest]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))
    
    w_2_inj=w_2[where_isinj]
    suma_test = torch.einsum('ijk,ijk->ik',PA_test,w_2_inj)
    sumA_test = torch.zeros(w_2_inj.shape[0],4,4,dtype=dtype, device=device)
    del PA_test, w_2_inj

    w_2_ini=w_2[where_isini]
    sumb_test = torch.einsum('ijk,ijk->ij',PB_test,w_2_ini)
    sumB_test = torch.zeros(w_2_ini.shape[0],4,4,dtype=dtype, device=device)
    del PB_test, w_2_ini    

    sumA_test[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma_test
    sumB_test[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb_test

    del suma_test, sumb_test

    iii=lookup_tensor[loc_i]
    indi_of_new_diag_in_old = maskd_sub[iii]

    jjj = lookup_tensor[loc_j]
    indj_of_new_diag_in_old = maskd_sub[jjj]

    F.index_add_(0,indi_of_new_diag_in_old, sumB_test)
    F.index_add_(0,indj_of_new_diag_in_old, sumA_test)
    del sumB_test, sumA_test

    ####################################################

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

    F0 = F.reshape(nmol,len(block_indices),len(block_indices),4,4) \
                     .transpose(2,3) \
                     .reshape(nmol, 4*len(block_indices), 4*len(block_indices))
    F0.add_(F0.triu(1).transpose(1,2));       
    return F0

def get_hcore_pyseqm(coords,symbols,atomTypes, device='cpu', verb=False):
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

  # Vectorized approach to combine the arrays
  combined_array = np.column_stack((atom_internal_indices[:, np.newaxis], coords)).tolist()

  # Convert to the desired format
  molecule_elem_coord = [[int(item[0]), tuple(item[1:])] for item in combined_array]


  species = torch.as_tensor(np.array([atom_internal_indices,]),
                          dtype=torch.int64, device=device)
  
  coordinates = torch.tensor(np.array([coords,]), device=device, dtype=torch.float64)
  
  #print(coordinates)

 
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


def get_molecule_pyseqm(sdc, coords, symbols, atomTypes, do_large_tensors=True, device='cpu', verb=False):
  # move to a sep file $$$
  torch.cuda.empty_cache()
  """PYSEQM"""
  
  # COHO
  # symbols: (C, O, H)
  # atomsTypes (0,1,2,1)
  # construc dict: 


  if(PYSEQM == False):
    print("ERROR: No PySCF installed")

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
  species = torch.as_tensor(np.array([atom_internal_indices,]),
                          dtype=dtype_int, device=device)
  
  
  if torch.is_tensor(coords):
    coordinates = coords
  else:
    coordinates = torch.tensor(np.array([coords]), 
                             device=device, dtype=sdc.torch_dt)
  
  #print(coordinates)

 
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

  
  molecule = Molecule(const, seqm_parameters, coordinates, species, do_large_tensors=do_large_tensors).to(device)

  ### Create electronic structure driver:
  
  return molecule, molecule.nocc.item()


def get_eVals_pyseqm(H, Nocc, Tel, mu0, coreSize, core_ham_dim, molecule=None, verb=False, calcD=False):

  from seqm.seqm_functions.diag import sym_eig_trunc

  kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
  if(verb): print("Computing the renormalized Density matrix")

  E_val, Q = sym_eig_trunc( H, molecule.nHeavy, molecule.nHydro, Nocc, eig_only=True)
  Q = Q[0]
  N = len(Q)
  E_val = E_val[0,:N]

  homoIndex = Nocc - 1
  lumoIndex = Nocc
  mu_test = 0.5*(E_val[homoIndex] + E_val[lumoIndex]) #don't need it 
  print(' SubSys HOMO/LUMO:', np.round(E_val[homoIndex].item(),4), np.round(E_val[lumoIndex].item(),4), end=" ")

  # rho = Q@f_vector@Q.T
  # or
  # rho_ij = SUM_k Q_ik * f_kk * Q_jk

  dVals = torch.tensor([], device = core_ham_dim.device, dtype=H.dtype)
  for i in range(N):
    dVals = torch.cat((dVals, torch.inner(Q[core_ham_dim,i],Q[core_ham_dim, i]).unsqueeze(0)) )

  return E_val, dVals.detach().cpu().numpy(), Q, [molecule.nHeavy, molecule.nHydro, H.shape[-1]]


def get_densityMatrix_renormalized_pyseqm(E_val, Q, Tel, mu0, NH_Nh_Hs, Nocc):
  
  kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
  beta = 1./(kB*Tel)
  #print(type(E_val), E_val)
  f = 1/(torch.exp(beta*(E_val - mu0)) + 1)
  
  # two lines below are vectorization of this: D = 2*sum(torch.outer(Q[:, i],Q[:, i]*f[i]) for i in range(Nocc))
  Q_weighted = Q * f  # Broadcasting multiplication
  D = 2 * Q @ Q_weighted.T

  #D = 2*sum(torch.outer(Q[:, i],Q[:, i]) for i in range(Nocc))
  D = unpack(D, NH_Nh_Hs[0], NH_Nh_Hs[1], NH_Nh_Hs[2])

  return D.detach()

def get_hamiltonian_pyseqm_uhf(coords,symbols,atomTypes, hindex, verb=False):
  # move to a sep file $$$
  torch.cuda.empty_cache()
  """PYSEQM"""
  
  # COHO
  # symbols: (C, O, H)
  # atomsTypes (0,1,2,1)
  # construc dict: 


  if(PYSEQM == False):
    print("ERROR: No PySCF installed")

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
  seqm.seqm_functions.scf_loop.debug=False

  torch.autograd.set_detect_anomaly(True)


  
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


  species = torch.as_tensor([atom_internal_indices,
                          ],
                          dtype=torch.int64, device=device)
  
  coordinates = torch.tensor([
                               coords,
                            ], device=device, dtype=torch.float64)
  
  #print(coordinates)

 
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
                    'UHF' : True,
                    }

  molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
  make_dm_guess(molecule, seqm_parameters, mix_homo_lumo=True, mix_coeff=0.3, overwrite_existing_dm=True);

  ### Create electronic structure driver:
  esdriver = Electronic_Structure(seqm_parameters).to(device)

  ### Run esdriver on molecules:
  esdriver(molecule, P0=molecule.dm.detach())

  with torch.no_grad():
    M, w, rho0xi, rho0xj = hcore(molecule)
    W = torch.tensor([0], device=molecule.nocc.device)
    W_exch = torch.tensor([0], device=molecule.nocc.device)

    H = fock_u_batch(molecule.nmol, molecule.molsize, molecule.dm, M, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, w, W, \
                  molecule.parameters['g_ss'],
                  molecule.parameters['g_pp'],
                  molecule.parameters['g_sp'],
                  molecule.parameters['g_p2'],
                  molecule.parameters['h_sp'],
                  molecule.method,
                  molecule.parameters['s_orb_exp_tail'],
                  molecule.parameters['p_orb_exp_tail'],
                  molecule.parameters['d_orb_exp_tail'],
                  molecule.Z,
                  molecule.parameters['F0SD'],
                  molecule.parameters['G2SD'])
  
  return H, molecule.nocc, molecule



def get_densityMatrix_renormalized_pyseqm_uhf(H, Nocc, Tel, mu0, coreSize, core_ham_dim, molecule=None, verb=False):
  # start FROM THIS!!!!!

  from seqm.seqm_functions.diag import sym_eig_trunc

  # m - limit of a partial trace
  kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
  if(verb): print("Computing the renormalized Density matrix")

  E_val, Q = sym_eig_trunc( H, molecule.nHeavy, molecule.nHydro, molecule.nocc, eig_only=True)
  E_val = E_val[0]
  Q = Q[0]
  molecule.e_mo.numpy()[0]


  N = len(H[0])

  #print('Q\n', Q[:,0])

  homoIndex_a, homoIndex_b = molecule.nocc[0,0].unsqueeze(0).T - 1, molecule.nocc[0,1].unsqueeze(0).T -1
  lumoIndex_a, lumoIndex_b = molecule.nocc[0,0].unsqueeze(0).T, molecule.nocc[0,1].unsqueeze(0).T

  print(homoIndex_a, homoIndex_b)

  print('a HOMO, LUMO:', E_val[0,homoIndex_a], E_val[0,lumoIndex_a])
  print('b HOMO, LUMO:', E_val[1,homoIndex_b], E_val[1,lumoIndex_b])
  mu_test = 0.5*(E_val[0,homoIndex_a] + E_val[0,lumoIndex_a]), 0.5*(E_val[1,homoIndex_b] + E_val[1,lumoIndex_b]) #don't need it 
  print('!!!! mu test:\n', mu_test)

  # use mu0 as a guess

  OccErr = 1.0
  beta = 1./(kB*Tel)
  f = 1/(torch.exp(beta*(E_val - mu0)) + 1)
  #f = f[0]

  D_a = sum(torch.outer(Q[0,:, i],Q[0,:, i]*f[0,i]) for i in range(molecule.nocc[0,0]))
  D_b = sum(torch.outer(Q[0,:, i],Q[0,:, i]*f[0,i]) for i in range(molecule.nocc[0,0]))

  D = sum(torch.outer(Q[:, i],Q[:, i]*f[i]) for i in range(molecule.nocc.item()))*2
  #np.savetxt('co2_32_dm.txt',D)


  # rho = Q@f_vector@Q.T
  # or
  # rho_ij = SUM_k Q_ik * f_kk * Q_jk


  print('core_ham_dim', core_ham_dim)
  dVals = torch.tensor([])
  #print(N)
  for i in range(N):
    #for j in range(coreSize):
    #  d_i += D[j,i]**2
    dVals = torch.cat((dVals, torch.inner(Q[:core_ham_dim,i],Q[:core_ham_dim, i]).unsqueeze(0)) )
  
  #print('dVals', dVals)
  return D.detach().numpy(), E_val.detach().numpy(), dVals.detach().numpy()



def get_overlap_pyseqm(coords,symbols,atomTypes, hindex, verb=False):
    # move to a sep file $$$
    torch.cuda.empty_cache()
    """PYSEQM"""

    # COHO
    # symbols: (C, O, H)
    # atomsTypes (0,1,2,1)
    # construc dict: 


    if(PYSEQM == False):
        print("ERROR: No PySCF installed")

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


    species = torch.as_tensor(np.array([np.array(atom_internal_indices)]),
                            dtype=torch.int64, device=device)

    coordinates = torch.tensor(np.array([coords]),
                                device=device, dtype=torch.float64)

    #print(coordinates)


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
    
    #torch.save(di, 'di.pt')
    #del di, overlap_pairs, zeta, qn_int, coordinates
    di_full = torch.zeros((molecule.nmol*molecule.molsize*molecule.molsize, 4, 4),dtype=dtype, device=device)
    mask_H = molecule.Z==1
    mask_heavy = molecule.Z>1
    

    H_self_ovr = torch.zeros((4,4), dtype=dtype, device=device)
    H_self_ovr[0,0] = 1.0

    di_full[molecule.maskd[mask_H]] = H_self_ovr
    di_full[molecule.maskd[mask_heavy]] = torch.eye(4, dtype=dtype, device=device)
    di_full[molecule.mask] = di
    di_full[molecule.mask_l] = di.transpose(1,2)

    di_full = di_full.reshape(molecule.nmol,molecule.molsize,molecule.molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(molecule.nmol, 4*molecule.molsize, 4*molecule.molsize)

        
    di_full = pack(di_full, molecule.nHeavy, molecule.nHydro)


    return di_full[0]

def get_diag_guess_pyseqm(molecule, sy, verb=False):
    tore = molecule.const.tore
    
    method = 'PM6_SP'

    if method == 'PM6':
        P0 = torch.zeros(sy.nats,9,9,dtype=molecule.coordinates.dtype, device=tore.device)  # density matrix
        P0[molecule.Z>1,0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[:,1,1] = P0[:,0,0]
        P0[:,2,2] = P0[:,0,0]
        P0[:,3,3] = P0[:,0,0]
        P0[molecule.Z==1,0,0] = 1.0
        # P = P0.reshape(nmol,molecule.molsize,molecule.molsize,9,9) \
        #     .transpose(2,3) \
        #     .reshape(nmol, 9*molecule.molsize, 9*molecule.molsize)
        
    else:
        P0 = torch.zeros(sy.nats,4,4,dtype=molecule.coordinates.dtype, device=tore.device)  # density matrix
        P0[molecule.Z>1,0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[:,1,1] = P0[:,0,0]
        P0[:,2,2] = P0[:,0,0]
        P0[:,3,3] = P0[:,0,0]
        P0[molecule.Z==1,0,0] = 1.0
        # P = P0.reshape(nmol,molecule.molsize,molecule.molsize,4,4) \
        #     .transpose(2,3) \
        #     .reshape(nmol, 4*molecule.molsize, 4*molecule.molsize)
        
    return P0

class ParamContainer():
   def __init__(self,):
      self