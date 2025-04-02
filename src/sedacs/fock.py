from sedacs.interface_pyseqm import get_full_fock_pyseqm
import torch

__all__ = ["get_fock"]


## Build the density matrix.
# @brief This will build a density matrix. Typically this will be done interfacing with an engine.
# @param eng Engine object. See sdc_engine.py for a full explanation
# @param nocc Number of occupied states
# @param ham Hamiltonian matrix
# @verbose Verbosity
#
def get_fock(eng, obj):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Fock")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own Fock")
        exit()
    elif eng.interface == "PySEQM":

        W_whole = torch.tensor([0], device=obj.molecule_whole.nocc.device)
        return get_full_fock_pyseqm(obj.molecule_whole.nmol, obj.molecule_whole.molsize, obj.molecule_whole.dm, obj.M_whole,
                                    obj.molecule_whole.maskd, obj.molecule_whole.mask,
                                    obj.molecule_whole.idxi, obj.molecule_whole.idxj, obj.w_whole, W_whole,
                                    obj.molecule_whole.parameters['g_ss'],
                                    obj.molecule_whole.parameters['g_pp'],
                                    obj.molecule_whole.parameters['g_sp'],
                                    obj.molecule_whole.parameters['g_p2'],
                                    obj.molecule_whole.parameters['h_sp'],
                                    obj.molecule_whole.method,
                                    obj.molecule_whole.parameters['s_orb_exp_tail'],
                                    obj.molecule_whole.parameters['p_orb_exp_tail'],
                                    obj.molecule_whole.parameters['d_orb_exp_tail'],
                                    obj.molecule_whole.Z,
                                    obj.molecule_whole.parameters['F0SD'],
                                    obj.molecule_whole.parameters['G2SD'])
        
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho
