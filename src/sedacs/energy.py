from sedacs.interface.pyseqm import get_nucAB_energy_pyseqm, get_total_energy_pyseqm
import torch

__all__ = ["get_fock"]


def get_eNuc(eng, obj):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Fock")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own Fock")
        exit()
    elif eng.interface == "PySEQM":
        alpha = obj.molecule_whole.parameters['alpha']
        K = torch.stack((   obj.molecule_whole.parameters['Gaussian1_K'],
                            obj.molecule_whole.parameters['Gaussian2_K'],
                            obj.molecule_whole.parameters['Gaussian3_K'],
                            obj.molecule_whole.parameters['Gaussian4_K']),dim=1)
        
        L = torch.stack((   obj.molecule_whole.parameters['Gaussian1_L'],
                            obj.molecule_whole.parameters['Gaussian2_L'],
                            obj.molecule_whole.parameters['Gaussian3_L'],
                            obj.molecule_whole.parameters['Gaussian4_L']),dim=1)

        M = torch.stack((   obj.molecule_whole.parameters['Gaussian1_M'],
                            obj.molecule_whole.parameters['Gaussian2_M'],
                            obj.molecule_whole.parameters['Gaussian3_M'],
                            obj.molecule_whole.parameters['Gaussian4_M']),dim=1)
        parnuc = (alpha, K, L, M)
        ev = 27.21
        if 'g_ss_nuc' in obj.molecule_whole.parameters:
            g = obj.molecule_whole.parameters['g_ss_nuc']
            rho0a = 0.5 * ev / g[obj.molecule_whole.idxi]
            rho0b = 0.5 * ev / g[obj.molecule_whole.idxj]
            gam = ev / torch.sqrt(obj.molecule_whole.rij**2 + (rho0a + rho0b)**2)
        else:
            #print('w',obj.w_whole[...,0,0].shape)
            #gam = obj.w_whole[...,0,0]
            gam = obj.w_ssss
        return get_nucAB_energy_pyseqm(obj.molecule_whole.Z, obj.molecule_whole.const, obj.molecule_whole.nmol, obj.molecule_whole.ni, obj.molecule_whole.nj,
                                         obj.molecule_whole.idxi, obj.molecule_whole.idxj, obj.molecule_whole.rij, \
                                         obj.rho0xi_whole, obj.rho0xj_whole, obj.molecule_whole.alp, obj.molecule_whole.chi,
                                         gam, obj.molecule_whole.method, parnuc)
        
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho

def get_eTot(eng, obj, eNucAB, eElec):
    if eng.interface == "None":
        print("ERROR!!! - Write your own Fock")

    # Tight interface using modules or an external code compiled as a library
    elif eng.interface == "Module":
        # We will call proxyA directly as it will be loaded as a module.
        print("ERROR!!! - Write your own Fock")
        exit()
    elif eng.interface == "PySEQM":
        eTot, eNuc = get_total_energy_pyseqm(obj.molecule_whole.nmol, obj.molecule_whole.pair_molid, eNucAB, eElec)
        return eTot[0], eNuc[0]
    else:
        print("ERROR!!!: Interface type not recognized. Use any of the following: Module,File,Socket,MDI")
        exit()
    return rho


