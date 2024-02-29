from proxy_a import *

def sdc_get_hamiltonian_module(eng,coords,atomTypes,symbols,verb):
    ham = proxyA_get_hamiltonian(coords,atomTypes=np.zeros((1),dtype=int),verb=False)
    return ham

