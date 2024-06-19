#!/usr/bin/env python3


from sdc_system import *
from sdc_ffields import *

def test_ffield(coords,types,symbols,latticeVectors,nl,nlTrX,nlTrY,nlTrZ):
    forces = get_sdc_classical_forces(coords,types,symbols,latticeVectors,nl,nlTrX,nlTrY,nlTrZ,"HarmonicAll",verb=True)
    print("forces",forces)
    exit(0)


