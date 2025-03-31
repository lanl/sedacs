Proxy codes for sedacs
=======================

Instead of directly using external quantum chemistry or ML codes (that can be quite complex 
with many separate libraries) SEDACS provides generic proxy codes which are a minimalistic 
representation of the external (more complex) guest codes. 
These proxy codes are implemented in Python, Fortran, and C, which are commonly used in 
electronic structure calculations. The proxy codes make a simplified step-by-step development possible
and they provide transparent examples of how to interface guest codes to SEDACS. 

        - Proxy code A: Simple proxy tight-binding Hamiltonians. 
        - Proxy code B: Distributed calculations of the quantum response and the preconditioned Krylov subspace approximation. 
        - Proxy code C: This version includes the full dynamics.



