LATTE: Los Alamos Transferable Tight-binding for Energetics
=============================================================
LATTE is a quantum molecular dynamics (QMD) program developed at LANL. It has several
important algorithmic implementations for the fast construction of the density matrix (DM),
which is an essential part of a quantum chemistry calculation. LATTE operates on the
foundational principles of tight-binding self-consistent-charge density functional theory (DFTB),
allowing for a fast construction of the Hamiltonian matrix. LATTE offers an approximation
to the DFTB method by analytically calculating the elements of the Hamiltonian matrix. This
is achieved through a “physically motivated function” that incorporates adjustable parameters.
To obtain the parameter set, high-level DFT calculations are typically employed, utilizing a
diverse training set comprising various molecular configurations.

The code can be found here: `[LATTE] <https://github.com/lanl/LATTE>`_



