""" Periodic table of elements.
 
 This data was generated with pybabel and openbable packages
 Openbabel: http://openbabel.org/dev-api/index.shtml
 Pybel: https://openbabel.org/docs/dev/UseTheLibrary/Python_Pybel.html#
 Other sources includes NIST: http://www.nist.gov/pml/data/ion_energy.cfm
"""

import numpy as np
import argparse

class ptable:
    """A simple periodic table.
    """
    def __init__(self):
        self.ntypes = 104 #Number of atom types. Each atomic number is a type.

        ## Element symbols for every atom type (atomic number)
        # The Bl (Bolonium) is added to have index = atomic number
        self.symbols = [ "Bl" ,                               
        "H" ,          "He" ,         "Li" ,         "Be" ,         \
        "B" ,          "C" ,          "N" ,          "O" ,          \
        "F" ,          "Ne" ,         "Na" ,         "Mg" ,         \
        "Al" ,         "Si" ,         "P" ,          "S" ,          \
        "Cl" ,         "Ar" ,         "K" ,          "Ca" ,         \
        "Sc" ,         "Ti" ,         "V" ,          "Cr" ,         \
        "Mn" ,         "Fe" ,         "Co" ,         "Ni" ,         \
        "Cu" ,         "Zn" ,         "Ga" ,         "Ge" ,         \
        "As" ,         "Se" ,         "Br" ,         "Kr" ,         \
        "Rb" ,         "Sr" ,         "Y" ,          "Zr" ,         \
        "Nb" ,         "Mo" ,         "Tc" ,         "Ru" ,         \
        "Rh" ,         "Pd" ,         "Ag" ,         "Cd" ,         \
        "In" ,         "Sn" ,         "Sb" ,         "Te" ,         \
        "I" ,          "Xe" ,         "Cs" ,         "Ba" ,         \
        "La" ,         "Ce" ,         "Pr" ,         "Nd" ,         \
        "Pm" ,         "Sm" ,         "Eu" ,         "Gd" ,         \
        "Tb" ,         "Dy" ,         "Ho" ,         "Er" ,         \
        "Tm" ,         "Yb" ,         "Lu" ,         "Hf" ,         \
        "Ta" ,         "W" ,          "Re" ,         "Os" ,         \
        "Ir" ,         "Pt" ,         "Au" ,         "Hg" ,         \
        "Tl" ,         "Pb" ,         "Bi" ,         "Po" ,         \
        "At" ,         "Rn" ,         "Fr" ,         "Ra" ,         \
        "Ac" ,         "Th" ,         "Pa" ,         "U" ,          \
        "Np" ,         "Pu" ,         "Am" ,         "Cm" ,         \
        "Bk" ,         "Cf" ,         "Es" ,         "Fm" ,         \
        "Md" ,         "No" ,         "Lr"                          \
        ]


        ## Element name
        #
        self.names = ["Bolonium", \
        "Hydrogen" ,     "Helium" ,       "Lithium" ,      "Beryllium" ,    \
        "Boron" ,        "Carbon" ,       "Nitrogen" ,     "Oxygen" ,       \
        "Fluorine" ,     "Neon" ,         "Sodium" ,       "Magnesium" ,    \
        "Aluminium" ,    "Silicon" ,      "Phosphorus" ,   "Sulfur" ,       \
        "Chlorine" ,     "Argon" ,        "Potassium" ,    "Calcium" ,      \
        "Scandium" ,     "Titanium" ,     "Vanadium" ,     "Chromium" ,     \
        "Manganese" ,    "Iron" ,         "Cobalt" ,       "Nickel" ,       \
        "Copper" ,       "Zinc" ,         "Gallium" ,      "Germanium" ,    \
        "Arsenic" ,      "Selenium" ,     "Bromine" ,      "Krypton" ,      \
        "Rubidium" ,     "Strontium" ,    "Yttrium" ,      "Zirconium" ,    \
        "Niobium" ,      "Molybdenum" ,   "Technetium" ,   "Ruthenium" ,    \
        "Rhodium" ,      "Palladium" ,    "Silver" ,       "Cadmium" ,      \
        "Indium" ,       "Tin" ,          "Antimony" ,     "Tellurium" ,    \
        "Iodine" ,       "Xenon" ,        "Caesium" ,      "Barium" ,       \
        "Lanthanum" ,    "Cerium" ,       "Praseodymium" , "Neodymium" ,    \
        "Promethium" ,   "Samarium" ,     "Europium" ,     "Gadolinium" ,   \
        "Terbium" ,      "Dysprosium" ,   "Holmium" ,      "Erbium" ,       \
        "Thulium" ,      "Ytterbium" ,    "Lutetium" ,     "Hafnium" ,      \
        "Tantalum" ,     "Tungsten" ,     "Rhenium" ,      "Osmium" ,       \
        "Iridium" ,      "Platinum" ,     "Gold" ,         "Mercury" ,      \
        "Thallium" ,     "Lead" ,         "Bismuth" ,      "Polonium" ,     \
        "Astatine" ,     "Radon" ,        "Francium" ,     "Radium" ,       \
        "Actinium" ,     "Thorium" ,      "Protactinium" , "Uranium" ,      \
        "Neptunium" ,    "Plutonium" ,    "Americium" ,    "Curium" ,       \
        "Berkelium" ,    "Californium" ,  "Einsteinium" ,  "Fermium" ,      \
        "Mendelevium" ,  "Nobelium" ,     "Lawrencium"                      \
        ]

        ## Element mass in atomic mass units (1.66 x 10-27 kg)
        #
        self.mass = np.zeros(self.ntypes) 
        self.mass[:] =  0.000,                                     \
        1.007825032 ,  4.002603254 ,  7.01600455 ,   9.0121822 ,    \
        11.0093054 ,   12.0 ,         14.003074005 , 15.99491462 ,  \
        18.99840322 ,  19.992440175 , 22.989769281 , 23.9850417 ,   \
        26.98153863 ,  27.976926532 , 30.97376163 ,  31.972071 ,    \
        34.96885268 ,  39.962383123 , 38.96370668 ,  39.96259098 ,  \
        44.9559119 ,   47.9479463 ,   50.9439595 ,   51.9405075 ,   \
        54.9380451 ,   55.9349375 ,   58.933195 ,    57.9353429 ,   \
        62.9295975 ,   63.929142 ,    68.925573 ,    73.921177 ,    \
        74.921596 ,    79.916521 ,    78.918337 ,    83.911507 ,    \
        84.911789 ,    87.905612 ,    88.905848 ,    89.904704 ,    \
        92.906378 ,    97.905408 ,    97.907216 ,    101.904349 ,   \
        102.905504 ,   105.903486 ,   106.905097 ,   113.903358 ,   \
        114.903878 ,   119.902194 ,   120.903815 ,   129.906224 ,   \
        126.904473 ,   131.904153 ,   132.905451 ,   137.905247 ,   \
        138.906353 ,   139.905438 ,   140.907652 ,   141.907723 ,   \
        144.912749 ,   151.919732 ,   152.92123 ,    157.924103 ,   \
        158.925346 ,   163.929174 ,   164.930322 ,   165.930293 ,   \
        168.934213 ,   173.938862 ,   174.940771 ,   179.94655 ,    \
        180.947995 ,   183.950931 ,   186.955753 ,   191.96148 ,    \
        192.962926 ,   194.964791 ,   196.966568 ,   201.970643 ,   \
        204.974427 ,   207.976652 ,   208.980398 ,   208.98243 ,    \
        209.987148 ,   222.017577 ,   223.019735 ,   226.025409 ,   \
        227.027752 ,   232.038055 ,   231.035884 ,   238.050788 ,   \
        237.048173 ,   244.064204 ,   243.061381 ,   247.070354 ,   \
        247.070307 ,   251.079587 ,   252.08298 ,    257.095105 ,   \
        258.098431 ,   259.10103 ,    262.10963                     
        

        ## van der Waals radius (in Angstroms)
        #
        self.vdwr = np.zeros(self.ntypes) 
        self.vdwr[:] = 0.000,                                      \
        1.1 ,          1.4 ,          1.81 ,         1.53 ,         \
        1.92 ,         1.7 ,          1.55 ,         1.52 ,         \
        1.47 ,         1.54 ,         2.27 ,         1.73 ,         \
        1.84 ,         2.1 ,          1.8 ,          1.8 ,          \
        1.75 ,         1.88 ,         2.75 ,         2.31 ,         \
        2.3 ,          2.15 ,         2.05 ,         2.05 ,         \
        2.05 ,         2.05 ,         2.0 ,          2.0 ,          \
        2.0 ,          2.1 ,          1.87 ,         2.11 ,         \
        1.85 ,         1.9 ,          1.83 ,         2.02 ,         \
        3.03 ,         2.49 ,         2.4 ,          2.3 ,          \
        2.15 ,         2.1 ,          2.05 ,         2.05 ,         \
        2.0 ,          2.05 ,         2.1 ,          2.2 ,          \
        2.2 ,          1.93 ,         2.17 ,         2.06 ,         \
        1.98 ,         2.16 ,         3.43 ,         2.68 ,         \
        2.5 ,          2.48 ,         2.47 ,         2.45 ,         \
        2.43 ,         2.42 ,         2.4 ,          2.38 ,         \
        2.37 ,         2.35 ,         2.33 ,         2.32 ,         \
        2.3 ,          2.28 ,         2.27 ,         2.25 ,         \
        2.2 ,          2.1 ,          2.05 ,         2.0 ,          \
        2.0 ,          2.05 ,         2.1 ,          2.05 ,         \
        1.96 ,         2.02 ,         2.07 ,         1.97 ,         \
        2.02 ,         2.2 ,          3.48 ,         2.83 ,         \
        2.0 ,          2.4 ,          2.0 ,          2.3 ,          \
        2.0 ,          2.0 ,          2.0 ,          2.0 ,          \
        2.0 ,          2.0 ,          2.0 ,          2.0 ,          \
        2.0 ,          2.0 ,          2.0                           
        

        ## Covalent radius (in Angstroms)
        #
        self.covr = np.zeros(self.ntypes) 
        self.covr[:] = 0.00,                                               \
        0.31 ,           0.28 ,           1.28 ,           0.96 ,           \
        0.84 ,           0.76 ,           0.71 ,           0.66 ,           \
        0.57 ,           0.58 ,           1.66 ,           1.41 ,           \
        1.21 ,           1.11 ,           1.07 ,           1.05 ,           \
        1.02 ,           1.06 ,           2.03 ,           1.76 ,           \
        1.7 ,            1.6 ,            1.53 ,           1.39 ,           \
        1.39 ,           1.32 ,           1.26 ,           1.24 ,           \
        1.32 ,           1.22 ,           1.22 ,           1.2 ,            \
        1.19 ,           1.2 ,            1.2 ,            1.16 ,           \
        2.2 ,            1.95 ,           1.9 ,            1.75 ,           \
        1.64 ,           1.54 ,           1.47 ,           1.46 ,           \
        1.42 ,           1.39 ,           1.45 ,           1.44 ,           \
        1.42 ,           1.39 ,           1.39 ,           1.38 ,           \
        1.39 ,           1.4 ,            2.44 ,           2.15 ,           \
        2.07 ,           2.04 ,           2.03 ,           2.01 ,           \
        1.99 ,           1.98 ,           1.98 ,           1.96 ,           \
        1.94 ,           1.92 ,           1.92 ,           1.89 ,           \
        1.9 ,            1.87 ,           1.87 ,           1.75 ,           \
        1.7 ,            1.62 ,           1.51 ,           1.44 ,           \
        1.41 ,           1.36 ,           1.36 ,           1.32 ,           \
        1.45 ,           1.46 ,           1.48 ,           1.4 ,            \
        1.5 ,            1.5 ,            2.6 ,            2.21 ,           \
        2.15 ,           2.06 ,           2.0 ,            1.96 ,           \
        1.9 ,            1.87 ,           1.8 ,            1.69 ,           \
        1.6 ,            1.6 ,            1.6 ,            1.6 ,            \
        1.6 ,            1.6 ,            1.6                               
        

        ## Ionization energy (in eV)
        #
        self.ip = np.zeros(self.ntypes)
        self.ip[:] =  0.00,                                        \
        13.5984 ,      24.5874 ,      5.3917 ,       9.3227 ,       \
        8.298 ,        11.2603 ,      14.5341 ,      13.6181 ,      \
        17.4228 ,      21.5645 ,      5.1391 ,       7.6462 ,       \
        5.9858 ,       8.1517 ,       10.4867 ,      10.36 ,        \
        12.9676 ,      15.7596 ,      4.3407 ,       6.1132 ,       \
        6.5615 ,       6.8281 ,       6.7462 ,       6.7665 ,       \
        7.434 ,        7.9024 ,       7.881 ,        7.6398 ,       \
        7.7264 ,       9.3942 ,       5.9993 ,       7.8994 ,       \
        9.7886 ,       9.7524 ,       11.8138 ,      13.9996 ,      \
        4.1771 ,       5.6949 ,       6.2173 ,       6.6339 ,       \
        6.7589 ,       7.0924 ,       7.28 ,         7.3605 ,       \
        7.4589 ,       8.3369 ,       7.5762 ,       8.9938 ,       \
        5.7864 ,       7.3439 ,       8.6084 ,       9.0096 ,       \
        10.4513 ,      12.1298 ,      3.8939 ,       5.2117 ,       \
        5.5769 ,       5.5387 ,       5.473 ,        5.525 ,        \
        5.582 ,        5.6437 ,       5.6704 ,       6.1498 ,       \
        5.8638 ,       5.9389 ,       6.0215 ,       6.1077 ,       \
        6.1843 ,       6.2542 ,       5.4259 ,       6.8251 ,       \
        7.5496 ,       7.864 ,        7.8335 ,       8.4382 ,       \
        8.967 ,        8.9588 ,       9.2255 ,       10.4375 ,      \
        6.1082 ,       7.4167 ,       7.2855 ,       8.414 ,        \
        0.0 ,          10.7485 ,      4.0727 ,       5.2784 ,       \
        5.17 ,         6.3067 ,       5.89 ,         6.1941 ,       \
        6.2657 ,       6.026 ,        5.9738 ,       5.9914 ,       \
        6.1979 ,       6.2817 ,       6.42 ,         6.5 ,          \
        6.58 ,         6.65 ,         4.9                           
        

        ## Electron affinity (in eV)
        #
        self.ea = np.zeros(self.ntypes)
        self.ea[:] =  0.00,                                        \
        0.75420375 ,   0.0 ,          0.618049 ,     0.0 ,          \
        0.279723 ,     1.262118 ,     -0.07 ,        1.461112 ,     \
        3.4011887 ,    0.0 ,          0.547926 ,     0.0 ,          \
        0.43283 ,      1.389521 ,     0.7465 ,       2.0771029 ,    \
        3.612724 ,     0.0 ,          0.501459 ,     0.02455 ,      \
        0.188 ,        0.084 ,        0.525 ,        0.67584 ,      \
        0.0 ,          0.151 ,        0.6633 ,       1.15716 ,      \
        1.23578 ,      0.0 ,          0.41 ,         1.232712 ,     \
        0.814 ,        2.02067 ,      3.363588 ,     0.0 ,          \
        0.485916 ,     0.05206 ,      0.307 ,        0.426 ,        \
        0.893 ,        0.7472 ,       0.55 ,         1.04638 ,      \
        1.14289 ,      0.56214 ,      1.30447 ,      0.0 ,          \
        0.404 ,        1.112066 ,     1.047401 ,     1.970875 ,     \
        3.059038 ,     0.0 ,          0.471626 ,     0.14462 ,      \
        0.47 ,         0.5 ,          0.5 ,          0.5 ,          \
        0.5 ,          0.5 ,          0.5 ,          0.5 ,          \
        0.5 ,          0.5 ,          0.5 ,          0.5 ,          \
        0.5 ,          0.5 ,          0.5 ,          0.0 ,          \
        0.322 ,        0.815 ,        0.15 ,         1.0778 ,       \
        1.56436 ,      2.1251 ,       2.30861 ,      0.0 ,          \
        0.377 ,        0.364 ,        0.942363 ,     1.9 ,          \
        2.8 ,          0.0 ,          0.0 ,          0.0 ,          \
        0.0 ,          0.0 ,          0.0 ,          0.0 ,          \
        0.0 ,          0.0 ,          0.0 ,          0.0 ,          \
        0.0 ,          0.0 ,          0.0 ,          0.0 ,          \
        0.0 ,          0.0 ,          0.0                           
        

        ## The Pauling electronegativity for this element
        #
        self.en = np.zeros(self.ntypes)
        self.en[:] =  0.00,                                        \
        2.2 ,          0.0 ,          0.98 ,         1.57 ,         \
        2.04 ,         2.55 ,         3.04 ,         3.44 ,         \
        3.98 ,         0.0 ,          0.93 ,         1.31 ,         \
        1.61 ,         1.9 ,          2.19 ,         2.58 ,         \
        3.16 ,         0.0 ,          0.82 ,         1.0 ,          \
        1.36 ,         1.54 ,         1.63 ,         1.66 ,         \
        1.55 ,         1.83 ,         1.88 ,         1.91 ,         \
        1.9 ,          1.65 ,         1.81 ,         2.01 ,         \
        2.18 ,         2.55 ,         2.96 ,         3.0 ,          \
        0.82 ,         0.95 ,         1.22 ,         1.33 ,         \
        1.6 ,          2.16 ,         1.9 ,          2.2 ,          \
        2.28 ,         2.2 ,          1.93 ,         1.69 ,         \
        1.78 ,         1.96 ,         2.05 ,         2.1 ,          \
        2.66 ,         2.6 ,          0.79 ,         0.89 ,         \
        1.1 ,          1.12 ,         1.13 ,         1.14 ,         \
        0.0 ,          1.17 ,         0.0 ,          1.2 ,          \
        0.0 ,          1.22 ,         1.23 ,         1.24 ,         \
        1.25 ,         0.0 ,          1.27 ,         1.3 ,          \
        1.5 ,          2.36 ,         1.9 ,          2.2 ,          \
        2.2 ,          2.28 ,         2.54 ,         2.0 ,          \
        1.62 ,         2.33 ,         2.02 ,         2.0 ,          \
        2.2 ,          0.0 ,          0.7 ,          0.9 ,          \
        1.1 ,          1.3 ,          1.5 ,          1.38 ,         \
        1.36 ,         1.28 ,         1.3 ,          1.3 ,          \
        1.3 ,          1.3 ,          1.3 ,          1.3 ,          \
        1.3 ,          1.3 ,          0.0                           
        

        ## The maximum expected number of bonds to this element
        #
        self.maxbonds = np.zeros(self.ntypes,dtype=int)
        self.maxbonds[:] =  0,                                   
        1 ,            0 ,            1 ,            2 ,            \
        4 ,            4 ,            4 ,            2 ,            \
        1 ,            0 ,            1 ,            2 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        1 ,            0 ,            1 ,            2 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        8 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            3 ,            4 ,            \
        3 ,            2 ,            1 ,            0 ,            \
        1 ,            2 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        3 ,            4 ,            3 ,            2 ,            \
        1 ,            0 ,            1 ,            2 ,            \
        12 ,           6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        3 ,            4 ,            3 ,            2 ,            \
        1 ,            0 ,            1 ,            2 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6 ,            6 ,            \
        6 ,            6 ,            6                             
        

        ## Last shell number of electrons
        #
        self.numel = np.zeros(self.ntypes,dtype=int)
        self.numel[:] = 0,   \
        1 ,        2 ,       1 ,       2 ,     \
        3 ,        4 ,       5 ,       6 ,     \
        7 ,        8 ,       1 ,       2 ,     \
        3 ,        4 ,       5 ,       6 ,     \
        7 ,        8 ,       1 ,       2 ,     \
        3 ,        4 ,       5 ,       6 ,     \
        7 ,        8 ,       9 ,      10 ,     \
        11 ,       12 ,      13 ,      14 ,    \
        15 ,       16 ,      17 ,      18 ,    \
        1 ,        2 ,       3 ,       4 ,     \
        5 ,        6 ,       7 ,       8 ,     \
        9 ,       10 ,      11 ,      12 ,     \
        13 ,       14 ,      15 ,      16 ,    \
        17 ,       18 ,       1 ,       2 ,    \
        3 ,        4 ,       5 ,       6 ,     \
        7 ,        8 ,       9 ,      10 ,     \
        11 ,       12 ,      13 ,      14 ,    \
        15 ,       16 ,      17 ,      18 ,    \
        19 ,       20 ,      21 ,      22 ,    \
        23 ,       24 ,      25 ,      26 ,    \
        27 ,       28 ,      29 ,      30 ,    \
        31 ,       32 ,       1 ,       2 ,    \
        3 ,        4 ,       5 ,       6 ,     \
        7 ,        8 ,       9 ,      10 ,     \
        11 ,       12 ,      13 ,      14 ,    \
        15 ,       16 ,      17                
        

        ## The electronic configuration
        #
        self.econf = ['0s',
        "1s" ,              "1s2" ,             "1s22s" ,           "1s22s2" ,          \
        "1s22s22p" ,        "1s22s22p2" ,       "1s22s22p3" ,       "1s22s22p4" ,       \
        "1s22s22p5" ,       "1s22s22p6" ,       "[Ne]3s" ,          "[Ne]3s2" ,         \
        "[Ne]3s23p" ,       "[Ne]3s23p2" ,      "[Ne]3s23p3" ,      "[Ne]3s23p4" ,      \
        "[Ne]3s23p5" ,      "[Ne]3s23p6" ,      "[Ar]4s" ,          "[Ar]4s2" ,         \
        "[Ar]3d4s2" ,       "[Ar]3d24s2" ,      "[Ar]3d34s2" ,      "[Ar]3d54s" ,       \
        "[Ar]3d54s2" ,      "[Ar]3d64s2" ,      "[Ar]3d74s2" ,      "[Ar]3d84s2" ,      \
        "[Ar]3d104s" ,      "[Ar]3d104s2" ,     "[Ar]3d104s24p" ,   "[Ar]3d104s24p2" ,  \
        "[Ar]3d104s24p3" ,  "[Ar]3d104s24p4" ,  "[Ar]3d104s24p5" ,  "[Ar]3d104s24p6" ,  \
        "[Kr]5s" ,          "[Kr]5s2" ,         "[Kr]4d5s2" ,       "[Kr]4d25s2" ,      \
        "[Kr]4d45s" ,       "[Kr]4d55s" ,       "[Kr]4d55s2" ,      "[Kr]4d75s" ,       \
        "[Kr]4d85s" ,       "[Kr]4d10" ,        "[Kr]4d105s" ,      "[Kr]4d105s2" ,     \
        "[Cd]5p" ,          "[Cd]5p2" ,         "[Cd]5p3" ,         "[Cd]5p4" ,         \
        "[Cd]5p5" ,         "[Cd]5p6" ,         "[Xe]6s" ,          "[Xe]6s2" ,         \
        "[Xe]5d6s2" ,       "[Xe]4f5d6s2" ,     "[Xe]4f36s2" ,      "[Xe]4f46s2" ,      \
        "[Xe]4f56s2" ,      "[Xe]4f66s2" ,      "[Xe]4f76s2" ,      "[Xe]4f75d6s2" ,    \
        "[Xe]4f96s2" ,      "[Xe]4f106s2" ,     "[Xe]4f116s2" ,     "[Xe]4f126s2" ,     \
        "[Xe]4f136s2" ,     "[Xe]4f146s2" ,     "[Xe]4f145d6s2" ,   "[Xe]4f145d26s2" ,  \
        "[Xe]4f145d36s2" ,  "[Xe]4f145d46s2" ,  "[Xe]4f145d56s2" ,  "[Xe]4f145d66s2" ,  \
        "[Xe]4f145d76s2" ,  "[Xe]4f145d96s" ,   "[Xe]4f145d106s" ,  "[Xe]4f145d106s2" , \
        "[Hg]6p" ,          "[Hg]6p2" ,         "[Hg]6p3" ,         "[Hg]6p4" ,         \
        "[Hg]6p5" ,         "[Hg]6p6" ,         "[Rn]7s" ,          "[Rn]7s2" ,         \
        "[Rn]6d7s2" ,       "[Rn]6d27s2" ,      "[Rn]5f26d7s2" ,    "[Rn]5f36d7s2" ,    \
        "[Rn]5f46d7s2" ,    "[Rn]5f67s2" ,      "[Rn]5f77s2" ,      "[Rn]5f76d7s2" ,    \
        "[Rn]5f97s2" ,      "[Rn]5f107s2" ,     "[Rn]5f117s2" ,     "[Rn]5f127s2" ,     \
        "[Rn]5f137s2" ,     "[Rn]5f147s2" ,     "[Rn]5f147s27p"                         \
        ]

    ## Get the atomic number of a paerticular element    
    def get_atomic_number(self,symb):
        return self.symbols.index(symb) 

if(__name__ == '__main__'):

    parser = argparse.ArgumentParser(description="""Parser for the ptable""")

    parser.add_argument("-name", help="Getting the name of a particular element", type=str, default="Bl")
    parser.add_argument("-mass", help="Getting the mass of a particular element (a.u.)", type=str, default="Bl")
    parser.add_argument("-vdwr", help="Getting the van Der Waals radius of a particular element (Ang)", type=str, default="Bl")
    parser.add_argument("-covr", help="Getting the covalency radius of a particular element (Ang)", type=str, default="Bl")
    parser.add_argument("-ip", help="Getting the ionization energy of a particular element (eV)", type=str, default="Bl")
    parser.add_argument("-ea", help="Getting the electron affinity of a particular element (eV)", type=str, default="Bl")
    parser.add_argument("-en", help="Getting the electronegativity of a particular element", type=str, default="Bl")
    parser.add_argument("-maxbonds", help="Getting the maximun allowed bonds of a particular element", type=str, default="Bl")
    parser.add_argument("-numel", help="Getting the last shell number of electrons of a particular element", type=str, default="Bl")
    parser.add_argument("-econf", help="Getting the electron configuration of a particular element", type=str, default="Bl")
    parser.add_argument("-atnum", help="Getting the atomic number of a particular element", type=str, default="Bl")

    options = parser.parse_args()
    pt = ptable()
    if(options.name != "Bl"):
        print("Element name: ",pt.names[pt.get_atomic_number(options.name)])
    if(options.mass != "Bl"):
        print("Element mass (a.u.) = ",pt.mass[pt.get_atomic_number(options.mass)])
    if(options.vdwr != "Bl"):
        print("Element van Der Waals radius (Ang) = ",pt.vdwr[pt.get_atomic_number(options.vdwr)])
    if(options.covr != "Bl"):
        print("Element covalency radius (Ang) = ",pt.covr[pt.get_atomic_number(options.covr)])
    if(options.ip != "Bl"):
        print("Element ionization potential (eV) = ",pt.ip[pt.get_atomic_number(options.ip)])
    if(options.ea != "Bl"):
        print("Element electron affinity (eV) = ",pt.ea[pt.get_atomic_number(options.ea)])
    if(options.en != "Bl"):
        print("Element electronegativity  = ",pt.en[pt.get_atomic_number(options.en)])
    if(options.maxbonds != "Bl"):
        print("Element max bonds = ",pt.maxbonds[pt.get_atomic_number(options.maxbonds)])
    if(options.numel != "Bl"):
        print("Element number of electrons = ",pt.numel[pt.get_atomic_number(options.numel)])
    if(options.econf != "Bl"):
        print("Element electron configuration = ",pt.econf[pt.get_atomic_number(options.econf)])
    if(options.atnum != "Bl"):
        print("Element atomic number = ",pt.get_atomic_number(options.atnum))
