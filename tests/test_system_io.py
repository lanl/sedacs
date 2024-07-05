import textwrap
import unittest

import numpy as np
from sedacs.periodic_table import PeriodicTable
from sedacs.system_io import (
    are_files_equivalent,
    read_pdb_file,
    read_xyz_file,
    read_xyz_trajectory,
    write_pdb_coordinates,
    write_xyz_coordinates,
)


class TestReaders(unittest.TestCase):
    def test_read_xyz_file(self):
        passed = True
        xyz_file_content = textwrap.dedent("""4
        Test
        H 0.0 0.0 0.0
        He 1.0 1.0 1.0
        C 2.0 2.0 2.0
        C 3.0 3.0 3.0""")
        coordsRef = np.zeros((4, 3))
        coordsRef[1, :] = [1.0, 1.0, 1.0]
        coordsRef[2, :] = [2.0, 2.0, 2.0]
        coordsRef[3, :] = [3.0, 3.0, 3.0]
        typesRef = np.zeros((4), dtype=int)
        typesRef = [0, 1, 2, 2]
        file = "test_xyz.xyz"
        with open(file, "w") as f:
            f.write(xyz_file_content)
        try:
            _, symbols, types, coords = read_xyz_file(file, lib=None, verb=False)
            if symbols != ["H", "He", "C"]:
                passed = False
            if not np.allclose(coords, coordsRef):
                passed = False
            if not np.allclose(types, typesRef):
                passed = False
        except Exception:
            passed = False

        self.assertTrue(passed)

    def test_write_xyz_coordinates(self):
        passed = True
        try:
            pt = PeriodicTable()
            nats = len(pt.symbols)
            nsymb = 8
            symbols = [] * nsymb
            symbols[:] = pt.symbols[0:nsymb]
            coords = np.zeros((nats, 3))
            types = np.zeros((nats), dtype=int)
            for i in range(len(pt.symbols)):
                coords[i, 0] = float(i)
                coords[i, 1] = float(i) + 2.0
                coords[i, 2] = float(i) + 3.0
                types[i] = i % (nsymb - 1)

            with open("ref.xyz", "w") as f:
                print(nats, file=f)
                print("xyz format", file=f)
                for i in range(nats):
                    symb = symbols[types[i]]
                    print(symb, coords[i, 0], coords[i, 1], coords[i, 2], file=f)

            write_xyz_coordinates("actual.xyz", coords, types, symbols)
            if not are_files_equivalent("actual.xyz", "ref.xyz"):
                passed = False
        except Exception:
            passed = False

        self.assertTrue(passed)

    def test_read_xyz_trajectory(self):
        passed = True
        try:
            pt = PeriodicTable()
            nsymb = 8
            symbols = [] * nsymb
            symbols[:] = pt.symbols[0:nsymb]
            nats = len(symbols)
            coords = np.zeros((2, nats, 3))
            types = np.zeros((nats), dtype=int)
            values = np.zeros((2, nats))
            for i in range(nats):
                coords[0, i, 0] = float(i)
                coords[0, i, 1] = float(i) + 2.0
                coords[0, i, 2] = float(i) + 3.0
                values[0, i] = 0.09 * i % 10
                coords[1, :, :] = coords[0, :, :] + 0.1
                values[1, i] = 0.08 * i % 10

            with open("ref.xyz", "w") as f:
                for j in range(2):
                    print(nats, file=f)
                    print("frame {}".format(j), file=f)
                    for i in range(nats):
                        print(symbols[i], coords[j, i, 0], coords[j, i, 1], coords[j, i, 2], values[j, i], file=f)

            e, c, v = read_xyz_trajectory("ref.xyz")
            if np.any(e != symbols):
                passed = False
            if not np.allclose(coords, c):
                passed = False
            if not np.allclose(values, v):
                passed = False
        except Exception:
            passed = False

        self.assertTrue(passed)

    def test_read_pdb_file(self):
        passed = True
        pdb_file_content = textwrap.dedent("""TITLE test
        REMARK This is a test file
        CRYST1   31.230   31.230   31.230  90.00  90.00  90.00 P 1           1
        MODEL        1
        ATOM      1  O   HOH     1      15.427  15.434  15.615  1.00  0.00           O
        ATOM      2  H   HOH     0      15.009  16.295  15.615  1.00  0.00           H
        ATOM      3  H   HOH     0      16.408  15.115  15.615  1.00  0.00           H
        ATOM      4  OW  SOL     1       5.690  12.751  11.651  1.00  0.00           O
        ATOM      5  HW1 SOL     1       4.760  12.681  11.281  1.00  0.00           H
        ATOM      6  HW2 SOL     1       5.800  13.641  12.091  1.00  0.00           H
        TER
        ENDMDL""")

        coordsRef = np.zeros((6, 3))
        coordsRef[0, :] = [15.427, 15.434, 15.615]
        coordsRef[1, :] = [15.009, 16.295, 15.615]
        coordsRef[2, :] = [16.408, 15.115, 15.615]
        coordsRef[3, :] = [5.690, 12.751, 11.651]
        coordsRef[4, :] = [4.760, 12.681, 11.281]
        coordsRef[5, :] = [5.800, 13.641, 12.091]
        typesRef = np.zeros((6), dtype=int)
        typesRef[:] = [0, 1, 1, 0, 1, 1]
        symbRef = ["O", "H"]
        file = "test_pdb.pdb"
        with open(file, "w") as f:
            f.write(pdb_file_content)
        try:
            _, symbols, types, coords = read_pdb_file(file, lib=None, verb=False)
            if symbols != symbRef:
                passed = False
            if not np.allclose(coords, coordsRef):
                passed = False
            if not np.allclose(types, typesRef):
                passed = False
        except Exception:
            passed = False

        self.assertTrue(passed)

    def test_write_pdb_coordinates(self):
        passed = True
        pdb_file_content = textwrap.dedent("""TITLE  PDB written by SEDACS
        CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
        MODEL
        ATOM      1  O   MOL     1      15.427  15.434  15.615  1.00  0.00           O
        ATOM      2  H   MOL     1      15.009  16.295  15.615  1.00  0.00           H
        ATOM      3  H   MOL     1      16.408  15.115  15.615  1.00  0.00           H
        ATOM      4  O   MOL     1      5.690  12.751  11.651  1.00  0.00           O
        ATOM      5  H   MOL     1      4.760  12.681  11.281  1.00  0.00           H
        ATOM      6  H   MOL     1      5.800  13.641  12.091  1.00  0.00           H
        TER
        END""")

        coords = np.zeros((6, 3))
        coords[0, :] = [15.427, 15.434, 15.615]
        coords[1, :] = [15.009, 16.295, 15.615]
        coords[2, :] = [16.408, 15.115, 15.615]
        coords[3, :] = [5.690, 12.751, 11.651]
        coords[4, :] = [4.760, 12.681, 11.281]
        coords[5, :] = [5.800, 13.641, 12.091]
        types = np.zeros((6), dtype=int)
        types[:] = [0, 1, 1, 0, 1, 1]
        symb = ["O", "H"]
        with open("ref.pdb", "w") as f:
            f.write(pdb_file_content)
        try:
            write_pdb_coordinates("actual.pdb", coords, types, symb, molIds=np.zeros((0), dtype=int))
            if not are_files_equivalent("actual.pdb", "ref.pdb"):
                passed = False
        except Exception:
            passed = False

        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
