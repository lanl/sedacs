import unittest

from sedacs.periodic_table import PeriodicTable


class TestPeriodicTable(unittest.TestCase):
    def test_carbon(self):
        passed = False
        tolerance = 1e-4
        # Get values for Carbon
        pt = PeriodicTable()
        atnum = pt.get_atomic_number("C")
        passed = True

        if atnum != 6:
            passed = False
            # print("0")
        if pt.names[atnum] != "Carbon":
            passed = False
            # print("1")
        if abs(pt.mass[atnum] - 12.0) > tolerance:
            passed = False
            # print("2")
        if abs(pt.vdwr[atnum] - 1.7) > tolerance:
            passed = False
            # print("3")
        if abs(pt.covr[atnum] - 0.76) > tolerance:
            passed = False
            # print("4")
        if abs(pt.ip[atnum] - 11.2603) > tolerance:
            passed = False
            # print("5")
        if abs(pt.ea[atnum] - 1.262118) > tolerance:
            passed = False
            # print("6")
        if abs(pt.en[atnum] - 2.55) > tolerance:
            passed = False
            # print("7")
        if pt.maxbonds[atnum] != 4:
            passed = False
            # print("8")
        if pt.numel[atnum] != 4:
            passed = False
            # print("9")
        if pt.econf[atnum] != "1s22s22p2":
            passed = False

        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
