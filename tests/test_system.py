import unittest

import numpy as np
from sedacs.message import sdc_test_fail, sdc_test_pass
from sedacs.periodic_table import PeriodicTable
from sedacs.system import (
    build_nlist,
    coords_cart_to_frac,
    coords_dvec_nlist,
    coords_frac_to_cart,
    get_volBox,
    parameters_to_vectors,
    vectors_to_parameters,
)


class TestConversion(unittest.TestCase):
    def test_parameters_to_vectors(self):
        paramA, paramB, paramC = 2.0, 3.0, 4.0
        angleAlpha, angleBeta, angleGamma = 90.0, 90.0, 90.0
        latticeVectors = np.zeros((3, 3))
        latticeVectors = parameters_to_vectors(
            paramA, paramB, paramC, angleAlpha, angleBeta, angleGamma, latticeVectors
        )
        expected_result = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

        np.testing.assert_allclose(latticeVectors, expected_result, atol=1e-15)

    def test_vectors_to_parameters(self):
        latticeVectors = np.zeros((3, 3))
        latticeVectors[0, :] = [1.0, 2.0, 3.0]
        latticeVectors[1, :] = [1.0, 0.0, 0.0]
        latticeVectors[2, :] = [3.0, 2.0, 1.0]
        paramsRef = np.zeros((6))
        paramsRef = [3.74165739, 1.0, 3.74165739, 36.6992252, 44.4153086, 74.49864043]
        params = vectors_to_parameters(latticeVectors, verb=False)

        np.testing.assert_allclose(paramsRef, params)

    def test_coords_cart_to_frac(self):
        passed = True
        pt = PeriodicTable()
        nats = len(pt.symbols)
        coords = np.zeros((nats, 3))
        for i in range(len(pt.symbols)):
            coords[i, 0] = float(i)
            coords[i, 1] = float(i) + 2.0
            coords[i, 2] = float(i) + 3.0
        latticeVectors = np.array([
            [(np.max(coords[:, 0]) + 2.0) / 2.0, 0.0, 0.0],
            [0.0, (np.max(coords[:, 1]) + 2.0) / 2.0, 0.0],
            [0.0, 0.0, (np.max(coords[:, 2]) + 2.0) / 2.0],
        ])
        ref_coords = np.matmul(coords, np.linalg.inv(latticeVectors))
        test_coords = coords_cart_to_frac(coords, latticeVectors)

        np.testing.assert_allclose(test_coords, ref_coords)

        if passed:
            sdc_test_pass("coords_cart_to_frac")
        else:
            sdc_test_fail("coords_cart_to_frac")

        self.assertTrue(passed)

    def test_coords_frac_to_cart(self):
        passed = True
        pt = PeriodicTable()
        nats = len(pt.symbols)
        coords = np.zeros((nats, 3))
        for i in range(len(pt.symbols)):
            coords[i, 0] = float(i)
            coords[i, 1] = float(i) + 2.0
            coords[i, 2] = float(i) + 3.0
        latticeVectors = np.array([
            [(np.max(coords[:, 0]) + 2.0) / 2.0, 0.0, 0.0],
            [0.0, (np.max(coords[:, 1]) + 2.0) / 2.0, 0.0],
            [0.0, 0.0, (np.max(coords[:, 2]) + 2.0) / 2.0],
        ])
        coords = np.matmul(coords, np.linalg.inv(latticeVectors))
        ref_coords = np.matmul(coords, latticeVectors)
        test_coords = coords_frac_to_cart(coords, latticeVectors)

        np.testing.assert_allclose(test_coords, ref_coords)

        if passed:
            sdc_test_pass("coords_frac_to_cart")
        else:
            sdc_test_fail("coords_frac_to_cart")

        self.assertTrue(passed)


class TestNeighborList(unittest.TestCase):
    def test_get_volBox(self):
        volBoxRef = 4.0
        latticeVectors = np.zeros((3, 3))
        latticeVectors[0, :] = [1.0, 2.0, 3.0]
        latticeVectors[1, :] = [1.0, 0.0, 0.0]
        latticeVectors[2, :] = [3.0, 2.0, 1.0]
        volBox = get_volBox(latticeVectors, verb=False)
        if abs(volBox - volBoxRef) == 0.0:
            passed = True
        else:
            passed = False

        self.assertTrue(passed)

    def test_coords_dvec_nlist(self):
        passed = True
        try:
            pt = PeriodicTable()
            nats = len(pt.symbols)
            coords = np.zeros((nats, 3))
            for i in range(len(pt.symbols)):
                coords[i, 0] = float(i)
                coords[i, 1] = float(i) + 2.0
                coords[i, 2] = float(i) + 3.0
            latticeVectors = np.array([
                [(np.max(coords[:, 0]) + 2.0) / 2.0, 0.0, 0.0],
                [0.0, (np.max(coords[:, 1]) + 2.0) / 2.0, 0.0],
                [0.0, 0.0, (np.max(coords[:, 2]) + 2.0) / 2.0],
            ])
            latticeLengths = latticeVectors.diagonal()
            rcut = 4.0
            nn, nl, nlTr = build_nlist(coords, latticeVectors, rcut=rcut, api="new")
            dvec = np.zeros((nl.shape[0], nl.shape[1], 3), dtype=coords.dtype)
            for i in range(coords.shape[0]):
                for k in range(3):
                    dvec[i, 0 : nn[i], k] = (coords[i, k] - coords[nl[i, 0 : nn[i]], k]) - nlTr[
                        i, 0 : nn[i], k
                    ] * latticeLengths[k]

            dr = np.zeros(nl.shape, dtype=coords.dtype)
            for i in range(dvec.shape[0]):
                dr[i, 0 : nn[i]] = np.linalg.norm(dvec[i, 0 : nn[i], :], axis=1)
                # dr[i,0:nn[i]] = np.sqrt(np.sum(dvec[:,i,0:nn[i]]**2,axis=0))
            ref_dr = dr
            ref_dvec = dvec
            test_dvec, test_dr = coords_dvec_nlist(coords, nn, nl, nlTr, latticeVectors)

            np.testing.assert_allclose(test_dr, ref_dr)
            np.testing.assert_allclose(test_dvec, ref_dvec)
        except Exception:
            print("test_coords_dvec_nlist failed to execute")
            passed = False

        if passed:
            sdc_test_pass("coords_dvec_nlist")
        else:
            sdc_test_fail("coords_dvec_nlist")

        self.assertTrue(passed)

    # FIXME: this will fail!
    def test_build_nlist(self):
        passed = True
        try:
            pt = PeriodicTable()
            nats = len(pt.symbols)
            coords = np.zeros((nats, 3))
            for i in range(len(pt.symbols)):
                coords[i, 0] = float(i)
                coords[i, 1] = float(i) + 2.0
                coords[i, 2] = float(i) + 3.0
            latticeVectors = np.array([
                [(np.max(coords[:, 0]) + 2.0) / 2.0, 0.0, 0.0],
                [0.0, (np.max(coords[:, 1]) + 2.0) / 2.0, 0.0],
                [0.0, 0.0, (np.max(coords[:, 2]) + 2.0) / 2.0],
            ])
            coords = np.matmul(coords, np.linalg.inv(latticeVectors))
            rcut = 4.0
            density = 1.0
            maxneigh = np.min([int(3.14592 * (4.0 / 3.0) * density * rcut**3), nats])
            dvec = np.empty(coords.shape, dtype=coords.dtype)
            nlTrvec = np.empty(coords.shape, dtype=int)
            nl = np.zeros([nats, maxneigh], dtype=int)
            nlTr = np.empty([nats, maxneigh, 3], dtype=int)
            for i in range(nats):
                for k in range(3):
                    # Compute the integer lattice vector translation first
                    dvec[:, k] = coords[i, k] - coords[:, k] + 0.5
                    nlTrvec[:, k] = np.floor(dvec[:, k])
                    # Now use the translation to compute the periodic displacement
                    dvec[:, k] = dvec[:, k] - nlTrvec[:, k] - 0.5
                distance = np.linalg.norm(coords_frac_to_cart(dvec, latticeVectors), axis=1)
                # Filter the list according to the threshold
                nlSel = np.where(distance < rcut)[0]
                nlSel = nlSel[nlSel != i]
                cnt = len(nlSel)
                nl[i, 1 : cnt + 1] = nlSel
                nlTr[i, 1 : cnt + 1] = nlTrvec[:cnt]
                nl[i, 0] = cnt
                nlTr[i, 0] = cnt
            ref_nl = nl[:, 1:]
            ref_nlTr = nlTr[:, 1:, :]
            coords = np.matmul(coords, latticeVectors)
            test_nn, test_nl, test_nlTr = build_nlist(coords, latticeVectors, rcut=rcut, api="new")
            for i in range(nats):
                sort_indices = np.argsort(test_nl[i, : test_nn[i]])
                test_nl[i, : test_nn[i]] = test_nl[i, sort_indices]
                test_nlTr[i, : test_nn[i], :] = test_nlTr[i, sort_indices, :]

            np.testing.assert_array_equal(test_nl, ref_nl)
            np.testing.assert_array_equal(test_nlTr, ref_nlTr)
        except Exception:
            print("test_build_nlist failed to execute")
            passed = False

        if passed:
            sdc_test_pass("build_nlist")
        else:
            sdc_test_fail("build_nlist")

        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
