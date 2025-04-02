import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh


###
# @brief Constructs a Hamiltonian and Overlap for LATTE Tight-binding.
# This is a python version of bldnewH.F90 LATTE routine
#
##
def build_HS(
    nats,
    coords_in,
    box,
    elempointer,
    basis,
    hes,
    hep,
    hed,
    hef,
    hdim,
    cutoffList,
    iglMap,
    tabh,
    tabs,
    sspl,
    hspl,
    lentabint,
    tabr,
    hcut,
    scut,
    verbose,
):
    if verbose:
        print("In build_HS ...")

    # Indexing transformation coumn mayor to row mayor
    coords = np.zeros((3, nats))
    coords[0, :] = coords_in[:, 0]
    coords[1, :] = coords_in[:, 1]
    coords[2, :] = coords_in[:, 2]

    pi = 3.1415926535
    ham = np.zeros((hdim, hdim))
    smat = np.eye((hdim))
    mybondint = np.zeros((4))
    myoverlapint = np.zeros((4))

    # Build diagonal elements (pre-calculated)
    ham_onsite = gen_Honsite(nats, elempointer, basis, hes, hep, hed, hef, hdim, verbose)
    count = 0
    for i in range(hdim):
        ham[i, i] = ham_onsite[i]

    orbitalList = get_orbitalList(nats, elempointer, basis)

    matindList = get_matindlist(nats, elempointer, basis)
    rij = np.zeros((3))
    for i in range(nats):
        basisi = orbitalList[:, i]
        indi = matindList[i]
        for j in range(nats):
            for ix in range(-1, 2):
                tx = ix * box[0, :]
                for iy in range(-1, 2):
                    ty = iy * box[1, :]
                    for iz in range(-1, 2):
                        tz = iz * box[2, :]
                        neigh = coords[:, j] + tx + ty + tz

                        # print(neigh)
                        rij[0] = neigh[0] - coords[0, i]
                        rij[1] = neigh[1] - coords[1, i]
                        rij[2] = neigh[2] - coords[2, i]
                        magr2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]
                        rcuttb = cutoffList[j, i]

                        if (magr2 < rcuttb * rcuttb) and (magr2 > 1.0e-12):
                            magr = np.sqrt(magr2)
                            basisj = orbitalList[:, j]
                            indj = matindList[j]
                            magrp = np.sqrt(rij[0] * rij[0] + rij[1] * rij[1])
                            if abs(rij[0]) > 1.0e-12:
                                if (rij[0] > 0.0) and (rij[1] >= 0.0):
                                    phi = 0.0
                                elif (rij[0] > 0.0) and (rij[1] < 0.0):
                                    phi = 2.0 * pi
                                else:
                                    phi = pi
                                alpha = np.arctan(rij[1] / rij[0]) + phi
                            elif abs(rij[1]) > 1.0e-12:
                                if rij[1] > 1.0e-12:
                                    alpha = pi / 2.0
                                else:
                                    alpha = 3.0 * pi / 2.0
                            else:
                                alpha = 0.0

                            cosbeta = rij[2] / magr
                            beta = np.arccos(rij[2] / magr)

                            # Build matrix elements using eqns (1)-(9) in PRB 72 165107

                            # The loops over LBRA and LKET need to take into account
                            # the orbitals assigned to each atom, e.g., sd rather than
                            # spd...
                            ibra = indi + 1
                            lbrainc = 1
                            while (basisi[lbrainc - 1]) != -1:
                                lbra = basisi[lbrainc - 1]
                                lbrainc = lbrainc + 1
                                for mbra in range(-lbra, lbra + 1):
                                    # We can calculate these two outside the
                                    # MKET loop
                                    iket = indj + 1
                                    lketinc = 1
                                    while basisj[lketinc - 1] != -1:
                                        lket = basisj[lketinc - 1]
                                        lketinc = lketinc + 1
                                        # Precompute the integrals outside Mket loop
                                        for mp in range(min(lbra, lket) + 1):
                                            mybondint[mp] = univscale(
                                                i,
                                                j,
                                                lbra,
                                                lket,
                                                mp,
                                                magr,
                                                tabs,
                                                tabh,
                                                sspl,
                                                hspl,
                                                iglMap,
                                                lentabint,
                                                tabr,
                                                tabh,
                                                hcut,
                                                scut,
                                                "H",
                                            )
                                            myoverlapint[mp] = univscale(
                                                i,
                                                j,
                                                lbra,
                                                lket,
                                                mp,
                                                magr,
                                                tabs,
                                                tabh,
                                                sspl,
                                                hspl,
                                                iglMap,
                                                lentabint,
                                                tabr,
                                                tabh,
                                                hcut,
                                                scut,
                                                "S",
                                            )

                                        for mket in range(-lket, lket + 1):
                                            # This is the sigma bonds (mp = 0)
                                            # Hamiltonian build
                                            # Pre-compute the angular part so we can use it
                                            # again later if we're building the S matrix too
                                            myangfactor = angfactor(lbra, lket, mbra, mket, 0, alpha, cosbeta)
                                            ham[ibra - 1, iket - 1] = (
                                                ham[ibra - 1, iket - 1] + myangfactor * mybondint[0]
                                            )
                                            smat[ibra - 1, iket - 1] = (
                                                smat[ibra - 1, iket - 1] + myangfactor * myoverlapint[0]
                                            )
                                            # Everything else
                                            count = count + 1
                                            for mp in range(1, min(lbra, lket) + 1):
                                                myangfactor = angfactor(lbra, lket, mbra, mket, mp, alpha, cosbeta)
                                                ham[ibra - 1, iket - 1] = (
                                                    ham[ibra - 1, iket - 1] + myangfactor * mybondint[mp]
                                                )
                                                smat[ibra - 1, iket - 1] = (
                                                    smat[ibra - 1, iket - 1] + myangfactor * myoverlapint[mp]
                                                )

                                            iket = iket + 1

                                    ibra = ibra + 1

    if verbose:
        print("\nHamiltonian Matrix")
        print(ham)
        print("\nOverlap Matrix")
        print(smat)

    return (smat, ham)


def get_hdim(nats, elempointer, basis, verbose):
    if verbose:
        print("\nIn get_hdim ...")

    hdim = 0
    for i in range(nats):
        if basis[elempointer[i]] == "s":
            numorb = 1
        elif basis[elempointer[i]] == "p":
            numorb = 3
        elif basis[elempointer[i]] == "d":
            numorb = 5
        elif basis[elempointer[i]] == "f":
            numorb = 7
        elif basis[elempointer[i]] == "sp":
            numorb = 4
        elif basis[elempointer[i]] == "sd":
            numorb = 6
        elif basis[elempointer[i]] == "sf":
            numorb = 8
        elif basis[elempointer[i]] == "pd":
            numorb = 8
        elif basis[elempointer[i]] == "pf":
            numorb = 10
        elif basis[elempointer[i]] == "df":
            numorb = 12
        elif basis[elempointer[i]] == "spd":
            numorb = 9
        elif basis[elempointer[i]] == "spf":
            numorb = 11
        elif basis[elempointer[i]] == "sdf":
            numorb = 13
        elif basis[elempointer[i]] == "pdf":
            numorb = 15
        elif basis[elempointer[i]] == "spdf":
            numorb = 16
        hdim = hdim + numorb

    return hdim


###
# @brief Get onsite Hamiltonian elements for LATTE Tight-binding.
# This is a python version of genHonsite.F90 LATTE routine
#
##
def gen_Honsite(nats, elempointer, basis, hes, hep, hed, hef, hdim, verbose):
    ham_onsite = np.zeros((hdim))
    index = -1
    for i in range(nats):
        k = elempointer[i]

        if basis[k] == "s":
            index = index + 1
            ham_onsite[index] = hes[k]
        elif basis[k] == "p":
            for subi in range(3):
                index = index + 1
                ham_onsite[index] = hep[k]
        elif basis[k] == "d":
            for subi in range(5):
                index = index + 1
                ham_onsite[index] = hed[k]
        elif basis[k] == "f":
            for subi in range(7):
                index = index + 1
                ham_onsite[index] = hef[k]
        elif basis[k] == "sp":
            for subi in range(4):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                else:
                    ham_onsite[index] = hep[k]
        elif basis[k] == "sd":
            for subi in range(6):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                else:
                    ham_onsite[index] = hed[k]
        elif basis[k] == "sf":
            for subi in range(8):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "pd":
            for subi in range(8):
                index = index + 1
                if subi <= 2:
                    ham_onsite[index] = hep[k]
                else:
                    ham_onsite[index] = hed[k]
        elif basis[k] == "pf":
            for subi in range(10):
                index = index + 1
                if subi <= 2:
                    ham_onsite[index] = hep[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "df":
            for subi in range(12):
                index = index + 1
                if subi <= 4:
                    ham_onsite[index] = hed[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "spd":
            for subi in range(9):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                elif subi > 0 & subi <= 3:
                    ham_onsite[index] = hep[k]
                else:
                    ham_onsite[index] = hed[k]
        elif basis[k] == "spf":
            for subi in range(11):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                elif subi > 0 & subi <= 3:
                    ham_onsite[index] = hep[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "sdf":
            for subi in range(13):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                elif subi > 0 & subi <= 5:
                    ham_onsite[index] = hed[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "pdf":
            for subi in range(15):
                index = index + 1
                if subi <= 2:
                    ham_onsite[index] = hep[k]
                elif subi > 2 & subi <= 7:
                    ham_onsite[index] = hed[k]
                else:
                    ham_onsite[index] = hef[k]
        elif basis[k] == "spdf":
            for subi in range(16):
                index = index + 1
                if subi == 0:
                    ham_onsite[index] = hes[k]
                elif subi > 0 & subi <= 3:
                    ham_onsite[index] = hep[k]
                elif subi > 3 & subi <= 8:
                    ham_onsite[index] = hed[k]
                else:
                    ham_onsite[index] = hef[k]
    return ham_onsite


def get_orbitalList(nats, elempointer, basis):
    orbital_list = np.zeros((5, nats), dtype=int)
    for i in range(nats):
        if basis[elempointer[i]] == "s":
            orbital_list[0, i] = 0
            orbital_list[1, i] = -1
        elif basis[elempointer[i]] == "p":
            orbital_list[0, i] = 0
            orbital_list[1, i] = -1
        elif basis[elempointer[i]] == "d":
            orbital_list[0, i] = 2
            orbital_list[1, i] = -1
        elif basis[elempointer[i]] == "f":
            orbital_list[0, i] = 3
            orbital_list[1, i] = -1
        elif basis[elempointer[i]] == "sp":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 1
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "sd":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 2
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "sf":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 3
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "pd":
            orbital_list[0, i] = 1
            orbital_list[1, i] = 2
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "pf":
            orbital_list[0, i] = 1
            orbital_list[1, i] = 3
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "df":
            orbital_list[0, i] = 2
            orbital_list[1, i] = 3
            orbital_list[2, i] = -1
        elif basis[elempointer[i]] == "spd":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 1
            orbital_list[2, i] = 2
            orbital_list[3, i] = -1
        elif basis[elempointer[i]] == "spf":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 1
            orbital_list[2, i] = 3
            orbital_list[3, i] = -1
        elif basis[elempointer[i]] == "sdf":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 2
            orbital_list[2, i] = 3
            orbital_list[3, i] = -1
        elif basis[elempointer[i]] == "pdf":
            orbital_list[0, i] = 1
            orbital_list[1, i] = 2
            orbital_list[2, i] = 3
            orbital_list[3, i] = -1
        elif basis[elempointer[i]] == "spdf":
            orbital_list[0, i] = 0
            orbital_list[1, i] = 1
            orbital_list[2, i] = 2
            orbital_list[3, i] = 3
            orbital_list[4, i] = -1

    return orbital_list


def get_matindlist(nats, elempointer, basis):
    matindlist = np.zeros((nats), dtype=int)

    for i in range(nats):
        indi = 0
        for j in range(i):
            if basis[elempointer[j]] == "s":
                numorb = 1
            elif basis[elempointer[j]] == "p":
                numorb = 3
            elif basis[elempointer[j]] == "d":
                numorb = 5
            elif basis[elempointer[j]] == "f":
                numorb = 7
            elif basis[elempointer[j]] == "sp":
                numorb = 4
            elif basis[elempointer[j]] == "sd":
                numorb = 6
            elif basis[elempointer[j]] == "sf":
                numorb = 8
            elif basis[elempointer[j]] == "pd":
                numorb = 8
            elif basis[elempointer[j]] == "pf":
                numorb = 10
            elif basis[elempointer[j]] == "df":
                numorb = 12
            elif basis[elempointer[j]] == "spd":
                numorb = 9
            elif basis[elempointer[j]] == "spf":
                numorb = 11
            elif basis[elempointer[j]] == "sdf":
                numorb = 13
            elif basis[elempointer[j]] == "pdf":
                numorb = 15
            elif basis[elempointer[j]] == "spdf":
                numorb = 16
            indi = indi + numorb
        matindlist[i] = indi

    return matindlist


## ANDERS CHANGES get the net partial occupations of the fully isolated atoms
def get_znuc(atele, ele, atocc, verbose):
    if verbose:
        print("\nIn get_znuc ...")

    znuc = np.zeros(len(atele))

    for i in range(len(atele)):
        for j in range(len(ele)):
            if atele[i] == ele[j]:
                znuc[i] = atocc[j]
    return znuc


## ANDERS CHANGES  easier indexing for matrices
def get_hindex(nats, elempointer, basis, hdim, verbose):
    if verbose:
        print("\nIn get_hindex ...")

    matindList = get_matindlist(nats, elempointer, basis)
    h_start = np.ones(nats, dtype=int)
    h_stop = np.ones(nats, dtype=int)
    h_start[0:nats] = matindList[0:nats]
    h_stop[0 : nats - 1] = matindList[1:nats]
    h_stop[nats - 1] = hdim

    return (h_start, h_stop)


def get_btypeInt(noint, btype, atele, ele1, ele2, nats, verbose):
    if verbose:
        print("\nIn get_btypeInt ...")

    lMax = 0
    mpMax = 0
    btypeInt = np.zeros((3, noint), dtype=int)
    for i in range(noint):
        if btype[i] == "sss":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 0
            btypeInt[2, i] = 0
        elif btype[i] == "sps":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 1
            btypeInt[2, i] = 0
        elif btype[i] == "sds":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "sfs":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "pps":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 1
            btypeInt[2, i] = 0
        elif btype[i] == "ppp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 1
            btypeInt[2, i] = 1
        elif btype[i] == "pds":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "pdp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 2
            btypeInt[2, i] = 1
        elif btype[i] == "pfs":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "pfp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "dds":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "ddp":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 1
        elif btype[i] == "ddd":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 2
        elif btype[i] == "dfs":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "dfp":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "dfd":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 2
        elif btype[i] == "ffs":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "ffp":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "ffd":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 2
        elif btype[i] == "fff":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 3
        else:
            print("Warning! Missed assigning a bond type in readtb")
            sys.exit(0)

        if btypeInt[1, i] > lMax:
            lMax = btypeInt[1, i]
        if btypeInt[2, i] > mpMax:
            mpMax = btypeInt[2, i]

    return lMax, mpMax, btypeInt


def build_integralMap_I(noint, btype, atele, ele1, ele2, nats, lMax, mpMax, btypeInt, i):
    iglMapI = np.zeros((mpMax + 1, lMax + 1, lMax + 1, nats), dtype=int)

    for j in range(nats):
        for l1 in range(lMax + 1):
            for l2 in range(lMax + 1):
                if l1 > l2:
                    ip1 = l2
                    ip2 = l1
                else:
                    ip1 = l1
                    ip2 = l2
                for mp in range(min(l1, l2) + 1):
                    # Build basis strind from L and M values - pure hackery
                    if ip1 == 0:
                        igltype = "s"
                    elif ip1 == 1:
                        igltype = "p"
                    elif ip1 == 2:
                        igltype = "d"
                    elif ip1 == 3:
                        igltype = "f"

                    if ip2 == 0:
                        igltype = igltype + "s"
                    elif ip2 == 1:
                        igltype = igltype + "p"
                    elif ip2 == 2:
                        igltype = igltype + "d"
                    elif ip2 == 3:
                        igltype = igltype + "f"

                    if mp == 0:
                        igltype = igltype + "s"
                    elif mp == 1:
                        igltype = igltype + "p"
                    elif mp == 2:
                        igltype = igltype + "d"
                    elif mp == 3:
                        igltype = igltype + "f"

                    # It makes a difference if our atoms are of the species or not...

                    # Easier case first ATELE(I) = ATELE(J)

                    if atele[i] == atele[j]:
                        for ic in range(noint):
                            if (atele[i] == ele1[ic]) and (atele[j] == ele2[ic]) and (igltype == btype[ic]):
                                iglMapI[mp, l2, l1, j] = ic

                    else:
                        # Elements are different - care must be taken with p-s, s-p etc
                        if l1 == l2:
                            for ic in range(noint):
                                if (
                                    ((atele[i] == ele1[ic]) and (atele[j] == ele2[ic]))
                                    or ((atele[i] == ele2[ic]) and (atele[j] == ele1[ic]))
                                ) and (igltype == btype[ic]):
                                    # Now we've ID'ed our bond integral
                                    iglMapI[mp, l2, l1, j] = ic
                                    breakloop = 1
                        else:  # l1 ne l2
                            if l1 < l2:
                                for ic in range(noint):
                                    if ((atele[i] == ele1[ic]) and (atele[j] == ele2[ic])) and (igltype == btype[ic]):
                                        # Now we've ID'ed our bond integral
                                        iglMapI[mp, l2, l1, j] = ic
                            else:
                                for ic in range(noint):
                                    if ((atele[i] == ele2[ic]) and (atele[j] == ele1[ic])) and (igltype == btype[ic]):
                                        # Now we've ID'ed our bond integral
                                        iglMapI[mp, l2, l1, j] = ic

    return iglMapI


def build_integralMap_parPool(noint, btype, atele, ele1, ele2, nats, verbose, parallel=1):
    if verbose:
        print("\nIn build_integralMap_parPool ...")

    lMax, mpMax, btypeInt = get_btypeInt(noint, btype, atele, ele1, ele2, nats, verbose)
    iglMap = np.zeros((mpMax + 1, lMax + 1, lMax + 1, nats, nats), dtype=int)

    if parallel > 1:
        pool = Pool(processes=parallel)
        results = [
            pool.apply_async(build_integralMap_I, [noint, btype, atele, ele1, ele2, nats, lMax, mpMax, btypeInt, val])
            for val in range(0, nats)
        ]
        for idx, val in enumerate(results):
            iglMap[:, :, :, :, idx] = val.get()
        pool.close()
    elif parallel == 1:
        for val in range(nats):
            iglMap[:, :, :, :, val] = build_integralMap_I(
                noint, btype, atele, ele1, ele2, nats, lMax, mpMax, btypeInt, val
            )

    return iglMap


def build_integralMap(noint, btype, atele, ele1, ele2, nats, verbose):
    if verbose:
        print("\nIn build_integralMap ...")

    lMax = 0
    mpMax = 0
    btypeInt = np.zeros((3, noint), dtype=int)
    for i in range(noint):
        if btype[i] == "sss":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 0
            btypeInt[2, i] = 0
        elif btype[i] == "sps":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 1
            btypeInt[2, i] = 0
        elif btype[i] == "sds":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "sfs":
            btypeInt[0, i] = 0
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "pps":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 1
            btypeInt[2, i] = 0
        elif btype[i] == "ppp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 1
            btypeInt[2, i] = 1
        elif btype[i] == "pds":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "pdp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 2
            btypeInt[2, i] = 1
        elif btype[i] == "pfs":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "pfp":
            btypeInt[0, i] = 1
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "dds":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 0
        elif btype[i] == "ddp":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 1
        elif btype[i] == "ddd":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 2
            btypeInt[2, i] = 2
        elif btype[i] == "dfs":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "dfp":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "dfd":
            btypeInt[0, i] = 2
            btypeInt[1, i] = 3
            btypeInt[2, i] = 2
        elif btype[i] == "ffs":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 0
        elif btype[i] == "ffp":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 1
        elif btype[i] == "ffd":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 2
        elif btype[i] == "fff":
            btypeInt[0, i] = 3
            btypeInt[1, i] = 3
            btypeInt[2, i] = 3
        else:
            print("Warning! Missed assigning a bond type in readtb")
            sys.exit(0)

        if btypeInt[1, i] > lMax:
            lMax = btypeInt[1, i]
        if btypeInt[2, i] > mpMax:
            mpMax = btypeInt[2, i]

    iglMap = np.zeros((mpMax + 1, lMax + 1, lMax + 1, nats, nats), dtype=int)
    for i in range(nats):
        for j in range(nats):
            for l1 in range(lMax + 1):
                for l2 in range(lMax + 1):
                    if l1 > l2:
                        ip1 = l2
                        ip2 = l1
                    else:
                        ip1 = l1
                        ip2 = l2
                    for mp in range(min(l1, l2) + 1):
                        # Build basis strind from L and M values - pure hackery
                        if ip1 == 0:
                            igltype = "s"
                        elif ip1 == 1:
                            igltype = "p"
                        elif ip1 == 2:
                            igltype = "d"
                        elif ip1 == 3:
                            igltype = "f"

                        if ip2 == 0:
                            igltype = igltype + "s"
                        elif ip2 == 1:
                            igltype = igltype + "p"
                        elif ip2 == 2:
                            igltype = igltype + "d"
                        elif ip2 == 3:
                            igltype = igltype + "f"

                        if mp == 0:
                            igltype = igltype + "s"
                        elif mp == 1:
                            igltype = igltype + "p"
                        elif mp == 2:
                            igltype = igltype + "d"
                        elif mp == 3:
                            igltype = igltype + "f"

                        # It makes a difference if our atoms are of the species or not...

                        # Easier case first ATELE(I) = ATELE(J)

                        if atele[i] == atele[j]:
                            for ic in range(noint):
                                if (atele[i] == ele1[ic]) and (atele[j] == ele2[ic]) and (igltype == btype[ic]):
                                    iglMap[mp, l2, l1, j, i] = ic

                        else:
                            # Elements are different - care must be taken with p-s, s-p etc
                            if l1 == l2:
                                for ic in range(noint):
                                    if (
                                        ((atele[i] == ele1[ic]) and (atele[j] == ele2[ic]))
                                        or ((atele[i] == ele2[ic]) and (atele[j] == ele1[ic]))
                                    ) and (igltype == btype[ic]):
                                        # Now we've ID'ed our bond integral
                                        iglMap[mp, l2, l1, j, i] = ic
                                        breakloop = 1
                            else:  # l1 ne l2
                                if l1 < l2:
                                    for ic in range(noint):
                                        if ((atele[i] == ele1[ic]) and (atele[j] == ele2[ic])) and (
                                            igltype == btype[ic]
                                        ):
                                            # Now we've ID'ed our bond integral
                                            iglMap[mp, l2, l1, j, i] = ic
                                else:
                                    for ic in range(noint):
                                        if ((atele[i] == ele2[ic]) and (atele[j] == ele1[ic])) and (
                                            igltype == btype[ic]
                                        ):
                                            # Now we've ID'ed our bond integral
                                            iglMap[mp, l2, l1, j, i] = ic

    return iglMap


def univscale(i, j, l1, l2, mp, r, tabs, tbh, sspl, hspl, iglMap, lentabint, tabr, tabh, hcut, scut, whichint):
    myIntegral = iglMap[mp, l2, l1, j, i]
    klo = 1
    khi = lentabint[myIntegral]
    while (khi - klo) > 1:
        k = (khi + klo) / 2
        if tabr[int(k), myIntegral] > r:
            khi = k
        else:
            klo = k
    khi = int(khi)
    klo = int(klo)
    dx = tabr[(khi), (myIntegral)] - tabr[(klo), (myIntegral)]
    sa = (tabr[(khi), (myIntegral)] - r) / dx
    sb = (r - tabr[(klo), (myIntegral)]) / dx
    if whichint == "H":
        univscale = (
            sa * tabh[klo, myIntegral]
            + sb * tabh[khi, myIntegral]
            + ((sa * sa * sa - sa) * hspl[klo, myIntegral] + (sb * sb * sb - sb) * hspl[khi, myIntegral])
            * (dx * dx / 6.0)
        )
        if r > hcut[myIntegral]:
            univscale = 0.0
    else:
        univscale = (
            sa * tabs[klo, myIntegral]
            + sb * tabs[khi, myIntegral]
            + ((sa * sa * sa - sa) * sspl[klo, myIntegral] + (sb * sb * sb - sb) * sspl[khi, myIntegral])
            * (dx * dx / 6.0)
        )
        if r > scut[myIntegral]:
            univscale = 0.0

    if (l1 > l2) and ((l1 + l2 % 2) != 0):
        univscale = -1.0 * univscale

    return univscale


def angfactor(lbra, lket, mbra, mket, mp, alpha, cosbeta):
    mytlmmp = 0.0
    wig_lbra_mbra_mp = 0.0
    wig_lket_mket_mp = 0.0
    if mp == 0:
        wig_lbra_mbra_0 = wignerd(lbra, abs(mbra), 0, cosbeta)
        wig_lket_mket_0 = wignerd(lket, abs(mket), 0, cosbeta)
        angfactor = 2.0 * (am(mbra, alpha) * wig_lbra_mbra_0) * (am(mket, alpha) * wig_lket_mket_0)
    else:
        wig_lbra_mbra_mp = wignerd(lbra, abs(mbra), mp, cosbeta)
        wig_lket_mket_mp = wignerd(lket, abs(mket), mp, cosbeta)
        wig_lbra_mbra_negmp = wignerd(lbra, abs(mbra), -mp, cosbeta)
        wig_lket_mket_negmp = wignerd(lket, abs(mket), -mp, cosbeta)

        angfactor = (
            am(mbra, alpha)
            * (((-1.0) ** mp) * wig_lbra_mbra_mp + wig_lbra_mbra_negmp)
            * am(mket, alpha)
            * (((-1.0) ** mp) * wig_lket_mket_mp + wig_lket_mket_negmp)
        )

        if mbra == 0:
            mytlmmp = 0.0
        else:
            mytlmmp = bm(mbra, alpha) * (((-1.0) ** mp) * wig_lbra_mbra_mp - wig_lbra_mbra_negmp)

        if mket == 0:
            mytlmmp = 0.0
        else:
            mytlmmp = mytlmmp * bm(mket, alpha) * (((-1.0) ** mp) * wig_lket_mket_mp - wig_lket_mket_negmp)
        angfactor = angfactor + mytlmmp

    return angfactor


def am(m, alpha):
    am = 0.0
    if m == 0:
        am = 0.707106781186548  # 1/SQRT(2)
    elif m > 0:
        am = ((-1) ** m) * np.cos(abs(m) * alpha)
    elif m < 0:
        am = ((-1) ** m) * np.sin(abs(m) * alpha)
    return am


def bm(m, alpha):
    bm = 0.0
    if m == 0:
        bm = 0.0
    elif m > 0:
        bm = (-((-1) ** m)) * np.sin(abs(m) * alpha)
    elif m < 0:
        bm = ((-1) ** m) * np.cos(abs(m) * alpha)
    return bm


# Builds Wigner d function
# notation conforms to that in PRB 72 165107 (2005), eq. (9)
def wignerd(l, m, mp, cosbeta):
    wignerd = 0.0
    if abs(mp) > l:
        wignerd = 0.0
    else:
        pref = (
            (((-1.0) ** (l - mp)) / ((2.0) ** l))
            * sqrtfact(l + m)
            * sqrtfact(l - m)
            * sqrtfact(l + mp)
            * sqrtfact(l - mp)
        )
        wignered = 0.0
        for k in range(max(0, -m - mp), min(l - m, l - mp) + 1):
            powtmp = float(m + mp) / 2.0
            power1 = float(l - k) - powtmp
            power2 = float(k) + powtmp

            wignerd = wignerd + ((-1.0) ** k) * ((1.0 - cosbeta) ** power1) * ((1.0 + cosbeta) ** power2) / float(
                factorial(k) * factorial(l - m - k) * factorial(l - mp - k) * factorial(m + mp + k)
            )

        wignerd = pref * wignerd
    return wignerd


def factorial(k):
    fact = 1.0

    if k > 0:
        for l in range(1, k + 1):
            fact = fact * l

    return fact


def sqrtfact(k):
    sqrtfact = np.sqrt(factorial(k))

    return sqrtfact


###
# @brief Gets the bond integral paramters of the LATTE Tight-binding.
# This is a python version of readtb.F90 LATTE routine
#
# \param bondintsFileName Name of the file containing bond integrals
# \param electronsFileName Name of the file containing atomic data
#
# @detailed Bond integrals are the distance dependent Hamiltonian elements
# \f[
# H_{ij} = <R_i | H | R_j>
# \f]
# and
# \f[
# H_{ii} = U_i
# \f]
#
##
def read_bondints(electronsFileName, bondintsFileName, verbose):
    if verbose:
        print("In read_bondints ...")

    # Checking if the file is in the local directory
    if os.path.isfile(bondintsFileName):
        if verbose >= 1:
            print("\nWorking with file:", os.path.abspath(bondintsFileName), " ...")
    else:
        sys.exit("\nI can't find bondints in this directory")
    if os.path.isfile(electronsFileName):
        if verbose >= 1:
            print("\nWorking with file:", os.path.abspath(electronsFileName), " ...")
    else:
        sys.exit("\nI can't find bondints in this directory")

    myBondintsFile = open(bondintsFileName, "r")
    myElectronsFile = open(electronsFileName, "r")

    count = 0
    for lines in myElectronsFile:
        count = count + 1
        lines_split = lines.split()
        if len(lines_split) == 0:
            break
        if count == 1:
            noelem = int(lines_split[1])
            ele = []
            basis = []
            atocc = np.zeros((noelem))
            hes = np.zeros((noelem))
            hep = np.zeros((noelem))
            hed = np.zeros((noelem))
            hef = np.zeros((noelem))
            mass = np.zeros((noelem))
            hubbardu = np.zeros((noelem))
        if count == 2:
            pass
        if count >= 3:
            ele.append(lines_split[0])
            basis.append(lines_split[1])
            atocc[count - 3] = float(lines_split[2])
            hes[count - 3] = float(lines_split[3])
            hep[count - 3] = float(lines_split[4])
            hed[count - 3] = float(lines_split[5])
            hef[count - 3] = float(lines_split[6])
            mass[count - 3] = float(lines_split[7])
            hubbardu[count - 3] = float(lines_split[8])

    if verbose:
        print("\nTable data")
        print("Nunber of elements =", noelem)
        print("Elements =", ele)
        print("Basis sets =", basis)
        print("Atomic occupation =", atocc)
        print("Self energy s =", hes)
        print("Self energy p =", hep)
        print("Self energy d =", hed)
        print("Self energy f =", hef)
        print("Atomic masses =", mass)
        print("Hubbard Us =", hubbardu)

    count = 0
    listOfLines = []
    for lines in myBondintsFile:
        count = count + 1
        lines_split = lines.split()
        if len(lines_split) == 0:
            break
        if count == 1:
            noint = int(lines_split[1])
        if count >= 2:
            listOfLines.append(lines_split)

    # Getting the max number of entries
    maxentry = 0
    for i in range(len(listOfLines)):
        if len(listOfLines[i]) == 1:
            maxentry = max(maxentry, int(listOfLines[i][0]))

    tabr = np.zeros((maxentry, noint))
    tabh = np.zeros((maxentry, noint))
    tabs = np.zeros((maxentry, noint))
    lentabint = np.zeros((noint), dtype=int)
    hspl = np.zeros((maxentry, noint))
    sspl = np.zeros((maxentry, noint))
    hcut = np.zeros((noint))
    scut = np.zeros((noint))

    count = 0
    ele1 = []
    ele2 = []
    btype = []
    for i in range(noint):
        ele1.append(listOfLines[count][0])
        ele2.append(listOfLines[count][1])
        btype.append(listOfLines[count][2])
        count = count + 1
        lentabint[i] = int(listOfLines[count][0])
        count = count + 1
        for j in range(lentabint[i]):
            tabr[j, i] = float(listOfLines[count][0])
            tabs[j, i] = float(listOfLines[count][1])
            tabh[j, i] = float(listOfLines[count][2])
            count = count + 1
        for j in range(lentabint[i]):
            if tabr[j, i] > hcut[i]:
                hcut[i] = tabr[j, i]
                scut[i] = hcut[i]

    paramU = np.zeros((maxentry))

    for i in range(noint):
        n = lentabint[i] - 1
        hspl[0, i] = 0.0
        paramU[0] = 0.0

        for j in range(1, n - 1):
            sig = (tabr[j, i] - tabr[j - 1, i]) / (tabr[j + 1, i] - tabr[j - 1, i])
            p = sig * hspl[j - 1, i] + 2.0
            hspl[j, i] = (sig - 1.0) / p
            paramU[j] = (
                6.0
                * (
                    (tabh[j + 1, i] - tabh[j, i]) / (tabr[j + 1, i] - tabr[j, i])
                    - (tabh[j, i] - tabh[j - 1, i]) / (tabr[j, i] - tabr[j - 1, i])
                )
                / (tabr[j + 1, i] - tabr[j - 1, i])
                - sig * paramU[j - 1]
            ) / p
        qn = 0.0
        un = 0.0
        hspl[n, i] = (un - qn * paramU[n - 1]) / (qn * hspl[n - 1, i] + 1.0)

        # 1 > 0
        # n > n-1 (range n)
        # n-1 > n-2 (range n-1)

        for k in range(n - 2, 0):
            hspl[k, i] = hspl[k, i] * hspl[k + 1, i] + paramU[k]

    # Now for the overlap
    for i in range(noint):
        n = lentabint[i] - 1
        sspl[0, i] = 0.0
        paramU[0] = 0.0

        for j in range(1, n - 1):
            sig = (tabr[j, i] - tabr[j - 1, i]) / (tabr[j + 1, i] - tabr[j - 1, i])
            p = sig * sspl[j - 1, i] + 2.0
            sspl[j, i] = (sig - 1.0) / p
            paramU[j] = (
                6.0
                * (
                    (tabh[j + 1, i] - tabh[j, i]) / (tabr[j + 1, i] - tabr[j, i])
                    - (tabh[j, i] - tabh[j - 1, i]) / (tabr[j, i] - tabr[j - 1, i])
                )
                / (tabr[j + 1, i] - tabr[j - 1, i])
                - sig * paramU[j - 1]
            ) / p
        qn = 0.0
        un = 0.0
        sspl[n, i] = (un - qn * paramU[n - 1]) / (qn * sspl[n - 1, i] + 1.0)

        # 1 > 0
        # n > n-1 (range n)
        # n-1 > n-2 (range n-1)

        for k in range(n - 2, 0):
            sspl[k, i] = hspl[k, i] * sspl[k + 1, i] + paramU[k]

    # New: translate the 'pp pi's to L1 L2 MP for use later

    btype_int = np.zeros((3, noint), dtype=int)
    for i in range(1, noint):
        if btype[i] == "sss":
            btype_int[0, i] = 0
            btype_int[1, i] = 0
            btype_int[2, i] = 0
        elif btype[i] == "sps":
            btype_int[0, i] = 0
            btype_int[1, i] = 0
            btype_int[2, i] = 0
        elif btype[i] == "sds":
            btype_int[0, i] = 0
            btype_int[1, i] = 1
            btype_int[2, i] = 0
        elif btype[i] == "sfs":
            btype_int[0, i] = 0
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "pps":
            btype_int[0, i] = 0
            btype_int[1, i] = 0
            btype_int[2, i] = 0
        elif btype[i] == "ppp":
            btype_int[0, i] = 0
            btype_int[1, i] = 0
            btype_int[2, i] = 0
        elif btype[i] == "pds":
            btype_int[0, i] = 0
            btype_int[1, i] = 1
            btype_int[2, i] = 0
        elif btype[i] == "pdp":
            btype_int[0, i] = 0
            btype_int[1, i] = 1
            btype_int[2, i] = 0
        elif btype[i] == "pfs":
            btype_int[0, i] = 0
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "pfp":
            btype_int[0, i] = 0
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "dds":
            btype_int[0, i] = 1
            btype_int[1, i] = 1
            btype_int[2, i] = 0
        elif btype[i] == "ddp":
            btype_int[0, i] = 1
            btype_int[1, i] = 1
            btype_int[2, i] = 0
        elif btype[i] == "ddd":
            btype_int[0, i] = 1
            btype_int[1, i] = 1
            btype_int[2, i] = 1
        elif btype[i] == "dfs":
            btype_int[0, i] = 1
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "dfp":
            btype_int[0, i] = 1
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "dfd":
            btype_int[0, i] = 1
            btype_int[1, i] = 2
            btype_int[2, i] = 1
        elif btype[i] == "ffs":
            btype_int[0, i] = 2
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "ffp":
            btype_int[0, i] = 2
            btype_int[1, i] = 2
            btype_int[2, i] = 0
        elif btype[i] == "ffd":
            btype_int[0, i] = 2
            btype_int[1, i] = 2
            btype_int[2, i] = 1
        elif btype[i] == "fff":
            btype_int[0, i] = 2
            btype_int[1, i] = 2
            btype_int[2, i] = 2
        else:
            print("Warning! Missed assigning a bond type in readtb")
            sys.exit(0)

    # return(noelem,ele,ele,basis,atocc,hes,hep,hed,hef,mass,hubbardu,btype_int,)
    return (
        noelem,
        ele,
        ele1,
        ele2,
        basis,
        hes,
        hep,
        hed,
        hef,
        hcut,
        scut,
        noint,
        btype,
        tabh,
        tabs,
        sspl,
        hspl,
        lentabint,
        tabr,
        atocc,
        mass,
        hubbardu,
    )


def get_elempointer(atele, ele, verbose):
    if verbose:
        print("\nIn get_elempointer ...")

    noelem = len(ele)
    nats = len(atele)

    elempointer = np.zeros((nats), dtype=int)
    for i in range(nats):
        for j in range(noelem):
            if atele[i] == ele[j]:
                elempointer[i] = j

    return elempointer


def get_cutoffList(nats, atele, ele1, ele2, hcut, scut, noint, verbose):
    if verbose:
        print("\nIn get_cutoffList ...")

    cutoffList = np.zeros((nats, nats))

    for i in range(nats):
        for j in range(nats):
            rcuttb = 0.0

            for k in range(noint):
                if (atele[i] == ele1[k] and atele[j] == ele2[k]) or (atele[j] == ele1[k] and atele[i] == ele2[k]):
                    if hcut[k] > rcuttb:
                        rcuttb = hcut[k]
                    if scut[k] > rcuttb:
                        rcuttb = scut[k]

            cutoffList[j, i] = rcuttb
    return cutoffList


def get_cutoffList_I(nats, atele, ele1, ele2, hcut, scut, noint, i):
    cutoffList = np.zeros((nats, nats))
    for j in range(nats):
        rcuttb = 0.0

        for k in range(noint):
            if (atele[i] == ele1[k] and atele[j] == ele2[k]) or (atele[j] == ele1[k] and atele[i] == ele2[k]):
                if hcut[k] > rcuttb:
                    rcuttb = hcut[k]
                if scut[k] > rcuttb:
                    rcuttb = scut[k]

        cutoffList[j, i] = rcuttb
    return cutoffList


def get_cutoffList_par(nats, atele, ele1, ele2, hcut, scut, noint, verbose):
    if verbose:
        print("\nIn get_cutoffList_par ...")

    cutoffList = Parallel(n_jobs=12)(
        delayed(get_cutoffList_I)(nats, atele, ele1, ele2, hcut, scut, noint, i) for i in range(nats)
    )
    cutoffList = sum(cutoffList)

    return cutoffList


def get_cutoffList_I_vect(nats, atele, ele1, ele2, hcut, scut, noint, i):
    cutoffListI = np.zeros((nats))
    for j in range(nats):
        rcuttb = 0.0
        for k in range(noint):
            if (atele[i] == ele1[k] and atele[j] == ele2[k]) or (atele[j] == ele1[k] and atele[i] == ele2[k]):
                if hcut[k] > rcuttb:
                    rcuttb = hcut[k]
                if scut[k] > rcuttb:
                    rcuttb = scut[k]

        cutoffListI[j] = rcuttb

    return cutoffListI


def get_cutoffList_parPool(nats, atele, ele1, ele2, hcut, scut, noint, verbose, parallel=1):
    if verbose:
        print("\nIn get_cutoffList_parPool ...")

    cutoffList = np.zeros((nats, nats))
    if parallel > 1:
        pool = Pool(processes=parallel)
        results = [
            pool.apply_async(get_cutoffList_I_vect, [nats, atele, ele1, ele2, hcut, scut, noint, val])
            for val in range(0, nats)
        ]
        for idx, val in enumerate(results):
            cutoffList[idx, :] = val.get()
        pool.close()
    elif parallel == 1:
        for val in range(nats):
            cutoffList[val, :] = get_cutoffList_I_vect(nats, atele, ele1, ele2, hcut, scut, noint, val)

    return cutoffList


###
# @brief Constructs inverse overlap factors given S.
# This is a python version of genX.F90 LATTE routine
#
##
def genX(smat, method, verbose):
    if verbose:
        print("In genX ...")

    hdim = len(smat)
    if method == "Diag":
        # e,v = np.linalg.eig(smat)
        e, v = eigh(smat)
        s = 1.0 / np.sqrt(e)
        zmat = np.zeros((hdim, hdim))
        for i in range(hdim):
            zmat[i, :] = s[i] * v[:, i]
        zmat = np.matmul(v, zmat)
    elif method == "Cholesky":
        pass
    else:
        print("ERROR: Method not implemented")
        sys.exit(0)

    if verbose:
        print("\nZmat Matrix")
        print(zmat)

    return zmat


def fermi_exp(hmat, zmat, nocc, T):
    hmat_orth = np.matmul(np.matmul(np.transpose(zmat), hmat), np.transpose(zmat))
    e, C = np.linalg.eigh(hmat_orth)
    mu0 = 0.5 * (e[nocc - 1] + e[nocc])

    kB = 8.61739e-5
    beta = 1.0 / (kB * T)
    hdim = len(hmat)

    OccErr = 100.0
    iter = 0
    focc = np.zeros(hdim)
    while OccErr > 1e-12:  # Adjusts the chemical potential for a given Fermi-Dirac
        iter = iter + 1
        occ = 0.0
        docc = 0.0
        for i in range(hdim):
            tmp = beta * (e[i] - mu0)
            if tmp < -100:
                tmp = -100.0
            if tmp > 100:
                tmp = 100.0
            focc[i] = 1.0 / (np.exp(tmp) + 1.0)
        occ = np.sum(focc)
        docc = np.sum(beta * focc * (1.0 - focc))
        OccErr = np.absolute(nocc - occ)
        print(OccErr, nocc, occ)
        if np.absolute(OccErr) > 1e-10:
            mu0 = mu0 + (nocc - occ) / docc

        if iter > 20:
            print(" Not Converging in feri_exp  OccErr = ", OccErr)
            OccErr = 0.0

    if docc > 1e-14:
        focc = focc + ((nocc - occ) / docc) * beta * focc * (1.0 - focc)

    dmat = np.zeros((hdim, hdim))
    for i in range(hdim):
        dmat[:, i] = C[:, i] * focc[i]
    dmat = np.matmul(dmat, np.transpose(C))
    dmat = np.matmul(np.matmul(zmat, dmat), np.transpose(zmat))

    entropy = 0.0
    for i in range(hdim):
        if focc[i] > 1e-14 and focc[i] < 1.0 - 1e-14:
            entropy = entropy - kB * (focc[i] * np.log(focc[i]) + (1.0 - focc[i]) * np.log(1.0 - focc[i]))

    return (dmat, focc, C, e, mu0, entropy)
