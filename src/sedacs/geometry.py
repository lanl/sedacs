import numpy as np

## Minimum Image Convention Distances
# @brief Builds MIC interatomic distances for periodic systems. Distances in Angstrom.
# @param R torch Tensor of shape (N_atoms, 3) representing atomic positions.
# @param cell torch Tensore of shape (3,3) containging lattice parameters.
# @return R_mic Torch tensor of shape (N_atoms, N_atoms) containing minimum image distance (0 on diagonals).
def get_mic_distances(R, cell, torchLib=True):
    # This can simply be extended for cases where we have skewed and/or small cells.
    # Probably this can be gotten rid of for Cagri's implementation once it is in main branch.
    translations = [
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
        [-1, -1, 1],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    if torchLib:
        import torch

        # Numpy array to hold the minimum image distances between all atoms in the simulation.
        R_mic = torch.cdist(R, R)
        R_comparison = torch.zeros((2, *R_mic.shape))
        R_comparison[0] = R_mic

        # Pull min distance for each element along each translation.
        # This can in principle be faster for some sensible memory
        # estimates that allow us to stack more translations into R_comparison.
        for trans in translations:
            R_shift = (
                R.clone() + trans[0] * cell[0] + trans[1] * cell[1] + trans[2] * cell[2]
            )
            R_comparison[1] = torch.cdist(R, R_shift)
            mics = torch.min(R_comparison, dim=0, keepdim=True).values
            R_comparison[0] = mics

        return R_comparison[0]

    else:
        from scipy.spatial.distance import cdist
        # Numpy array to hold the minimum image distances between all atoms in the simulation.
        R_mic = cdist(R, R)
        R_comparison = np.zeros((2, *R_mic.shape))
        R_comparison[0] = R_mic

        # Pull min distance for each element along each translation.
        # This can in principle be faster for some sensible memory
        # estimates that allow us to stack more translations into R_comparison.
        for trans in translations:
            R_shift = (
                R.copy() + trans[0] * cell[0] + trans[1] * cell[1] + trans[2] * cell[2]
            )
            R_comparison[1] = cdist(R, R_shift)
            mics = np.min(R_comparison, axis=0)
            R_comparison[0] = mics

        return R_comparison[0]

## Quick Contact Map
# @brief Builds quick contact map from interatomic distances (respecting MIC). 
# @param Rij: torch Tensor of shape (N_atoms, N_atoms) representing interatomic distances.
# @param cutoff: Global cutoff in angstrom for what counts as a contact.
# @param torchLib: Whether to use torch or numpy.
# @param visualize: Whether to visualize the contact map as an imshow.
# @return contact_map:  nl[i,j] will give the j'th connection of i. 
def get_contact_map(Rij, cutoff=1.8, torchLib=True, visualize=False):

    if torchLib:
        import torch

        Rij.fill_diagonal_(cutoff + 1)
        contact_map = torch.zeros_like(Rij)
        mask = Rij <= cutoff
        notmask = Rij > cutoff
        contact_map[mask] = 1
        contact_map[notmask] = 0

        if 0:
            sz = 100
            plt.subplots(1,2)
            plt.subplot(1,2,1)
            plt.imshow(Rij[:sz, :sz])
            plt.subplot(1,2,2)
            plt.imshow(contact_map[:sz, :sz])
            plt.show()
    else:
        np.fill_diagonal(Rij, cutoff+1)
        contact_map = np.zeros_like(Rij)
        mask = Rij <= cutoff
        notmask = Rij > cutoff
        contact_map[mask] = 1
        contact_map[notmask] = 0

        if 0:
            sz = 100
            plt.subplots(1,2)
            plt.subplot(1,2,1)
            plt.imshow(Rij[:sz, :sz])
            plt.subplot(1,2,2)
            plt.imshow(contact_map[:sz, :sz])
            plt.show()

    return contact_map
