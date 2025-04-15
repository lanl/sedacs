"""
Help file for common Union/Custom types.
"""
from typing import Union
import numpy as np
import torch


# If we are moving towards torch only, we should eventually get rid of this, 
# however a lot of the code currently is mixed.
ArrayLike = Union[np.ndarray, torch.Tensor]

