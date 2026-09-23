import numpy as np
from qcelemental.models.molecule import Molecule
from .constants import *

def proc_Hess(mol:Molecule, hessian:np.ndarray) -> tuple:
    # Get frequencies and normal modes from a Hessian
    mw = np.repeat(np.sqrt(mol.masses)**-1, 3)
    mw_hess = np.multiply(mw[:,None], np.multiply(hessian, mw[None,:]))
    fcs, nmodes = np.linalg.eigh(mw_hess)
    nmodes /= mw
    signs = np.where(fcs<0.0, -1, 1)
    freqs = np.sqrt(fcs * signs) * freq_to_hartree
    freqs *= signs
    return freqs, nmodes
