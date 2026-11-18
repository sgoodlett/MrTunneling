import chempotpy
import numpy as np
import qcelemental

from mrtunneling import TaskDriver
from qcelemental.models.v2.molecule import Molecule


class ChemPotPyRunner(TaskDriver):
    def __init__(self, system, surface):
        super().__init__()
        self._system = system
        self._surface = surface

    def energy(self, mol: Molecule):
        chempotpy_geometry = ChemPotPyRunner.reshape_and_zip(mol.symbols, mol.geometry)
        p = chempotpy.p(self._system, self._surface, chempotpy_geometry)
        return p[0] / qcelemental.constants.hartree2ev

    def gradient(self, mol):
        chempotpy_geometry = ChemPotPyRunner.reshape_and_zip(mol.symbols, mol.geometry)
        #print(chempotpy_geometry)
        # chempotpy will return the numerical gradient if the surface does provide an analytical gradient
        _, g = chempotpy.pg(self._system, self._surface, chempotpy_geometry)
        #_, g = chempotpy.pn(self._system, self._surface, chempotpy_geometry)
        return g[0] * qcelemental.constants.bohr2angstroms / qcelemental.constants.hartree2ev

    def hessian(self, mol, h=1e-5):
        geometry = np.asarray(mol.geometry).flatten()
        n = len(geometry)
        H = np.zeros((n, n))

        for i in range(n):
            geom_plus = geometry.copy()
            geom_minus = geometry.copy()

            geom_plus[i] += h
            geom_minus[i] -= h

            mol_plus = mol.model_copy(update={"geometry": geom_plus.reshape(-1, 3)})
            mol_minus = mol.model_copy(update={"geometry": geom_minus.reshape(-1, 3)})

            grad_plus = self.gradient(mol_plus).flatten()
            grad_minus = self.gradient(mol_minus).flatten()

            H[:, i] = (grad_plus - grad_minus) / (2 * h)

        return H

    @staticmethod
    def reshape_and_zip(symbols, geometry):
        return [[symbol, *xyz] for symbol, xyz in zip(symbols, np.array(geometry).reshape(-1, 3) * qcelemental.constants.bohr2angstroms)]


class CH5VBMMRunner(ChemPotPyRunner):
    """
    CH5_VBMM_CBG_2009 called directly, bypassing chempotpy's atom reordering.

    chempotpy.CH5.check() puts the H farthest from C in position 1, but the Fortran
    (DATA nnc=2, nnb=6, nnh=3,4,5,1) expects the order  H, C, H, H, H, H_attacking.
    With chempotpy's order the surface is distorted (CH4 minimum near 1.2 A, no sensible TS).

    attacking_H: index of the attacking/transferring H in your molecule. Give a fixed index
                 so the labeling cannot flip during an optimization. If None, the H farthest
                 from C is used on each call (same rule chempotpy uses).
    """
    def __init__(self, attacking_H=None):
        super().__init__("CH5", "CH5_VBMM_CBG_2009")
        import importlib
        self._pes = importlib.import_module("chempotpy.CH5.CH5_VBMM_CBG_2009").pes
        self._attacking_H = attacking_H

    def _order(self, symbols, xyz):
        c = symbols.index("C")
        hs = [i for i in range(len(symbols)) if i != c]
        if self._attacking_H is not None:
            b = self._attacking_H
        else:
            b = max(hs, key=lambda i: np.linalg.norm(xyz[i] - xyz[c]))
        hs.remove(b)
        return [hs[0], c, hs[1], hs[2], hs[3], b]

    def _evaluate(self, mol):
        xyz = np.asarray(mol.geometry).reshape(-1, 3) * qcelemental.constants.bohr2angstroms
        order = self._order(list(mol.symbols), xyz)
        v, g, _ = self._pes(xyz[order], 1)          # eV, eV/Angstrom
        g_mol = np.zeros((len(order), 3))
        g_mol[order] = g[0]                          # map back to the molecule's atom order
        E = v[0] / qcelemental.constants.hartree2ev
        G = g_mol * qcelemental.constants.bohr2angstroms / qcelemental.constants.hartree2ev
        return E, G

    def energy(self, mol):
        return self._evaluate(mol)[0]

    def gradient(self, mol):
        return self._evaluate(mol)[1]
