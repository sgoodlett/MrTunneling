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
        # chempotpy will return the numerical gradient if the surface does provide an analytical gradient
        _, g = chempotpy.pg(self._system, self._surface, chempotpy_geometry)
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