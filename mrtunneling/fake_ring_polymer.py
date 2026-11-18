import numpy as np
from .constants import kb
from .bead import Bead
from .ring_polymer import RingPolymer

class FakeRingPolymer(RingPolymer):
    """
        Collapsed ring polymer: all N beads sit at the same structure (e.g. a
        reactant or transition state), with N taken from a reference ring polymer.
    """
    def __init__(self, refRingPolymer:RingPolymer, bead:Bead) -> None:
        self.beads = [bead] # Bead 1 to N/2, indexed minus 1 bc Python
        self.N = len(refRingPolymer.beads) * 2
        self.T = refRingPolymer.T
        self.beta = (kb*self.T)**-1
        self.betaN = self.beta / self.N

    def update_beadN(self, N:int) -> None:
        # Update N to match N of another RP
        self.N = N 
        self.betaN = self.beta / self.N

    def full_ring_geometries(self) -> list[np.ndarray]:
        # All N beads are at the same structure
        return [self.beads[0].mol.geometry] * self.N

    def com(self) -> np.ndarray:
        # Center of mass (same for one bead or the full collapsed ring polymer)
        m = self.atom_masses()
        return m @ self.beads[0].mol.geometry / np.sum(m)

    def moit(self) -> np.ndarray:
        # Moment of inertia tensor of the full collapsed ring polymer:
        # N times that of a single bead, in m_e * bohr^2
        c = self.com()
        m = self.atom_masses()
        r = self.beads[0].mol.geometry - c
        I = np.sum(m * np.sum(r*r, axis=1)) * np.eye(3) - (r.T * m) @ r
        return I * self.N
