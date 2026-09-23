import numpy as np
from .constants import kb
from .bead import Bead
from .ring_polymer import RingPolymer

class FakeRingPolymer(RingPolymer):
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

    def com(self) -> np.ndarray:
        out = np.zeros(3)
        bead = self.beads[0]
        out += np.sum(np.diag(bead.mol.masses) @ bead.mol.geometry, axis=0)
        return out / (self.N * np.sum(self.beads[0].mol.masses))

    def moit(self) -> np.ndarray:
        c = self.com()
        I = np.zeros((3,3))
        bead = self.beads[0]
        for i in range(3):
            # i + 1 mod 3 and i + 2 mod 3 essentially give not i
            I[i,i] += np.sum(bead.mol.masses * 
                             ( (bead.mol.geometry[:,(i+1)%3] - c[(i+1)%3])**2 
                             + (bead.mol.geometry[:,(i+2)%3] - c[(i+1)%3])**2))
            for j in range(i+1,3):
                I[i,j] += -1.0 * np.sum(bead.mol.masses
                    * (bead.mol.geometry[:,i] - c[i])
                    * (bead.mol.geometry[:,j] - c[j]))
                I[j,i] = I[i,j]
        return I * self.N
