import numpy as np
from scipy.interpolate import CubicSpline
from .constants import kb, hbar, ref_mass, au_to_amu
from .task_driver import TaskDriver
from .bead import Bead

class RingPolymer():
    """
        Ring polymer is essentially a list of beads.
        Potential, gradient, and Hessian of the Ring Polymer are defined
        with respect to the beads and the fictitious spring forces.
    """
    def __init__(self, T:float, beads:list[Bead], task_driver:TaskDriver) -> None:
        """
            beads: list of half ring polymer mrtunneling.bead.Bead
            N: number of beads for full ring polymer
            beta: thermodynamic quantity 1/kb*T
            betaN: ring polymer beta
            task_driver: defines routines to get energies, gradients,
                and Hessians for beads
        """
        self.beads = beads # Bead 1 to N/2, indexed minus 1 bc Python
        self.N = len(self.beads) * 2
        self.T = T
        self.beta = (kb*self.T)**-1
        self.betaN = self.beta / self.N
        #self.exact_hessian = False
        self.task_driver = task_driver

    def M(self) -> np.ndarray:
        # Mass-weighting matrix
        return np.diag(np.concatenate([np.diag(bead.M) for bead in self.beads]))
            
    def Minv(self) -> np.ndarray:
        # Inverse mass-weighting matrix
        return np.diag(np.concatenate([np.diag(bead.Minv) for bead in self.beads]))

    def U(self, ref_E=0.0) -> float:
        # Half ring polymer potential
        out = 0.0
        for b, bead in enumerate(self.beads): # Sum from 1 to N
            # Bead energy from task driver
            Vx = bead.energy - ref_E
            out += Vx
            if b == 0: # Only count each "spring energy" once between each bead
                continue
            # Spring force terms
            Vharm = np.sum(bead.masses * (bead.x - self.beads[b-1].x)**2)
            Vharm /= 2*(self.betaN**2)*(hbar**2)
            out += Vharm
        return out

    def gradient(self) -> np.ndarray:
        # Gradient of half ring polymer
        prefactor = 1.0 / (self.betaN**2 * hbar**2)
        # Number of coords in mol
        lmol = len(self.beads[0])
        big_grad = np.zeros(len(self.beads)*lmol)
        for bi, bead in enumerate(self.beads):
            # Gradient of bead from task_driver
            bead_grad = np.copy(bead.gradient).flatten()
            # Spring terms
            lil_grad = np.zeros_like(bead_grad)
            if bi == 0:
                lil_grad += prefactor * (bead.x - self.beads[1].x)
            elif bi == len(self.beads) - 1:
                lil_grad += prefactor * (bead.x - self.beads[bi-1].x)
            else:
                lil_grad += prefactor * 2.0*bead.x
                lil_grad -= prefactor * self.beads[bi-1].x
                lil_grad -= prefactor * self.beads[bi+1].x
            big_grad[bi*lmol:(bi+1)*lmol] = (bead.masses * lil_grad) + bead_grad
        return big_grad
    
    def spring_subhess(self) -> np.ndarray:
        # Hessians of spring terms between adjacent beads
        A = np.eye(len(self.beads[0]))
        A *= self.beads[0].masses / (self.betaN**2 * hbar**2)
        return A

    def hessian(self) -> np.ndarray:
        # Hessian of half ring polymer
        lmol = len(self.beads[0])
        #A = np.eye(len(self.beads[0]))
        #A *= self.beads[0].masses / (self.betaN**2 * hbar**2)
        A = self.spring_subhess()
        hN = lmol * len(self.beads)
        bigHess = np.zeros((hN, hN))
        for bi, bead in enumerate(self.beads):
            pre_dstart = lmol * (bi-1)
            dstart = lmol * bi
            dend = lmol * (bi+1)
            post_dend = lmol * (bi+2)
            # Hessian of bead from task_driver
            bigHess[dstart:dend, dstart:dend] += bead.hessian
            # Spring terms
            if bi == 0:
                bigHess[dstart:dend, dstart:dend] += A
                bigHess[dend:post_dend, dstart:dend] -= A
            elif bi == len(self.beads) - 1:
                bigHess[dstart:dend, dstart:dend] += A
                bigHess[pre_dstart:dstart, dstart:dend] -= A
            else:
                bigHess[dstart:dend, dstart:dend] += 2*A
                bigHess[pre_dstart:dstart, dstart:dend] -= A
                bigHess[dend:post_dend, dstart:dend] -= A
        return bigHess

    def full_U(self, ref_E=0.0) -> float:
        # Full ring polymer potential
        return 2.0 * self.U(ref_E=ref_E)
    
    def full_gradient(self) -> np.ndarray:
        # Gradient of full ring polymer
        # Structured as Bead_0, Bead_1, ..., Bead_N/2-1, Bead_0, Bead_1, ..., Bead_N/2-1
        half_grad = self.gradient()
        return np.hstack((half_grad, half_grad))

    def full_hessian(self) -> np.ndarray:
        # Hessian of full ring polymer
        # Structured as Bead_0, Bead_1, ..., Bead_N/2-1, Bead_0, Bead_1, ..., Bead_N/2-1
        lmol = len(self.beads[0])
        A = self.spring_subhess()
        half_Hess = self.hessian()
        # Diagonal is just the half Hessians with corrections for N_0 and N_N/2-1
        half_Hess[:lmol,:lmol] += A
        half_Hess[-lmol:,-lmol:] += A
        # There are some couplings from the spring forces between the copies of N_0 and N_N/2-1 with their originals
        off_diag = np.zeros_like(half_Hess)
        off_diag[:lmol,:lmol] -= A
        off_diag[-lmol:,-lmol:] -= A
        return np.block([[half_Hess, off_diag],[off_diag, half_Hess]])

    def mw_full_hess(self) -> np.ndarray:
        # Mass-weighting matrix for full ring polymer Hessian
        fh = self.full_hessian()
        Mweight = np.diag(np.tile(np.diag(self.Minv()), 2))
        return Mweight @ fh @ Mweight

    def evaluate_all_beads(self, der_lvl=1, update_hess=False) -> None:
        """
            For each bead, evaluate request features.
            Always evaluate energy of beads.
            For der_lvl (derivative level) greater than 0 also 
                evaluate gradients.
            For der_lvl greater than 1 evaluate Hessians.
            update_hess: Whether to do a Hessian update for the beads when not 
                evaluating exact Hessian
        """
        assert type(der_lvl) == int
        if der_lvl < 0 or der_lvl > 2:
            raise ValueError("Derivative level (der_lvl) needs to be 0, 1, or 2!")
        loading_bar_length = len(self.beads)
        print("\tEvaluating beads")
        print("\t"+"="*loading_bar_length)
        print("\t", end="", flush=True)
        for bi, bead in enumerate(self.beads):
            # If a bead already has a value for a quantity, skip computing
            if not bead.has_V:
                bead.energy = self.task_driver.energy
            if der_lvl >= 1 and not bead.has_grad:
                bead.gradient = self.task_driver.gradient
            # Check for der_lvl and whether an exact Hess is known
            if der_lvl == 2:
                if not (bead.has_hess and bead.exact_hessian):
                    bead.hessian = self.task_driver.hessian
                    bead.exact_hessian = True
            elif update_hess:
                # Kinda weird but I defined bead.hessian to take a function
                bead.hessian = lambda mol: bead.update_hessian(mol)
                bead.exact_hessian = False
            else:
                bead.exact_hessian = False
            print("*", end="", flush=True)
        print("")

    def double_beads(self) -> None:
        # For each bead pair, add another in the middle by interpolating attributes
        n = np.arange(len(self.beads))
        ref_mol = self.beads[0].mol
        natoms = self.beads[0].natoms
        geoms    = np.vstack([bead.mol.geometry.flatten() for bead in self.beads])
        hessians = np.vstack([bead.hessian.flatten() for bead in self.beads])
        geom_spline = CubicSpline(n, geoms)
        hess_spline = CubicSpline(n, hessians)
        new_beads = [self.beads[0]]
        for bidx in range(1,len(self.beads)):
            new_geom = geom_spline(bidx - 0.5)
            new_hess = hess_spline(bidx - 0.5)
            newBead = Bead(ref_mol.copy(update={"geometry": new_geom.reshape((natoms,3))}))
            newBead._hess = new_hess.reshape((3*natoms,3*natoms))
            newBead.has_hess = True
            newBead.exact_hessian = False
            new_beads.append(newBead)
            new_beads.append(self.beads[bidx])
        self.beads = new_beads # Bead 1 to N/2, indexed minus 1 bc Python
        self.N = len(self.beads) * 2
        self.betaN = self.beta / self.N

    def align_beads(self) -> None:
        # Align the structures of the beads by translation and rotation
        for b in range(1,len(self.beads)):
            if self.beads[b].has_data:
                print(Warning("Aligning beads with computed data will overwrite the computed data."))
            self.beads[b].mol = self.beads[b].mol.align(
                self.beads[b-1].mol, atoms_map=True)[0].geometry
            
    def atom_masses(self) -> np.ndarray:
        # Atomic masses in m_e (atomic units), shape (natoms,)
        return np.asarray(self.beads[0].mol.masses) / au_to_amu

    def full_ring_geometries(self) -> list[np.ndarray]:
        # Geometries of all N beads of the full ring polymer.
        # The half ring polymer stores beads 1 to N/2; beads N/2+1 to N are the
        # same structures traversed in reverse.
        geoms = [bead.mol.geometry for bead in self.beads]
        return geoms + geoms[::-1]

    def com(self) -> np.ndarray:
        # Center of mass of the full ring polymer
        m = self.atom_masses()
        geoms = self.full_ring_geometries()
        out = np.sum([m @ g for g in geoms], axis=0)
        return out / (len(geoms) * np.sum(m))

    def moit(self) -> np.ndarray:
        # Moment of inertia tensor of the full ring polymer (all N beads) about
        # its common center of mass, in m_e * bohr^2
        c = self.com()
        m = self.atom_masses()
        I = np.zeros((3,3))
        for g in self.full_ring_geometries():
            r = g - c
            I += np.sum(m * np.sum(r*r, axis=1)) * np.eye(3) - (r.T * m) @ r
        return I


