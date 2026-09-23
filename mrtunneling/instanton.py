from __future__ import annotations
import pathlib
import numpy as np
from copy import deepcopy
import re
import scipy
import qcelemental as qcel
from qcelemental.models.molecule import Molecule
from .constants import hbar, h2cm, kb, au_to_amu, ref_mass
from .utils import proc_Hess
from .task_driver import TaskDriver
from .bead import Bead
from .ring_polymer import RingPolymer
#from .partition_functions import trans_pfxn, rot_pfxn, vib_pfxn_R, vib_pfxn_inst

class Instanton(RingPolymer):

    _default_opt_plan = {
        "step_limit": 0.1,
        "max_iter": 50,
        "grad_rms_convergence": 1e-3,
        "grad_max_convergence": 1e-3,
        "step_rms_convergence": 1e-2,
        "step_max_convergence": 1e-2,
        "hess_every": 10,
        "opt_method": "nr",
        "trust_radius": 0.3,
        "tr_scale_factor": 2.0,
        "tr_lower_bound": 0.0,
        "tr_upper_bound": 2.0,
        "tr_good_step_lower": 0.25,
        "tr_good_step_upper": 1.75
    }

    def __init__(self, wb:float, T:float, beads:list[Bead], 
                 task_driver:TaskDriver, restart_at=0, align=False) -> None:
        self.wb = wb
        #self.rate_convergence = 1e-2
        self.restart_at = restart_at
        self.working_dir = pathlib.Path.cwd()
        super().__init__(T, beads, task_driver)
        if align:
            self.align_beads()

    def crossover_T(self) -> float:
        # Crossover temperature, don't run instantons above this temperature
        return hbar * np.abs(self.wb) / (2*np.pi*kb)

    def optimize(self, opt_plan:dict, save_dir_prefix="") -> int:
        self.old_Hevals = None
        self.old_Hevecs = None
        self.read_opt_plan(opt_plan)
        # Compute exact Hessian for first step
        self.evaluate_all_beads(der_lvl=2)
        self.save_state(save_dir_prefix + f"step_{self.restart_at}")
        for iter in range(1,self.max_iter):
            print(f"\tStep {iter+self.restart_at}")
            # Calculate step
            if self.opt_method == "nr":
                step = self.take_step_NR()
                # Demass weight step
                #step = self.Minv() @ h
                # Scale down step
                if np.max(np.abs(step)) > self.step_limit:
                    step *= self.step_limit / np.max(np.abs(step))
            elif self.opt_method == "evf":
                good, step = self.take_step_EF()
                if not good:
                    self.evaluate_all_beads(der_lvl=2)
                    good, step = self.take_step_EF()
                    if not good:
                        print("FOOK")
                #step = self.Minv() @ h
                # Scale down step
                if np.max(np.abs(step)) > self.step_limit:
                    step *= self.step_limit / np.max(np.abs(step))
            elif self.opt_method == "rnr":
                step = self.take_step_RNR()
            else:
                raise ValueError(f"Unrecognized optimization algorithm: {self.opt_method}")

            # Apply step to bead geoms
            new_bead_steps = step.reshape((-1, self.beads[0].natoms, 3))
            for b,bead in enumerate(self.beads):
                new_geom_i = bead.mol.geometry + new_bead_steps[b]
                bead.update(new_geom_i)
            
            # Evaluate new beads
            if self.do_exact_hess_updates and iter % self.hess_every == 0:
                self.evaluate_all_beads(der_lvl=2)
            else:
                self.evaluate_all_beads(der_lvl=1, update_hess=True)
            
            # Save and check for convergence
            self.save_state(save_dir_prefix + f"step_{iter+self.restart_at}")
            converged = self.check_convergence(step, self.gradient())
            if converged:
                print("Done")
                return 0
        raise Exception("Max number of iterations reached without convergence!")

    def read_opt_plan(self, opt_plan:dict) -> None:
        # Get relevant optimization settings
        new_opt_plan = self._default_opt_plan.copy()
        invalid = set(opt_plan.keys()) - set(self._default_opt_plan.keys())
        if invalid:
            raise ValueError(f"Invalid config options: {invalid}")
        # Override with provided values
        new_opt_plan.update(opt_plan)
        # Set as attributes
        for key, value in new_opt_plan.items():
            setattr(self, key, value)
            kval = eval(f"self.{key}")
            if type(kval) is float:
                print(f"\t{key:20s} = {kval:5.2e}")
            elif type(kval) is str:
                print(f"\t{key:20s} = {kval:8s}")
        print("\n")
        if self.hess_every < 1:
            self.do_exact_hess_updates = False
        else:
            self.do_exact_hess_updates = True

    def take_step_NR(self) -> np.ndarray:
        # Newton-Raphson
        # Assume n, and n,n for grad and hess shapes
        g = self.gradient()
        n = len(g)
        eps = 0.028 # 0.1 Eh/Ang^2 as defined by Richardson
        H = self.hessian() + eps*np.eye(n)
        h = -1.0 * np.linalg.pinv(H, hermitian=True) @ g
        return h

    def take_step_EF(self) -> np.ndarray:
        g = self.gradient()
        H = self.hessian()
        Heval, Hevec = np.linalg.eigh(H)

        # Check H
        if self.old_Hevals is None or self.old_Hevecs is None:
            pass
        else:
            if np.isclose(self.old_Hevals[1], 0, atol=1e-5) and Heval[1] < -1:
                print("Oh shit, evals exploded")
                return False, None
            evec_overlap = self.old_Hevecs.T @ Hevec[:,0:2]
            if np.abs(evec_overlap[0,0]) < 0.8:
                print("Oh shit, evec changed!")
                print("\t", evec_overlap)

        # Save the two relevant eigenpairs
        self.old_Hevals = Heval[0:2]
        self.old_Hevecs = Hevec[:,0:2]

        F = g @ Hevec
        if (Heval[0]/2.0) < Heval[1]:
            alpha = 1.0
            labda = (Heval[0] + 2*Heval[1]) / 4.0
        else:
            alpha = (Heval[0] - Heval[1]) / Heval[1]
            labda = (Heval[0] + 3*Heval[1]) / 4.0
        print(f"\tlambda = {labda:5.2e}, h_1 = {Heval[0]:5.2e}, h_2 = {Heval[1]:5.2e}")
        xi = alpha * F / (labda - Heval)
        h = xi @ Hevec.T
        return True, h

    def check_convergence(self, step:np.ndarray, grad:np.ndarray) -> bool:
        n = self.beads[0].natoms * len(self.beads)
        step_norm = np.linalg.norm(step) / np.sqrt(n)
        grad_norm = np.linalg.norm(grad) / np.sqrt(n)
        max_step = np.max(np.abs(step))
        max_grad = np.max(np.abs(grad))
        rms_step_chk = step_norm < self.step_rms_convergence
        rms_grad_chk = grad_norm < self.grad_rms_convergence
        max_step_chk = max_step  < self.step_max_convergence
        max_grad_chk = max_grad  < self.grad_max_convergence
        print("\t-----Convergence Check-----")
        print_these = [["\tRMS Step size"   , self.step_rms_convergence, rms_step_chk, step_norm], 
                       ["\tMax Step size"   , self.step_max_convergence, max_step_chk, max_step], 
                       ["\tRMS Grad norm"   , self.grad_rms_convergence, rms_grad_chk, grad_norm], 
                       ["\tMax Grad Element", self.grad_max_convergence, max_grad_chk, max_grad]]
        for i in print_these:
            sign = ">" if i[2] else "<"
            print(f"{i[0]:20s}   {i[1]:10.7e} {sign} {i[3]:10.7e}")
        print("\t---------------------------")
        if rms_step_chk and rms_grad_chk and max_step_chk and max_grad_chk:
            return True
        else:
            return False

    @classmethod
    def initiate_from_TS(cls, T:float, TSmol:Molecule, TShess:np.ndarray, 
                         Nbeads:int, task_driver:TaskDriver,
                           delta=0.1) -> Instanton:
        """
            Initiate ring polymer from a transition state structure and Hessian
            T: Temperature
            TSmol: QCElemental molecule for transition state
            TShess: Hessian at transition state (3N,3N) np.array
            Nbeads: Initial number of beads
            task_driver: Functions for getting energies, gradients, and Hessians
            delta: Step scale from transition state coordinates for new beads
        """
        # Do Hessian analysis on TShess
        freqs, nmodes = proc_Hess(TSmol, TShess)
        if freqs[0] > 0.0:
            raise ValueError("Imaginary mode has positive frequency!")
        print(f"Selecting mode with frequency {freqs[0]*h2cm:5.2f} cm-1")
        qi = nmodes[:,0] # Imaginary freq mode is assumed to be the largest negative freq
        geoms = []
        # Make half of ring polymer, the other half wraps back around and is equivalent
        # Nbeads = N/2
        for i in range(Nbeads):
            pf = delta * np.cos(np.pi*i/(Nbeads-1)) # Go from +q_i to -q_i once
            newgeom = TSmol.geometry + pf * qi.reshape(*TSmol.geometry.shape)
            mol_i = TSmol.copy(update={"geometry": newgeom})
            geoms.append(mol_i)
        beads = [Bead(g) for g in geoms]
        return cls(freqs[0], T, beads, task_driver, align=True)

    def save_state(self, dirname:str) -> None:
        # Save data to directory: dirname
        d = pathlib.Path(dirname)
        d.mkdir(parents=True, exist_ok=True)
        self.write_to_mXYZ(d / "all_structs.xyz")
        for b, bead in enumerate(self.beads):
            bead.write(d, b)
        with open(d / "other_data.dat", "w") as f:
            f.write(f"wb:{self.wb}\n")
            f.write(f"T:{self.T}")

    def write_to_mXYZ(self, fn:str) -> None:
        # Write an XYZ file with all of the beads
        with open(f"{fn}", "w") as f:
            f.write("\n>\n".join([str(i) for i in self.beads]))
    
    @classmethod
    def read_state(cls, restart_dir:str) -> Instanton:
        # Load previous instanton data for restarting
        p = pathlib.Path(restart_dir)
        # Read instanton conditions
        with open(p / "other_data.dat", "r") as f:
            s = f.read()
        wb = float(re.search(r"wb:(-?\d+\.\d+)", s).groups()[0])
        T = float(re.search(r"T:(-?\d+\.\d+)", s).groups()[0])
        # Read beads and their previous state
        nbeads = len([i for i in p.iterdir() if i.suffix == ".mol"])
        beads = []
        for i in range(nbeads):
            # Read mol and energy
            qcmol = qcel.models.Molecule.from_file(p / f"bead_{i}.mol")
            E = np.loadtxt(p / f"bead_{i}.mol", skiprows=1, max_rows=1)
            b = Bead(qcmol)
            if E == 0.0:
                pass
            else:
                b._V = E
                b.has_V = True
            # Read gradient
            if (p / f"bead_{i}.grad").exists():
                b._grad = np.loadtxt(p / f"bead_{i}.grad")
                b.has_grad = True
            # Read Hessian
            if (p / f"bead_{i}.hess").exists():
                b._hess = np.loadtxt(p / f"bead_{i}.hess")
                b.has_hess = True
            beads.append(b)
        return cls(wb, T, beads, None)

    @classmethod
    def restart(cls, restart_dir:str, task_driver:TaskDriver, temp=None, 
                restart_at=0) -> Instanton:
        # Set up instanton for restarting
        #ra = int(restart_dir.strip("step_")) + 1
        # NEW
        c = cls.read_state(restart_dir)
        if temp is not None:
            c.T = temp
        c.task_driver = task_driver
        c.restart_at = restart_at
        return c

        #p = pathlib.Path(restart_dir)
        ## Read instanton conditions
        #with open(p / "other_data.dat", "r") as f:
        #    s = f.read()
        #wb = float(re.search(r"wb:(-?\d+\.\d+)", s).groups()[0])
        #if temp is None:
        #    T = float(re.search(r"T:(-?\d+\.\d+)", s).groups()[0])
        #else:
        #    T = temp
        ## Read beads and their previous state
        #nbeads = len([i for i in p.iterdir() if i.suffix == ".mol"])
        #beads = []
        #for i in range(nbeads):
        #    # Read mol and energy
        #    qcmol = qcel.models.Molecule.from_file(p / f"bead_{i}.mol")
        #    E = np.loadtxt(p / f"bead_{i}.mol", skiprows=1, max_rows=1)
        #    b = Bead(qcmol)
        #    if E == 0.0:
        #        pass
        #    else:
        #        b._V = E
        #        b.has_V = True
        #    # Read gradient
        #    if (p / f"bead_{i}.grad").exists():
        #        b._grad = np.loadtxt(p / f"bead_{i}.grad")
        #        b.has_grad = True
        #    # Read Hessian
        #    if (p / f"bead_{i}.hess").exists():
        #        b._hess = np.loadtxt(p / f"bead_{i}.hess")
        #        b.has_hess = True
        #    beads.append(b)
        #return cls(wb, T, beads, task_driver, restart_at=ra)

 