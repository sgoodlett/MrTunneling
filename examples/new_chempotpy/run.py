import qcelemental as qcel
import numpy as np
import mrtunneling
from chempotpy_driver import CH5VBMMRunner

# Atom 6 (index 5) is the attacking H in the geometry below
# Define task driver
pRun = CH5VBMMRunner(attacking_H=5)


"""
    Example instanton computation for the CH4 + H ---> CH3 + H2 (doublet electronic state) reaction.
    Precomputed are the reactant and transition state (TS) optimized geometries.
    The instanton is initialized from the TS, and then optimized.
    In the example, the optimization completed on step_39.
    Exact Hessians are computed, and the rates determined.

"""

# Transition state optimized geometry, get energy, gradient, and Hessian for rate computation and instanton initialization
molstr = """
0 2

C                     0.115521709411    -0.010438993532    -0.127035742917
H                     0.627933083211    -0.965663424341    -0.011510548545
H                     0.794197457884     0.833944748360    -0.005460562002
H                    -0.818028899839     0.072358416243     0.899960422062
H                    -0.439823309722     0.044017179220    -1.063514058055
H                    -1.404056239438     0.124333688218     1.544647302464

"""

qcmol = qcel.models.Molecule.from_data(molstr)

do_ts_computation = True
if do_ts_computation:
    E = pRun.energy(qcmol)
    G = pRun.gradient(qcmol)
    H = pRun.hessian(qcmol)
    np.savetxt("tsE.dat", np.array([E]))
    np.savetxt("tsG.dat", G)
    np.savetxt("tsH.dat", H)
else:
    E = np.loadtxt("tsE.dat")
    G = np.loadtxt("tsG.dat")
    H = np.loadtxt("tsH.dat")


# Manually set the necessary values of the bead, I probably need a better way to do this
ts_bead = mrtunneling.bead.Bead(qcmol)
ts_bead._V = E
ts_bead.grad = G
ts_bead.has_hess = True
ts_bead.exact_hess = True
ts_bead._hess = H


# Get frequencies of Hessian (in Hartree)
freqs, nmodes = mrtunneling.proc_Hess(qcmol, H)
print(freqs)

# Define plan for optimizing instanton
opt_plan = {
        "step_limit": 0.05,
        "max_iter": 50,
        "grad_rms_convergence": 1e-6,
        "grad_max_convergence": 1e-6,
        "step_rms_convergence": 1e-6,
        "step_max_convergence": 1e-6,
        "hess_every": 0,
        "opt_method": "evf"
        }

# Initiate instanton from transition state geometry
I = mrtunneling.Instanton.initiate_from_TS(300.0, qcmol, H, 64, pRun, delta=0.1)
print(f"Instanton crossover temperature: {I.crossover_T()}")
# Or restart from directory "step_39"
#I = mrtunneling.Instanton.restart("step_39", pRun, temp=400.0)
I.optimize(opt_plan)

# For final iteration, evaluate exact Hessians
I.evaluate_all_beads(der_lvl=2)
# Save for future use
#I.save_state("step_39_exact_hess")
# Or read in if you already have it
#I = mrtunneling.Instanton.read_state("step_39_exact_hess")

# Set up "fake" ring polymers for rate (r) and tunneling correction (k) computations
tsRP = mrtunneling.fake_ring_polymer.FakeRingPolymer(I, ts_bead)

k = mrtunneling.partition_functions.kappa(I, tsRP)
print(k)

