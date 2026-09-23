import numpy as np
from .utils import proc_Hess
from .constants import hbar, time_au_to_s
from .ring_polymer import RingPolymer
# I have no clue if any of these work...

def trans_pfxn(ring_polymer:RingPolymer) -> float:
    M = np.sum(ring_polymer.beads[0].mol.masses)
    out = ((ring_polymer.N*M) / (2*np.pi*ring_polymer.betaN*hbar**2))**(3.0/2.0)
    out /= ring_polymer.N**3
    return out

def rot_pfxn(ring_polymer:RingPolymer) -> float:
    dI = np.linalg.det(ring_polymer.moit())
    out = np.sqrt(8*np.pi*dI / (ring_polymer.betaN**3 * hbar**6))
    out /= ring_polymer.N**3
    return out

def analyze_hessian(ring_polymer:RingPolymer) -> np.ndarray:
    mwHess = ring_polymer.mw_full_hess()
    eigvals, eigvecs = np.linalg.eigh(mwHess)
    print(eigvals[:20])
    # Check Hessian structure
    zeros = np.isclose(eigvals, 0.0, atol=1e-7)
    negatives = eigvals < 0.0
    n_zero = np.sum(zeros)
    if n_zero != 7:
        print(f"Wrong number of zero modes {n_zero}.")
    n_negatives = np.sum(negatives & ~zeros)
    if n_negatives != 1:
        raise ValueError(f"Wrong number of negative nonzero modes {n_negatives}.")
    # Return relevant eigenvalues
    return np.sqrt(np.abs(eigvals[~zeros]))

def vib_pfxn_R(ring_polymer:RingPolymer, n_ignore=6) -> float:
    # Vibrational partition function for collapsed Ring Polymer
    #omegas = np.linalg.eigh(ring_polymer.beads[0].hessian)[0][n_ignore:]
    omegas = proc_Hess(ring_polymer.beads[0].mol, ring_polymer.beads[0].hessian)[0][n_ignore:]
    # Solve: sinh(beta_N hbar wt_k / 2) = beta_N hbar w_k / 2
    arcsinh_rhs = np.arcsinh(ring_polymer.betaN * hbar * omegas / 2.0)
    tomegas = 2 * arcsinh_rhs / (ring_polymer.betaN * hbar)
    out = np.prod((2*np.sinh(0.5*ring_polymer.beta*hbar*tomegas))**-1)
    return out

def vib_pfxn_inst(ring_polymer:RingPolymer) -> float:
    etas = analyze_hessian(ring_polymer)
    beads = ring_polymer.beads
    BN = np.sum([beads[0].masses * (beads[i].x - beads[i-1].x)**2 for i in range(1, len(beads))])
    out = np.prod([1.0/(ring_polymer.betaN * hbar * np.abs(eta)) for eta in etas])
    out *= np.sqrt(2*np.pi*BN / (ring_polymer.betaN * hbar**2))
    out *= ring_polymer.N**7
    return out

def classical_rate(ts_rp:RingPolymer, r_rp:RingPolymer) -> float:
    Q_t_ts = trans_pfxn(ts_rp)
    Q_r_ts = rot_pfxn(ts_rp)
    Q_v_ts = vib_pfxn_R(ts_rp, n_ignore=7)
    Q_t_r = trans_pfxn(r_rp)
    Q_r_r = rot_pfxn(r_rp)
    Q_v_r = vib_pfxn_R(r_rp, n_ignore=6) # Assume not linear
    print(Q_t_ts, Q_r_ts, Q_v_ts)
    print(Q_t_r, Q_r_r, Q_v_r)
    prefactor = np.log(Q_t_ts) + np.log(Q_r_ts) + np.log(Q_v_ts)
    prefactor += -np.log(Q_t_r) - np.log(Q_r_r) - np.log(Q_v_r)
    prefactor += -np.log(2*np.pi*hbar*ts_rp.beta)
    Vdd = ts_rp.beads[0]._V - r_rp.beads[0]._V 
    exponent = (-Vdd * ts_rp.beta)
    print(prefactor, exponent)
    return np.exp(prefactor + exponent) / time_au_to_s

def rate(inst_rp:RingPolymer, r_rp:RingPolymer) -> float:
    Q_t_inst = trans_pfxn(inst_rp)
    Q_r_inst = rot_pfxn(inst_rp)
    Q_v_inst = vib_pfxn_inst(inst_rp)
    Q_t_r = trans_pfxn(r_rp)
    Q_r_r = rot_pfxn(r_rp)
    Q_v_r = vib_pfxn_R(r_rp, n_ignore=6) # Assume not linear
    print(Q_t_inst, Q_r_inst, Q_v_inst)
    print(Q_t_r, Q_r_r, Q_v_r)
    prefactor = np.log(Q_t_inst) + np.log(Q_r_inst) + np.log(Q_v_inst)
    prefactor += -np.log(Q_t_r) - np.log(Q_r_r) - np.log(Q_v_r)
    prefactor += -np.log(2*np.pi*hbar*inst_rp.beta)
    r_V = r_rp.beads[0]._V
    inst_S_over_h = inst_rp.full_U(ref_E=r_V) * inst_rp.betaN
    exponent = (-inst_S_over_h)
    print(prefactor, exponent)
    return np.exp(prefactor + exponent) / time_au_to_s

def kappa(inst_rp:RingPolymer, ts_rp:RingPolymer) -> float:
    Q_t_inst = trans_pfxn(inst_rp)
    Q_r_inst = rot_pfxn(inst_rp)
    Q_v_inst = vib_pfxn_inst(inst_rp)
    Q_t_ts = trans_pfxn(ts_rp)
    Q_r_ts = rot_pfxn(ts_rp)
    Q_v_ts = vib_pfxn_R(ts_rp, n_ignore=7) # 7 because TS
    #print(Q_t_inst, Q_r_inst, Q_v_inst)
    #print(Q_t_ts, Q_r_ts, Q_v_ts)
    inst_S_over_h = inst_rp.full_U() * inst_rp.betaN
    ts_V = ts_rp.beads[0]._V
    exponent = (-inst_S_over_h) + (ts_rp.beta*ts_V)
    prefactor = np.log(Q_t_inst) + np.log(Q_r_inst) + np.log(Q_v_inst)
    prefactor += -np.log(Q_t_ts) - np.log(Q_r_ts) - np.log(Q_v_ts)
    #print(np.exp(prefactor), np.exp(exponent))
    return np.exp(prefactor + exponent)
