import numpy as np
from .fake_ring_polymer import FakeRingPolymer
from .instanton import Instanton
from .partition_functions import rate

def optimize_rate(instanton:Instanton, reactant:FakeRingPolymer, 
                  opt_plan:dict) -> int:
    # Define rate convergence criteria as relative to absolute rate!
    print("Running rate optimization routine:")
    k_conv = 1e-4
    k_max_steps = 10
    print(f"\t{"k_conv":20s} = {k_conv:5.2e}")
    print(f"\t{"k_max_steps":20s} = {k_max_steps:5d}")
    working_dir = f"k_opt_{len(instanton.beads)}/"
    instanton.optimize(opt_plan, working_dir)
    instanton.evaluate_all_beads(der_lvl=2)
    instanton.save_state(working_dir + "final")
    k_old = rate(instanton, reactant)
    print(f"\t ***Rate computed: {k_old}")
    for i in range(k_max_steps):
        instanton.double_beads()
        instanton.restart_at = 0 # Reset in case of restart
        reactant.update_beadN(instanton.N)
        print(f"Beads doubled to {len(instanton.beads)}")
        working_dir = f"k_opt_{len(instanton.beads)}/"
        opt_plan["step_limit"] *= 2
        #opt_plan["hess_every"] = 5
        #opt_plan["opt_method"] = "nr"
        instanton.optimize(opt_plan, save_dir_prefix=working_dir)
        instanton.save_state(working_dir + "final")
        k_new = rate(instanton, reactant)
        print(f"\t ***Rate computed: {k_new:7.5e}  (old: {k_old:7.5e})")
        print(f"\t Rate conv ({k_conv}): {k_new - k_old: 5.2e}   {(k_new-k_old)*k_old:5.2e}")
        if np.isclose(k_new, k_old, rtol=k_conv):
            return 0
        k_old = k_new
    raise InterruptedError("Max number of iterations reached for k optimization.")
