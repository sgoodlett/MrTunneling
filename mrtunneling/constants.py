import numpy as np
import qcelemental as qcel

# Assume atomic units throughout program
hbar = 1.0
kb = 3.166811563e-6 # Eh/K
c_cm = 1.0
au_to_amu = qcel.constants.get("electron mass in u")

h2j = qcel.constants.conversion_factor("hartree", "joule")
h2cm = qcel.constants.conversion_factor("hartree", "wavenumber")
b2m = qcel.constants.conversion_factor("bohr", "m")
amu2kg = qcel.constants.conversion_factor("amu", "kg")

# k from Hessian is in Eh / a_0**2 amu
k_to_rad_per_sec = np.sqrt(h2j * (b2m**-2) * (amu2kg**-1))
k_to_rad_per_sec /= 2*np.pi * qcel.constants.c * 100
freq_to_hartree = np.sqrt((5.48579909065 * (10 ** (-4)))) # WTF is this number

ref_mass = 1.0 #/ au_to_amu

# Time
time_au_to_s = qcel.constants.get("atomic unit of time")
