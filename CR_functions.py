import math
import numpy as np
import astropy.constants as con
import astropy.units as u
import scipy.integrate as integrate
import scipy.special as special
from galprep import *

# CR Diffusion fit function (Equation 20 in Sampson+(2022)

def fitfunc(f, M_A0, chi):
    '''
    Fit function for the parallel diffusion coefficient and ratio
    of diffusion coefficients k_parallel/k_perpendicular as derived
    in Sampson+(2022).
    
    INPUTS:
    f [string] ------- value to calculate. Choose from {"k_ratio",
                       "k_perp", "u_v"}.
    M_A0 [array] ----- array of Alfven Mach numbers from RAMSES snapshot.
                       Should be dimensionless.
    chi [array] ------ ionization fraction of each cell in RAMSES snapshot
                       as found using the cooling tables of RAMSES.
                       
    OUTPUTS:
    fvals [array] ---- fit values for {"k_ratio", "k_perp", "u_v"}. Note 
    that the units for the diffusion coefficient are in l_0 * sigma_1D, 
    where l_0 = dx / 2.
    '''
    def get_params(f):
        if f == 'k_ratio':
            pvec = [1.077,-0.0175,5.65,-0.403,-5.94,-0.201]
            uvec = [0.069,0.0097,0.69,0.015,0.42,0.022]
        elif f == 'k_perp':
            pvec = [0.0541,-0.017,0.0804,-0.324,5.59,0.074]
            uvec = [0.0050,0.016,0.0074,0.011,0.62,0.019]
        elif f == 'u_v':
            pvec = [1.546,0.223,0.306,-0.110,-7.1,-0.132]
            uvec = [0.060,0.0058,0.071,0.024,1.9,0.041]

        return pvec, uvec
    
    pvec, uvec = get_params(f)
    p0, p1, p2, p3, p4, p5 = pvec
    fvals = p0 * chi ** p1 + p2 * chi ** p3 * 0.5 * (np.tanh(p4 * (np.log10(M_A0)-p5)) + 1)
    
    return fvals


# Approximation functions:

def kappa_ani(M, M_A0, chi):
    k_par =  M / (22 * M_A0 * chi)
    k_ratio = 1 / (2 * M_A0 * np.sqrt(chi))
    
    return k_par, k_ratio

def kappa_trans_iso(M_A0, chi):
    
    k_par = (M_A0 * np.sqrt(chi)) ** (-1)
    k_ratio = 1
    
    return k_par, k_ratio
    
# =========================================================================
# =========================================================================
# Yan & Lazarian approximations:
# =========================================================================
# =========================================================================

# Relevant papers
#    Yan & Lazarian (2008)
#    Commerçon+(2019)
#    Xu+(2016)

# Assumptions: 
#     lambda_par >> L_inj in full Alfven Mach number regime
#     CR velocity ~ c
#     In the sub-Alfvenic regime, D_par >> D_perp and
#     we can assume lambda_par > lambda_crit and lambda_par ~ L_inj

def YL_approx(data):
    '''
    INPUTS:
    data ---------  RAMSES snapshot that's been run through galprep
    
    OUTPUTS:
    Dpars [array] - diffusion coefficient parallel to magnetic field
    Perps [array] - diffusion coefficient perpendicular to magnetic field
    '''
    def highM(L, MA, v):
        lA = L / MA ** 3
        D = lA * v / 3
        return D, D

    def lowM(L, MA, v):
        Dpar = L * v / 3
        Dperp = MA ** 4 * Dpar
        return Dpar, Dperp
    
    L = data.gas['smooth'].in_units('cm')  # cell size dx
    MA = data.gas['M_A0']  # Alfven mach number 
    v = con.c.to('cm / s')
    
    Dpars = pynbody.array.SimArray(np.zeros(len(L)), 
                                   units='cm**2 s**-1')
    Dperps = pynbody.array.SimArray(np.zeros(len(L)), 
                                   units='cm**2 s**-1')
    high = MA >= 1
    low = MA < 1
    
    Dpars[high], Dperps[high] = highM(L[high], MA[high], v)
    Dpars[low], Dperps[low] = lowM(L[low], MA[low], v)
    
    Dpars = pynbody.array.SimArray(Dpars.in_units('cm**2 s**-1'), 
                                   units='cm**2 s**-1')
    Dperps = pynbody.array.SimArray(Dperps.in_units('cm**2 s**-1'), 
                                   units='cm**2 s**-1')
    
    
    return Dpars, Dperps