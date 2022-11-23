import math
import numpy as np
import astropy.constants as con
import astropy.units as u
import scipy.integrate as integrate
import scipy.special as special
from galprep import *

# Diffusion coefficient calcs from Yan & Lazarian (2008) :

def larmor_freq(data):
    Omega = - (con.e.to('C') * 1 * data.gas['B_norm'] * u.Gauss / (2 * con.m_p)).to('Hz').value
    
    Omega = - Omega  # !! ?? !!
    return Omega

def rigidity(data):
    '''
    Calculates rigidity of CRs associated with each cell
    '''
    omega = larmor_freq(data)
    v_CR, _, _, _ = CR_velocity(data)
    L = data.gas['smooth'].in_units('m')
    R = v_CR / (L * omega)
    
    return R    

def CR_velocity(data):
    '''
    Calculate CR particle velocity from the plasma beta value of each cell.
    Assign random direction to the CRs assuming a spherically symmetric distribution
    (which might not be accurate!!! ???)
    '''
    np.random.seed(300)
    v_CR = pynbody.array.SimArray(data.gas['beta'] * con.c.to('m/s').value, 
                                             units='m s**-1')
    vecdirs = np.random.normal(0, 1, (len(data.gas), 3))
    vnormed = (vecdirs.T / np.linalg.norm(vecdirs, axis=1)).T
    vCRs = (vnormed.T * v_CR).T
    vx = vCRs[:,0]
    vy = vCRs[:,1]
    vz = vCRs[:,2]
    
    return v_CR, vx, vy, vz

def get_B_dir(data):
    '''
    Get vector direction of magnetic field in each snapshot cell
    '''
    norms = data.gas['B_norm']
    bx = data.gas['B_xc'] / norms
    by = data.gas['B_yc'] / norms
    bz = data.gas['B_zc'] / norms
    
    return bx, by, bz

def CR_vB(data):
    vmags, vx, vy, vz = CR_velocity(data)
    bx, by, bz = get_B_dir(data)
    vxpar = vx * bx
    vxperp = vx - vxpar
    vypar = vy * by
    vyperp = vy - vypar
    vzpar = vz * bz
    vzperp = vz - vzpar
    
    vparmag = np.sqrt(vxpar**2 + vypar**2 + vzpar**2)
    vperpmag = np.sqrt(vxperp**2 + vyperp**2 + vzperp**2)
    
    return vmags, vparmag, vperpmag

def CR_kvals(data):
    '''
    Calculates the wavenumbers of the CRs in each cell
    '''
    vmag, vx, vy, vz = CR_velocity(data)
    vxnorm = vx / vmag
    vynorm = vy / vmag
    vznorm = vz / vmag
    
    omega = larmor_freq(data)
    k = omega / vmag
    
    kx = k * vxnorm
    ky = k * vynorm
    kz = k * vznorm
    
    bx, by, bz = get_B_dir(data)
    
    kpar = np.sqrt((kx*bx)**2 + (ky*by)**2 + (kz*bz)**2)
    kperp = np.sqrt((kx*(1-bx))**2 + (ky*(1-by))**2 + (kz*(1-bz))**2)
    
    return k, kpar, kperp

def lAval(data, L):
    '''
    - data is the RAMSES snapshot that has been run through galprep
    - L is the injection scale of the turbulence (need to find out units!!)
    '''
    MA = data.gas['M_A0']
    
    return L / MA**3

def kmax_L(zeta, data):
    MA = data.gas['M_A0']
    mi = con.m_p
    me = con.m_e
    beta = data.gas['beta']
    sin2 = 1 - zeta**2
    coeff = 4 * MA**4 * mi / (np.pi * me * beta) * zeta**2 / sin2 ** 2
    return coeff * np.exp(2*me/(beta*mi*zeta**2))

def D_G(data, mu, dmu, dmupar, v, vA, R, L):
    '''
    data:          RAMSES galaxy snapshot that has been run through galprep
    mu [vector]:   Pitch-angle cosine of CRs
    R:             CR rigidities
    L:             Injection scale of the turbulence
    dmu:           Pitch-angle cosine dispersion 
    '''
    k, kpar, kperp = CR_kvals(data)
    
    def f(zeta, x, Jp, mu, dmu, dmupar):
        coeff = x ** (-5/2) * zeta / dmupar * Jp ** 2
        exp = np.exp(-(mu - 1 / (x*zeta*R))**2 / dmu ** 2)
        return coeff * exp
    
    w = kperp * L * R * np.sqrt(1-mu**2)
    Jp = special.jvp(1, w, n=1)
    coeff = v * np.sqrt(np.pi) * (1-mu**2) / (2 * L * R**2)
    kmaxL = 1e7 # ?? !! ?? kmax_L(zeta, data)
    DG = integrate.nquad(f, [[0, 1],[1, kmaxL]], args=[Jp,mu,dmu,dmupar])
    DG = coeff * DG
    
    return DG
    
def D_T(data, mu, dmu, dmupar, v, vA, R, L):
    k, kpar, kperp = CR_kvals(data)
    
    def f(zeta, x, Jv, vA, v, mu, dmu, dmupar):
        coeff = x ** (-5/2) * zeta / dmupar * Jv ** 2
        exp = np.exp(-(mu - vA / v)**2 / dmu ** 2)
        return coeff * exp    
    
    w = kperp * L * R * np.sqrt(1-mu**2)
    Jv = special.jv(1, w)
    coeff = v * np.sqrt(np.pi) * (1-mu**2) / (2 * L * R**2)
    kmaxL = 1e7 # !! ?? !!kmax_L(zeta, data)
    DT = integrate.nquad(f, [[0, 1], [1, kmaxL]], args=[Jv,vA,v,mu,dmu,dmupar])
    DT = coeff * DT
    
    return DT
    
def lambda_par(data, L, v, vA, R, dmu, dmupar):
    def f(mu, data=data, dmu=dmu, dmupar=dmupar, v=v, vA=vA, R=R, L=L):
        coeff = 3 * v / (4 * L)
        DG = D_G(data, mu, dmu, dmupar, v, vA, R, L)
        DT = D_T(data, mu, dmu, dmupar, v, vA, R, L)
        
        return coeff * (1 - mu**2)**2 / (DG + DT)
        
    lampar = L * integrate.quad(f, 0, 1)
    return lampar

def YL_Dcoeffs_large_scales(data):
    '''
    data:        RAMSES galaxy snapshot that has been run through galprep
    L:           Injection scale of the turbulence (need units!!)
    lam_par:  Mean free path of the CRs parallel to the mean magnetic field
    v_CR:        CR particle veclocity vector (does this get sampled from a distribution?)
    '''
    def high_M(data, lam_par, v_CR, lA):
        Dperp = np.zeros(len(v_CR))
        Dperp[lam_par>lA] = 1 / 3 * l_A * v_CR[lam_par>lA]
        Dperp[lam_par<lA] = 1 / 3 * (lam_par * v_CR)[lam_par<lA]
        return Dperp, Dperp
    
    def low_M(data, lam_par, v_CR, L):
        Dpar = lam_par * v_CR / 3
        Dperp = Dpar * data.gas['M_A0'] ** 4
        Dperp[lam_par>L] *= (L/lam_par)[lam_par>L]
        return Dperp, Dpar
    
    L = data.gas['smooth'].in_units('m')
    M_A = data.gas['M_A0']
    vA = data.gas['v_a'].in_units('m s**-1')
    Omega = larmor_freq(data)
    v, _, _, _ = CR_velocity(data)  # !! ?? !!
    R = rigidity(data)
    lA = lAval(data, L)
    dmu = M_A**(1/2)
    dmupar = dmu # !! ?? !!
    lam_par = lambda_par(data, L, v, vA, R, dmu, dmupar)
    print(lam_par)
    
    Dperp = np.zeros(len(v))
    Dpar = np.zeros(len(v))
    high = M_A > 1
    low = M_A < 1
    
    Dperp[low], Dpar[low] = low_M(data.gas[low], lam_par[low], v[low], L[low])
    Dperp[high], Dpar[high] = low_M(data.gas[high], lam_par[high], v[high], L[high])
    
    return Dperp, Dpar
    
    