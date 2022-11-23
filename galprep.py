import numpy as np
from scipy.stats import linregress
import pylab
import pynbody
from scipy.interpolate  import griddata
from astropy.table import Table
import astropy.constants as co
import astropy.units as u
from licplot import lic_internal
import matplotlib.pyplot as plt
from matplotlib import rc, colors
from matplotlib import ticker, cm
import matplotlib.animation as animation
from matplotlib.artist import Artist
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
import astropy.constants as con
from michaels_functions import center_and_r_vir, remove_bulk_velocity, read_unit_from_info
from michaels_functions import s2, scrit, eff, a_stoa_m, s2m, scritm, effm, vec_transform
from read_ramses_cooling import *
from CR_functions import *
import pandas as pd

def magfieldcalcs(data):
    data.physical_units()
    omega_b, omega_m, unit_l, unit_d, unit_t = read_unit_from_info(data)
    
    unit_v=unit_l/unit_t #cm/s
    unit_b = np.sqrt(4*np.pi*unit_d)*unit_v #Gaussian units
    
    # Magnetic field - cartesian components
    data.gas['B_xc'] = 0.5 * (data.gas['B_x_left'] + data.gas['B_x_right']) * unit_b
    data.gas['B_yc'] = 0.5 * (data.gas['B_y_left'] + data.gas['B_y_right']) * unit_b
    data.gas['B_zc'] = 0.5 * (data.gas['B_z_left'] + data.gas['B_z_right']) * unit_b
    data.gas['B2'] = data.gas['B_xc']**2 + data.gas['B_yc']**2 + data.gas['B_zc']**2
    data.gas['B_norm'] = np.sqrt(data.gas['B2'])
    # conversion to microGauss
    data.gas['Bx_uG']=data.gas['B_xc'] * 1e6
    data.gas['By_uG']=data.gas['B_yc'] * 1e6
    data.gas['Bz_uG']=data.gas['B_zc'] * 1e6
    data.gas['B_norm_uG'] = data.gas['B_norm'] * 1e6
    data.gas['Bx_norm_uG']=np.abs(data.gas['Bx_uG'])
    data.gas['By_norm_uG']=np.abs(data.gas['By_uG'])
    data.gas['Bz_norm_uG']=np.abs(data.gas['Bz_uG'])

    # Magnetic field - cylindrical components
    data.gas['Brxy']=(data.gas['B_xc']*data.gas['x'].in_units('cm') + (data.gas['B_yc']*data.gas['y'].in_units('cm')))/data.gas['rxy'].in_units('cm')
    data.gas['Btxy']=(data.gas['B_yc']*data.gas['x'].in_units('cm') - (data.gas['B_xc']*data.gas['y'].in_units('cm')))/data.gas['rxy'].in_units('cm')
    # conversion to microGauss
    data.gas['Brxy_uG'] = data.gas['Brxy'] * 1e6
    data.gas['Btxy_uG'] = data.gas['Btxy'] * 1e6
    data.gas['Bz_uG'] = data.gas['B_zc'] * 1e6
    
    return data

def alfvencalcs(data):
    #Alvfen speed & beta values
    #import pdb
    #pdb.set_trace()
    data.gas['v_a'] = pynbody.array.SimArray(data.gas['B_norm'] / np.sqrt(4.*np.pi*data.gas['rho'].in_units('g cm**-3')), 
                                             units='cm s**-1')
    data.gas['beta'] = (data.gas['c_s'].in_units('cm s**-1') / data.gas['v_a'])**2. 
    data.gas['M_A0'] = pynbody.array.SimArray(data.gas['c_s'].in_units('cm s**-1').view(type=np.ndarray) * data.gas['M_s'] / data.gas['v_a'].in_units('cm s**-1').view(type=np.ndarray))

    return data

def alfvir_eff_calcs(data):
    #Sonic/magnetic alpha_vir and efficiencies
    data.gas['alpha_vir_s'] = (15. / np.pi) * (c_s**2. / (G * rho * dx**2.)) * (1. + M_s**2.)
    data.gas['alpha_vir_m'] = a_stoa_m(data.gas['alpha_vir_s'].view(type=np.ndarray), 
                                       data.gas['M_s'].view(type=np.ndarray), 
                                       data.gas['beta'].view(type=np.ndarray))
    data.gas['eff_s'] = eff(data.gas['alpha_vir_s'].view(type=np.ndarray), 
                            data.gas['M_s'].view(type=np.ndarray))
    data.gas['eff_m'] = effm(data.gas['alpha_vir_s'].view(type=np.ndarray), 
                             data.gas['M_s'].view(type=np.ndarray), 
                             data.gas['beta'].view(type=np.ndarray))

    return data

def xioncalcs(data, path, output):
    coolcode = (path+output)[-5:]
    coolpath = path+output+'/cooling_{}.out'.format(coolcode)
    c, lognH, logT2 = rd_cool(coolpath)
    
    lognHvals, logT2vals = np.meshgrid(lognH, logT2)
    nHvals = 10 ** lognHvals
    T2vals = 10 ** logT2vals
    points = np.array([np.concatenate(nHvals), np.concatenate(T2vals)]).T
    
    nHgal = (data.gas['rho'].in_units('g cm**-3')/co.m_p.cgs)
    T2gal = data.gas['temp'] 
    
    logxion = griddata(points, np.concatenate(c.xion), np.array([nHgal, T2gal]).T, method='nearest')
    xion = np.array(10**logxion)
    
    data.gas['xion'] = np.array(xion)
    
    return data
    
def galprep(path, output, magfields=True, alfven=True, xion=True, alphavir=False, **kwargs):
    data = pynbody.load(path + output)
    data.physical_units()
    omega_b, omega_m, unit_l, unit_d, unit_t = read_unit_from_info(data)
    
    print(data.properties)
    print(data.properties['time'].in_units('Myr'))
    
    unit_v=unit_l/unit_t #cm/s
    unit_b = np.sqrt(4*np.pi*unit_d)*unit_v #Gaussian units
    gamma = 5./3.

    data.gas['sigma1d'] = pynbody.array.SimArray(np.sqrt(2/3*data.gas['scalar_01'])*unit_v/1e5, 
                                                 units='km s**-1')
    data.gas['c_s'] = np.sqrt(gamma * data.gas['p'] / data.gas['rho'])
    data.gas['M_s'] = data.gas['sigma1d'] / data.gas['c_s'].in_units('km s**-1')

    if magfields == True:
        data = magfieldcalcs(data)
        
    if alfven == True:
        data = alfvencalcs(data)
        
    # convert everything to cgs units for computing alpha_virs and efficiencies
    rho = data.gas['rho'].in_units('g cm**-3').view(type=np.ndarray)
    dx = data.gas['smooth'].in_units('cm').view(type=np.ndarray)
    G = float(pynbody.array.SimArray(6.67e-8, units='cm**3 g**-1 s**-2'))
    c_s = data.gas['c_s'].in_units('cm s**-1').view(type=np.ndarray)
    M_s = data.gas['M_s'].view(type=np.ndarray)
    beta = data.gas['beta'].view(type=np.ndarray)

    if alphavir == True:
        data = alphavir_eff_calcs(data)

    if xion == True:
        data = xioncalcs(data, path, output)

    return data
