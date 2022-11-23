import pynbody
import sys
import os
import numpy as np
import glob
import re
from scipy.special import erfc, erf

# Eden's additional functions

def s2(M, b=0.4):
    return np.log(1.0 + (b**2.0 * M**2.0))

def scrit(a, M):
    return np.log(a * (1.0 + ((2.0 * M**4.0)/(1.0 + M**2.0))))

def eff(a, M):
    return 0.5 * np.exp((3.0/8.0) * s2(M)) * (1.0 + erf( (s2(M) - scrit(a, M)) / np.sqrt(2.0*s2(M)) ))

def a_stoa_m(a_s, M, beta=0.39):
    mag_corr = 1. + 1./beta # (1+beta)/beta
    return a_s * ((mag_corr + M**2.)/(1. + M**2.))

def s2m(M, beta=0.39, b=0.4):
    mag_corr = 1. + 1./beta # (1+beta)/beta
    return np.log(1. + (b**2. * M**2. / mag_corr))

def scritm(a_s, M, beta=0.39):
    mag_corr = 1. + 1./beta # (1+beta)/beta
    a = a_stoa_m(a_s, M, beta)
    #return np.log(a * (1. + ((2. * M**4.)/(mag_corr + M**2.))) / mag_corr)
    return np.log(a * (1.0 + (2. * M**4 / (mag_corr * (mag_corr + M**2)))))

def effm(a_s, M, beta=0.39):
    mag_corr = 1. + 1./beta # (1+beta)/beta
    a = a_stoa_m(a_s, M, beta)
    sigs = s2m(M, beta)
    scritt = scritm(a, M, beta)
    return 0.5 * np.exp((3.0/8.0) * sigs) * (1.0 + erf( (sigs - scritt) / np.sqrt(2.0*sigs) ))

# transforming vector array such that components are correctly projected onto face or side views

def vec_transform(region, vec, view=None):
    
    assert vec.shape[0]==3
    
    dregion = region.gas[pynbody.filt.Sphere(radius = '5.0 kpc')]
    amv = pynbody.analysis.angmom.ang_mom_vec(dregion)
    
    if view == 'Faceon':
        mf = pynbody.analysis.angmom.calc_faceon_matrix(amv)
        vec_new = pynbody.array.SimArray(np.dot(mf, vec).T)
    elif view == 'Sideon':
        ms = pynbody.analysis.angmom.calc_sideon_matrix(amv)
        vec_new = pynbody.array.SimArray(np.dot(ms, vec).T)
    else:
        raise KeyError
        
    return vec_new

# Original functions from Michael

def center_and_r_vir(data, aexp, path):
    if os.path.exists(path + "/centers.txt"):
        centering(path + "/centers.txt", data, aexp)

    else:
        try:
            pynbody.analysis.halo.center(data.s)
            sys.stdout.write('no catalogue: centering on stars \n')
        except:
            pynbody.analysis.halo.center(data.dm, vel=False)
            sys.stdout.write('no catalogue: centering on dm \n')

    sph_s_vir = pynbody.filt.Sphere(radius='%f kpc' % (320.0 * aexp - 10.0))
    h_1kpc = data[sph_s_vir]

    if len(h_1kpc.s) > 0:
        cent_stars = pynbody.analysis.halo.center(h_1kpc.s, retcen=True,
                                                  move_all=False, vel=False)
        print("shifting on Stars:", cent_stars)

        pynbody.analysis.halo.transformation.translate(data.s, -cent_stars)
        pynbody.analysis.halo.transformation.translate(data.g, -cent_stars)
        pynbody.analysis.halo.transformation.translate(data.d, -cent_stars)

    else:
        cent_stars = np.zeros(3)
        print("no Stars to center")

    try:
        r_vir_guess = pynbody.analysis.halo.virial_radius(
            h_1kpc.d, cen=(0, 0, 0), overden=200, rho_def="critical")
    except:
        sys.stdout.write('R_vir not converged \n')
        return None

    if float(r_vir_guess) / 2. < np.sqrt(np.sum(np.square(cent_stars))):
        print("WARNING: stars shifted by more than R_vir /2 ! REVERSING !")

        pynbody.analysis.halo.transformation.translate(data.s, cent_stars)
        pynbody.analysis.halo.transformation.translate(data.g, cent_stars)
        pynbody.analysis.halo.transformation.translate(data.d, cent_stars)

    dm_mass = np.array(data.dm["mass"].in_units("Msol"))

    dm_x = np.array(data.dm["x"].in_units("kpc"))
    dm_y = np.array(data.dm["y"].in_units("kpc"))
    dm_z = np.array(data.dm["z"].in_units("kpc"))

    dm_r_3d = np.sqrt(dm_x**2 + dm_y**2 + dm_z**2)

    r_list = np.linspace(.5, r_vir_guess * 2., 150)

    m_dm_inside_r = np.array([])
    for x in r_list:
        m_dm_inside_r = np.append(
            m_dm_inside_r, np.sum(dm_mass[dm_r_3d < x]))

    rho_avg = 3. * m_dm_inside_r / (4.0 * np.pi * r_list**3)

    rho_bar_200 = pynbody.analysis.cosmology.rho_crit(
        data, unit="Msol kpc^-3", z=(1. / aexp - 1.)) * 200.0
    r_vir = r_list[rho_avg < rho_bar_200][0]
    r_vir = pynbody.array.SimArray(r_vir, "kpc")
    print("virial radius:", r_vir)

    if float(r_vir) / 2. < np.sqrt(np.sum(np.square(cent_stars))):
        print("WARNING: stars shifted by more than R_vir /2 !")
    '''
    print pynbody.analysis.halo.virial_radius(
            h_1kpc.d, cen=(0, 0, 0), overden=200, rho_def="critical")
    '''
    del h_1kpc, sph_s_vir, dm_mass, dm_r_3d, dm_x, dm_y, dm_z
    return r_vir


def centering(path_here, data, aexp):
    cat_a, cat_x, cat_y, cat_z = np.genfromtxt(path_here, unpack=True)

    center = np.array([np.interp(aexp, cat_a, cat_x),
                       np.interp(aexp, cat_a, cat_y),
                       np.interp(aexp, cat_a, cat_z)])

    if np.nanmax(cat_a) < aexp:
        pynbody.analysis.halo.center(data.s)
        sys.stdout.write('aexp not in catalogue: centering on stars \n')

    else:
        pynbody.analysis.halo.transformation.translate(data.s, -center)
        pynbody.analysis.halo.transformation.translate(data.g, -center)
        pynbody.analysis.halo.transformation.translate(data.d, -center)

    return None


def config_check(path):
    log_files = os.listdir(path + '/../log/')
    log_files.sort()

    with open(path + "/../log/" + log_files[0], "r") as f:
        fi = f.readlines()

    cfg = pynbody.config_parser.get("ramses", "hydro-blocks")
    print("Pynbody config: nvar =", len(cfg.split(",")))
    print("Ramses log:    ", fi[12][-10:-1], "\n")

    assert int(fi[12][-3:-1]) == len(pynbody.config_parser.get("ramses",
                                     "hydro-blocks").split(",")), \
        "Problem with nvar!"

    for line in fi[22:30]:
        if line[:13] == "_____________":
            break
        else:
            print(cfg.split(",")[int(line[-3:-1]) - 1].ljust(6),
                  " <->", line[4:-1])

    print("\n", cfg)


def read_unit_from_info(data, path=None):
    '''
    Reads the info files and returns:
    omega_b, unit_l, unit_d, unit_t
    '''
    try:
        filename = glob.glob(data.filename + "/info_*")[0]
    except:
        filename = glob.glob(path + "info_*")[0]

    with open(filename, "r") as info_file:

        for line in info_file:
            if line[0:13] == "unit_l      =":
                unit_l = float(line[14:-1])
            if line[0:13] == "unit_d      =":
                unit_d = float(line[14:-1])
            if line[0:13] == "unit_t      =":
                unit_t = float(line[14:-1])
            if line[0:13] == "omega_b     =":
                omega_b = float(line[14:-1])
            if line[0:13] == "omega_m     =":
                omega_m = float(line[14:-1])

    return omega_b, omega_m, unit_l, unit_d, unit_t


def scale_list(path, a_min=0.1):
    info_list = np.sort(glob.glob(path + "output_?????/info_?????.txt"))

    a_list = []
    for info in info_list:
        with open(info, 'r') as f:
            for line in f:
                if re.search('aexp', line):
                    a_list.append(float(line.split()[-1]))

    a_list = np.array(a_list)
    a_list = a_list[a_list > a_min]

    return np.sort(a_list)[::-1]


def get_bulk_vel(data):
    sph_1kpc = pynbody.filt.Sphere(radius='1. kpc')
    h_1kpc = data[sph_1kpc]

    if len(h_1kpc.s) == 0:
        return None

    bulk_vel = pynbody.array.SimArray(
        (np.average(h_1kpc.s["vx"].in_units("km s^-1"),
         weights=h_1kpc.s["mass"]),
         np.average(h_1kpc.s["vy"].in_units("km s^-1"),
         weights=h_1kpc.s["mass"]),
         np.average(h_1kpc.s["vz"].in_units("km s^-1"),
         weights=h_1kpc.s["mass"])), 'km s^-1')
    return bulk_vel


def remove_bulk_velocity(data):

    bulk_vel = get_bulk_vel(data)

    pynbody.analysis.halo.transformation.v_translate(data.s, -bulk_vel)
    pynbody.analysis.halo.transformation.v_translate(data.g, -bulk_vel)
    pynbody.analysis.halo.transformation.v_translate(data.d, -bulk_vel)


def behroozi_2013(z=0):
    """Loads stellar mass from Moster+2013 relation at given redshift"""
    rca = "/home/cluster/mkrets/BehrooziAbundance/"

    if z == 0:
        z = 0.10  # relation for z=0 is missing, closest is z=0.1

    adat = np.genfromtxt(rca + 'c_smmr_z{z:.02f}_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat'.format(z=z))

    am_halomass = adat[:, 0]
    am_stellarmass = adat[:, 1] + am_halomass
    am_stellarmass_ue = (adat[:, 1]+adat[:, 2]) + am_halomass
    am_stellarmass_le = (adat[:, 1]-adat[:, 3]) + am_halomass

    return am_halomass, am_stellarmass, am_stellarmass_ue, am_stellarmass_le


def halo_abundance(m200, z):
    a = 1./(1+z)
    nu = np.exp(-4*a*a)
    log_eps = -1.777+(-0.006*(a-1.)+0.000*z)*nu-0.119*(a-1.)
    log_M1 = 11.514+(-1.793*(a-1.)+(-0.251)*z)*nu
    alpha = -1.412+(0.731*(a-1.))*nu
    delta = 3.508+(2.608*(a-1.)+(-0.043)*z)*nu
    beta = 0.316+(1.319*(a-1.)+0.279*z)*nu
    x = 0.0
    f0 = -np.log10(np.power(10., alpha*x)+1.)+delta * \
        np.power(np.log10(1.+np.exp(x)), beta)/(1.+np.exp(np.power(10., -x)))
    x = np.log10(m200)-log_M1
    fx = -np.log10(np.power(10., alpha*x)+1.)+delta * \
        np.power(np.log10(1.+np.exp(x)), beta)/(1.+np.exp(pow(10., -x)))
    mstar = np.power(10., (log_M1+log_eps)+fx-f0)

    return mstar


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def rotation(R, x, y, z):
    return np.dot(R, [x, y, z])


def Rz(phi):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(phi)
    R[0, 1] = -np.sin(phi)
    R[1, 0] = np.sin(phi)
    R[1, 1] = np.cos(phi)
    R[2, 2] = 1.
    return R


def Ry(phi):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(phi)
    R[0, 2] = np.sin(phi)
    R[2, 0] = -np.sin(phi)
    R[2, 2] = np.cos(phi)
    R[1, 1] = 1.
    return R


def Rx(phi):
    R = np.zeros((3, 3))
    R[0, 0] = 1.
    R[1, 1] = np.cos(phi)
    R[1, 2] = -np.sin(phi)
    R[2, 1] = np.sin(phi)
    R[2, 2] = np.cos(phi)
    return R


def rotMatToZ(vec):
    th = np.arccos(vec[2])
    phi = np.arctan2(vec[1], vec[0])
    return np.dot(Rz(phi), np.dot(Ry(th), np.dot(Rz(0), np.identity(3)))).T


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
