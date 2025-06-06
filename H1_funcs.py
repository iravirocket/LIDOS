import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u


import os

import glob


from astropy import constants as const
from astropy.io import ascii
import scipy.special as sp


from scipy.signal import convolve
from scipy.special import expit
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit,minimize


def h1abs_lya(N_log,vdop):
    #Read in H1 data
    df = pd.read_csv('h1.dat', delimiter='\t',header=None)
    df = df.drop(columns=[5, 7]) #Remove the NaNs
    
    lya = df[4][0] *u.AA
    f_s = df[6][0]
    gam = df[9][0]/u.s
    
    N = 10**N_log *(u.cm **-2)#colum density
    vshifts=0 *u.km / u.s
    vdop=vdop * (u.km/u.s)
    lamshifts = lya * vshifts/const.c
    ncompd=1
    n_vshifts=1
    n_vdop=1
    #nlam = 6000 #number of synthetic data points
    
    #Doppler broadening parament and oscillator strength*lamda
    bs = lya*vdop/const.c
    f_lam = lya*f_s
    
    #Make wavelength scale
    #lam_edge = 1200 *u.AA
    #diff = (lya - lam_edge)
    diff=30*u.AA
    nlam = (diff.value*2)*100 #number of synthetic data points
    
    lama = np.linspace(lya-diff,lya+diff,int(2*nlam))
    
    #Frequency of lya (nu0) and nus for lamdas
    nus = const.c / (lama)
    nu0s = (const.c/lya)
    
    #Doppler freq width
    del_nu = (vdop/const.c) * nu0s

    #Calculate Voigt parameters a and u
    voigt_a = (gam/(4*np.pi*del_nu))
    voigt_u = (nus-nu0s)/del_nu
    
    #cross section (x-section) of line core
    x0s = np.sqrt(2*np.pi) *(const.e.esu**2)* f_lam/(const.m_e *const.c*bs*nu0s)
    
    #Voigt profile using u,a,and the error function
    voigt_prof = sp.wofz(voigt_u + 1j * voigt_a).real

    #cross-section of core and in wings of profile (line core smeared by voigt)
    xphis = (-x0s*voigt_prof)
    
    #The negative optical depth of line core and wings and transmission profile
    negative_tau = (N*xphis).cgs
    trans = np.exp(negative_tau)
    
    return trans, negative_tau, lama

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def make_kernel(fwhm, grid, nfwhm=4.0, gauss=False, box=False, triangle=False):
    if fwhm is None or grid is None:
        raise ValueError("Both fwhm and grid must be specified")
    
    
    ngrd = len(grid)
    spacing = (grid[-1] - grid[0]) / (ngrd - 1)
    nkpts = int(np.ceil(nfwhm * fwhm / spacing))
    if nkpts % 2 == 0:
        nkpts += 1
    
    
    kernel = np.arange(nkpts,dtype=float) - (nkpts // 2)
    kernel *= spacing
    
    if box:
        kernel_box = np.zeros_like(kernel)
        kernel_box[np.abs(kernel) <= fwhm / 2] = 1.0
        kernel = kernel_box
    elif triangle:
        kernel_box = np.zeros_like(kernel)
        kernel_box[np.abs(kernel) <= fwhm / 2] = 1.0
        kernel = convolve(kernel_box, kernel_box, mode='same')
    else:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        kernel = np.exp(-0.5 * (kernel / sigma) ** 2)

    kernel /= np.sum(kernel)  # Normalize the kernel

    return kernel

def interp_func(x_model,y_model,x_data):
    #Interpolate to match length of observed data
    interp_func = interp1d(x_model, y_model, kind='linear')

    interp_y_val= interp_func(x_data)
    return interp_y_val