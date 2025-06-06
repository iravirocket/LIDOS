# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.modeling.models import Gaussian1D, Linear1D, Voigt1D
from astropy.modeling.fitting import LinearLSQFitter,LMLSQFitter,DogBoxLSQFitter,TRFLSQFitter

from astropy import units as u
import glob
from scipy.signal import convolve
from astropy.io import ascii

from astropy import constants as const
from H1_funcs import make_kernel, h1abs_lya,interp_func
from scipy.integrate import quad,simpson
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d

from EW_Ravi_v3 import SpectrumAnalyzer



def rotational_energy(J, B):
    correction_term = (4.71e-2/u.cm)*((J*(J+1))**2)
    return ((B * J * (J + 1)) - correction_term)* const.h * const.c

def vibrational_energy(v, we):
    correction_term = (121.33/u.cm) * ((v+0.5)**2)
    return (((v + 0.5) * we) - correction_term) * const.h*const.c

def cross_energy_term(v,J):
    Xa_e = 3.062/u.cm
    return (-Xa_e*(v + 0.5)*(J*(J+1)))*const.h*const.c

def boltzmann_factor(E, T):
    return np.exp(-E / (const.k_B * T))

def partition_function_2model(B, we, Tv,TJ,Tcross, max_J=15, max_v=9):
    #Z_rot = sum((2*J + 1) *(2*(J%2) +1)* boltzmann_factor(rotational_energy(J, B), T) for J in range(max_J + 1))
    #Z_vib = sum(boltzmann_factor(vibrational_energy(v, we), T) for v in range(max_v + 1))
    #Zcross= sum((2*J + 1) *(2*(J%2) +1)* boltzmann_factor(cross_energy_term(v,J),T) for J, v in zip(range(max_J + 1), range(max_v+1)))
    Z=0
    for v in range(max_v+1):
        for J in range(max_J+1):
            Z+=(2*J + 1) *(2*(J%2) +1)* boltzmann_factor(cross_energy_term(v,J),Tcross) * boltzmann_factor(rotational_energy(J, B), TJ)* boltzmann_factor(vibrational_energy(v, we), Tv)
    return Z

def population_ratio_2model(v, J, Tv,TJ):
    nuc_spin = J % 2
    Tcross = np.sqrt((Tv**2)+(TJ**2)) / np.sqrt(2)
    g = (2*J + 1)  # Rotational degeneracy
    para_term = (2*nuc_spin + 1)

    # Constants for H2 (in cm^-1)
    B_H2 = 59.335  /(u.cm)# Rotational constant in cm^-1
    #we_H2 = const.c/(2.1*u.um)/(2*np.pi)
    we_H2 = 4161.1 /(u.cm) # Vibrational frequency in cm^-1


    
    E1,E2,E3 = rotational_energy(J, B=B_H2), vibrational_energy(v, we=we_H2), cross_energy_term(v,J)
    E=E1+E2+E3
    B1 = boltzmann_factor(E1,TJ) * boltzmann_factor(E2,Tv) * boltzmann_factor(E3,Tcross)
    
    #Zv = partition_function_2model(B, we, Tv,TJ,Tcross, max_J=15, max_v=9)
    #ZJ = partition_function_2model(B, we, Tv,TJ,Tcross, max_J=15, max_v=9)
    Z = partition_function_2model(B_H2, we_H2, Tv,TJ,Tcross, max_J=15, max_v=9)
    
    frac = g*para_term*(B1) / Z # / (((Zv**3)+(ZJ**3)+(Zcross**3))/3)**(1/3))
    #frac = ([g*para_term*(B1) / (Zv + ZJ + Zcross)])
    
    return (frac),E.to(u.eV)

def find_2pop_model(Tv,TJ=300*u.K):

    #There are the ranges of the tau templates H2ools
    Js = np.arange(0,16) 
    vs = np.arange(0,8)
    
    pop2=[]
    Es2=[]
    for v in vs:
        pop_js = []
        Ejs=[]
        for J in Js:
            pop_js.append(population_ratio_2model(v, J, Tv,TJ)[0])
            Ejs.append(population_ratio_2model(v, J, Tv,TJ)[1].value)
            
        pop2.append(pop_js)
        Es2.append(Ejs)
        
    Es2=np.array(Es2)
    pop2=np.array(pop2)
    return pop2,Es2



def H2_dist_plot(pop,logscale=False):

    Js = np.arange(0,16) 
    vs = np.arange(0,8)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))  # Create a 2x2 grid

    # Remove the top-right subplot
    fig.delaxes(axes[0, 1])
    
    axes[0, 0].plot(Js,pop.T)
    axes[0,0].plot(Js,np.sum(pop.T,axis=1),ls='--',lw=3,color='black',label='Total')
    axes[0, 0].legend([f"v={v:.0f}" for v in vs])
    
    cax = axes[1, 0].imshow(np.log10(pop), aspect='auto', origin='lower', cmap='viridis', extent=[Js.min(), Js.max(), vs.min(), vs.max()])
    cbar = plt.colorbar(cax)
    cbar.set_label("Intensity", fontsize=12)
    
    axes[1, 1].plot(pop,vs);
    axes[1,1].plot(np.sum(pop,axis=1),vs,ls='--',lw=3,color='black',label='Total')
    axes[1,1].legend([f"J={J:.0f}" for J in Js])
    if (logscale == True):
        axes[0, 0].set_yscale('log')
        axes[1, 1].set_xscale('log')





def pop_tau_templates(pop,n_tot,b='3'):
    vector_length_true = 59000
    n=n_tot/1e21
    

    Js = np.arange(0,16) 
    #vs = np.arange(0,8)

    b_list = glob.glob('h2v0-7/tauh2n21b'+b+'j*')
    
    vector_length = 59000
    with open(b_list[0], 'rb') as f:
        # Read the first vector: wavelength grid (59000 values)
        wavelength_grid = np.fromfile(f, dtype=np.float64, count=vector_length).byteswap()
    
    tau = np.zeros_like(wavelength_grid)
    for count,file in enumerate(b_list):
        #Some files are longer than others, this will reduce to shortest common length
        if count >= 4:
            vector_length = 79000
            # Open the file in binary mode ('rb')
            with open(file, 'rb') as f:
                # Read the next 16 vectors: rotational state templates (each of length 59000)
                rotational_templates = np.fromfile(f, dtype=np.float64).byteswap() 
            
            
            # Calculate total expected size for the last 16 vectors
            total_vals_needed = vector_length * 16
            # Extract last 16 vectors
            rotational_data = rotational_templates[-total_vals_needed:]
            
            # Reshape the rotational state templates into a 16x79000 array
            rotational_templates = rotational_data.reshape((16, vector_length))
            #Mask to 59000 long array of other wavelength grid
            rotational_templates = rotational_templates[:, :vector_length_true]
            
            
        else:
            vector_length=vector_length_true
            # Open the file in binary mode ('rb')
            with open(file, 'rb') as f:
                # Read the next 16 vectors: rotational state templates (each of length 59000)
                rotational_templates = np.fromfile(f, dtype=np.float64).byteswap() 
                
            # Calculate total expected size for the last 16 vectors
            total_vals_needed = vector_length * 16
            # Extract last 16 vectors
            rotational_data = rotational_templates[-total_vals_needed:]
            # Reshape the rotational state templates into a 16x59000 array
            rotational_templates = rotational_data.reshape((16, vector_length))
                                        
        #print(np.shape(rotational_templates),count)
        for J in Js:
            tau+= (n*pop[count,J])*rotational_templates[J]
    
    
    return wavelength_grid, np.exp(tau)

def lya_pop(pop):
    
    #Read in Lyman-Werner transitions v'' = 0-7
    H2_data = glob.glob('highjsh*.dat')
    df=[]
    for data in (H2_data):
        df.append(np.loadtxt(data))
    df = np.array(df)
    df = df.reshape(-1, df.shape[2])
    
    #Identify population transitions near lya
    wav = df[:,4]
    lines = np.where(np.isclose(wav,1215.7,atol=5))[0]
    new_df = df[lines]
    
    wavelengths = new_df[:,4]
    init_vJ = new_df[:,[1,3]]
    init_vJ = init_vJ.astype(int)
    
    lya_pops = pop[init_vJ[:,0],init_vJ[:,1]]
    return wavelengths,lya_pops


def syn_lamp_func(amp,fwhm_G,line_ratio,other_I,plots=True):
    #amp, fwhm_G, fwhm_L parameters for Lya Voight1D model
    #line_ratio fraction of 'amp' to fix 1216 to lya
    #other_I strength of background lines from H2 template
    
    data = np.loadtxt('h2ft01b01n18.dat')
    lam = data[:,0]
    units = data[:,1]
    length = np.where(np.isclose(lam,1215.7,atol=7))[0]
    lam = lam[length]
    #units=units[length]
    units=units[length]
    
    #Artifically boost blue flux to match lamp output
    units2=np.concatenate([units[0:int(len(lam)/2)],units[int(len(lam)/2):]])
    
    left_edge = np.where(np.isclose(lam,1214.2,atol=0.7))[0]
    
    units2[left_edge]*=10
    #Data to create pumped 1216 line 
    new_lam = data[:,0]
    new_units = data[:,1]
    length = np.where(np.isclose(new_lam,1216.07,atol=0.1))[0]
    new_lam = new_lam[length]
    new_units=new_units[length]*5e4
    
    l1 = Gaussian1D(mean=1216.07,amplitude=1,stddev=0.01)
    fitter =TRFLSQFitter()
    
    gaussian_kernel = make_kernel(1, lam)
    tru_fit_model = fitter(l1,new_lam,new_units)


    lya_model = Gaussian1D(mean=1215.67,amplitude=amp,stddev=fwhm_G)
    #lya_model = Voigt1D(x_0=1215.67,amplitude_L=amp,fwhm_G=fwhm_G,fwhm_L=fwhm_L)

    #Combine Amplified Lya with pumped line and scaled H2 background lines
    lya_data = lya_model(lam) + (tru_fit_model(lam)*amp*line_ratio)+(units2*other_I)
    broad_lya_units = convolve(lya_data,gaussian_kernel,mode='same')
    
    
    if (plots==True):
        plt.figure()
        plt.plot(lam,lya_data)
        #plt.xlim(1215,1217)

        plt.figure()
        plt.plot(lam,broad_lya_units)
        plt.vlines(x=1215.67,ymin=0,ymax=max(broad_lya_units))
        #plt.xlim(1212,1220)
        plt.grid();
    

    return np.array(lam),np.array(lya_data),np.array(broad_lya_units)


def lamp_func(x,amp,fwhm_G,line_ratio,other_I):
    data = np.loadtxt('h2ft01b01n18.dat')
    lam = data[:,0]
    units = data[:,1]
    length = np.where(np.isclose(lam,1215.7,atol=7))[0]
    lam = lam[length]
    #units=units[length]
    units=units[length]
    
    #Artifically boost blue flux to match lamp output
    units2=np.concatenate([units[0:int(len(lam)/2)],units[int(len(lam)/2):]])
    
    left_edge = np.where(np.isclose(lam,1214.2,atol=0.7))[0]
    
    units2[left_edge]*=10
    
    #Data to create pumped 1216 line 
    new_lam = data[:,0]
    new_units = data[:,1]
    length = np.where(np.isclose(new_lam,1216.07,atol=0.1))[0]
    new_lam = new_lam[length]
    new_units=new_units[length]*5e4
    
    l1 = Gaussian1D(mean=1216.07,amplitude=1,stddev=0.01)
    fitter =TRFLSQFitter()
    
    gaussian_kernel = make_kernel(1, lam)
    tru_fit_model = fitter(l1,new_lam,new_units)
    
    
    lya_model = Gaussian1D(mean=1215.67,amplitude=amp,stddev=fwhm_G)
    #lya_model = Voigt1D(x_0=1215.67,amplitude_L=amp,fwhm_G=fwhm_G,fwhm_L=fwhm_L)
    
    #Combine Amplified Lya with pumped line and scaled H2 background lines
    lya_data = lya_model(lam) + (tru_fit_model(lam)*amp*line_ratio)+(units2*other_I)
    broad_lya_units = convolve(lya_data,gaussian_kernel,mode='same')
    interp_vals = np.interp(x,lam,broad_lya_units)
    
    return interp_vals

def lamp_fit_func(x,y,amp,fwhm_G,line_ratio,other_I):
    
    p0=[amp,fwhm_G,line_ratio,other_I]
    
    #bounds = lower,upper
    bounds=([10,1e-3,0,0],[np.inf,np.inf,1,np.inf])
    popt,pconv = curve_fit(lamp_func,x,y,p0=p0,bounds=bounds)
    print(popt)
    vals_array = syn_lamp_func(*popt)
    plt.step(x,y,where='mid')
    return vals_array


def model_cell(Nums,lam,lya_data):
    gaussian_kernel = make_kernel(1, lam)
    length2=np.where(np.isclose(lam,1215.67,atol=0.32))[0]
    
    y1s=[]
    y2s=[]
    t1s=[]
    Is=[]
    for N in Nums:
        trans,negative_tau,lama= h1abs_lya((N),2.5)

        abs_units = interp_func(lama,trans,lam)
        trans_units = abs_units*lya_data
        I_cell = simpson(trans_units[length2],lam[length2])
        broad_trans_units = convolve(trans_units,gaussian_kernel,mode='same')
        y1s.append(broad_trans_units)
        y2s.append(trans_units)
        t1s.append(abs_units)
        Is.append(I_cell)
    
    I0 = simpson(lya_data[length2],lam[length2])
    #new = y2s[0][length2]
    #new_l = lam[length2]
    #mask= new!=0
    
    return np.array(y1s),np.array(y2s),np.array(t1s),np.array(Is)/I0


    

def lidos_data(directory_date,date,id_lamp,id_cell,window=5):
    #directory_date = '02_06_2025'
    sub_directory = 'Processed'
    #date = '02_06_2025'
    base_dir = os.path.join(directory_date)
    sub_dir = os.path.join(sub_directory)


    #id_lamp = 'lamp_3T_6V_3m'
    #id_filament_dark = 'darkf_3.5A_2T'
    #id_cell = 'cell_3T_6V_3m'
    
    lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
    #filament_dark_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_filament_dark}_processed.csv')
    cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')
    
    name = lamp_filename[23:49]
    
    
    analyzer = SpectrumAnalyzer(lamp_filename, cell_filename)
    trans_est, trans_real, trans_err = analyzer.analyze('calibration_plot.png',name)
    
    wav_cell_1 = analyzer.lamda_cell
    wav_l_1 = analyzer.lamda_lamp
    lamp_y_1 = analyzer.y_e_lamp
    cell_y_1 = analyzer.y_e_cell
    
    if window==0:
        return wav_l_1,wav_cell_1,lamp_y_1,cell_y_1
    
    length = np.where(np.isclose(wav_l_1,1215.7,atol=window))[0]
    
    
    mask=np.array([wav_l_1>1050])
    lam=wav_l_1[mask.flatten()]
    trans = analyzer.y_e_cell_bin/analyzer.y_e_lamp_bin
    trans=trans[mask.flatten()]
    
    return wav_l_1[length],wav_cell_1[length],lamp_y_1[length],cell_y_1[length],lam,trans



def total_model_lya(wav_l,data_l,pop,n_tot):
    
    lam,lya_data,broad_lya_units = lamp_fit_func(wav_l,data_l,amp=250,fwhm_G=0.08,fwhm_L=0.02,line_ratio=0.5,other_I=7e3)
    wavelength_range, cell_trans = pop_tau_templates(pop, n_tot)
    
    #interpolate entire H2 template onto lam (window around lya)
    lya_cell_trans = np.interp(lam, wavelength_range, cell_trans)
    model_cell_out = lya_cell_trans * lya_data
    
    gaussian_kernel = make_kernel(1, lam)
    broad_cell_out = convolve(model_cell_out,gaussian_kernel,mode='same')
    
    return lam,model_cell_out,broad_cell_out
    
    
    
    