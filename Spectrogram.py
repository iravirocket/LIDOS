# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:36:37 2025

@author: jackf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from scipy.integrate import simps
import os
from scipy.signal import find_peaks
from astropy.modeling.polynomial import Polynomial1D
import glob
from scipy.ndimage import median_filter

class SpectrumAnalyzer:
    def __init__(self, lamp_file):
        #self.df_lamp = pd.read_csv(lamp_file)
        #self.df_darkf = pd.read_csv(dark_file)
        #self.df_cell = pd.read_csv(cell_file)
        #self.integration_time = np.max(self.df_lamp['Timestamp']) # seconds
        self.bandpass = 1650 - 900  # Angstroms
        #self.avg_dark = self.df_darkf['y'].mean()
        #self.guessed_Angstroms =  [1492.6, 1304, 1334, 1200]
        #self.guessed_Angstroms =  [1492.6, 1200]
        self.guessed_pixels_index = []
        self.full_x_range = np.arange(0, 2049)
        
        self.df_lamp = lamp_file
        

        
    def preprocess_data(self,y_lim_min,y_lim_max):
        #self.xr_lamp = (self.df_lamp['xr'])
        #self.y_lamp = self.df_lamp['y'] #- self.avg_dark
        #self.xr_darkf = (self.df_darkf['xr'])
        #self.y_darkf = self.df_darkf['y']
        #self.xr_cell = (self.df_cell['xr'])
        #self.y_cell = self.df_cell['y'] - self.avg_dar
        
        
        
        #self.df_lamp['xr'] = np.round(self.df_lamp['xr']).astype(int)
        #self.df_lamp['y'] = np.round(self.df_lamp['y']).astype(int)
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_lim_min) & (self.df_lamp['y'] <= y_lim_max) ]
        #lamp_hist, xedges, yedges = np.histogram2d(self.filtered_lamp['xr'], self.filtered_lamp['y'], bins=(xbins, ybins))
        self.y_lamp, self.xr_lamp = np.histogram(self.filtered_lamp['xr'], bins=self.full_x_range)
        
        self.xr_lamp = (self.xr_lamp[:-1])
        #self.y_lamp = (self.y_lamp)/self.integration_time
        
        
        '''
        # Filter data
        y_cut_min, y_cut_max = y_lim_min, y_lim_max
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_cut_min) & (self.df_lamp['y'] <= y_cut_max)]
        #self.filtered_darkf = self.df_darkf[(self.df_darkf['y'] >= y_cut_min) & (self.df_darkf['y'] <= y_cut_max)]
        #self.filtered_cell = self.df_cell[(self.df_cell['y'] >= y_cut_min) & (self.df_cell['y'] <= y_cut_max)]
        
        #self.filtered_lamp['xr'] = np.round(self.filtered_lamp['xr'])
        #self.filtered_cell['xr'] = np.round(self.filtered_cell['xr'])
        
        # Round 'xr' to the nearest integer before grouping
        self.lamp_counts = self.filtered_lamp.assign(xr_rounded=self.filtered_lamp['xr'].round()) \
            .groupby('xr_rounded')['y'].count().reset_index()
        
        
            
        # Group and sum counts
        #self.lamp_counts = self.filtered_lamp.groupby('xr')['y'].count().reset_index()
        #self.darkf_counts = self.filtered_darkf.groupby('xr')['y'].sum().reset_index()
        #self.cell_counts = self.filtered_cell.groupby('xr')['y'].sum().reset_index()

        # Update values
        
        self.xr_lamp = (self.lamp_counts['xr_rounded'].values)
        self.y_lamp = (self.lamp_counts['y'].values)/self.integration_time
        
        #self.y_lamp = (self.lamp_counts['y'].values)/self.integration_time 
        #self.y_lamp = np.clip(self.y_lamp, a_min=0.001, a_max=None)
        
        #self.xr_darkf = (self.darkf_counts['xr'].values)
        #self.y_darkf = self.darkf_counts['y'].values
        #self.xr_cell = (self.cell_counts['xr'].values)
        
        #self.y_cell = self.cell_counts['y'].values - self.avg_dark
        #self.y_cell = np.clip(self.y_cell, a_min=0.001, a_max=None)
        '''
            
    def calibrate_wavelength(self,lamp_wav_fit=None):
        
        if (lamp_wav_fit==None):
            def px_2_wav(data,y_data,g1,g4,gl):
               px1 = np.where(np.isclose(data, g1, atol=2))[0][0]
               #px2 = np.where(np.isclose(self.xr_lamp, 1100, atol=5))[0][0]
               #px3 = np.where(np.isclose(self.xr_lamp, 1190, atol=5))[0][0]
               px4 = np.where(np.isclose(data, g4, atol=2))[0][0]
               pxl = np.where(np.isclose(data, gl, atol=2))[0][0]
               
               
               from scipy.signal import find_peaks
               def find_peak(data, expected_position, y_data, atol=25):
                   peaks, _ = find_peaks(y_data)
                   # Find the peaks within the specified tolerance
                   valid_peaks = peaks[np.abs(peaks - expected_position) <= atol]
                   
                   if len(valid_peaks) == 0:
                       raise ValueError(f"No peak found within {atol} pixels of position {expected_position}")
                   
                   # Return the peak closest to the expected position
                   closest_peak = valid_peaks[np.abs(valid_peaks - expected_position).argmin()]
                   return closest_peak
               
               px1 = find_peak(data, g1, y_data)
               px4 = find_peak(data, g4, y_data)
               pxl = find_peak(data, gl, y_data)
               self.guessed_pixels_index = [px1,pxl]
               #print(self.guessed_pixels_index)
               
               #self.guessed_Angstroms =  [1492.6, 1304, 1334, 1200,1215.67]
               
               self.guessed_Angstroms =  [1492.6,1215.67]
               
               #npixels = 30
               from scipy.ndimage import gaussian_filter1d
    
               def gaussian_weighted_average(data, index, y_data, window_size=10):
                   region = y_data[index - window_size:index + window_size]
                   smoothed = gaussian_filter1d(region,sigma=2)  # Adjust sigma as needed
                   return np.average(data[index - window_size:index + window_size], weights=smoothed)
                   
               self.improved_xval_guesses = [gaussian_weighted_average(data, g, y_data) 
                                     for g in self.guessed_pixels_index] 
               #print(self.improved_xval_guesses)
               '''
               self.improved_xval_guesses = [np.average(data[g-npixels:g+npixels], 
                                                   weights=y_data[g-npixels:g+npixels]) 
                                       for g in self.guessed_pixels_index]
               #print(self.improved_xval_guesses)
               '''
               linfitter = LinearLSQFitter()
               poly_order = 1  # You can try different orders
               wlmodel = Polynomial1D(degree=poly_order)
               linfit_wlmodel = linfitter(model=wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
               return linfit_wlmodel
            
            self.lamp_linfit_wlmodel = px_2_wav(self.xr_lamp,self.y_lamp,1636,800,835-3)
            #lamp_linfit_wlmodel = Linear1D(slope=0.347546, intercept=922.3011821391464)
            self.lamda_lamp = self.lamp_linfit_wlmodel(self.xr_lamp)
             #self.lamda_cell = self.linfit_wlmodel(self.xr_cell)
             #self.lamda_darkf = self.linfit_wlmodel(self.xr_darkf)
             
            self.y_e_lamp = self.y_lamp
            #print(self.lamp_linfit_wlmodel)
        
        else:
            self.lamda_lamp = lamp_wav_fit(self.xr_lamp)
            self.y_e_lamp = self.y_lamp
            #print(lamp_wav_fit)
        
    def plot_data(self, save_path,name):
        fig, (ax1, ax2, ax3, ax5) = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
        ax1.plot(self.xr_lamp, self.y_lamp, ls='-', markersize=3.5, label='Source')
        #ax1.plot(self.xr_cell, self.y_cell, ls='-', markersize=3.5, label='Cell')
        #for x in self.improved_xval_guesses:
            #ax1.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Counts * s-1')
        ax1.set_title(f'Plots for Lamp and Cell {name}')
        ax1.legend()

        ax2.plot(self.lamda_lamp, self.y_e_lamp, 'o-', markersize=.5, label='Source')
        #ax2.plot(self.lamda_cell, self.y_cell, 'o-', markersize=.5, label='Cell')
        for x in self.guessed_Angstroms:
            ax2.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        #ax2.plot(self.guessed_Angstroms, [2600]*4, 'x', label='Identified Peaks')
        ax2.axvline(x=1215.67, label='LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Angstroms')
        ax2.set_ylabel('Counts * s-1 *AA-1')
        ax2.legend()

        ax3.plot(self.lamda_lamp, self.y_e_lamp, 'o-', label='Source')
        #ax3.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        for x in self.guessed_Angstroms:
            ax3.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Angstroms')
        ax3.set_ylabel('Counts * s-1 *AA-1')
        #ax3.set_xlim(1198,1205)
        ax3.set_xlim(1490,1500)
        ax3.legend()
        '''
        ax4.plot(self.lamda_lamp, self.y_lamp, 'o-', label='Source')
        ax4.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        
        ax4.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Angstroms')
        ax4.set_ylabel('Counts * s-1 *AA-1')
        ax4.set_xlim(1213,1219.5)
        ax4.legend()
        '''
        ax5.plot(self.lamda_lamp, self.y_e_lamp, 'o-', label='Source')
        #ax5.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        
        for x in self.guessed_Angstroms:
            ax5.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax5.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax5.set_xlabel('Angstroms')
        ax5.set_ylabel('Counts * s-1 *AA-1')
        ax5.set_xlim(1190,1221)
        ax5.legend()
        
        fig.savefig(save_path)
        
        
        
        
    '''    
    def bin_data(self, lamda, y, new_spacing=0.366):
        start_point = lamda.min()
        end_point = lamda.max()
        num_bins = int(np.ceil((end_point - start_point) / new_spacing))
        bin_sums = np.zeros(num_bins)
        x_bin = []
        for i in range(num_bins):
            bin_start = start_point + i * new_spacing
            bin_end = bin_start + new_spacing
            points_in_bin = y[(lamda >= bin_start) & (lamda < bin_end)]
            x_bin.append(bin_start)
            bin_sums[i] = np.sum(points_in_bin)

        bin_sums = np.clip(bin_sums, a_min=0.01, a_max=None)
        return x_bin, bin_sums
    '''
    def find_EW(self):
        
        #print(np.isclose(self.lamda_cell, 1213.5, atol=1.2))
        lamp_window_start = np.where(np.isclose(self.lamda_lamp, 1190, atol=1))[0][0]
        lamp_window_end = np.where(np.isclose(self.lamda_lamp, 1240, atol=1))[0][-1]
        #cell_window_start = np.where(np.isclose(self.lamda_cell, 1213.5, atol=1.2))[0][0]
        #cell_window_end = np.where(np.isclose(self.lamda_cell, 1218.5, atol=1.2))[0][-1]
               
        
        data_l = self.y_e_lamp[lamp_window_start:lamp_window_end]
        wav_l = self.lamda_lamp[lamp_window_start:lamp_window_end]
        #print(data_l)
        #data_cell = self.y_cell[cell_window_start:cell_window_end]
        #wav_cell = self.lamda_cell[cell_window_start:cell_window_end]
        
        norm_data_lamp = data_l 
        #norm_data_cell = data_cell / max(data_l)
        
        
        #print(min(norm_data_cell))
        spec_l = Spectrum1D(spectral_axis=wav_l*u.AA, flux=norm_data_lamp*u.Unit('erg cm-2 s-1 AA-1'))
        EW_l = equivalent_width(spec_l, continuum=0.001*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_l)*u.AA, max(wav_l)*u.AA))
        
        #spec_cell = Spectrum1D(spectral_axis=wav_cell*u.AA, flux=norm_data_cell*u.Unit('erg cm-2 s-1 AA-1'))
        #EW_cell = equivalent_width(spec_cell, continuum=0.001*u.Unit('erg cm-2 s-1 AA-1'), 
                              #regions=SpectralRegion(min(wav_cell)*u.AA, max(wav_cell)*u.AA))

        al = (0.001 - data_l) / 0.001
        I0 = simps(data_l, wav_l)
        EW_2L = simps(al, wav_l)
        
        #acell = (0.001 - data_cell) / 0.001
        #I_cell = simps(data_cell, wav_cell)
        #EW_2cell = simps(acell, wav_cell)

        return -1*EW_l, -1*EW_2L, I0, wav_l, data_l
    

    def analyze(self, save_path,name,y_lim_min, y_lim_max,lamp_wav_fit=None):
        self.preprocess_data(y_lim_min,y_lim_max)
        self.calibrate_wavelength(lamp_wav_fit)
        #self.plot_data(save_path,name)



        EW_lamp1, EW_lamp2, I0_lamp, wav_l, data_l = self.find_EW()
    

        return wav_l, data_l
    

def waterfall_plot_one(directory_date,date,id_lamp,time_res,time_start=0,make_plot=True,save_array=False):
    
    # Usage
    #directory_date = '12_13_2024'
    sub_directory = 'Processed'
    #date = '12_13_2024'
    base_dir = os.path.join(directory_date)
    sub_dir = os.path.join(sub_directory)


    #id_lamp = 'lamp_2T_6V_3m_start'

    lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
    
    df = pd.read_csv(lamp_filename)

    integration_time = np.max(df['Timestamp']) # seconds

    analyzer = SpectrumAnalyzer(df)
    wav_l, data_l = analyzer.analyze('calibration_plot.png',lamp_filename,600,1000)
    lamp_wav_fit0 = analyzer.lamp_linfit_wlmodel

    # Create time bins using np.linspace
    #time_res = 10
    num_bins = int((integration_time-1)/time_res)
    time_bins = np.linspace(time_start, integration_time-1, num_bins+1)  # +1 to create boundaries
    # Assign each row to a bin using pd.cut
    df['time_bin'] = pd.cut(df['Timestamp'], bins=time_bins, labels=False)
    grouped_data = [df[df['time_bin'] == i] for i in range(num_bins)]

    waves=[]
    datas=[]
    mean_x=[]
    mean_y=[]
    means_diff=[]
    for i,group in enumerate(grouped_data):
        analyzer = SpectrumAnalyzer(group)
        wav_l, data_l = analyzer.analyze('calibration_plot.png',lamp_filename,600,1000,lamp_wav_fit=lamp_wav_fit0)
        #data_l = np.clip(data_l, a_min=0.01, a_max=None)
        mean_y.append(np.mean(analyzer.df_lamp['y']))
        mean_x.append(np.mean(analyzer.df_lamp['xr']))
        means_diff.append(mean_x[0]-mean_x[i])
        
        
        waves.append(wav_l)
        datas.append(data_l)
    
    
    cmap = plt.get_cmap('jet')  # Choose your colormap
    num_datasets = len(datas)
    colors = [cmap(i / num_datasets) for i in range(num_datasets)]


    
    from matplotlib.colors import LogNorm
    from astropy.visualization import ImageNormalize, ZScaleInterval

    time = time_bins
    x_values = np.array(waves,dtype=object)
    unique_x = np.unique(np.concatenate(x_values))
    counts_values=np.array(datas,dtype=object)
    # 2. Initialize a 2D array for counts with shape (len(time), len(unique_x))
    counts_grid = np.full((len(time[1:]), len(unique_x)),0)

    # 3. Populate the 2D counts grid
    for i, (x, counts) in enumerate(zip(x_values, counts_values)):
        # Find indices in unique_x where each x value in this time slice should go
        indices = np.searchsorted(unique_x, x)
        counts_grid[i, indices] = counts  # Place counts in the correct positions

   
    if (make_plot==True):
        plt.figure()
        plt.grid()
        
        mid = int(num_datasets/2)
        plt.step(waves[0],datas[0],'-', linewidth=1, label=f'Time index {0}',color=colors[0])
        plt.step(waves[mid],datas[mid],'-', linewidth=1, label=f'Time index {mid}',color=colors[mid])
        plt.step(waves[-1],datas[-1],'-', linewidth=1, label=f'Time index {-1}',color=colors[-1])
        plt.vlines(x=1215.67,ymin=0,ymax=20)   
        
        #plt.xlim(1200,1220)
        plt.xlabel('Wavelength(Angstroms)')
        plt.ylabel('Counts')
        plt.title('Spectra as a function of time (Blue=0s, Red=End)')    
            
    
        # 4. Plot the heatmap with LogNorm to handle the varying magnitudes
        #znorm1 = ImageNormalize(counts_grid,interval=ZScaleInterval())
        plt.figure(figsize=(10, 6))
        plt.imshow(counts_grid, aspect='auto', extent=[unique_x.min(), unique_x.max(), time.max(), time.min()],
                   cmap='jet')
        plt.colorbar(label='Counts')
        #plt.xlim(1480,1540)
        # Label the axes
        plt.xlabel('Wavelength(Angstroms)')
        plt.ylabel(f'Time[s] ({(integration_time-1)/num_bins:.1f} sec bins)')
        plt.title('Counts vs Wavelength and Time')
        plt.minorticks_on();
        #plt.gca().invert_yaxis()  # Reverse y-axis if needed
   
        
    if (save_array==True):
        np.savez("waterfall_data.npz", counts_grid=counts_grid, time=time)
    
    return lamp_wav_fit0, datas, waves


def waterfall_plot_two(directory_date,date,id_lamp,time_res,id_cell):
    
    lamp_wav_fit0,datas,waves = waterfall_plot_one(directory_date, date, id_lamp, time_res,make_plot=False,save_array=True)
    
    
    sub_directory = 'Processed'
    
    base_dir = os.path.join(directory_date)
    sub_dir = os.path.join(sub_directory)


    #lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
    #filament_dark_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_filament_dark}_processed.csv')
    cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')
    
    df = pd.read_csv(cell_filename)

    integration_time = np.max(df['Timestamp']) # seconds

    # Create time bins using np.linspace

    num_bins = int((integration_time-1)/time_res)
    time_bins = np.linspace(0, integration_time-1, num_bins+1)  # +1 to create boundaries
    # Assign each row to a bin using pd.cut
    df['time_bin'] = pd.cut(df['Timestamp'], bins=time_bins, labels=False)
    grouped_data = [df[df['time_bin'] == i] for i in range(num_bins)]
    
    waves2=[]
    datas2=[]
    mean_x=[]
    mean_y=[]
    means_diff=[]
    for i,group in enumerate(grouped_data):
        analyzer = SpectrumAnalyzer(group)
        wav_l, data_l = analyzer.analyze('calibration_plot.png',cell_filename,600,1000,lamp_wav_fit0)
        #data_l = np.clip(data_l, a_min=0.01, a_max=None)
        mean_y.append(np.mean(analyzer.df_lamp['y']))
        mean_x.append(np.mean(analyzer.df_lamp['xr']))
        means_diff.append(mean_x[0]-mean_x[i])
        
        waves2.append(wav_l)
        datas2.append(data_l)
        
    time = time_bins
    x_values = np.array(waves2,dtype=object)
    unique_x = np.unique(np.concatenate(x_values))
    counts_values=np.array(datas2,dtype=object)
    # 2. Initialize a 2D array for counts with shape (len(time), len(unique_x))
    counts_grid = np.full((len(time[1:]), len(unique_x)),0)
    
    # 3. Populate the 2D counts grid
    for i, (x, counts) in enumerate(zip(x_values, counts_values)):
        # Find indices in unique_x where each x value in this time slice should go
        indices = np.searchsorted(unique_x, x)
        counts_grid[i, indices] = counts  # Place counts in the correct positions
        
        
    saved_data = np.load("waterfall_data.npz")
    prev_counts_grid = saved_data["counts_grid"]
    prev_time = saved_data["time"]

    # Stack the counts grids along the time axis
    combined_counts_grid = np.vstack([prev_counts_grid, counts_grid])

    # Concatenate the time arrays
    boundary_time = prev_time[-1]
    combined_time = np.concatenate([prev_time, time[1:]+boundary_time])
    
    
    plt.figure(figsize=(10, 6))
    plt.imshow(combined_counts_grid, aspect='auto',
               extent=[unique_x.min(), unique_x.max(), combined_time.max(), combined_time.min()],
               cmap='jet')

    plt.colorbar(label='Counts')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel(f'Time (s) ({(integration_time-1)/num_bins:.1f} sec bins)')
    plt.title(r'Ly$\alpha$ Profile vs. Time')

    plt.axhline(y=boundary_time, color='white', linestyle='--', linewidth=2, label="Filament Turned On")
    plt.legend();


    cmap = plt.get_cmap('jet')  # Choose your colormap
    num_datasets = len(datas)
    colors = [cmap(i / num_datasets) for i in range(num_datasets)]
    
    plt.figure()
    plt.grid()
    diffs=[]
    #for i in range(0,len(datas),3):
    mid = int(num_datasets/2)
    plt.step(waves[0],sum(datas[0:3]),'-', linewidth=1, label=f'Start of Lamp',color=colors[0])
    plt.step(waves2[0],sum(datas2[0:3]),'-', linewidth=1, label=f'Start of Cell',color=colors[mid])
    plt.step(waves2[-1],sum(datas2[-3:]),'-', linewidth=1, label=f'End of Cell',color=colors[-1])
    plt.vlines(x=1215.67,ymin=0,ymax=max(sum(datas[0:3])),ls='--',color='black',label=r'Ly$\alpha$',lw=1)   
    
    plt.minorticks_on()
    plt.legend();
    #plt.xlim(1200,1220)
    plt.xlabel('Wavelength(Angstroms)')
    plt.ylabel('Counts')
    plt.title('Spectra as a function of time (Blue=0s, Red=End)')    
    