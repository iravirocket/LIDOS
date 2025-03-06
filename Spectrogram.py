# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:36:37 2025

@author: jackf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.fitting import LinearLSQFitter
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from scipy.integrate import simps
import os
from scipy.signal import find_peaks
from astropy.modeling.polynomial import Polynomial1D
from scipy.ndimage import uniform_filter1d, gaussian_filter1d,median_filter
import matplotlib.colors as mcolors
from astropy.visualization import ImageNormalize, ZScaleInterval



class SpectrumAnalyzer:
    #All data is refered to as lamp, but can at times be cell data.
    #Just means only one file being used at a time, unlike EW_Ravi_v3.py
    
    def __init__(self, lamp_file_data):

        self.df_lamp = lamp_file_data
        self.integration_time = np.max(self.df_lamp['Timestamp']) # seconds
        self.res=2
        self.full_x_range = np.arange(0, (2048*self.res))

        
    def preprocess_data(self,y_lim_min,y_lim_max):
        
        #create x-pixel grid
        lower,upper = 0,2048*self.res
        self.df_lamp['xr'] = self.df_lamp['xr']*self.res
        
        #Mask out any extraneous values
        mask = ((self.df_lamp['xr'])>lower) & ((self.df_lamp['xr'])<upper)# & (self.df_lamp['den']<7000)
        self.df_lamp = self.df_lamp[mask]
        
        #y limits defined outside function. For current pinhole (3-6-2025) should still be 1000-1500
        
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_lim_min) & (self.df_lamp['y'] <= y_lim_max) ]
        
        #Histogram counts into spectrum
        self.y_lamp,self.xr_lamp = np.histogram(self.filtered_lamp['xr'], bins=self.full_x_range)
        
        self.xr_lamp = (self.xr_lamp[:-1])
      
            
    def calibrate_wavelength(self,lamp_wav_fit=None):
        
        if (lamp_wav_fit==None):
            def px_2_wav(data,y_data,guesses):
                
                
                def find_peak(data, expected_position, y_data, atol=20*self.res):
                    peaks, _ = find_peaks(y_data)
                    # Find the peaks within the specified tolerance
                    valid_peaks = peaks[np.abs(peaks - expected_position) <= atol]
                    
                    if len(valid_peaks) == 0:
                        raise ValueError(f"No peak found within {atol} pixels of position {expected_position}")
                    
                    peak_values = y_data[valid_peaks]  
                    highest_peak_index = np.argmax(peak_values)  
                    highest_peak_position = valid_peaks[highest_peak_index] 
                    
                    # Return the peak closest to the expected position
                    #closest_peak = valid_peaks[np.abs(valid_peaks - expected_position).argmin()]
                    return highest_peak_position
                
                self.guessed_pixels_index =[]
                for guess in guesses:
                    
                    px1 = find_peak(data, guess, y_data)
                    self.guessed_pixels_index.append(px1)
                
                
                #print(self.guessed_pixels_index)
                def gaussian_weighted_average(data, index, y_data, window_size=3*self.res):
                    region = y_data[index - window_size:index + window_size]
                    smoothed = gaussian_filter1d(region,sigma=2)  # Adjust sigma as needed
                    return np.average(data[index - window_size:index + window_size], weights=smoothed)
                    
                self.improved_xval_guesses = [gaussian_weighted_average(data, g, y_data) 
                                      for g in self.guessed_pixels_index]
                
                
                #self.guessed_Angstroms =  [1492.6, 1302, 1334, 1200,1215.67]
                
                self.guessed_Angstroms =  [1494.6,1302.2,1230,1215.67]
                
 
                #print(self.improved_xval_guesses)

                linfitter = LinearLSQFitter()
                poly_order =  2 # You can try different orders
                wlmodel = Polynomial1D(degree=poly_order)
                num_guesses = len(guesses)
                self.guessed_Angstroms=self.guessed_Angstroms[:num_guesses]
                linfit_wlmodel = linfitter(model=wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
                return linfit_wlmodel
            
            #self.linfit_wlmodel = Linear1D(slope=0.3522/2, intercept=917.4538)
            self.lamp_linfit_wlmodel = px_2_wav(self.xr_lamp,self.y_lamp,[1620*self.res,1055*self.res,860*self.res])
            self.lamda_lamp = self.lamp_linfit_wlmodel(self.xr_lamp)
            
            #Counts for every time bin being analyzed. Set outside class down below
            self.y_e_lamp = self.y_lamp
            #print(self.lamp_linfit_wlmodel)
        
        #Use wavelength fit from file as a whole, then apply to individual time bins to preserve
        #resolution of spectrogram
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
        #for x in self.guessed_Angstroms:
            #.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        #ax2.plot(self.guessed_Angstroms, [2600]*4, 'x', label='Identified Peaks')
        ax2.axvline(x=1215.67, label='LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Angstroms')
        ax2.set_ylabel('Counts * s-1 *AA-1')
        ax2.legend()

        ax3.plot(self.lamda_lamp, self.y_e_lamp, 'o-', label='Source')
        #ax3.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        #for x in self.guessed_Angstroms:
            #ax3.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Angstroms')
        ax3.set_ylabel('Counts * s-1 *AA-1')
        #ax3.set_xlim(1198,1205)
        ax3.set_xlim(1490,1500)
        ax3.legend()

        ax5.plot(self.lamda_lamp, self.y_e_lamp, 'o-', label='Source')
        #ax5.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        
        #for x in self.guessed_Angstroms:
            #ax5.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax5.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax5.set_xlabel('Angstroms')
        ax5.set_ylabel('Counts * s-1 *AA-1')
        ax5.set_xlim(1190,1221)
        ax5.set_ylim(0,3*self.integration_time)
        ax5.legend()
        
        fig.savefig(save_path)
        

    def find_EW(self):
        

        lamp_window_start = np.where(np.isclose(self.lamda_lamp, 1200, atol=1))[0][0]
        lamp_window_end = np.where(np.isclose(self.lamda_lamp, 1230, atol=1))[0][-1]

        
        data_l = self.y_e_lamp[lamp_window_start:lamp_window_end]
        wav_l = self.lamda_lamp[lamp_window_start:lamp_window_end]


        
        
        '''
        spec_l = Spectrum1D(spectral_axis=wav_l*u.AA, flux=norm_data_lamp*u.Unit('erg cm-2 s-1 AA-1'))
        EW_l = equivalent_width(spec_l, continuum=0.001*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_l)*u.AA, max(wav_l)*u.AA))
        '''


        al = (0.001 - data_l) / 0.001
        I0 = simps(data_l, wav_l)
        EW_2L = simps(al, wav_l)
        
        
                
        
        return wav_l, data_l
    

    def analyze(self, save_path,name,y_lim_min, y_lim_max,lamp_wav_fit=None):
        self.preprocess_data(y_lim_min,y_lim_max)
        self.calibrate_wavelength(lamp_wav_fit)
        
        #do not want to plot data everytime
        #self.plot_data(save_path,name)

        #Do not need TE measurements
        wav_l, data_l = self.find_EW()

        return wav_l, data_l
     
#Plot font size that steve likes
plt.rcParams['font.size'] = 18

def waterfall_plot_one(directory_date,date,id_lamp,time_res,t_start=90,make_plot=True,save_array=False):
    
    # Usage
    sub_directory = 'Processed'
    base_dir = os.path.join(directory_date)
    sub_dir = os.path.join(sub_directory)
    
    lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
    
    #read in file
    df = pd.read_csv(lamp_filename)
    integration_time = np.max(df['Timestamp']) # seconds
    
    
    analyzer = SpectrumAnalyzer(df)
    wav_l, data_l = analyzer.analyze('calibration_plot.png',lamp_filename,1000,1500)
    
    #Save wavelength fit and wavelength array for later use
    lamp_wav_fit0 = analyzer.lamp_linfit_wlmodel
    wavelength = analyzer.lamda_lamp
    
    
    #Cut off beginning of lamp if data looks wonky with t_start
    # Create time bins using np.linspace, with time_res as the seconds per bin
    num_bins = int((integration_time-1-t_start)/time_res)
    time_bins = np.linspace(t_start,(integration_time-1),num_bins+1)  # +1 to create boundaries
    
    
    # Assign each row to a bin using pd.cut
    #grouped data result is entire df sliced into time bins
    #READ IN AGAIN CUZ CODE ABOVE CHANGES X POSITION NOT COPY... MB
    df = pd.read_csv(lamp_filename)
    df['time_bin'] = pd.cut(df['Timestamp'], bins=time_bins, labels=False)
    grouped_data = [df[df['time_bin'] == i] for i in range((num_bins))]
    
    waves=[]
    datas=[]
    mean_x=[]
    mean_y=[]
    means_diff=[]
    lamp=[]
    
    for i,group in enumerate(grouped_data):
        analyzer = SpectrumAnalyzer(group)
        wav_l, data_l = analyzer.analyze('calibration_plot.png',lamp_filename,1000,1500,lamp_wav_fit0)
        
        lamp.append(analyzer.y_e_lamp)
        
        #Used to diagnose MCP walk over time
        mean_y.append(np.mean(analyzer.df_lamp['y']))
        mean_x.append(np.mean(analyzer.df_lamp['xr']))
        means_diff.append(mean_x[0]-mean_x[i])
        
        waves.append(wav_l)
        datas.append(data_l)
    
    #Spectra per time bin, summed together then divided by integration time.
    #Likely better way to do this but oh well  
    lamp=np.array(lamp)
    lamp2 = np.sum(lamp,axis=0)
    lamp_time = len(lamp)*time_res
    lamp3 = (lamp2/lamp_time)

    
    cmap = plt.cm.jet
    
    # Create a new colormap with 0 mapped to white
    new_cmap = cmap(np.linspace(0, 1, 256))  # Get the color array
    new_cmap[0] = [0.85, 0.85, 0.9, 1]  # Set the lowest value to white (RGBA)
    
    # Convert back to a colormap
    #cmap = mcolors.ListedColormap(new_cmap)
    num_datasets = len(datas)
    jet_map = plt.cm.jet
    colors1 = [jet_map(i / num_datasets) for i in range(num_datasets)]

    #Creation of spectrogram
    time = time_bins

    #Creates wavelength bins for whole spectrogram. If more than one wavelength fit is used then
    #unique_x values skyrockets and bins become too narrow
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
        #Plots lya profile at time intervals to check lamp stability
        plt.figure()
        plt.grid()

        mid = int(num_datasets/2)
        plt.step(waves[0],(datas[0]),'-', linewidth=1, label=f'Time index {0}',color=colors1[0])
        plt.step(waves[mid],datas[mid],'-', linewidth=1, label=f'Time index {mid}',color=colors1[mid])
        plt.step(waves[-1],datas[-1],'-', linewidth=1, label=f'Time index {-1}',color=colors1[-1])
        plt.vlines(x=1215.67,ymin=0,ymax=20,ls='--')   
        
        plt.xlabel('Wavelength(Angstroms)')
        plt.ylabel('Counts')
        plt.title('Spectra as a function of time (Blue=0s, Red=End)')    
        
        
        #Plots MCP walk means. If noticable trend in any direction then mcp is walking
        plt.figure()
        plt.grid();
        #plt.scatter(analyzer.df_lamp['xr'],analyzer.df_lamp['y'],s=0.01)
        #plt.hlines(y=600, xmin=0, xmax=2049, color='red',ls='--')
        #plt.hlines(y=850, xmin=0, xmax=2049, color='red',ls='--')
        for i,x in enumerate(mean_x):
            plt.scatter(x,mean_y[i],marker='*',label=f'Time{(integration_time-1-t_start)*i/num_bins}s',color=colors1[i])
    
    
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
    
    return lamp_wav_fit0, datas, waves, lamp3,wavelength


def waterfall_plot_two(directory_date,date,id_lamp,time_res,id_cell,t_start=90):
    
    lamp_wav_fit0,datas,waves,lamp3,wavelength = waterfall_plot_one(directory_date, date, id_lamp, time_res,t_start,make_plot=False,save_array=True)
    
    sub_directory = 'Processed'
    base_dir = os.path.join(directory_date)
    sub_dir = os.path.join(sub_directory)

    
    cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')
    
    df = pd.read_csv(cell_filename)
    integration_time = np.max(df['Timestamp']) # seconds

    # Create time bins using np.linspace
    num_bins = int((integration_time-1)/time_res)
    #time_bins = np.arange(t_start,(integration_time-1)+time_res,time_res)  # +1 to create boundaries
    time_bins = np.linspace(0,(integration_time-1),num_bins+1)

    # Assign each row to a bin using pd.cut
    #Do not need to read in df again this time
    df['time_bin'] = pd.cut(df['Timestamp'], bins=time_bins, labels=False)
    grouped_data = [df[df['time_bin'] == i] for i in range(1,num_bins)] #Remove first datapoint for cell
    
    waves2=[]
    datas2=[]
    mean_x=[]
    mean_y=[]
    means_diff=[]
    cell=[]
    
    for i,group in enumerate(grouped_data):
        analyzer = SpectrumAnalyzer(group)
        wav_l, data_l = analyzer.analyze('calibration_plot.png',cell_filename,1000,1500,lamp_wav_fit0)
        
        mean_y.append(np.mean(analyzer.df_lamp['y']))
        mean_x.append(np.mean(analyzer.df_lamp['xr']))
        means_diff.append(mean_x[0]-mean_x[i])
        
        cell.append(analyzer.y_e_lamp)
        
        waves2.append(wav_l)
        datas2.append(data_l)
    
    num_datasets = len(datas)
    jet_map = plt.cm.jet
    colors1 = [jet_map(i / num_datasets) for i in range(num_datasets)]
    mid = int(num_datasets/2)
#==========================================================================================================        
    #FOR USE IN BINNED TRANSMISSION AS FUNCTION OF TIME ONLY
    #BINNED TIME BINS LOOK AT HOW ABSORPTION WITHIN THE CELL CHANGES OVER TIME (BINS)
    #Nomeclature about to get real confusing
    cell=np.array(cell)

    #decides how many time bins are summed together (num_row_sum*time_res = 'binned time bin' in seconds)
    num_row_sum = 7
    cell_bin_time=num_row_sum*time_res


    indices = np.arange(0, len(cell), num_row_sum)  # Start indices for summing
    #Data summed into 'binned time bins'
    cell2 = np.add.reduceat(cell, indices, axis=0)

    #Decides WAVELENGTH bin size (not time)
    bin_size=30*analyzer.res
    binned_cell2 = uniform_filter1d((cell2),size=bin_size)

    #lamp3 is binned at counts per time_res. Multiply by cell_bin_time to get lamp counts per cell time bin
    #Now that it matches integration times of binned_cell2, smooth wavlength by same bin_size
    binned_lamp3 = uniform_filter1d((lamp3 * cell_bin_time),size=bin_size) 
    disp = wavelength[1]-wavelength[0]
    
    #Remake colormap for new number of datasets
    num_datasets = len(binned_cell2)
    jet_map = plt.cm.jet
    colors = [jet_map(i / num_datasets) for i in range(num_datasets)]


    #Plot transmission for every 'binned time bin'
    plt.figure(figsize=(12,8))

    for i in range(len(binned_cell2)):
        
        #Creates time bin boundaries for legend [ti,tf]
        time_bin = np.array([i*cell_bin_time,(i+1)*cell_bin_time]) +time_res
        
        #because of np.add.reduceat, if cell_bin_time is not integer multiple of integration time of cell =175s
        #then last element will have wrong integration time scaling, must remove if tf > 180s-time_res
        
        if time_bin[1] <= integration_time:
            transmission = binned_cell2[i]/binned_lamp3
            plt.step(wavelength,transmission, label=f't= {time_bin[0]}-{time_bin[1]}s',color=colors[i],lw=1)
        
    plt.vlines(x=1215.67,ymin=0,ymax=1.5,color='green',ls='--')
    plt.hlines(y=1,xmin=min(wavelength),xmax=max(wavelength),color='red',ls='--')
    plt.title(fr'Transmission Data (Bins={bin_size*disp:.1f}$\AA$) {id_lamp}')
    plt.minorticks_on()
    plt.legend()
    plt.ylim(0,1.4)
    plt.xlim(1050,1640)
    plt.grid();
#=============================================================================================================         
    
    #Load in previously saved data 
    saved_data = np.load("waterfall_data.npz")
    prev_counts_grid = saved_data["counts_grid"]
    prev_time = saved_data["time"]
    
    #Add 180s to cell time bins to create sequential times
    #Repeat of code above... function would be nice
    time = time_bins+prev_time[-1]
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

    # Stack the counts grids along the time axis
    combined_counts_grid = np.vstack([prev_counts_grid, counts_grid])
    
    # Concatenate the time arrays
    boundary_time = prev_time[-1]
    combined_time = np.concatenate([prev_time,time])
    
    
    znorm1 = ImageNormalize(combined_counts_grid,interval=ZScaleInterval())
    cmap = plt.cm.jet
    #Plot combined spectrogram
    plt.figure(figsize=(13, 7))
    plt.imshow(combined_counts_grid, aspect='auto',
               extent=[unique_x.min(), unique_x.max(), combined_time.max(),combined_time.min()],
               cmap=cmap)#,norm=znorm1)

    plt.colorbar(label='Counts')
    plt.xlabel('Wavelength (Å)',fontsize=24)
    plt.ylabel(f'Time (s) ({(prev_time[-1]-prev_time[-2]):.1f} second bins)',fontsize=24)
    plt.title(r'Ly$\alpha$ Profile vs. Time  (2 Torr + 21 Watts)',fontsize=24)
    #plt.vlines(1215.67, t_start, max(combined_time), color='black',lw=3,ls=':')

    plt.axhline(y=prev_time[-1], color='red', linestyle='--', linewidth=2, label="Filament Activated")
    plt.legend(framealpha=1);
    plt.minorticks_on();
    plt.tick_params(axis="both", which="major", length=8, width=2.5)  # Major ticks
    plt.tick_params(axis="both", which="minor", length=5, width=2)  # Minor ticks

    #Cut off extra 0s added to arrays
    plt.ylim(combined_time.max()-time_res,t_start)


    #Slice of spectrogram at different time intervals, showing attenuation initally and over time
    plt.figure(figsize=(12,7))
    plt.step(waves[0],sum(datas[1:4]),'-', linewidth=1, label=f'Start of Lamp',color=colors1[0])
    plt.step(waves2[0],sum(datas2[0:3]),'-', linewidth=1, label=f'Start of Cell',color=colors1[mid])
    plt.step(waves2[0],sum(datas2[-3:]),'-', linewidth=1, label=f'End of Cell',color=colors1[-1])
    plt.vlines(x=1215.67,ymin=0,ymax=max(sum(datas[0:3])),ls='--',color='black',label=r'Ly$\alpha$',lw=1)   
    #plt.xlim(1212,1219)
    plt.minorticks_on()
    plt.legend();
    #plt.xlim(1200,1220)
    plt.xlabel('Wavelength(Angstroms)')
    plt.ylabel('Counts')
    plt.title('Spectra as a function of time (Blue=0s, Red=End)')    
    