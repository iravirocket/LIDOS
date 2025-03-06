import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width,fwhm
from scipy.integrate import simps
import os
from astropy.modeling import models, fitting
from astropy.convolution import convolve_models
import glob
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks


class SpectrumAnalyzer:
    def __init__(self, lamp_file, cell_file):
        
        self.df_lamp = pd.read_csv(lamp_file)
        self.df_cell = pd.read_csv(cell_file)
        
        self.integration_time = np.max(self.df_lamp['Timestamp']) # seconds
        self.integration_time_cell = np.max(self.df_cell['Timestamp']) # seconds
        
        #select only last minute of data from cell
        self.time_window = self.integration_time_cell - 60
        self.time_window2 = self.integration_time_cell
        mask = (self.df_cell['Timestamp'] > self.time_window) & (self.df_cell['Timestamp']<self.time_window2)
        self.df_cell = self.df_cell[mask]
        
        #Clip beginning of lamp data in case it looks wonky
        self.t_start = 90
        self.df_lamp = self.df_lamp[self.df_lamp['Timestamp'] > self.t_start]
                
        #Resolution for x data points. res=1 coresponds to 2048 pixels, 2= 2*2048, ...
        self.res = 2
        
    def preprocess_data(self):
      
        self.df_lamp['xr'] = self.df_lamp['xr']*self.res
        self.df_cell['xr'] = self.df_cell['xr']*self.res
        lower,upper = 0,2048*self.res
        
        mask = ((self.df_lamp['xr'])>lower) & ((self.df_lamp['xr'])<upper)# & (self.df_lamp['den']<7000)
        self.df_lamp = self.df_lamp[mask]
        
        mask2 = ((self.df_cell['xr'])>lower) & ((self.df_cell['xr'])<upper)# & (self.df_cell['den']<7000)
        self.df_cell = self.df_cell[mask2]
        
        #Plots x,y positions of counts on detector
# =============================================================================
#         plt.figure()
#         plt.scatter(self.df_lamp['xr'],self.df_lamp['y'],s=0.01)
#         plt.scatter(self.df_cell['xr'],self.df_cell['y'],s=0.01)
#         plt.hlines(y=1000,xmin=0,xmax=2048*self.res,color='red',ls='--')
#         #plt.ylim(1000,1400)
#         #plt.xlim(1800*self.res,1900*self.res)
#         #plt.grid()
# =============================================================================
        #Data falls in this range for current pinhole (3-6-2025), rest is noise
        y_cut_min, y_cut_max = 1000,1500
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_cut_min) & (self.df_lamp['y'] <= y_cut_max)]
        self.filtered_cell = self.df_cell[(self.df_cell['y'] >= y_cut_min) & (self.df_cell['y'] <= y_cut_max)]
        
        #Histogram the counts for spectrum (test?)
        bins = np.linspace(lower,upper,upper-lower)
        lamp_x,_ = np.histogram(self.filtered_lamp['xr'],bins=bins)
        cell_x,_ = np.histogram(self.filtered_cell['xr'],bins=bins)
        
        lamp_den_hist, lamp_den_bins = np.histogram(self.filtered_lamp['den'],bins=72)
        cell_den_hist, cell_den_bins = np.histogram(self.filtered_cell['den'],bins=72)
        
        #Plots pulse height dist.
# =============================================================================      
#         
#         plt.figure()
#         
#         plt.bar(lamp_den_bins[:-1], lamp_den_hist, width=np.diff(lamp_den_bins)[0], align='edge')
#         plt.bar(cell_den_bins[:-1], cell_den_hist, width=np.diff(cell_den_bins)[0], align='edge',alpha=0.75)
#         plt.xlabel('(Y1+Y2)')
#         plt.ylabel('Total (y1+y2) per bin')
#         time_int = self.df_lamp['Timestamp'].max()
#         plt.title(f'Pulse height for total time interval {time_int:.2f}s')
# =============================================================================
        
        #Same histogram but saves data
        self.full_x_range = np.arange(0, (2048*self.res))
        
        self.xr_lamp, self.y_lamp = self.full_x_range, np.histogram(self.filtered_lamp['xr'], bins=self.full_x_range)[0]
        self.xr_lamp = (self.xr_lamp[1:])

    
        self.xr_cell, self.y_cell = self.full_x_range, np.histogram(self.filtered_cell['xr'], bins=self.full_x_range)[0]
        self.xr_cell = (self.xr_cell[1:])
        
       
        return self.xr_lamp,self.xr_cell
    
    #Wavelength calibration from pixel space to wavelength space
    def calibrate_wavelength(self):
        
        def px_2_wav(data,y_data,guesses):
            #Requires x_data (pixel space), counts, and [guesses] array for each pixel position
            #guesses may contain up to number of lines in guessed_Angstroms
            
            
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
            
            #Find x-peaks near each guess
            self.guessed_pixels_index =[]
            for guess in guesses:
                
                px1 = find_peak(data, guess, y_data)
                self.guessed_pixels_index.append(px1)
            
            #print(self.guessed_pixels_index)
            
            #Takes weighted average to find true peak in case crowded line
            def gaussian_weighted_average(data, index, y_data, window_size=3*self.res):
                region = y_data[index - window_size:index + window_size]
                smoothed = gaussian_filter1d(region,sigma=2)  # Adjust sigma as needed
                return np.average(data[index - window_size:index + window_size], weights=smoothed)
            
            #Use pixel guess after find_peak
            self.improved_xval_guesses = [gaussian_weighted_average(data, g, y_data) 
                                  for g in self.guessed_pixels_index]
            
            
            #self.guessed_Angstroms =  [1492.6, 1302, 1334, 1200,1215.67]
            
            #Known lines in the data
            self.guessed_Angstroms =  [1494.6,1302.2,1230,1215.67]
   
            #print(self.improved_xval_guesses)
            
            #Create dispersion fit. Linear fit sometimes breaks so parabolic fit better
            #(Dispersion I think is non-linear near edges of detector where air lines are)
            linfitter = LinearLSQFitter()
            poly_order =  2 # You can try different orders
            wlmodel = Polynomial1D(degree=poly_order)
            
            num_guesses = len(guesses)
            #Only use number of lines guessed
            self.guessed_Angstroms=self.guessed_Angstroms[:num_guesses]
            
            #Obtain wavelength calibration
            linfit_wlmodel = linfitter(model=wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
            return linfit_wlmodel
        
        #Apply wavelength calibration with x-pixel guesses
        
        #self.linfit_wlmodel = Linear1D(slope=0.3522/2, intercept=917.4538)
        lamp_linfit_wlmodel = px_2_wav(self.xr_lamp,self.y_lamp,[1620*self.res,1055*self.res,860*self.res])
        self.lamda_lamp = lamp_linfit_wlmodel(self.xr_lamp)
        
        #Apply lamp fit to cell data (to retain any real asymettry in data)(Assumes no MCP walk)
        
        #cell_linfit_wlmodel = px_2_wav(self.xr_cell,self.y_cell,[1620*self.res,1055*self.res,860*self.res])
        #self.lamda_cell = cell_linfit_wlmodel(self.xr_cell)
        self.lamda_cell = lamp_linfit_wlmodel(self.xr_cell)
       # print(lamp_linfit_wlmodel)
        
        #Convert from counts to counts per sec
        self.y_e_lamp = self.y_lamp/(self.integration_time-self.t_start)#-98)
        self.y_e_cell = self.y_cell/(self.time_window2 -self.time_window)
        
        #plots residuals for wavelength calibration
# =============================================================================
#         plt.figure()
#         plt.plot(self.improved_xval_guesses,self.guessed_Angstroms[:],'o')
#         plt.plot(np.linspace(0,2048*self.res),lamp_linfit_wlmodel(np.linspace(0,2048*self.res)))
#         plt.show()
#         plt.figure()
#         plt.scatter(self.guessed_Angstroms,lamp_linfit_wlmodel(self.improved_xval_guesses)-self.guessed_Angstroms)
# =============================================================================

        
    def plot_data(self, save_path,name):
        
        fig, (ax2, ax1, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        ax1.plot(self.lamda_lamp, self.y_e_lamp, '-', label='Source')
        ax1.plot(self.lamda_cell, self.y_e_cell, '-', label='Cell')
        ax1.set_xlim(1480,1630)
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Counts * s-1')
        ax2.set_title(f'Plots for Lamp and Cell {name}')
        ax1.legend()
        ax1.set_ylim(0,1)
        for x in self.guessed_Angstroms:
            ax1.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        


        ax2.plot(self.lamda_lamp, self.y_e_lamp, 'o-', markersize=.5, label='Source')
        ax2.plot(self.lamda_cell, self.y_e_cell, 'o-', markersize=.5, label='Cell')
        for x in self.guessed_Angstroms:
            ax2.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        
        ax2.axvline(x=1215.67, label='LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Angstroms')
        ax2.set_ylabel('Counts * s-1 *AA-1')
        ax2.minorticks_on()
        ax2.legend()
        ax2.set_xlim(1000,)


        ax3.step(self.lamda_lamp, self.y_e_lamp, label='Source')
        ax3.step(self.lamda_cell, self.y_e_cell, label='Cell')
        for x in self.guessed_Angstroms:
            ax3.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Angstroms')
        ax3.set_ylabel('Counts * s-1 *AA-1')
        #ax3.set_xlim(1198,1205)
        ax3.set_xlim(1050,1200)
        ax3.set_ylim(0,0.6)
        ax3.minorticks_on()
        ax3.legend()
        
        
        ax4.step(self.lamda_lamp, self.y_e_lamp, label='Source')
        ax4.step(self.lamda_cell, self.y_e_cell, label='Cell')
        for x in self.guessed_Angstroms:
            ax4.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Angstroms')
        ax4.axvline(x=1302, label='OI', color='red', linestyle='--', linewidth=0.5)
        ax4.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Angstroms')
        ax4.set_ylabel('Counts * s-1 *AA-1')
        ax4.axvline(x=1107, label='Lyman band min', color='green', linestyle='-', linewidth=1)
        ax4.axvline(x=1008, label='Werner band min', color='green', linestyle='-', linewidth=1)
        ax4.set_xlim(1190,1320)
        ax4.set_ylim(0,1.2)
        ax4.minorticks_on()
        
        fig.savefig(save_path)
        
        
        #Plots binned transmission of entire bandpass
        bin_size=self.res*30
        self.y_e_cell_bin = uniform_filter1d((self.y_e_cell),size=bin_size)
        self.y_e_lamp_bin = uniform_filter1d((self.y_e_lamp),size=bin_size)
        
        disp = self.lamda_lamp[1]-self.lamda_lamp[0]
        
        plt.figure(figsize=(15,10))
        plt.step(self.lamda_lamp,self.y_e_cell_bin/self.y_e_lamp_bin)
        plt.vlines(x=1215.67,ymin=0,ymax=1.5,color='black',ls='--',label=r'Ly$\alpha$')
        plt.hlines(y=1,xmin=min(self.lamda_lamp),xmax=max(self.lamda_lamp),color='black',ls='--')
        plt.title(fr'Transmission Data (Bins={bin_size*disp:.1f}$\AA$) {name[12:22]}')
        plt.minorticks_on()
        plt.legend()
        plt.ylim(0,1.5)
        plt.xlim(1050,1640)
        plt.grid();
        #plt.xlim(1050,1350)
        
        #Plots H2 transition wavelengths from v' 1-7 and J'=0,1,2 overtop binned transmission
        
        H2_data = glob.glob('highjsh*.dat')
        
        lyman=[]
        werner=[]
        for data in (H2_data):
            df = np.loadtxt(data)
            
            werner_data = df[1088:,:]
            lyman_data = df[:1088,:]
            
            values = [0,1]
            werner.append(werner_data[np.isin(werner_data[:, 3], values)])
            lyman.append(lyman_data[np.isin(lyman_data[:, 3], values)])
            
            
        #Very dense so oscillator strength f is used as alpha value, highlighting strong transitions
        self.lyman = np.array(lyman)
        f = self.lyman[:,:,5]
        
        self.werner = np.array(werner)
        f_w = self.werner[:,:,5]
        
        # Normalize to [0, 1] range
        normalized_alpha_l = (f - np.min(f)) / (np.max(f) - np.min(f))
        normalized_alpha_w = (f_w - np.min(f_w)) / (np.max(f_w) - np.min(f_w))

        for i,line in enumerate(self.lyman[:,:,4]):
            plt.vlines(x=line,ymin=0,ymax=1.5,ls='--',color='red',alpha=normalized_alpha_l[i])
            
        for i,line in enumerate(self.werner[:,:,4]):
            plt.vlines(x=line,ymin=0,ymax=1.5,ls='-',color='green',alpha=normalized_alpha_w[i])

    def find_EW(self):
        
        #Define finite window in wavelength space in which to run EW spec util code
        
        #selecting index for which to slice the data based on wavelength ex. ~1213-1218
        lamp_window_start = np.where(np.isclose(self.lamda_lamp, 1210, atol=1))[0][0]
        lamp_window_end = np.where(np.isclose(self.lamda_lamp, 1221, atol=1))[0][-1]
        cell_window_start = np.where(np.isclose(self.lamda_cell, 1210, atol=1))[0][0]
        cell_window_end = np.where(np.isclose(self.lamda_cell, 1221, atol=1))[0][-1]
        
        data_l = self.y_e_lamp[lamp_window_start:lamp_window_end]
        wav_l = self.lamda_lamp[lamp_window_start:lamp_window_end]
        
        
        data_cell = self.y_e_cell[cell_window_start:cell_window_end]
        wav_cell = self.lamda_cell[cell_window_start:cell_window_end]

        
        #Intensity calculations for lamp and cell
        
        I0 = simps(data_l, wav_l)
        I_cell = simps(data_cell, wav_cell)
        
        plt.figure(figsize=(12,7))
        plt.step(wav_l, data_l,label='Lamp',color='red')
        plt.step(wav_cell, data_cell,label="Cell",color='black')
        plt.vlines(x=1215.67,ymax=max(data_l),ymin=0)
        plt.vlines(x=1217.65,ymax=max(data_l),ymin=0)
        plt.vlines(x=1217.30,ymax=max(data_l),ymin=0)
        #plt.errorbar(wav_l, data_l,yerr=np.sqrt(data_l),fmt='ro-',label='Lamp',ms=3,capsize=1)
        #plt.errorbar(wav_cell, data_cell,yerr=np.sqrt(data_cell),fmt='ko-',label='Cell',ms=3,capsize=1)
        
        #print(c1_model)
        #print(c1_model[0])
        #plt.plot(wav_space,c1_model[0](wav_space)*max(data_l),'--',color='teal',label='Lamp Fit Model')
        #plt.plot(wav_space,l1_model(wav_space),'--',color='black',label='Lamp Fit Model')
        #plt.plot(wav_space,c1_model(wav_space)*max(data_l),'g--', label='Cell Fit Model')
        #plt.hlines(y=0.01, xmin=min(wav_l), xmax=max(wav_l), color='blue')
        #plt.hlines(y=(np.median(norm_data_lamp)), xmin=min(wav_l), xmax=max(wav_l), color='green')
        plt.minorticks_on()
        plt.grid()
        plt.legend()
        plt.title(fr'Ly$\alpha$ Line Profiles ({(I_cell/I0)*100:.2f}% Transmission)')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Counts')
        
        
        #Transmission data used for attenuation plots (different notebook)
        #lamp and cell wavelengths should be same unless different fit applied to each
        wav = self.lamda_lamp
        lamp_y = self.y_e_lamp
        y_e_cell = self.y_e_cell
        
        length = np.where(np.isclose(wav,1215.7,atol=1))[0]
        self.new_wav_space = wav[length]
        dx = np.diff(self.new_wav_space) #Ang per pixel
        
        #Calculating error in lamp + integration (Poisson noise only)
        self.lamp = lamp_y[length]
        self.lamp_err = np.sqrt(self.lamp)/ (self.integration_time-self.t_start)
        I0 = (simps(self.lamp,self.new_wav_space))

        weights = np.ones_like(self.lamp)
        weights[1:-1:2] = 4  # Odd indices get weight of 4
        weights[2:-1:2] = 2  # Even indices get weight of 2
        weights = weights.astype(float)
        weights *= dx[0] / 3  # Scale weights by rule factor

        # Propagate uncertainties
        lamp_integral_error = np.sqrt(np.sum((weights * self.lamp_err) ** 2)) 

        lamp_ratio = (lamp_integral_error/I0)**2
        
        
        
        #Calculating error in cell + integration (Poisson noise only)
        self.cell_y = y_e_cell[length]
        self.err = np.sqrt(self.cell_y)/(self.time_window2-self.time_window)
        If = simps(self.cell_y,self.new_wav_space)
        transmission = If/I0


        weights = np.ones_like(self.cell_y,dtype=float)
        weights[1:-1:2] = 4  # Odd indices get weight of 4
        weights[2:-1:2] = 2  # Even indices get weight of 2

        weights *= dx[0] / 3  # Scale weights by rule factor

        # Propagate uncertainties
        integral_error = np.sqrt(np.sum((weights * self.err) ** 2))

        cell_ratio = (integral_error/If)**2
        
        #Total error for attenuation estimate of profile
        tot_err = transmission * np.sqrt(cell_ratio + lamp_ratio)
        
        plt.figure()
        plt.errorbar(self.new_wav_space,self.lamp,yerr=self.lamp_err,fmt='-o',capsize=4)
        plt.errorbar(self.new_wav_space,self.cell_y,yerr=self.err,fmt='-o')
        plt.title(f'T={transmission*100:.2f}%')
        
        return (I_cell/I0), transmission,tot_err
    
   
    def analyze(self, save_path,name):
        xl,xc = self.preprocess_data()
        self.calibrate_wavelength()
        
        #When running using glob comment out for performance
        #self.plot_data(save_path,name)

        #trans_est is TE for wide profile
        #trans_real is TE for core of lya
        #trans_err is uncertainty of trans_real
        trans_est, trans_real, trans_err = self.find_EW()

        '''
        #print(f"Equivalent Width (Lamp): {EW_lamp1:.2f}")
        #print(f"Equivalent Width (Cell): {EW_cell1:.2f}")
        print(f'Equivalent Width (Absorption Profile): {EW_abs_prof:.2f}')
        print(f"Transmission from EW cell/lamp ratio: {transmission:.2f}")
        print(f"Optical Depth: {optical_depth:.2f}")
        print(f"Transmission from Intensity: {transmission_intensity:.2f}")
        print(f"Intensity Cell: {I0_cell:.2e} (idk just do the math)")
        print(f"Intensity Lamp: {I0_lamp:.2e} (idk just do the math)")
        print(f'b values (Cell, Abs Profile) = {b:.2f} ; {b2:.2f} (km/s)')
        print(f"Column Density of HI along line of sight: {N_col:.2e} cm^2")
        print(f"Column Density of HI along line of sight: {N_col2:.2e} cm^2")
        print(f"Column Density of HI along line of sight (abs): {N_col_consv:.2e} cm^2")
        '''
        return trans_est,trans_real,trans_err
    

#%%
#Comment out cell when importing this .py file into different notebook
'''
# Usage for single lamp/cell pair
directory_date = '02_05_2025'
sub_directory = 'Processed'
date = '02_05_2025'
base_dir = os.path.join(directory_date)
sub_dir = os.path.join(sub_directory)


id_lamp = 'lamp_2T_7V_3m_2'

id_cell = 'cell_2T_7V_3m_2'

lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')

cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')
    

name = lamp_filename[23:49]

analyzer = SpectrumAnalyzer(lamp_filename, cell_filename)
trans_est, trans_real, trans_err = analyzer.analyze('calibration_plot.png',name)
'''
#%%
#Comment out cell when importing this .py file into different notebook
'''
#Usage for lamp/cell data taken at single pressure (for use in attenuation plots notebook)
directory = 'Papa_bear'
sub_directory = '2T_new'
dir_name = os.path.join(directory,sub_directory)


lamp_filename= glob.glob(dir_name+'/*lamp*_*')
cell_filename= glob.glob(dir_name+'/*cell*_*')

#Print to keep track of files power and date
print(lamp_filename)
print(cell_filename)

ts=[]
t_errs=[]
for i in range(len(lamp_filename)):
    analyzer = SpectrumAnalyzer(lamp_filename[i], cell_filename[i])
    name = lamp_filename[i][13:40]
    trans_est, trans_real, trans_err = analyzer.analyze('calibration_plot.png',name)
    ts.append(trans_real)
    t_errs.append(trans_err)

#plots inital TE measurements for all lamp/cell pairs (no powers)
array = range(0,len(ts))
plt.figure()
plt.errorbar(array,ts,yerr=t_errs,fmt='--o')
'''
