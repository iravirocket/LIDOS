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

class SpectrumAnalyzer:
    def __init__(self, lamp_file, dark_file, cell_file):
        self.df_lamp = pd.read_csv(lamp_file)
        self.df_darkf = pd.read_csv(dark_file)
        self.df_cell = pd.read_csv(cell_file)
        self.int_time_dark = np.max(self.df_darkf['Timestamp']) # seconds
        self.integration_time = np.max(self.df_lamp['Timestamp']) # seconds
        self.bandpass = 1650 - 900  # Angstroms
        self.avg_dark = self.df_darkf['y'].mean()
        
        self.guessed_pixels_index = []
        
    def preprocess_data(self):
        #self.xr_lamp = (self.df_lamp['xr'])
        ##self.y_lamp = (self.df_lamp['y'] - self.avg_dark)
        #self.xr_darkf = (self.df_darkf['xr'])
        #self.y_darkf = self.df_darkf['y']
        #self.xr_cell = (self.df_cell['xr'])
        #self.y_cell = self.df_cell['y'] - self.avg_dark

        # Filter data
        y_cut_min, y_cut_max = 1100, 1300
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_cut_min) & (self.df_lamp['y'] <= y_cut_max)]
        self.filtered_darkf = self.df_darkf[(self.df_darkf['y'] >= y_cut_min) & (self.df_darkf['y'] <= y_cut_max)]
        self.filtered_cell = self.df_cell[(self.df_cell['y'] >= y_cut_min) & (self.df_cell['y'] <= y_cut_max)]
        
       # self.filtered_lamp['xr'] = np.round(self.filtered_lamp['xr']).astype(int)
       # self.filtered_cell['xr'] = np.round(self.filtered_cell['xr']).astype(int)
        # Group and sum counts
        self.lamp_counts = self.filtered_lamp.groupby('xr')['y'].count().reset_index()
        self.darkf_counts = self.filtered_darkf.groupby('xr')['y'].count().reset_index()
        self.cell_counts = self.filtered_cell.groupby('xr')['y'].count().reset_index()

        # Update values
        
        self.xr_lamp = (self.lamp_counts['xr'].values)
        self.y_lamp = (self.lamp_counts['y'].values)/self.integration_time
         
        #self.y_lamp = np.clip(self.y_lamp, a_min=0.001, a_max=None)
        
        self.xr_darkf = (self.darkf_counts['xr'].values)
        self.y_darkf = self.darkf_counts['y'].values / (self.integration_time)
        #self.xr_cell = (self.cell_counts['xr'].values)
        
        self.xr_cell = (self.cell_counts['xr'].values)
        self.y_cell = (self.cell_counts['y'].values)/self.integration_time
        
        
        matching_a_indices = np.where(np.isin(self.xr_lamp, self.xr_darkf))[0]
        matching_b_indices = np.where(np.isin(self.xr_darkf, self.xr_lamp))[0]
        
        result = self.y_lamp.copy()

        # Subtract the corresponding elements from arrays A and B where indices match
        result[matching_a_indices] = (self.y_lamp[matching_a_indices]) - self.y_darkf[matching_b_indices]
        self.y_lamp = np.clip(result, a_min=0, a_max=None)
        
        
        matching_a_indices = np.where(np.isin(self.xr_cell, self.xr_darkf))[0]
        matching_b_indices = np.where(np.isin(self.xr_darkf, self.xr_cell))[0]
        
        result = self.y_cell.copy()

        # Subtract the corresponding elements from arrays A and B where indices match
        result[matching_a_indices] = (self.y_cell[matching_a_indices]) - self.y_darkf[matching_b_indices]
        self.y_cell = np.clip(result, a_min=0, a_max=None) 
        
        
        return self.xr_lamp,self.xr_cell
    
    def calibrate_wavelength(self):
        
        px1 = np.where(np.isclose(self.xr_lamp, 1625, atol=2))[0][0]
        #px2 = np.where(np.isclose(self.xr_lamp, 1060, atol=5))[0][0]
        #px3 = np.where(np.isclose(self.xr_lamp, 1190, atol=5))[0][0]
        px4 = np.where(np.isclose(self.xr_lamp, 800, atol=5))[0][0]
        pxl = np.where(np.isclose(self.xr_lamp, 822, atol=2))[0][0]
        self.guessed_pixels_index = [px1,px4,pxl]
        #self.guessed_Angstroms =  [1492.6, 1304, 1334, 1200,1215.67]
        
        self.guessed_Angstroms =  [1492.6, 1200,1215.67]
        
        npixels = 30
        
        self.improved_xval_guesses = [np.average(self.xr_lamp[g-npixels:g+npixels], 
                                            weights=self.y_lamp[g-npixels:g+npixels]) 
                                for g in self.guessed_pixels_index]
        #print(self.improved_xval_guesses)
        
        #linfitter = LinearLSQFitter()
        #wlmodel = Linear1D()
        #self.linfit_wlmodel = linfitter(model=wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
        self.linfit_wlmodel = Linear1D(slope=0.35, intercept=923.7)
        
        self.lamda_lamp = self.linfit_wlmodel(self.xr_lamp)
        self.lamda_cell = self.linfit_wlmodel(self.xr_cell)
        #self.lamda_darkf = self.linfit_wlmodel(self.xr_darkf)
        
        
        self.y_e_lamp = self.y_lamp/self.lamda_lamp
        self.y_e_cell = self.y_cell/self.lamda_cell
        print(self.linfit_wlmodel)
        
        
    def bin_data(self, lamda, y):
        new_spacing= self.linfit_wlmodel.slope.value
        start_point = self.linfit_wlmodel.intercept.value
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

        #bin_sums = np.clip(bin_sums, a_min=0.001, a_max=None)
        return x_bin, bin_sums     
        
      
        
    def plot_data(self, save_path,name):
        
        self.lamda_lamp_bin, self.y_e_lamp_bin = self.bin_data(self.lamda_lamp,self.y_e_lamp)
        self.lamda_cell_bin, self.y_e_cell_bin = self.bin_data(self.lamda_cell,self.y_e_cell)
 
        
        fig, (ax1, ax2, ax3, ax4,ax5) = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))
        ax1.plot(self.xr_lamp, self.y_lamp, ls='-', markersize=3.5, label='Source')
        ax1.plot(self.xr_cell, self.y_cell, ls='-', markersize=3.5, label='Cell')
        for x in self.improved_xval_guesses:
            ax1.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Counts * s-1')
        ax1.set_title(f'Plots for Lamp and Cell {name}')
        ax1.legend()

        self.lamda_lamp = self.lamda_lamp_bin
        self.y_e_lamp = self.y_e_lamp_bin
        
        self.lamda_cell = self.lamda_cell_bin
        self.y_e_cell = self.y_e_cell_bin



        ax2.plot(self.lamda_lamp, self.y_e_lamp, 'o-', markersize=.5, label='Source')
        ax2.plot(self.lamda_cell, self.y_e_cell, 'o-', markersize=.5, label='Cell')
        for x in self.guessed_Angstroms:
            ax2.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        #ax2.plot(self.guessed_Angstroms, [2600]*4, 'x', label='Identified Peaks')
        ax2.axvline(x=1215.67, label='LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Angstroms')
        ax2.set_ylabel('Counts * s-1 *AA-1')
        ax2.legend()

        ax3.plot(self.lamda_lamp, self.y_e_lamp, 'o-', label='Source')
        ax3.plot(self.lamda_cell, self.y_e_cell, 'o-', label='Cell')
        for x in self.guessed_Angstroms:
            ax3.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Angstroms')
        ax3.set_ylabel('Counts * s-1 *AA-1')
        #ax3.set_xlim(1198,1205)
        ax3.set_xlim(1490,1500)
        ax3.legend()
        
        ax4.plot(self.lamda_lamp_bin, self.y_e_lamp_bin, label='Source')
        ax4.plot(self.lamda_cell_bin, self.y_e_cell_bin,'--', label='Cell')
        
        ax4.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Angstroms')
        ax4.set_ylabel('Counts * s-1 *AA-1')
        #ax4.set_ylim(0,0.2)
        ax4.set_xlim(1100,max(self.lamda_lamp))
        ax4.legend()
        
        ax5.plot(self.lamda_lamp_bin, self.y_e_lamp_bin, 'o-', label='Source')
        ax5.plot(self.lamda_cell_bin, self.y_e_cell_bin, 'o-', label='Cell')
        
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
        
        #Define finite window in wavelength space in which to run EW spec util code
        
        #selecting index for which to slice the data based on wavelength ex. ~1213-1218
        lamp_window_start = np.where(np.isclose(self.lamda_lamp, 1215, atol=2))[0][0]
        lamp_window_end = np.where(np.isclose(self.lamda_lamp, 1217, atol=2))[0][-1]
        cell_window_start = np.where(np.isclose(self.lamda_cell, 1215, atol=2))[0][0]
        cell_window_end = np.where(np.isclose(self.lamda_cell, 1217, atol=2))[0][-1]
        
        data_l = self.y_e_lamp[lamp_window_start:lamp_window_end]
        wav_l = self.lamda_lamp[lamp_window_start:lamp_window_end]
        #print(data_l)
        data_cell = self.y_e_cell[cell_window_start:cell_window_end]
        wav_cell = self.lamda_cell[cell_window_start:cell_window_end]
        
        #Normalize counts for EW spec util functions
        norm_data_lamp = data_l / max(data_l)
        norm_data_cell = data_cell / max(data_l)
        
        
        #print(min(norm_data_cell))
        #Find EW of Cell and Lamp
        spec_l = Spectrum1D(spectral_axis=wav_l*u.AA, flux=norm_data_lamp*u.Unit('erg cm-2 s-1 AA-1'))
        EW_l = equivalent_width(spec_l, continuum=0.01*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_l)*u.AA, max(wav_l)*u.AA))
        
        spec_cell = Spectrum1D(spectral_axis=wav_cell*u.AA, flux=norm_data_cell*u.Unit('erg cm-2 s-1 AA-1'))
        EW_cell = equivalent_width(spec_cell, continuum=0.01*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_cell)*u.AA, max(wav_cell)*u.AA))

        
        #Doppler Parameter of Cell Profile
        FWHM = fwhm(spec_cell,regions=SpectralRegion(min(wav_cell)*u.AA, max(wav_cell)*u.AA))
        b = FWHM.value * 1.66
        
        #Intensity and numerical EW calculations for lamp and cell
        al = (0.001 - data_l) / 0.001
        I0 = simps(data_l, wav_l)
        EW_2L = simps(al, wav_l)
        
        acell = (0.001 - data_cell) / 0.001
        I_cell = simps(data_cell, wav_cell)
        EW_2cell = simps(acell, wav_cell)
        
        
        #Fitting the profiles to obtain absorption profile of cell at Lya
        fitter = fitting.SLSQPLSQFitter()
        l1 = models.Voigt1D(x_0=1215.67)
        l1_model = fitter(l1,wav_l,norm_data_lamp)
        
        #Fix the lamp profile fit so that it cannot change
        l1_model.x_0.fixed=True
        l1_model.amplitude_L.fixed=True
        l1_model.fwhm_L.fixed=True
        l1_model.fwhm_G.fixed=True
        
        #Initial guess for cell absorption at Lya
        abs_model_infer = models.Voigt1D(x_0=1215.67,amplitude_L=-0.5)
        
        #Combined model of emission - absorption
        tot_output_model = l1_model + abs_model_infer
        '''
        c1 = models.Voigt1D(x_0=1215.5)
        c2 = models.Voigt1D(x_0=1216.25)
        ctot=c1+c2
        '''
        #Fit combined model to cell data
        
        #c1_model = fitter(ctot,wav_cell,norm_data_cell)
        c1_model = fitter(tot_output_model,wav_cell,norm_data_cell)
        
        #c_abs = l1_model-c1_model
        wav_space = np.linspace(min(wav_l),max(wav_l))
        #abs_profile = 1-c_abs(wav_space)
        abs_profile = 1+ c1_model[1](wav_space)
        print(c1_model)
        
        #Calculating column density from FHWM of models abs profile
        abs_prof_spec = Spectrum1D(spectral_axis=wav_space*u.AA, flux=abs_profile*u.Unit('erg cm-2 s-1 AA-1'))
        FWHM2 = fwhm(abs_prof_spec,regions=SpectralRegion(min(wav_space)*u.AA, max(wav_space)*u.AA))
        b2 = FWHM2.value * 1.66
        
        EW_abs_prof = equivalent_width(abs_prof_spec, continuum=1*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_space)*u.AA, max(wav_space)*u.AA))
        
        plt.figure()
        plt.plot(wav_l, norm_data_lamp, 'ro-',label='Lamp')
        plt.plot(wav_cell, norm_data_cell, 'ko-',label='cell')
        
        print(l1_model)
        plt.plot(wav_space,l1_model(wav_space),'--',color='teal')
        plt.plot(wav_space,c1_model(wav_space),'g--')
        plt.hlines(y=0.01, xmin=min(wav_l), xmax=max(wav_l), color='blue')
        #plt.hlines(y=(np.median(norm_data_lamp)), xmin=min(wav_l), xmax=max(wav_l), color='green')
        plt.grid()
        plt.title('EW')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Normalized Counts s-1 AA-1')
        
        #Plot just the absorption profile of cell at Lya
        plt.figure()
        plt.plot(wav_space,abs_profile,'o--')
        plt.ylabel('Modelled transmission through cell (Normalized)')
        plt.xlabel('Wavelength (AA)')
        plt.title('Cell Absorption Profile')
        plt.grid();
        
                
        
        return -1*EW_l, EW_cell*-1, -1*EW_2L, -1* EW_2cell, I0, I_cell,b,b2, EW_abs_prof
    
    '''   
    def find_EW(self, lamda, y):
        # Find peaks to determine the central wavelength of the spectral line
        peaks, _ = find_peaks(y, height=max(y))  # Adjust the height threshold as needed
        if len(peaks) == 0:
            raise ValueError("No peaks found in the data.")
        
        peak_index = peaks[0]  # Assuming the first peak is the line of interest
    
        # Define the window around the peak
        window_width = 10 # Adjust the window width as needed
        window_start = max(0, peak_index - window_width)
        window_end = min(len(lamda), peak_index + window_width)

        data = y[window_start:window_end]
        wav = lamda[window_start:window_end]
        spec = data / max(data)

        plt.figure()
        plt.plot(wav, spec, 'o-')
        plt.hlines(y=min(spec), xmin=min(wav), xmax=max(wav), color='red', linestyle='--')
        plt.grid()
        plt.title('Normalized Y-axis to find EW')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Normalized Counts s-1 AA-1')

        spec1 = Spectrum1D(spectral_axis=wav*u.AA, flux=spec*u.Unit('erg cm-2 s-1 AA-1'))
        EW = equivalent_width(spec1, continuum=min(spec)*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav)*u.AA, max(wav)*u.AA))

        a = (min(data) - data) / min(data)
        I0 = simps(data, wav)
        EW_2 = simps(a, wav)

        return EW * -1, EW_2 * -1, I0
        '''
    def analyze(self, save_path,name):
        xl,xc = self.preprocess_data()
        self.calibrate_wavelength()
        self.plot_data(save_path,name)

        #x_bin, bin_sums = self.bin_data(self.lamda_lamp, self.y_lamp)
        #x_cell, cell_sum = self.bin_data(self.lamda_cell, self.y_cell)

        EW_lamp1, EW_cell1, EW_lamp2,EW_cell2, I0_lamp, I0_cell,b,b2, EW_abs_prof = self.find_EW()
    
        #EW_lamp, EW_2_lamp, I0_lamp = self.find_EW(lamp_window_start, lamp_window_end, self.lamda_lamp, norm_data_lamp)
        #EW_cell, EW_2_cell, I0_cell = self.find_EW(cell_window_start, cell_window_end, self.lamda_cell, norm_data_cell)

        transmission = EW_cell1 / EW_lamp1
        optical_depth = -np.log(transmission)
        transmission_intensity = I0_cell/I0_lamp
        optical_depth_I = -np.log(transmission_intensity)
        #b in units of km/s
        #N_col = ((optical_depth / 0.7580) * (b/10))* (10**13)
        N_col2 = ((optical_depth / 0.7580) * (b2/10))* (10**13)
        N_col = (optical_depth /(0.416 * 1216) ) * 3.768e14 #From Ostlin paper basically same
        #print(f"Equivalent Width (Lamp): {EW_lamp1:.2f}")
        #print(f"Equivalent Width (Cell): {EW_cell1:.2f}")
        print(f'Equivalent Width (Absorption Profile): {EW_abs_prof:.2f}')
        print(f"Transmission from EW cell/lamp ratio: {transmission:.2f}")
        print(f"Optical Depth: {optical_depth:.2f}")
        print(f"Transmission Intensity: {transmission_intensity:.2f}")
        print(f"Intensity Cell: {I0_cell:.2e}")
        print(f"Intensity Lamp: {I0_lamp:.2e}")
        print(f'b values (Cell, Abs Profile) = {b:.2f} ; {b2:.2f}')
        print(f"Column Density of HI along line of sight: {N_col:.2e} cm^2")
        print(f"Column Density of HI along line of sight: {N_col2:.2e} cm^2")
        return transmission, transmission_intensity, optical_depth, optical_depth_I
    
'''
def write_trans(trans,trans_I,name):
    # Import writer class from csv module
    from csv import writer
     
    # List that we want to add as a new row
    List = [trans,trans_I,name]
     
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open('transmission.csv', 'a') as f_object:
     
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
     
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)
     
        # Close the file object
        f_object.close()
'''
        
# Usage
directory_date = '05_24_2024_2'
sub_directory = 'Processed'
date = '05_24_2024'
base_dir = os.path.join(directory_date)
sub_dir = os.path.join(sub_directory)


lamp_paths = glob.glob(f'{directory_date}/{sub_directory}/{date}_lamp*T_*')
cell_paths = glob.glob(f'{directory_date}/{sub_directory}/{date}_cell*')

#id_lamp = 'lamp_3.5A_2T_7'
id_filament_dark = 'darkf_3.5A_2T'
#id_cell = 'cell_3.5A_2T_7'

#lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
filament_dark_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_filament_dark}_processed.csv')
#cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')

trans = []
trans_I = []
taus = []
taus_I=[]

for lamp_filename, cell_filename in zip(lamp_paths,cell_paths):

    name = lamp_filename[23:49]
    analyzer = SpectrumAnalyzer(lamp_filename, filament_dark_filename, cell_filename)
    t, t_I, tau, tau_I = analyzer.analyze('calibration_plot.png',name)
    trans.append(t)
    trans_I.append(t_I)
    taus.append(tau)
    taus_I.append(tau_I)
    
    #write_trans(t,t_I,name)
    

