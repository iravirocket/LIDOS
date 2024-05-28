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

class SpectrumAnalyzer:
    def __init__(self, lamp_file, dark_file, cell_file):
        self.df_lamp = pd.read_csv(lamp_file)
        self.df_darkf = pd.read_csv(dark_file)
        self.df_cell = pd.read_csv(cell_file)
        self.integration_time = np.max(self.df_lamp['Timestamp']) # seconds
        self.bandpass = 1650 - 900  # Angstroms
        self.avg_dark = self.df_darkf['y'].mean()
        self.guessed_Angstroms =  [1492.6, 1304, 1334, 1200.2]
        self.guessed_pixels_index = []
        
    def preprocess_data(self):
        self.xr_lamp = (self.df_lamp['xr'])
        self.y_lamp = self.df_lamp['y'] - self.avg_dark
        self.xr_darkf = (self.df_darkf['xr'])
        self.y_darkf = self.df_darkf['y']
        self.xr_cell = (self.df_cell['xr'])
        self.y_cell = self.df_cell['y'] - self.avg_dark

        # Filter data
        y_cut_min, y_cut_max = 1000, 1200 #660, 1300
        self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_cut_min) & (self.df_lamp['y'] <= y_cut_max)]
        self.filtered_darkf = self.df_darkf[(self.df_darkf['y'] >= y_cut_min) & (self.df_darkf['y'] <= y_cut_max)]
        self.filtered_cell = self.df_cell[(self.df_cell['y'] >= y_cut_min) & (self.df_cell['y'] <= y_cut_max)]
        
        self.filtered_lamp['xr'] = np.round(self.filtered_lamp['xr'])
        self.filtered_cell['xr'] = np.round(self.filtered_cell['xr'])
        # Group and sum counts
        self.lamp_counts = self.filtered_lamp.groupby('xr')['y'].sum().reset_index()
        self.darkf_counts = self.filtered_darkf.groupby('xr')['y'].sum().reset_index()
        self.cell_counts = self.filtered_cell.groupby('xr')['y'].sum().reset_index()

        # Update values
        self.xr_lamp = (self.lamp_counts['xr'].values)
        
        self.y_lamp = self.lamp_counts['y'].values - self.avg_dark
        self.y_lamp = np.clip(self.y_lamp, a_min=0.001, a_max=None)
        
        self.xr_darkf = (self.darkf_counts['xr'].values)
        self.y_darkf = self.darkf_counts['y'].values
        self.xr_cell = (self.cell_counts['xr'].values)
        
        self.y_cell = self.cell_counts['y'].values - self.avg_dark
        self.y_cell = np.clip(self.y_cell, a_min=0.001, a_max=None)
    
    
        
        return self.xr_lamp,self.xr_cell
    
    def calibrate_wavelength(self):
        
        px1 = np.where(np.isclose(self.xr_lamp, 1627, atol=1))[0][0]
        px2 = np.where(np.isclose(self.xr_lamp, 1082, atol=1))[0][0]
        px3 = np.where(np.isclose(self.xr_lamp, 1178, atol=1))[0][0]
        px4 = np.where(np.isclose(self.xr_lamp, 795, atol= 1)) [0][0]
        self.guessed_pixels_index = [px1, px2, px3, px4]
        
        npixels = 10
        self.improved_xval_guesses = [np.average(self.xr_lamp[g-npixels:g+npixels], 
                                            weights=self.y_lamp[g-npixels:g+npixels]) 
                                for g in self.guessed_pixels_index]
        print(self.improved_xval_guesses)
        
        linfitter = LinearLSQFitter()
        wlmodel = Linear1D()
        self.linfit_wlmodel = linfitter(model=wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
        #self.linfit_wlmodel = Linear1D(slope=0.349, intercept=925.9)
        
        self.lamda_lamp = self.linfit_wlmodel(self.xr_lamp)
        self.lamda_cell = self.linfit_wlmodel(self.xr_cell)
        self.lamda_darkf = self.linfit_wlmodel(self.xr_darkf)
        print(self.linfit_wlmodel)
        
    def plot_data(self, save_path):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
        ax1.plot(self.xr_lamp, self.y_lamp, ls='-', markersize=3.5, label='Source')
        ax1.plot(self.xr_cell, self.y_cell, ls='-', markersize=3.5, label='Cell')
        for x in self.improved_xval_guesses:
            ax1.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Counts * s-1 *AA-1')
        ax1.set_title('Plots for Lamp and Cell')
        #ax1.legend()

        ax2.plot(self.lamda_lamp, self.y_lamp, 'o-', markersize=.5, label='Source')
        ax2.plot(self.lamda_cell, self.y_cell, 'o-', markersize=.5, label='Cell')
        for x in self.guessed_Angstroms:
            ax2.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        #ax2.plot(self.guessed_Angstroms, [2600]*4, 'x', label='Identified Peaks')
        ax2.axvline(x=1215.67, label='LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Angstroms')
        ax2.set_ylabel('Counts * s-1 *AA-1')
        #ax2.legend()

        ax3.plot(self.lamda_lamp, self.y_lamp, 'o-', label='Source')
        ax3.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        for x in self.guessed_Angstroms:
            ax3.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Angstroms')
        ax3.set_ylabel('Counts * s-1 *AA-1')
        ax3.set_xlim(1198,1205)
        #ax3.set_xlim(1490,1500)
        #ax3.legend()
        
        ax4.plot(self.lamda_lamp, self.y_lamp, 'o-', label='Source')
        ax4.plot(self.lamda_cell, self.y_cell, 'o-', label='Cell')
        
        ax4.axvline(x=1215.67, label='Rest LyAlpha', color='green', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Angstroms')
        ax4.set_ylabel('Counts * s-1 *AA-1')
        ax4.set_xlim(1210,1219.5)
        #ax4.legend()

        fig.savefig(save_path)
        
        
        
        
        
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
    
    def find_EW(self):
        
        lamp_window_start = np.where(np.isclose(self.lamda_lamp, 1210, atol=0.7))[0][0]
        lamp_window_end = np.where(np.isclose(self.lamda_lamp, 1220, atol=1))[0][0]
        cell_window_start = np.where(np.isclose(self.lamda_cell, 1210, atol=0.7))[0][0]
        cell_window_end = np.where(np.isclose(self.lamda_cell, 1220, atol=1))[0][0]
        
        integration_time = 200
        effective_area = 7.3644e-05
        solid_angle = (.0025 * .5)/28**2
        window_size = 10
        energy = 2.2e-6 #energy of lyman alpha photon in ergs
        counts_to_flux = energy / (effective_area * integration_time * window_size)
        
        data_l = self.y_lamp[lamp_window_start:lamp_window_end] * counts_to_flux
        wav_l = self.lamda_lamp[lamp_window_start:lamp_window_end]
        print(data_l)
        data_cell = self.y_cell[cell_window_start:cell_window_end] * counts_to_flux
        wav_cell = self.lamda_cell[cell_window_start:cell_window_end]
        
        norm_data_lamp = data_l / max(data_l)
        norm_data_cell = data_cell / max(data_l)
        
        
        print(min(norm_data_cell))
        spec_l = Spectrum1D(spectral_axis=wav_l*u.AA, flux=norm_data_lamp*u.Unit('erg cm-2 s-1 AA-1'))
        EW_l = equivalent_width(spec_l, continuum=0.001*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_l)*u.AA, max(wav_l)*u.AA))
        
        spec_cell = Spectrum1D(spectral_axis=wav_cell*u.AA, flux=norm_data_cell*u.Unit('erg cm-2 s-1 AA-1'))
        EW_cell = equivalent_width(spec_cell, continuum=0.001*u.Unit('erg cm-2 s-1 AA-1'), 
                              regions=SpectralRegion(min(wav_cell)*u.AA, max(wav_cell)*u.AA))

        al = (min(data_l) - data_l) / min(data_l)
        I0 = simps(data_l, wav_l)
        EW_2L = simps(al, wav_l)
        
        acell = (min(data_cell) - data_cell) / min(data_cell)
        I_cell = simps(data_cell, wav_cell)
        EW_2cell = simps(acell, wav_cell)
        
        rayleigh_conversion = 1e6 / (4* np.pi * solid_angle)
        I0_rayleighs = I0 * rayleigh_conversion
        Icell_rayleighs = I_cell * rayleigh_conversion

        
        plt.figure()
        plt.plot(wav, spec, 'o-')
        plt.hlines(y=min(spec), xmin=min(wav), xmax=max(wav), color='red', linestyle='--')
        plt.grid()
        plt.title('EW')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Normalized Counts s-1 AA-1')
        
        
        return -1*EW_l, EW_cell*-1, -1*EW_2L, -1* EW_2cell, I0, I_cell, I0_rayleighs, Icell_rayleighs
    
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
    def analyze(self, save_path):
        xl,xc = self.preprocess_data()
        self.calibrate_wavelength()
        self.plot_data(save_path)

        x_bin, bin_sums = self.bin_data(self.lamda_lamp, self.y_lamp)
        x_cell, cell_sum = self.bin_data(self.lamda_cell, self.y_cell)

        EW_lamp1, EW_cell1, EW_lamp2,EW_cell2, I0_lamp, I0_cell, I0_rayleighs, Icell_rayleighs = self.find_EW()
    
        #EW_lamp, EW_2_lamp, I0_lamp = self.find_EW(lamp_window_start, lamp_window_end, self.lamda_lamp, norm_data_lamp)
        #EW_cell, EW_2_cell, I0_cell = self.find_EW(cell_window_start, cell_window_end, self.lamda_cell, norm_data_cell)

        transmission = EW_cell1 / EW_lamp1
        #optical_depth = -np.log(transmission)
        transmission_intensity = I0_cell/I0_lamp
        optical_depth = -np.log(transmission_intensity)
        
        print(f"Equivalent Width (Lamp): {EW_lamp1}")
        print(f"Equivalent Width (Cell): {EW_cell1}")
        print(f"Transmission: {transmission}")
        print(f"Optical Depth: {optical_depth}")
        print(f"Transmission Intensity: {transmission_intensity}")
        print(f"Intensity Cell: {I0_cell} erg/cm^2 s")
        print(f"Intensity Lamp: {I0_lamp} erg/cm^2 s")
        print(f"Rayleigh Lamp: {I0_rayleighs}")
        print(f"Rayleigh Cell: {Icell_rayleighs}")
        return xl,xc
# Usage
directory_date = '05_24_2024'
sub_directory = 'Processed'
date = '05_24_2024'
base_dir = os.path.join(directory_date)
sub_dir = os.path.join(sub_directory)
id_lamp = 'lamp_2.75V_2T'
id_filament_dark = 'darkf_3.25V_1T'
id_cell = 'cell_2.75V_2T'

lamp_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_lamp}_processed.csv')
filament_dark_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_filament_dark}_processed.csv')
cell_filename = os.path.join(base_dir, sub_dir, f'{date}_{id_cell}_processed.csv')

analyzer = SpectrumAnalyzer(lamp_filename, filament_dark_filename, cell_filename)
xl,xc = analyzer.analyze('calibration_plot.png')
