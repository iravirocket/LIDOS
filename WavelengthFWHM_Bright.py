#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:02:34 2024

@author: Ravi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.fitting import fit_generic_continuum, find_lines_threshold, estimate_line_parameters, fit_lines
from specutils.analysis import equivalent_width, fwhm
from specutils.manipulation import noise_region_uncertainty, extract_region

from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.special import wofz
from matplotlib.backends.backend_pdf import PdfPages
from astropy.modeling import models, fitting
from astropy.convolution import convolve_models
from matplotlib.colors import Normalize, Colormap
import matplotlib.gridspec as gridspec


class WavelengthFWHM_Bright:
    
    def __init__(self, data_path):
        """
        Initialize the class with optional dataframes for lamp, cell, and darks.
        """
        self.df_data = pd.read_csv(data_path)
        self.x_lamp_all = self.df_data.get('x_corrected_lamp', None)
        self.y_lamp_all = self.df_data.get('y_lamp', None)
        self.x_lamp_filtered = self.df_data.get('corrected_x_filtered_lamp', None)
        self.y_lamp_filtered = self.df_data.get('y_filtered_lamp', None)
        self.y_data = self.df_data.get('lamp_y_filtered', None)
        self.x_data = self.df_data.get('lamp_x_filtered', None)
        
        # Check if cell data exists and load it conditionally
        if 'corrected_x_filtered_cell' in self.df_data.columns and 'y_filtered_cell' in self.df_data.columns:
            self.x_cell_filtered = self.df_data.get('corrected_x_filtered_cell', None)
            self.y_cell_filtered = self.df_data.get('y_filtered_cell', None)
        else:
            self.x_cell_filtered = None
            self.y_cell_filtered = None
        
        # Same for dark data
        if 'corrected_x_filtered_dark' in self.df_data.columns and 'y_filtered_dark' in self.df_data.columns:
            self.x_dark_filtered = self.df_data.get('corrected_x_filtered_dark', None)
            self.y_dark_filtered = self.df_data.get('y_filtered_dark', None)
        else:
            self.x_dark_filtered = None
            self.y_dark_filtered = None
        
        self.integration_time = self.df_data['Integration_Time'] if 'Integration_Time' in self.df_data.columns else None
        self.bandpass = 1650 - 900  # Angstroms
        self.guessed_Angstroms = [1199.5, 1215.7]  # Example guesses for wavelength calibration points
        self.guessed_pixels_index = []
        self.energy = 2.2e-6  # Energy of Lyman-alpha photon in ergs
    
   
    def preprocess_data(self, x_corrected_all, y_data_all, x_corrected_filtered, y_filtered):
        """
        Preprocess the data, calculate histograms, and return the counts.
        """
        
        #print(f"x data:  {self.x_data}")
        # Convert x_data and y_data to numeric, coercing errors to NaN
        self.x_data = pd.to_numeric(self.x_data, errors='coerce')
        self.y_data = pd.to_numeric(self.y_data, errors='coerce')
        
        
        xbins = 2058
        #ybins = 100
    
        #print(f"x data = {self.x_data}")
        
        # Remove NaN values from both full and filtered datasets
        full_mask = ~np.isnan(x_corrected_all) & ~np.isnan(y_data_all)
        x_corrected_all_clean = x_corrected_all[full_mask]
        y_data_clean = y_data_all[full_mask]
        
        filtered_mask = ~np.isnan(x_corrected_filtered) & ~np.isnan(y_filtered)
        x_corrected_filtered_clean = x_corrected_filtered[filtered_mask]
        y_filtered_clean = y_filtered[filtered_mask]
    
        orig_mask = ~np.isnan(self.x_data) & ~np.isnan(self.y_data)
        x_corrected = self.x_data[orig_mask]
        y_corrected = self.y_data[orig_mask]
        
     
    
        # Generate histograms without rounding or cutting y values
        full_counts_y, xedges_full = np.histogram(x_corrected_all_clean, bins=xbins, weights = y_data_clean)
        y_counts, xedges = np.histogram(x_corrected, bins= 2048, weights = y_corrected)
        filtered_counts_y, xedges_filtered = np.histogram(x_corrected_filtered_clean, bins=xbins, weights=y_filtered_clean)
        
    
        # Calculate the 1D histogram counts (summing along the y-axis)
        #full_counts_y = np.sum(full_hist, axis=1)  # Counts along x-axis summing over y
        #filtered_counts_y = np.sum(filtered_hist, axis=1)  # Counts along x-axis summing over y
        
        print(f"Filtered y counts:  {filtered_counts_y}")
        print(f"Full y counts: {full_counts_y}")
        
        # Plot the histograms for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
        # Plot full dataset histogram
        ax1.plot(xedges_full[:-1], full_counts_y, '-') #, width=np.diff(xedges_full), edgecolor='black', align='edge')
        
        ax1.set_title("Full Data Y-Counts")
        ax1.set_xlabel("X Data")
        ax1.set_ylabel("Counts")
        ax1.set_xlim(750, 875)
    
        # Plot filtered dataset histogram
        ax2.plot(xedges_full[:-1], full_counts_y, '-') #, width=np.diff(xedges_filtered), edgecolor='black', align='edge')
        ax2.set_title("Filtered Data Y-Counts")
        ax2.set_xlabel("X Data")
        ax2.set_ylabel("Counts")
        ax2.set_xlim(832,845)
        ax2.set_ylim()

        plt.tight_layout()
        plt.show()
        
        # Plot the histograms for visualization
        fig, (axo1, axo2) = plt.subplots(1, 2, figsize=(12, 6))
    
        # Plot full dataset histogram
        axo1.plot(xedges[:-1], y_counts, '-') #, width=np.diff(xedges_full), edgecolor='black', align='edge')
        
        axo1.set_title("Full Data Y-Counts")
        axo1.set_xlabel("X Data")
        axo1.set_ylabel("Counts")
        axo1.set_xlim(750, 875)
    
        # Plot filtered dataset histogram
        axo2.plot(xedges[:-1], y_counts, '-') #, width=np.diff(xedges_filtered), edgecolor='black', align='edge')
        axo2.set_title("Filtered Data Y-Counts")
        axo2.set_xlabel("X Data")
        axo2.set_ylabel("Counts")
        axo2.set_xlim(820, 840)
        axo2.set_ylim(0,2000000)

        plt.tight_layout()
        plt.show()
        
        
        return full_counts_y, xedges_full, filtered_counts_y, xedges_filtered, y_counts, xedges

        
    def calibrate_wavelength(self, xedges_full, full_counts_y):
        """
        Perform wavelength calibration using known pixel positions and guessed wavelengths.
        """
        # Calculate bin centers from bin edges
        bin_centers = (xedges_full[:-1] + xedges_full[1:]) / 2
        
        # Debugging output to check range of bin_centers
        print(f"bin_centers range: min={bin_centers.min()}, max={bin_centers.max()}")
        
        # Assuming you have pre-identified pixel positions corresponding to specific wavelengths
        if bin_centers is not None:
            px1_indices = np.where(np.isclose(bin_centers, 795, atol=2))[0]
            pxlyman_indices = np.where(np.isclose(bin_centers, 839, atol=2))[0]
            
            if len(px1_indices) == 0 or len(pxlyman_indices) == 0:
                raise ValueError("Could not find close pixel values for the guessed wavelengths.")
            
            px1 = px1_indices[0]
            pxlyman = pxlyman_indices[0]
            self.guessed_pixels_index = [px1, pxlyman]
            
            # Calculate the average positions for better accuracy
            npixels = 5
            self.improved_xval_guesses = [np.average(bin_centers[g-npixels:g+npixels], 
                                                     weights=full_counts_y[g-npixels:g+npixels]) 
                                          for g in self.guessed_pixels_index]
            print("Improved pixel guesses:", self.improved_xval_guesses)
        
            # Perform linear fitting for wavelength calibration
            linfitter = LinearLSQFitter()
            wlmodel = Linear1D()
            self.linfit_wlmodel = linfitter(wlmodel, x=self.improved_xval_guesses, y=self.guessed_Angstroms)
            print("Wavelength calibration model:", self.linfit_wlmodel)
            
            # Apply the wavelength calibration model to the lamp data
            self.lamda_lamp = self.linfit_wlmodel(bin_centers)
        else:
            print("No bin centers available for wavelength calibration.")
        
        # Perform calibration for cell data if available
        if self.x_cell_filtered is not None:
            self.lamda_cell = self.linfit_wlmodel(self.x_cell_filtered)
        else:
            print("No cell data available for wavelength calibration.")
     
            
    def plot_data(self, save_path, full_counts_y):
        """
        Plot the calibrated data and save it to a PDF.
        """
        with PdfPages(save_path) as pdf:
            fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
            
            # Plot lamp data
            ax1.plot(self.lamda_lamp, full_counts_y, ls='-', markersize=3.5, label='Lamp')
            for x in self.guessed_Angstroms:
                ax1.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
            ax1.set_xlabel('Wavelength (Angstroms)')
            ax1.set_ylabel('Counts * s-1')
            ax1.set_title('Lamp Data Wavelength Calibration')
            ax1.set_xlim(1190, 1220)
            ax1.legend()

            # Plot cell data if available
            if self.x_cell_filtered is not None:
                ax2.plot(self.lamda_cell, self.y_cell_filtered, ls='-', markersize=3.5, label='Cell')
                for x in self.guessed_Angstroms:
                    ax2.axvline(x=x, label='Identified Peaks', color='red', linestyle='--', linewidth=0.5)
                ax2.set_xlabel('Wavelength (Angstroms)')
                ax2.set_ylabel('Counts * s-1')
                ax2.set_title('Cell Data Wavelength Calibration')
                ax2.legend()
            else:
                ax2.set_visible(False)

            plt.show()
           # pdf.savefig(fig1)
            #plt.close(fig1)
            
            print(f"Data plots saved to {save_path}")
            
    
    def calculate_fwhm(self, wavelengths, counts, peak_idx):
        """
        Calculate the FWHM (Full Width at Half Maximum) for a given peak index.
        """
        peak_height = counts[peak_idx]
        half_max = peak_height / 2
        
        # Find the left side of the FWHM
        left_idx = np.where(counts[:peak_idx] < half_max)[0]
        if len(left_idx) > 0:
            left_idx = left_idx[-1]  # Closest point to the left of the peak
        else:
            left_idx = 0  # If no point is below half max, set to the first index
        
        # Find the right side of the FWHM
        right_idx = np.where(counts[peak_idx:] < half_max)[0]
        if len(right_idx) > 0:
            right_idx = peak_idx + right_idx[0]  # Closest point to the right of the peak
        else:
            right_idx = len(counts) - 1  # If no point is below half max, set to the last index
        
        # Calculate the FWHM in terms of wavelength
        fwhm = wavelengths[right_idx] - wavelengths[left_idx]
        
        return fwhm, wavelengths[left_idx], wavelengths[right_idx]
    
    def export_wavelength_data_with_fwhm(self, lamda_lamp, counts, peak1_idx, peak2_idx, fwhm1, fwhm2, save_path):
        """
        Export wavelength data, counts, and FWHM values to a CSV file.
        """
        df_export = pd.DataFrame({
            'Wavelength (Angstroms)': lamda_lamp,
            'Counts': counts
        })
        
        # Add the FWHM as a separate row at the end
        df_export.loc[len(df_export)] = pd.Series([f"FWHM Line 1: {fwhm1}", '', ''])
        df_export.loc[len(df_export)] = pd.Series([f"FWHM Line 2: {fwhm2}", '', ''])
        
        # Save to CSV
        df_export.to_csv(save_path, index=False)
        print(f"Wavelength data and FWHM values saved to {save_path}")
    
    def analyze(self):
        """
        Analyze the data: preprocess, calibrate wavelengths, compute FWHM, and export results.
        """
        # Preprocess data (counts calculation)
        full_counts_y, xedges_full, filtered_counts_y, xedges_filtered, y_counts, xedges = self.preprocess_data(
            self.x_lamp_all, self.y_lamp_all, self.x_lamp_filtered, self.y_lamp_filtered
        )
        
        # Perform wavelength calibration
        self.calibrate_wavelength(xedges_full, full_counts_y)
        
        # Identify peaks for the two calibration lines (assuming they're the two tallest peaks)
        peak_indices, _ = find_peaks(full_counts_y, height=0)
        
        # Use your knowledge of the pixel positions of the two lines (you mentioned around 800 and 840)
        peak1_idx = np.argmin(np.abs(xedges_full - 795))
        peak2_idx = np.argmin(np.abs(xedges_full - 839))
        
        # Calculate FWHM for each peak
        fwhm1, left_wavelength1, right_wavelength1 = self.calculate_fwhm(self.lamda_lamp, full_counts_y, peak1_idx)
        fwhm_lya, left_wavelength2, right_wavelength2 = self.calculate_fwhm(self.lamda_lamp, full_counts_y, peak2_idx)
        
        print(f"FWHM of first line: {fwhm1} Angstroms, peak at: {self.lamda_lamp[peak1_idx]}")
        print(f"FWHM of second line: {fwhm_lya} Angstroms, peak at: {self.lamda_lamp[peak2_idx]}")
        
        # Plot the data and save to a PDF (optional, you can remove this if not needed)
        save_plot_path = 'calibrated_data_plots.pdf'
        self.plot_data(save_plot_path, full_counts_y)
        
        # Export data and FWHM values to a CSV
     
        self.export_wavelength_data_with_fwhm(self.lamda_lamp, full_counts_y, peak1_idx, peak2_idx, fwhm1, fwhm_lya, save_path)


            
    '''
    def analyze(self):
        """
        Analyze the data: preprocess, calibrate wavelengths, and plot results.
        """
        # Preprocess data (counts calculation)
        full_counts_y, xedges_full, filtered_counts_y, xedges_filtered, y_counts, xedges  = self.preprocess_data(
            self.x_lamp_all, self.y_lamp_all, self.x_lamp_filtered, self.y_lamp_filtered
        )
        
        # Perform wavelength calibration
        self.calibrate_wavelength(xedges_full, full_counts_y)
        
        # Plot the data and save to a PDF
        save_path = 'calibrated_data_plots.pdf'
        self.plot_data(save_path, full_counts_y)
        '''
# Define paths and identifiers
directory_date = '05_09_2024'
sub_directory = 'Clean'
date = '05_09_2024'
base_dir = os.path.join(directory_date, sub_directory)

# Define identifiers
id_lamp = 'lamp_test8'
id_filament_dark = 'dark_test1'
#id_cell = 'cell_2.25V_a5'

# Construct file paths
data_path = os.path.join(base_dir, f'{date}_{id_lamp}_RANSAC_data.csv')
save_path = os.path.join(base_dir, f'{date}_{id_lamp}_Wavelength_data.csv')
#dark_filename = os.path.join(base_dir, f'{date}_{id_filament_dark}_wavelength.csv')
#cell_filename = os.path.join(base_dir, f'{date}_{id_cell}_processed.csv')

# Initialize and run the processor
processor = WavelengthFWHM_Bright(data_path)
processor.analyze()
