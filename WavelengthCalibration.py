#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:18:39 2024

@author: Ravi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from scipy.signal import find_peaks
import os


class WavelengthCalibration:
    def __init__(self, data_path):
        self.df_data = pd.read_csv(data_path)
        
        # Lamp data
        self.x_lamp_all = self.df_data.get('x_corrected_lamp', None)
        self.y_lamp_all = self.df_data.get('y_lamp', None)
        
        # Cell data
        self.x_cell_all = self.df_data.get('x_corrected_cell', None)
        self.y_cell_all = self.df_data.get('y_cell', None)
        
        # Calibration info
        self.guessed_Angstroms = [1200, 1215.7]
        self.guessed_pixels_index = []

    def preprocess_data(self, x_corrected_all, y_data_all):
        """
        Preprocess the data, clean NaNs, and return histogram counts.
        """
        mask = ~np.isnan(x_corrected_all) & ~np.isnan(y_data_all)
        x_clean = x_corrected_all[mask]
        y_clean = y_data_all[mask]
        
        ybins = 100  # Histogram bins for Y-axis
        full_hist, xedges_full, _ = np.histogram2d(x_clean, y_clean, bins=(len(x_clean), ybins))
        full_counts_y = np.sum(full_hist, axis=1)  # Summing counts along y-axis
        
        return full_counts_y, xedges_full

    def calibrate_wavelength(self, xedges_full, full_counts_y):
        """
        Perform wavelength calibration using known pixel positions and wavelengths.
        """
        bin_centers = (xedges_full[:-1] + xedges_full[1:]) / 2
        px1_indices = np.where(np.isclose(bin_centers, 792, atol=5))[0]
        pxlyman_indices = np.where(np.isclose(bin_centers, 845, atol=5))[0]

        if len(px1_indices) == 0 or len(pxlyman_indices) == 0:
            raise ValueError("Could not find close pixel values for the guessed wavelengths.")
        
        px1 = px1_indices[0]
        pxlyman = pxlyman_indices[0]
        self.guessed_pixels_index = [px1, pxlyman]
        npix = 10
        improved_xval_guesses = [
            np.average(bin_centers[g-npix:g+npix], weights=full_counts_y[g-npix:g+npix])
            for g in self.guessed_pixels_index
        ]

        linfitter = LinearLSQFitter()
        wlmodel = Linear1D()
        self.linfit_wlmodel = linfitter(wlmodel, x=improved_xval_guesses, y=self.guessed_Angstroms)
        lamda_data = self.linfit_wlmodel(bin_centers)
        
        return lamda_data

    def plot_data(self, xedges_full, lamda_data, counts, data_type):
        """
        Plot the pixel-space and wavelength-calibrated data side by side, marking the peak lines.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Pixel space plot
        ax1.plot(xedges_full[:-1], counts, label=f"{data_type} Pixel Space Data")
        ax1.set_xlabel("Pixel Index")
        ax1.set_ylabel("Counts")
        ax1.set_title(f"{data_type.capitalize()} Data in Pixel Space")
        ax1.set_xlim(780, 850)
        
        # Wavelength space plot
        ax2.plot(lamda_data, counts, label=f"{data_type} Wavelength Data")
        ax2.set_xlabel("Wavelength (Angstroms)")
        ax2.set_ylabel("Counts")
        ax2.set_title(f"{data_type.capitalize()} Wavelength Data")
        ax2.set_xlim(1190, 1220)

        # Add red lines at the calibrated peaks
        for wl in self.guessed_Angstroms:
            ax2.axvline(wl, color='red', linestyle='--', label=f"Peak at {wl} Å")

        plt.legend()
        plt.tight_layout()
        plt.show()

    def export_wavelength_data(self, lamp_lamda, lamp_counts, cell_lamda, cell_counts, save_path):
        """
        Export both lamp and cell wavelength data to the same CSV file.
        """
        export_dict = {
            'Lamp Wavelength (Angstroms)': pd.Series(lamp_lamda),
            'Lamp Counts': pd.Series(lamp_counts)
        }

        if cell_lamda is not None and cell_counts is not None:
            export_dict.update({
                'Cell Wavelength (Angstroms)': pd.Series(cell_lamda),
                'Cell Counts': pd.Series(cell_counts)
            })

        df_export = pd.DataFrame(export_dict)
        df_export.to_csv(save_path, index=False)
        print(f"Lamp and Cell wavelength data saved to {save_path}")

    def analyze(self):
        """
        Main analysis routine for both lamp and cell data.
        """
        lamp_lamda, lamp_counts = None, None
        cell_lamda, cell_counts = None, None

        # Analyze lamp data
        if self.x_lamp_all is not None and self.y_lamp_all is not None:
            lamp_counts, xedges_full = self.preprocess_data(self.x_lamp_all, self.y_lamp_all)
            lamp_lamda = self.calibrate_wavelength(xedges_full, lamp_counts)

            # Plot pixel and wavelength data for lamp
            self.plot_data(xedges_full, lamp_lamda, lamp_counts, "lamp")

        # Analyze cell data
        if self.x_cell_all is not None and self.y_cell_all is not None:
            cell_counts, xedges_full = self.preprocess_data(self.x_cell_all, self.y_cell_all)
            cell_lamda = self.calibrate_wavelength(xedges_full, cell_counts)

            # Plot pixel and wavelength data for cell
            self.plot_data(xedges_full, cell_lamda, cell_counts, "cell")

        # Export the results for both lamp and cell
        self.export_wavelength_data(lamp_lamda, lamp_counts, cell_lamda, cell_counts, save_path)


# Define paths and identifiers
directory_date = '07_30_2024'
sub_directory = 'Clean'
date = '07_30_2024'
base_dir = os.path.join(directory_date, sub_directory)
identifier = '2.5V_la_ca'
data_path = os.path.join(base_dir, f'{date}_{identifier}_RANSAC_data.csv')
save_path = os.path.join(base_dir, f'{date}_{identifier}_Wavelength_data.csv')

# Initialize and run the processor
processor = WavelengthCalibration(data_path)
processor.analyze()
