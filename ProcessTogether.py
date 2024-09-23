#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:07:08 2024

@author: Ravi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec
import os

"""
Class that can process the lamp, cell and dark data in tandem or separately.  The pixel space data is turned into integer 
quantities to better reflect the reality of pixels on the detector.  The data is then filtered for only the regions on the 
detector that contain spectrum.  The y-axis is then turned into counts with zero counts filled in to x-pixel locations where 
there is no data.  This is all put into a csv that also contains the integration time.  Both 2D and 1D histogram plots
are generated for the data sets that are processed.

"""

class ProcessTogether:
    def __init__(self, df_lamp=None, df_cell=None, df_dark=None):
        """
        Initialize the class with optional dataframes for lamp, cell, and darks.
        """
        self.df_lamp = df_lamp
        self.df_cell = df_cell
        self.df_dark = df_dark

        self.filtered_lamp = None
        self.filtered_cell = None
        self.filtered_dark = None

        self.xr_lamp = None
        self.xr_cell = None
        self.xr_dark = None

        self.y_lamp = None
        self.y_cell = None
        self.y_dark = None
        
        self.full_x_range = np.arange(0, 2049)
        
    def read_data(self, lamp_path=None, cell_path=None, dark_path=None):
        """
        Read data from CSV files.
        """
        if lamp_path:
            self.df_lamp = pd.read_csv(lamp_path)
        if cell_path:
            self.df_cell = pd.read_csv(cell_path)
        if dark_path:
            self.df_dark = pd.read_csv(dark_path)


    def preprocess_data(self):
        """
        Preprocess the data, filter it, and calculate histograms.
        """
        
        y_cut_min, y_cut_max = 650, 1350
        xbins = 2048
        ybins = 100

        if self.df_lamp is not None:
            self.df_lamp['xr'] = np.round(self.df_lamp['xr']).astype(int)
            self.df_lamp['y'] = np.round(self.df_lamp['y']).astype(int)
            self.filtered_lamp = self.df_lamp[(self.df_lamp['y'] >= y_cut_min) & (self.df_lamp['y'] <= y_cut_max)]
            lamp_hist, xedges, yedges = np.histogram2d(self.filtered_lamp['xr'], self.filtered_lamp['y'], bins=(xbins, ybins))
            self.xr_lamp, self.y_lamp = self.full_x_range, np.histogram(self.filtered_lamp['xr'], bins=self.full_x_range)[0]
            lamp_hist_1d = np.sum(lamp_hist, axis=1)
        else:
            lamp_hist = np.zeros((xbins, ybins))
            lamp_hist_1d = np.zeros(xbins)


        if self.df_cell is not None:
            self.df_cell['xr'] = np.round(self.df_cell['xr']).astype(int)
            self.df_cell['y'] = np.round(self.df_cell['y']).astype(int)
            self.filtered_cell = self.df_cell[(self.df_cell['y'] >= y_cut_min) & (self.df_cell['y'] <= y_cut_max)]
            cell_hist, _, _ = np.histogram2d(self.filtered_cell['xr'], self.filtered_cell['y'], bins=(xbins, ybins))
            self.xr_cell, self.y_cell = self.full_x_range, np.histogram(self.filtered_cell['xr'], bins=self.full_x_range)[0]
            cell_hist_1d = np.sum(cell_hist, axis=1)
        else:
            cell_hist = np.zeros((xbins, ybins))
            cell_hist_1d = np.zeros(xbins)


        if self.df_dark is not None:
            self.df_dark['xr'] = np.round(self.df_dark['xr']).astype(int)
            self.df_dark['y'] = np.round(self.df_dark['y']).astype(int)
            self.filtered_dark = self.df_dark[(self.df_dark['y'] >= y_cut_min) & (self.df_dark['y'] <= y_cut_max)]
            dark_hist, _, _ = np.histogram2d(self.filtered_dark['xr'], self.filtered_dark['y'], bins=(xbins, ybins))
            self.xr_dark, self.y_dark = self.full_x_range, np.histogram(self.filtered_dark['xr'], bins=self.full_x_range)[0]
            dark_hist_1d = np.sum(dark_hist, axis=1)
        else:
            dark_hist = np.zeros((xbins, ybins))
            dark_hist_1d = np.zeros(xbins)


        return lamp_hist, cell_hist, dark_hist, xedges, yedges, lamp_hist_1d, cell_hist_1d, dark_hist_1d
    

    def plot_1d_histograms(self, lamp_hist_1d, cell_hist_1d, dark_hist_1d, xedges):
        """Plots 1D histograms for lamp, cell, and dark data."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        if self.df_lamp is not None:
            # Plot Lamp histogram
            axs[0].bar(xedges[:-1], lamp_hist_1d, width=np.diff(xedges), edgecolor='purple', align='edge')
            axs[0].set_ylabel('Lamp Counts')
            axs[0].set_title('1D Histogram of Lamp Data')
        
        if self.df_cell is not None:
            # Plot Cell histogram
            axs[1].bar(xedges[:-1], cell_hist_1d, width=np.diff(xedges), edgecolor='purple', align='edge')
            axs[1].set_ylabel('Cell Counts')
            axs[1].set_title('1D Histogram of Cell Data')
        if self.df_dark is not None:
            
            # Plot Dark histogram
            axs[2].bar(xedges[:-1], dark_hist_1d, width=np.diff(xedges), edgecolor='purple', align='edge')
            axs[2].set_ylabel('Dark Counts')
            axs[2].set_xlabel('Pixel Position')
            axs[2].set_title('1D Histogram of Dark Data')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def plot_histograms(self, lamp_hist, cell_hist, dark_hist, xedges, yedges):
        """
        Plot 2D histograms for lamp, cell, and darks data.
        """
        fig = plt.figure(figsize=(10, 15))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plasma_mod = plt.cm.plasma.copy()
        plasma_mod.set_under('white')
        
        if self.df_lamp is not None:
            ax_lamp = fig.add_subplot(gs[0, 0])
            norm_lamp = Normalize(vmin=0.01, vmax=lamp_hist.max())
            im_lamp = ax_lamp.imshow(lamp_hist.T, interpolation='nearest', origin='lower', cmap=plasma_mod, norm=norm_lamp, aspect='auto', extent=extent)
            ax_lamp.set_aspect('equal')
            ax_lamp.set_ylim(550, 1400)
            ax_lamp.set_title('Lamp', fontsize=8)
        
        if self.df_cell is not None:
            ax_cell = fig.add_subplot(gs[1, 0])
            norm_cell = Normalize(vmin=0.01, vmax=cell_hist.max())
            im_cell = ax_cell.imshow(cell_hist.T, interpolation='nearest', origin='lower', cmap=plasma_mod, norm=norm_cell, aspect='auto', extent=extent)
            ax_cell.set_aspect('equal')
            ax_cell.set_ylim(550, 1400)
            ax_cell.set_title('Cell', fontsize=8)
        if self.df_dark is not None:
            ax_dark = fig.add_subplot(gs[2, 0])
            norm_dark = Normalize(vmin=0.01, vmax=dark_hist.max())
            im_dark = ax_dark.imshow(dark_hist.T, interpolation='nearest', origin='lower', cmap=plasma_mod, norm=norm_dark, aspect='auto', extent=extent)
            ax_dark.set_aspect('equal')
            ax_dark.set_ylim(550, 1400)
            ax_dark.set_title('Darks', fontsize=8)

        cax = fig.add_subplot(gs[:, 1])
        fig.colorbar(im_lamp, cax=cax).set_label('Counts', rotation=270, labelpad=10)

        plt.tight_layout()
        plt.show()

    def save_processed_data(self, directory_date, identifier, timestamp_col):
        """
        Save the processed data to a CSV file.
        """
        base_dir = os.path.join(directory_date, 'Clean')
        os.makedirs(base_dir, exist_ok=True)
        savefile = os.path.join(base_dir, f'{directory_date}_{identifier}_clean.csv')
           # Initialize an empty dictionary to hold the data
        data_dict = {}
    
        # Add integration time if lamp data is present
        if self.df_lamp is not None:
            data_dict['Integration_Time'] = pd.Series(np.round(max(self.df_lamp[timestamp_col].values - 1)))
        
        # Add lamp data if available
        if self.filtered_lamp is not None:
            data_dict['lamp_x_filtered'] = pd.Series(self.filtered_lamp['xr'].values)
            data_dict['lamp_y_filtered'] = pd.Series(self.filtered_lamp['y'].values)
            data_dict['lamp_counts'] = pd.Series(self.y_lamp)
    
        # Add cell data if available
        if self.filtered_cell is not None:
            data_dict['cell_x_filtered'] = pd.Series(self.filtered_cell['xr'].values)
            data_dict['cell_y_filtered'] = pd.Series(self.filtered_cell['y'].values)
            data_dict['cell_counts'] = pd.Series(self.y_cell)
    
        # Add dark data if available
        if self.filtered_dark is not None:
            data_dict['dark_x_filtered'] = pd.Series(self.filtered_dark['xr'].values)
            data_dict['dark_y_filtered'] = pd.Series(self.filtered_dark['y'].values)
            data_dict['darkf_counts'] = pd.Series(self.y_dark)
        
        # Add the full x_pixel range, which is always present
        data_dict['x_pixel'] = pd.Series(self.full_x_range)
    
        # Create DataFrame only with the available columns
        df_combined = pd.DataFrame(data_dict)
    
        # Save to CSV
        df_combined.to_csv(savefile, index=False)
        
        
    def run(self, directory_date, identifier, timestamp_col='Timestamp'):
        """
        Run the entire processing routine: preprocess data, plot histograms, and save the processed data.
        """
        
        lamp_hist, cell_hist, dark_hist, xedges, yedges, lamp_hist_1d, cell_hist_1d, dark_hist_1d = self.preprocess_data()
        self.plot_histograms(lamp_hist, cell_hist, dark_hist, xedges, yedges)
        # Plot the 1D histograms
        self.plot_1d_histograms(lamp_hist_1d, cell_hist_1d, dark_hist_1d, xedges)
        self.save_processed_data(directory_date, identifier, timestamp_col)
        
# Define paths and identifiers
directory_date = '07_30_2024'
sub_directory = 'Processed'
date = '07_30_2024'
base_dir = os.path.join(directory_date, sub_directory)
voltage = '2.5V'
lmark = 'a'
cmark = 'a'
# Define identifiers
#id_lamp = 'lamp_3V_a6'
id_lamp = f'lamp_{voltage}_{lmark}'
id_filament_dark = 'dark_30sec'
#id_cell = 'cell_3V_a5'
id_cell = f'cell_{voltage}_{cmark}'

# Construct file paths
lamp_filename = os.path.join(base_dir, f'{date}_{id_lamp}_processed.csv')
dark_filename = os.path.join(base_dir, f'{date}_{id_filament_dark}_processed.csv')
cell_filename = os.path.join(base_dir, f'{date}_{id_cell}_processed.csv')

# Initialize and run the processor
processor = ProcessTogether()
processor.read_data(lamp_path=lamp_filename, cell_path=cell_filename, dark_path=dark_filename)
processor.run(directory_date=directory_date, identifier=f'{voltage}_l{lmark}_c{cmark}')

