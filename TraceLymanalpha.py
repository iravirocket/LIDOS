#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:07:38 2024

@author: Ravi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pylab as pl
pl.rcParams['image.origin'] = 'lower'
pl.matplotlib.style.use('dark_background') # Optional!

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

import pickle

# Set the display options to show all rows and columns
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)




class TraceLymanalpha:
    def __init__(self, data_path, ransac_model_path=None, force_refit=False):
        self.df_data = pd.read_csv(data_path)
        self.x_lamp = self.df_data.get('lamp_x_filtered', None)
        self.y_lamp = self.df_data.get('lamp_y_filtered', None)
        self.x_cell = self.df_data.get('cell_x_filtered', None)
        self.y_cell = self.df_data.get('cell_y_filtered', None)
        self.x_dark = self.df_data.get('dark_x_filtered', None)
        self.y_dark = self.df_data.get('dark_y_filtered', None)

        self.ransac_model_path = ransac_model_path

        # Load the RANSAC model if a path is provided
        if ransac_model_path and not force_refit:
            try:
                with open(ransac_model_path, 'rb') as f:
                    self.ransac = pickle.load(f)
                print(f"Loaded RANSAC model from {ransac_model_path}")
            except FileNotFoundError:
                print(f"RANSAC model file {ransac_model_path} not found. Proceeding without loading model.")
                self.ransac = None
        else:
            self.ransac = None  # Initialize as None if no model is loaded
            
    
        
    def filter_data(self, x, y):
        # Filter out non-finite values
        mask_pixels_y = (x < 822) | (x > 842)
        mask_pixels_x = (y < 700) | (y > 1300)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x_filtered = np.array(x[finite_mask & ~mask_pixels_y & ~mask_pixels_x])
        y_filtered = np.array(y[finite_mask & ~mask_pixels_y & ~mask_pixels_x])
        
        # Ensure there are no NaN values in filtered data
        valid_mask = ~np.isnan(x_filtered) & ~np.isnan(y_filtered)
        x_filtered = x_filtered[valid_mask]
        y_filtered = y_filtered[valid_mask]


        if len(x_filtered) < 2 or len(y_filtered) < 2:
            raise ValueError("Not enough data points to fit a model")
    
            
        return x_filtered, y_filtered

    def initial_plot(self, x, y):

        fig = plt.figure()
        plt.plot(y, x, 'x')
        plt.ylabel('X Position')
        plt.xlabel('Trace Data')
        plt.ylim(800, 855)

        plt.show()

        
        fig = plt.figure()
        plt.plot(y, x, 'x')
        plt.ylim(800, 855)

        plt.show()
    
     
    def fit_with_ransac(self, x_filtered, y_filtered):
        
        # Reshape data for RANSAC
        y = np.array(y_filtered)
        y_filtered_reshaped = y_filtered.reshape(-1, 1)
        
        # Apply RANSAC regression model if no model exists
        if self.ransac is None:
            self.ransac = RANSACRegressor(LinearRegression(), min_samples = 100) #, residual_threshold = 2 )
            self.ransac.fit(y_filtered_reshaped, x_filtered)
            
        x_fitted = self.ransac.predict(y_filtered_reshaped)
    
        # Get the slope and intercept of the RANSAC fit
        slope = self.ransac.estimator_.coef_[0]  # Slope of the fitted line
        intercept = self.ransac.estimator_.intercept_  # Intercept of the fitted line
        
        # Print the fit line parameters
        print(f"RANSAC Fit Line: y = {slope} * x + {intercept}")
        
    
        # Plot the result
        plt.figure()
        plt.plot(y_filtered, x_filtered, 'x', alpha=0.5)
        plt.plot(y_filtered, x_fitted, color='r', label="RANSAC Fit")
        plt.ylabel("X Data")
        plt.xlabel("Y Data")
        plt.title("Robust RANSAC Fitted Line")
        plt.legend()
        
       
        return self.ransac, x_fitted
    
    def apply_ransac(self, x, y):
        y = np.array(y)
        y_reshaped = y.reshape(-1, 1)
        x_corrected = x - (self.ransac.predict(y_reshaped) - self.ransac.estimator_.intercept_)
        return x_corrected

        
    def correct_tilt_with_ransac(self, ransac, x, y, title):
        '''
        # Correct the tilt using the fitted RANSAC model
        y=np.array(y)
        y_reshaped = y.reshape(-1, 1)
        x_corrected = x - (ransac.predict(y_reshaped) - ransac.estimator_.intercept_)
        '''
        x_corrected = self.apply_ransac(x,y)
        
        fig, (axhor, axorig) = plt.subplots(1,2)
        # Plot the corrected data
        axhor.plot(y, x_corrected, 'x', label="Corrected Data", alpha=0.5)
        axhor.set_ylabel("X Data (Corrected)")
        axhor.set_xlabel("Y Data")
        axhor.set_title("RANSAC Corrected Tilt")
        axhor.set_ylim(700, 900)
        
        axorig.plot(y, x, 'x', label="Original Data", alpha = 0.5)
        axorig.set_title("Original Line")
        axorig.set_ylim(700,900)
        '''
        axvert.plot(x_corrected, y_reshaped, 'x', label="Corrected Data", alpha=0.5)
        axvert.set_ylabel("Y data")
        axvert.set_xlabel("X Data (Corrected)")
        #axvert.set_title("RANSAC Corrected Tilt")
        axvert.set_xlim(700,900)
        '''
        plt.show()
        
        print(f"Reshaped Y :  {y}")
        print(f"Corrected X:  {x_corrected}")
        
        
        return x_corrected

    
    def apply_all(self, ransac, x_data, y_data):
        # Apply the fitted model to the entire dataset for correction
        
        x_corrected_all = self.correct_tilt_with_ransac(ransac, x_data, y_data, "RANSAC Corrected Full Data")
        # Plot the entire corrected data for inspection before saving
        fig, (ax_new, ax_old, ax_together) = plt.subplots(3, 1, figsize=(12, 6))
        ax_new.plot(x_corrected_all, y_data, '.', label="Corrected")
        ax_new.set_title("Corrected Data (Full Dataset)")
        ax_old.plot(x_data, y_data, '.', label="Original")
        ax_old.set_title("Original Data")
        ax_together.plot(x_corrected_all,  y_data, 'x', label = "Corrected")
        ax_together.plot(x_data, y_data, 'x', label = "Original", alpha=0.1)
        ax_together.legend()
        ax_together.set_title("Data together")
        plt.show()
        
        
        # Plot the entire corrected data for inspection before saving
        fig, (ax_new, ax_old, ax_together) = plt.subplots(1, 3, figsize=(12, 6))
        ax_new.plot(x_corrected_all, y_data, '.', label="Corrected")
        ax_new.set_title("Corrected Data (Full Dataset)")
        ax_new.set_xlim(700, 900)
        ax_old.plot(x_data, y_data, '.', label="Original")
        ax_old.set_title("Original Data")
        ax_old.set_xlim(700, 900)
        ax_together.plot(x_corrected_all,  y_data, 'x', label = "Corrected")
        ax_together.plot(x_data, y_data, 'x', label = "Original", alpha=0.2)
        ax_together.legend()
        ax_together.set_xlim(700,900)
        ax_together.set_title("Data together")
        plt.show()
        
        
        return x_corrected_all
   
        
    def save_all_corrected_data(self, x_corrected_all, y_data, x_corrected_filtered, y_filtered, data_type):
       
        # Add corrected full dataset columns
        if data_type == 'lamp':
            self.df_data['x_corrected_lamp'] = pd.Series(x_corrected_all)
            self.df_data['y_lamp'] = y_data
            self.df_data['corrected_x_filtered_lamp'] = pd.Series(x_corrected_filtered)
            self.df_data['y_filtered_lamp'] = pd.Series(y_filtered)
        elif data_type == 'cell':
            self.df_data['x_corrected_cell'] = pd.Sereis(x_corrected_all)
            self.df_data['y_cell'] = y_data
            self.df_data['corrected_x_filtered_cell'] = pd.Series(x_corrected_filtered)
            self.df_data['y_filtered_cell'] = pd.Series(y_filtered)
    
        # Save the updated DataFrame to the same CSV
        save_path = os.path.join(base_dir, f'{directory_date}_{identifier}_RANSAC_data.csv')
        self.df_data.to_csv(save_path, index=False)
        print(f"Corrected data (both full and filtered) saved to {save_path}")


    def analyze(self, data_type='lamp'):
        """
        Analyze the data for lamp or cell with optional dark subtraction.
        """
        x_data = self.x_lamp if data_type == 'lamp' else self.x_cell
        y_data = self.y_lamp if data_type == 'lamp' else self.y_cell
        y_dark = self.y_dark if data_type == 'lamp' else None
        '''
        if y_dark is not None:
            y_data = self.subtract_dark(y_data, y_dark)
            
        '''
        
        x_filtered, y_filtered = self.filter_data(x_data, y_data)
        #Uncomment if you want to plot the initial data before processing
        #self.initial_plot(x_data, y_data)
        '''
        print(f"x_filtered has NaNs: {np.isnan(x_filtered).any()}")
        print(f"y_filtered has NaNs: {np.isnan(y_filtered).any()}")
        
        print(f"x_data has NaNs: {np.isnan(x_data).any()}")
        print(f"y_data has NaNs: {np.isnan(y_data).any()}")
        '''

        # Fit with RANSAC
        #ransac, x_fitted = self.fit_with_ransac(x_filtered, y_filtered)
        
        # Fit with RANSAC if model isn't already loaded
        if self.ransac is None:
            ransac, x_fitted = self.fit_with_ransac(x_filtered, y_filtered)
        else:
            ransac = self.ransac

        
        # Correct the entire dataset
        x_corrected = self.correct_tilt_with_ransac(ransac, x_filtered, y_filtered, "RANSAC Residuals")
        x_corrected_all = self.apply_all(ransac, x_data, y_data)
        
        # Save corrections (full dataset and filtered data) to the same CSV
        self.save_all_corrected_data(x_corrected_all, y_data, x_corrected, y_filtered, 'lamp')
        
        # If satisfied, save corrections to full data set, comment during analysis
        #self.save_corrected_data(x_corrected_all, 'lamp')
        #self.save_filtered_corrected_data(x_corrected, y_filtered, 'lamp')


# Example Usage
directory_date = '05_09_2024'
identifier = 'lamp_test8'
base_dir = os.path.join(directory_date, 'Clean')
filename = os.path.join(base_dir, f'{directory_date}_{identifier}_clean.csv')
ransac_model_path = 'ransac_model.pkl'

# If you want to re-run the model, just set force_refit=True
analyzer = TraceLymanalpha(filename, ransac_model_path=ransac_model_path, force_refit=False)

# Initialize and run the TraceLymanalpha class
analyzer.analyze(data_type='lamp') #, save_path = filename)  # Analyze lamp data
#analyzer.analyze(data_type='cell')  # Analyze cell data (if available)
# Analyze lamp data within a specific x-range
#analyzer.analyze(data_type='lamp', x_min=810, x_max=845)  


