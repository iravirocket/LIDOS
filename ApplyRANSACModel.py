#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:55:07 2024

@author: Ravi
"""
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

class ApplyRANSACModel:
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

    def save_ransac_model(self):
        """Saves the RANSAC model to a file."""
        if self.ransac and self.ransac_model_path:
            with open(self.ransac_model_path, 'wb') as f:
                pickle.dump(self.ransac, f)
            print(f"RANSAC model saved to {self.ransac_model_path}")
    
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
    '''
    def fit_with_ransac(self, x_filtered, y_filtered):
        # Reshape data for RANSAC
        y_filtered_reshaped = y_filtered.reshape(-1, 1)

        # Apply RANSAC regression model if no model exists
        if self.ransac is None:
            self.ransac = RANSACRegressor(LinearRegression(), min_samples=100)
            self.ransac.fit(y_filtered_reshaped, x_filtered)
            self.save_ransac_model()  # Save model after fitting

        # Predict the fitted line
        x_fitted = self.ransac.predict(y_filtered_reshaped)
        slope = self.ransac.estimator_.coef_[0]
        intercept = self.ransac.estimator_.intercept_
        print(f"RANSAC Fit Line: y = {slope} * x + {intercept}")

        return self.ransac, x_fitted
        '''
     
    def correct_tilt_with_ransac(self, x, y, x_filtered, y_filtered, label, plot_fitted_line=True):
        y=np.array(y)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        y_valid = y[valid_mask].reshape(-1, 1)
        x_valid = x[valid_mask]
        
        y_filtered = np.array(y_filtered).reshape(-1,1)
        
        # Apply RANSAC correction
        x_corrected_all = x_valid - (self.ransac.predict(y_valid) - self.ransac.estimator_.intercept_)
        x_corrected_filtered = x_filtered - (self.ransac.predict(y_filtered) - self.ransac.estimator_.intercept_)

        fig, (axhor, axorig) = plt.subplots(1,2)
        # Plot the corrected data
        axhor.plot(y_valid, x_corrected_all, 'x', label=f"Corrected {label} Data", alpha=0.5)
        axhor.set_ylabel("X Data (Corrected)")
        axhor.set_xlabel("Y Data")
        axhor.set_title(f"RANSAC Corrected Tilt ({label})")
        #axhor.set_ylim(700, 900)
        
        axorig.plot(y, x, 'x', label=f"Original {label} Data", alpha = 0.5)
        axorig.set_title("Original Line")
        #axorig.set_ylim(700,900)
        # Optionally plot the fitted line
        x_fitted = self.ransac.predict(y_valid)
        axorig.plot(y_valid, x_fitted, color='r', linestyle='-', linewidth=2, label="Fitted RANSAC Line")
        plt.legend()
    
        
        '''
        axvert.plot(x_corrected, y_reshaped, 'x', label="Corrected Data", alpha=0.5)
        axvert.set_ylabel("Y data")
        axvert.set_xlabel("X Data (Corrected)")
        #axvert.set_title("RANSAC Corrected Tilt")
        axvert.set_xlim(700,900)
        '''
        plt.show()   
        
        
        fig, (ax_filt_fit, ax_filt_og) = plt.subplots(1,2)
        ax_filt_fit.plot(y_filtered, x_corrected_filtered, 'x', label=f"Corrected {label} Data", alpha=0.5)
        ax_filt_fit.set_ylabel("X Data (Corrected")
        ax_filt_fit.set_xlabel("Y Data")
        ax_filt_fit.set_title(f"RANSAC Corrected tilt ({label})")
        
        ax_filt_og.plot(y_filtered, x_filtered, 'x', label = f"Original {label} Data", alpha = 0.5)
        ax_filt_og.set_ylabel("X Data (Corrected")
        ax_filt_og.set_xlabel("Y Data")
        ax_filt_og.set_title(f"RANSAC Corrected tilt ({label})")
        plt.show()
        
        return x_corrected_all, x_corrected_filtered

    def analyze(self):
        """
        Analyze the data for lamp and cell with optional dark subtraction.
        """
        for data_type in ['lamp', 'cell']:
            x_data = self.x_lamp if data_type == 'lamp' else self.x_cell
            y_data = self.y_lamp if data_type == 'lamp' else self.y_cell

            if x_data is None or y_data is None:
                print(f"No {data_type} data available, skipping...")
                continue

            # Filter and correct data
            x_filtered, y_filtered = self.filter_data(x_data, y_data)
            #if self.ransac is None:
             #   self.fit_with_ransac(x_filtered, y_filtered)
            
            x_corrected_all, x_corrected_filtered = self.correct_tilt_with_ransac(x_data, y_data, x_filtered, y_filtered,  data_type.capitalize())

            # Save corrected data to CSV
            #self.save_all_corrected_data(x_corrected_all, y_data, x_filtered, y_filtered, data_type)
            self.save_all_corrected_data(x_corrected_all, y_data, x_corrected_filtered, y_filtered, data_type)

    def save_all_corrected_data(self, x_corrected_all, y_data, x_corrected_filtered, y_filtered, data_type):
        if data_type == 'lamp':
            self.df_data['x_corrected_lamp'] = pd.Series(x_corrected_all)
            self.df_data['y_lamp'] = pd.Series(y_data)
            self.df_data['corrected_x_filtered_lamp'] = pd.Series(x_corrected_filtered)
            self.df_data['y_filtered_lamp'] = pd.Series(y_filtered)
        elif data_type == 'cell':
            self.df_data['x_corrected_cell'] = pd.Series(x_corrected_all)
            self.df_data['y_cell'] = pd.Series(y_data)
            self.df_data['corrected_x_filtered_cell'] = pd.Series(x_corrected_filtered)
            self.df_data['y_filtered_cell'] = pd.Series(y_filtered)

        # Save the updated DataFrame to CSV
        save_path = os.path.join(base_dir, f'{directory_date}_{identifier}_RANSAC_data.csv')
        self.df_data.to_csv(save_path, index=False)
        print(f"Corrected data (both full and filtered) saved to {save_path}")


# Example Usage
directory_date = '08_01_2024'
identifier = '2.25V_la6_ca5'
base_dir = os.path.join(directory_date, 'Clean')
filename = os.path.join(base_dir, f'{directory_date}_{identifier}_clean.csv')
ransac_model_path = 'ransac_model.pkl'

# Initialize and run the TraceLymanalpha class
analyzer = ApplyRANSACModel(filename, ransac_model_path=ransac_model_path, force_refit=False)
analyzer.analyze()
