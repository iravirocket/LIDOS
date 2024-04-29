#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:48:26 2024

@author: Ravi
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import astropy
from astropy.modeling import models, fitting
import pylab as pl
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Gaussian1D, Linear1D
from astropy.modeling.fitting import LinearLSQFitter

#date = datetime.now().strftime('%m_%d_%Y') #when plotting today's data
date = '04_24_2024' #to type in past date
id_lamp='lamp_10V_3T'
#id_dark = 'dark_6A_3T'
id_filament_dark = 'dark_9V_3T'
id_cell = 'cell_10V_3T'
# Base filenam
#lamp_filename = '03_11_24_lamp1_processed.csv'
lamp_filename = f'{date}_{id_lamp}_processed.csv'
#dark_filename = f'{date}_{id_dark}_processed.csv'
filament_dark_filename = f'{date}_{id_filament_dark}_processed.csv'
cell_filename = f'{date}_{id_cell}_processed.csv'
# Specific filenames

df_lamp = pd.read_csv(lamp_filename)
#df_dark = pd.read_csv(dark_filename)
df_darkf = pd.read_csv(filament_dark_filename)
df_cell = pd.read_csv(cell_filename)
#savefile for wavelength scale data
columns = ['wavelength', 'Source (counts)', 'wavelength', 'Cell (counts)', 'wavelength', 'Filament (counts)']
savefile = f'{date}_scaled_spectra.csv' 

xr_lamp=df_lamp['xr']
y_lamp = df_lamp['y']
#xr_dark = df_dark['xr']
#y_dark = df_dark['y']
xr_darkf = df_darkf['xr']
y_darkf = df_darkf['y']
xr_cell = df_cell['xr']
y_cell = df_cell['y']

integration_time = 10 #in seconds
bandpass = 1650 - 900 #Angstroms

#Cut down the spectra in y 
# After calculating y, y_raw, etc.
y_cut_min, y_cut_max = 660, 1300
filtered_lamp = df_lamp[(df_lamp['y'] >= y_cut_min) & (df_lamp['y'] <= y_cut_max)]
#filtered_dark = df_dark[(df_dark['y'] >= y_cut_min) & (df_dark['y'] <= y_cut_max)]
filtered_darkf = df_darkf[(df_darkf['y'] >= y_cut_min) & (df_darkf['y'] <= y_cut_max)]
filtered_cell = df_cell[(df_cell['y'] >= y_cut_min) & (df_cell['y'] <= y_cut_max)]
# Now, if you need to apply similar filtering to df_dark based on xr_lamp, ensure it's done based on a common key or parameter.

# Grouping and summing lamp counts after filtering
lamp_counts = filtered_lamp.groupby('xr')['y'].sum().reset_index()
#dark_counts = filtered_dark.groupby('xr')['y'].sum().reset_index()
darkf_counts = filtered_darkf.groupby('xr')['y'].sum().reset_index()
cell_counts = filtered_cell.groupby('xr')['y'].sum().reset_index()

lamp_flux = lamp_counts/(integration_time * bandpass)
cell_flux = cell_counts/(integration_time * bandpass)

# If you have dark counts prepared similarly, calculate the average
# Assuming dark_counts DataFrame is prepared correctly
avg_dark = df_darkf['y'].mean() 

# Prepare final xr_lamp and y_lamp for further calculations or plotting
xr_lamp = lamp_counts['xr'].values
y_lamp = lamp_counts['y'].values# - avg_dark
#xr_dark = dark_counts['xr'].values
#y_dark = dark_counts['y'].values
xr_darkf = darkf_counts['xr'].values
y_darkf = darkf_counts['y'].values
xr_cell = cell_counts['xr'].values
y_cell = cell_counts['y'].values

guessed_Angstroms = [1215.67, 1304, 1334]
guessed_pixels = [807, 1058, 1150.5]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
ax1.plot(xr_lamp, y_lamp,ls='-', markersize=.5, label='Source')
ax1.plot(xr_cell, y_cell,ls='-', markersize=.5, label='Cell')
#plt.plot(xr_dark, y_dark,ls='-', markersize=.5, label='Data')
#plt.plot(xr_darkf, y_darkf,ls='-', markersize=.5, label='Data')
ax1.plot(guessed_pixels, [2600]*3,'x', label='Identified Peaks')
ax1.set_xlabel('Pixels')
ax1.set_ylabel('Counts')
ax1.set_title('9 Amps, 3 Torr')
ax1.legend()
#plt.show()

linfitter = LinearLSQFitter()
wlmodel = Linear1D()
linfit_wlmodel = linfitter(model=wlmodel, x=guessed_pixels, y=guessed_Angstroms)
lamda_lamp = linfit_wlmodel(xr_lamp)
lamda_cell = linfit_wlmodel(xr_cell)
lamda_darkf = linfit_wlmodel(xr_darkf)

print(linfit_wlmodel)

#plt.figure(figsize=(10, 6))
ax2.plot(lamda_lamp, y_lamp,ls='-', markersize=.5, label='Source')
ax2.plot(lamda_cell, y_cell,ls='-', markersize=.5, label='Cell')

ax2.plot(guessed_Angstroms, [2600]*3,'x', label='Identified Peaks')
ax2.set_xlabel('Angstroms')
ax2.set_ylabel('Counts')
#ax2.set_title('Attenuation at 9  amps')
ax2.legend()
#plt.show()

#plt.figure(figsize=(10, 6))
ax3.plot(lamda_lamp, y_lamp,ls='-', markersize=.5, label='Source')
ax3.plot(lamda_cell, y_cell, ls='-', ms = .5, label='Cell')
#plt.plot(lamda_darkf, y_darkf, ls='-', ms=.5, label='Filament') #Only if lines are seen from darkf
ax3.plot(guessed_Angstroms, [2600]*3,'x', label='Identified Peaks')
ax3.set_xlabel('Angstroms')
ax3.set_ylabel('Counts')
#ax3.set_title('Lyman Alpha Comparison')
ax3.set_xlim(1214, 1218)
ax3.legend()
#plt.show()

#plt.figure()
#plt.plot(guessed_pixels, guessed_Angstroms, 'o')
#plt.plot(xr_lamp, lamda_lamp, '-')
#plt.xlabel('x(pixels)')
#plt.show()


#df = pd.DataFrame(np.column_stack((lamda_lamp, y_lamp, lamda_cell, y_cell, lamda_darkf, y_darkf)), columns=columns)
#df.to_csv(savefile, header=True, float_format='%.2f', index=False)




