#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:16:09 2024

@author: Ravi
"""
import datetime
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import time
from matplotlib.colors import Normalize, Colormap

#date = datetime.now().strftime('%m_%d_%Y') #when plotting today's data
date = '03_11_24' #to type in past date
id_lamp='lamp1'
id_dark = 'dark2'

lamp_filename = f'{date}_{id_lamp}_processed.csv'
dark_filename = f'{date}_{id_dark}_processed.csv'
plotname = f'{date}_spectra.pdf'

df_lamp = pd.read_csv(lamp_filename)
df_dark = pd.read_csv(dark_filename)
timebin = 1
bins = np.arange(df_lamp['Timestamp'].min(), df_lamp['Timestamp'].max() + timebin, timebin)
counts, _ = np.histogram(df_lamp['Timestamp'], bins=bins) #, weights=data_to_use)

# Convert counts to count rate (counts per second in this case)
count_rate = counts / timebin
avg_rate = np.mean(count_rate) if count_rate.size > 0 else 0

# Here you can add the count rate to your CSV or return it for further processing
print(f"Average count rate: {avg_rate} counts/sec")

xbins= 500
ybins= 100 # Bins adjusted to pixel size of detector
'''x_filtered = df_filtered['x_filtered']
y_filtered = df_filtered['y_filtered']
x=df_unfiltered['x']
y=df_unfiltered['y']'''
hist, xedges, yedges = np.histogram2d(df_lamp['xr'], df_lamp['y'],bins=(xbins, ybins))

# Compute the 1D histogram for x-values
x_hist, x_bins = np.histogram(df_lamp['xr'], bins=xbins)  

#with PdfPages(plotname) as pdf:
    
# Initialize figures and axes outside the loop
#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
# Create a figure and set up GridSpec
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1, 1])

# Create the subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])  
ax4 = fig.add_subplot(gs[3, 0])
tick_interval = 25                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
# Plot the data with the specified aspect ratio
#fig, ax1 = plt.subplots(figsize=(25, 10))
ax1.scatter(df_lamp['xr'], df_lamp['y_raw'], marker='.', linestyle='', s=1) #if using plot instead instead of 's', use 'markersize = 1'
#ax.set_aspect(25 / 10)
ax1.set_ylim(.3, .7)
#ax1.set_xlim(0, 2050)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=25)
plt.tick_params(length=25)
# Customize the plot (optional)
ax1.set_xlabel("X")
ax1.set_ylabel("Y = Y2 - Y1")
ax1.set_title(lamp_filename)
#ax1.set_xticks(range(950, 1650, 25))
# Optional: Set custom tick labels (e.g., empty string for no label)
#ax1.set_xticklabels(['' if (x % 50 != 0) else str(x) for x in range(950, 1650, tick_interval)])



 # 1D Projection

ax3.bar(x_bins[:-1], x_hist, width=np.diff(x_bins)[0], align='edge')
#ax3.set_xlim(0,2050)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
#ax3.set_title('1D Projection along X-axis', fontsize=10)
ax3.tick_params(axis='x', labelsize=9)
ax3.tick_params(axis='y', labelsize=10)
ax3.tick_params(axis='both', length=5)

#ax3.set_xticks(range(950, 1650, tick_interval))
# Optional: Set custom tick labels (e.g., empty string for no label)
#ax3.set_xticklabels(['' if (x % 50 != 0) else str(x) for x in range(950, 1650, tick_interval)])

#plt.pause(0.5)

# 2D Histogram


cax = fig.add_subplot(gs[1, 1])
# Create a modified colormap from 'plasma'
plasma_mod = plt.cm.plasma.copy()
plasma_mod.set_under('white')  # Set the color for zero counts

# Ensure that bins with zero counts are colored white
norm = Normalize(vmin=0.01, vmax=hist.max())  # Normalization
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax2.imshow(hist.T, interpolation='nearest', origin='lower', cmap=plasma_mod, norm = norm,aspect='auto', extent=extent)

cbar = fig.colorbar(im, cax=cax)
   
cbar.set_label('Counts', rotation=270, labelpad=10)

#ax2.set_xlim(0, 2050)
ax2.set_ylim(500, 1800)
ax2.tick_params(axis='x', labelsize=9)
ax2.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='both', length=5)
#ax2.set_xticks(range(950, 1650, 25))
#ax2.set_xticklabels(['' if (x % 50 != 0) else str(x) for x in range(950, 1650, tick_interval)])

#count rate histogram
#if bins[:-1].size > 0 and count_rate.size > 0:
ax4.bar(bins[:-1], count_rate, width=1)#, align='edge')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Count Rate (Hz)')
#ax4.title('Histogram of Count Rate over Time')
ax4.tick_params(axis='x', labelsize =10)
ax4.tick_params(axis='y', labelsize=10)
ax4.tick_params(axis='both', length=5)
ax4.set_xlim(left=0)
ax4.set_ylim(bottom= 0, top = 30)
#else:
#print("Empty bins or count_rate, skipping plot.")
 
# Display the plot
plt.show()
#pdf.savefig(fig)