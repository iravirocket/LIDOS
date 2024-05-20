#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:27:27 2024

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
import os


#date = datetime.now().strftime('%m_%d_%Y') #when plotting today's data
directory_date = '02_08_24'
identifier='cell_3.5A_2T_4'

# Constructing directory path
base_dir = os.path.join(directory_date)

# Make new directory for processed data
processed_dir = os.path.join(base_dir, 'Processed')
#os.makedirs(processed_dir)

# Form the base filename
base_filename = f'{directory_date}_{identifier}'

# Specific filenames uncomment to process only one file
#filename = os.path.join(base_dir, f'{base_filename}.csv')
#filenames = [filename]

# Uncomment the following line to process all CSV files in the directory
filenames = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.csv')]


#savefile = os.path.join(base_dir, f'{base_filename}_processed.csv')


# Create an empty DataF,,rame to store the time and 2-byte word
columns = ['Timestamp','x_raw', 'y_raw', 'xr', 'xscale', 'y', 'den', 'y1', 'y2']
df = pd.DataFrame(columns=columns)


def raw2ptsddlyyz(data, timestamps):
    k = 0
    cnt = 0
    many = len(data) // 3
    x = np.zeros(many, dtype=int)
    y1 = np.zeros(many, dtype=int)
    y2 = np.zeros(many, dtype=int)
    ts = np.zeros(many, dtype = timestamps.dtype)
    
    while k <= many * 3 - 3:
        y1t = (data[k] & 0x0003) == 0x0001
        y2t = (data[k+1] & 0x0003) == 0x0003
        xt = (data[k+2] & 0x0003) == 0x0002

        if xt and y1t and y2t:
            y1[cnt] = (data[k] >> 2) & 0x3fff
            y2[cnt] = (data[k+1] >> 2) & 0x3fff
            x[cnt] = (data[k+2] >> 2) & 0x3fff
            ts[cnt] = timestamps[k+2] #use k+2 to correctly match the timestamp to the x coordinate
            cnt += 1
            k += 3
        else:
            k += 1
    
    if cnt != 0:
        x = x[:cnt]
        y1 = y1[:cnt]
        y2 = y2[:cnt]
        ts = ts[:cnt]
    else:
            ts = []

   

    return x, y1, y2, ts

for filename in filenames:
    savefile = os.path.join(processed_dir,f'{os.path.splitext(os.path.basename(filename))[0]}_processed.csv' )
    # Read the CSV file
    data_df = pd.read_csv(filename,skiprows=1, header=None)
    #binary_data = data_df[1].apply(lambda x: int(x.strip("'"), 2)).values
    binary_data = np.array([int(row.strip("'"), 2) for row in data_df[1]])
    time_data = np.array([datetime.strptime(row, '%M:%S.%f') for row in data_df[0]])
    # Normalize timestamps to start at zero and convert to seconds
    # Subtract the first timestamp from all timestamps
    normalized_time = np.array([(t - time_data[0]).total_seconds() for t in time_data])
    
    #precise_time = [f"{t:.6g}" for t in normalized_time]
    # Call the raw2ptsddlyyz_py function
    x,y1, y2, ts = raw2ptsddlyyz(binary_data, normalized_time)
    ts_array = np.array(ts)
    scale=2
    dsize=1024*scale
    den = np.array(y1, dtype=np.int64) + np.array(y2, dtype=np.int64)
    nozero = np.where(den != 0)
    x, y1, y2, ts_array = [arr[nozero] for arr in [x, y1, y2, ts_array]]
    
    x_min, x_max = 0, 2048
    
    #xr=(x_raw/4)[::-1]
    x_new = ((x / 4) - 230 * scale)#[::-1]
    mask = (x_new >= x_min) & (x_new <= x_max)
    x, y1, y2, ts_array, x_new= [arr[mask] for arr in [x, y1, y2, ts_array, x_new]]
    den=y1+y2
    xr = x_max + x_min - x_new
    xscale = xr/(2.925714286)
    y = ((dsize * (4 / 3) * y1 )/ den - (dsize / 2) * (4 / 3) + 512 * scale)
    y_raw=y1/den         
    total_counts = len(x)
    
    
    timebin = 1
    
    bins = np.arange(ts_array.min(), ts_array.max() + timebin, timebin)
    counts, _ = np.histogram(ts_array, bins=bins) #, weights=data_to_use)
    
    # Convert counts to count rate (counts per second in this case)
    count_rate = counts / timebin
    avg_rate = np.mean(count_rate) if count_rate.size > 0 else 0
    
    df = pd.DataFrame(np.column_stack((ts_array, x, y_raw, xr, xscale,  y, den,y1, y2)), columns=columns)
    
    df.to_csv(savefile, header=True, float_format='%.2f', index=False)
    
    '''
    fig = plt.figure(figsize=(10, 15))
    
    ax1 = fig.add_subplot()
    ax1.scatter(xr, y, marker='.', linestyle='', s=1) #if using plot instead instead of 's', use 'markersize = 1'
    #ax.set_aspect(25 / 10)
    #ax1.set_ylim(500, 1500)
    #ax1.set_xlim(0, 2050)
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    plt.tick_params(length=25)
    # Customize the plot (optional)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y = Y2 - Y1")
    ax1.set_title(base_filename)
    
    plt.show
    '''
