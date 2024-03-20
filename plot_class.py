#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:40:36 2024

@author: Ravi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from datetime import datetime

date = datetime.now().strftime('%m_%d_%Y')
identifier='test_acquire_plot'
# Base filenam
base_filename = f'{date}_{identifier}'
filename = f'{base_filename}.csv'
processed_filename = f'{base_filename}_processed.csv'

class DataProcessorAndPlotter:
    def __init__(self, processed_filename):
        self.processed_filename = processed_filename
        self.fig, self.ax = plt.subplots(3, 1, figsize=(25, 10))
        self.stop_plotting = threading.Event()
        self.processed_data = pd.DataFrame()
       
        self.ani = None
        self.cbar = None
        self.heatmap=None
        self.scatter=self.ax[0].scatter([],[])
        self.hist1d=None
        self.init_plot()
   
        
    def stop_listener(self):
        input("Press 'Enter' to stop plotting and save the figure...")
        self.stop_plotting.set()
        
    def raw2coords(self, binary_words):
         # Initialize lists to hold the computed values
         processed_data = []
         data = [int(word.strip("'"), 2) for word in binary_words]
         scale = 2
         dsize = 1024 * scale
     
         k = 0
         cnt = 0
         many = len(data) // 3
         x = np.zeros(many, dtype=int)
         y1 = np.zeros(many, dtype=int)
         y2 = np.zeros(many, dtype=int)
     
         # Processing binary data
         while k <= many * 3 - 3:
             y1t = (data[k] & 0x0003) == 0x0001
             y2t = (data[k + 1] & 0x0003) == 0x0003
             xt = (data[k + 2] & 0x0003) == 0x0002
     
             if xt and y1t and y2t:
                 y1[cnt] = (data[k] >> 2) & 0x3FFF
                 y2[cnt] = (data[k + 1] >> 2) & 0x3FFF
                 x[cnt] = (data[k + 2] >> 2) & 0x3FFF
                 cnt += 1
                 k += 3
             else:
                 k += 1
     
         # Adjust for actual counts
         if cnt != 0:
             x = x[:cnt]
             y1 = y1[:cnt]
             y2 = y2[:cnt]
             
             # Further processing, previously in 'process'
             for idx in range(cnt):
                 den = y1[idx] + y2[idx]
                 if den != 0:
                     ya = y1[idx] / den
                     yb = ((dsize * (4/3) * y1[idx]) / den - (dsize / 2) * (4/3) + 512 * scale)
                     xra = x[::-1]
                     xrb = (x / 4 - 230 * scale)[::-1]
     
                     # Append processed data
                     processed_data.append({'X': x[idx], 'Y1': y1[idx], 'Y2': y2[idx], 'Yraw': ya, 'Yclean': yb, 'Xraw': xra[idx], 'Xclean': xrb[idx]})
     
         return pd.DataFrame(processed_data)

    def init_plot(self):
        """Initializes the plot with empty or default settings."""
        self.ax[0].set_xlabel("X")
        self.ax[0].set_ylabel("Y")
        self.ax[1].set_title("2D Histogram Heatmap")
        self.ax[2].set_xlim(0, 2000)
        self.ax[2].set_xlabel('X', fontsize=10)
        self.ax[2].set_ylabel('Count', fontsize=10)
        self.scatter = self.ax[0].scatter([], [], s=1)
        self.heatmap = self.ax[1].imshow(np.zeros((50, 50)), interpolation='nearest', origin='lower', cmap='plasma', aspect='auto')
        self.hist1d = self.ax[2].bar([], [])
        self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax[1])
        self.cbar.set_label('Counts')

    def load_and_process_data(self):
        
        binary_data = pd.read_csv(self.processed_filename)
        processed_data = self.raw2coords(binary_data['Binary Word'])
        self.processed_data = processed_data
            
    def update_plot(self, frame):
        if not self.processed_data.empty:            
            xraw = self.processed_data['Xraw'].values
            yraw = self.processed_data['Yraw'].values
            xclean = self.processed_data['Xclean'].values
            yclean = self.processed_data['Yclean'].values
            #print(xraw)
            #print(xrb)
            xbins= 500
            ybins= 100 # Bins adjusted to pixel size of detector
            self.ax[0].clear()
            # Update scatter plot of raw data
            self.scatter = self.ax[0].scatter(xraw, yraw, s=1)
            self.ax[1].clear()  # Clear the axis for the new heatmap
            #Update 2D histogram heatmap
            hist, xedges, yedges= np.histogram2d(xclean, yclean, bins=[xbins, ybins])
            # Update heatmap using imshow with the new histogram data
            self.heatmap = self.ax[1].imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='nearest', origin='lower', cmap='plasma', aspect='auto') 
            self.heatmap.set_data(hist.T)
            #self.heatmap.set_extent([xedges[0], xedges[-1], yedges[0], yedges[-1]],interpolation='nearest', origin='lower', cmap='plasma')
            #self.heatmap.autoscale()
           
            # Update 1D histogram
            self.ax[2].clear()
            self.ax[2].hist(xclean, bins=(xbins))

            #print("Updating plot...")
 
        
        
    def plot(self):
        """Sets up real-time plotting."""
        self.load_and_process_data()
        listener_thread = threading.Thread(target=self.stop_listener)
        listener_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, blit=False, interval=500, repeat=False, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    processor_plotter = DataProcessorAndPlotter(filename)
    processor_plotter.plot()
