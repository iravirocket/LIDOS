#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:40:36 2024

@author: Ravi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, animation

# Update to use your actual data and filenames
date = '12_14_23'
identifier = 'lamp4'
base_filename = f'{date}_{identifier}'

# Assume processed data is stored in this file
processed_filename = f'{base_filename}_processed.csv'

class DataProcessorAndPlotter:
    def __init__(self, processed_filename):
        self.processed_filename = processed_filename
        self.processed_data = pd.read_csv(processed_filename)
        self.fig, self.ax = plt.subplots(3, 1, figsize=(25, 10))
        self.scatter = self.ax[0].scatter([], [], s=1)
        self.heatmap = self.ax[1].imshow(np.zeros((50, 50)), interpolation='nearest', origin='lower', cmap='plasma', aspect='auto')
        self.hist1d = self.ax[2].bar([], [])
        self.cbar = None
        
    def init_plot(self):
        """Initializes the plot with empty or default settings."""
        self.ax[0].set_xlabel("X")
        self.ax[0].set_ylabel("Y")
        self.ax[1].set_title("2D Histogram Heatmap")
        self.ax[2].set_xlim(0, 2000)
        self.ax[2].set_xlabel('X', fontsize=10)
        self.ax[2].set_ylabel('Count', fontsize=10)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax[1])
            self.cbar.set_label('Counts')

    def update_plot(self, frame):
        """Updates the plot with new data."""
        # Assuming 'frame' indexes rows in 'self.processed_data' to simulate real-time plotting
        if not self.processed_data.empty and frame < len(self.processed_data):
            data_slice = self.processed_data.iloc[:frame+1]
            xra = data_slice['X']
            ya = data_slice['Y1']  # Example usage, adjust based on actual data columns
            xrb = data_slice['X']
            yb = data_slice['Y2']  # Adjust based on actual data columns

            # Update plots with data_slice
            self.scatter.set_offsets(np.column_stack([xra, ya]))
            # Here you would update the heatmap and histogram based on your data_slice
            # This is just a placeholder; actual implementation would depend on your data
            
            # For example, updating the 1D histogram (adjust as necessary)
            self.ax[2].cla()
            self.ax[2].hist(xrb, bins=50)  # Example bin size

    def plot(self):
        """Sets up real-time plotting."""
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, frames=range(len(self.processed_data)), blit=False, interval=500)
        plt.show()

if __name__ == "__main__":
    processor_plotter = DataProcessorAndPlotter(processed_filename)
    processor_plotter.plot()
