#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:40:36 2024

@author: Ravi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, animation
import threading

# Assume processed data is stored in this file
processed_filename = 'processed_data.csv'

class DataProcessorAndPlotter:
    def __init__(self, processed_filename):
        self.processed_filename = processed_filename
        self.processed_data = pd.read_csv(processed_filename)
        self.fig, self.ax = plt.subplots(3, 1, figsize=(25, 10))
        self.stop_plotting = threading.Event()
        self.init_plot()

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
        if not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax[1])
            self.cbar.set_label('Counts')

    def stop_listener(self):
        input("Press 'Enter' to stop plotting and save the figure...")
        self.stop_plotting.set()

    def update_plot(self, frame):
        if self.stop_plotting.is_set():
            plt.savefig('final_plot.pdf')
            plt.close()
            return
        # Assuming 'frame' indexes rows in 'self.processed_data' to simulate real-time plotting
        if not self.processed_data.empty and frame < len(self.processed_data):
            data_slice = self.processed_data.iloc[:frame+1]
            # Update plot logic here based on your actual data structure and needs

    def plot(self):
        """Sets up real-time plotting."""
        listener_thread = threading.Thread(target=self.stop_listener)
        listener_thread.start()
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, frames=range(len(self.processed_data)+1), blit=False, interval=500, repeat=False)
        plt.show()

if __name__ == "__main__":
    processor_plotter = DataProcessorAndPlotter(processed_filename)
    processor_plotter.plot()
