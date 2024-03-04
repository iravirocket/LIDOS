#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:34:54 2024

@author: Ravi
"""
import pandas as pd
import numpy as np
import datetime
import nidaqmx
import threading
from matplotlib import pyplot as plt, animation

date = '12_14_23'
identifier='lamp4'
# Base filename
base_filename = f'{date}_{identifier}'

# Specific filenames
filename = f'{base_filename}.csv'
processed_filename = f'{base_filename}_processedcsv'
plotname = f'{base_filename}.pdf'

class DataAcquisition:
    def __init__(self):
        self.stop_event = threading.Event()
        # Prepare for storing processed data
        self.processed_data = pd.DataFrame(columns = ['X', 'Y1', 'Y2'])
        self.unprocessed_data_batches = []
        self.fig, self.ax = plt.subplots(3,1,figsize=(25, 10))
        self.cbar = None
        self.heatmap=None
        self.scatter=None
        self.hist1d=None
        
        
    def listen_for_stop(self):
        input("Press Enter to stop data acquisition.")
        self.stop_event.set()

    def acquire_data(self):
        rate = 2000
        num_samples = 1000
        columns = ['Timestamp', 'Binary Word']  # Extend columns for processed data
        
        with nidaqmx.Task() as task:
            # Configure the digital input channels.
            for line in reversed(range(8)):
                task.di_channels.add_di_chan(f"Dev1/port1/line{line}", line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            for line in reversed(range(8)):
                task.di_channels.add_di_chan(f"Dev1/port0/line{line}", line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)

            # Set the sample rate and number of samples for the task.
            task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num_samples, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            task.start()
            BATCH_SIZE = 10  #Number of data reads before processing
            while not self.stop_event.is_set():
                data_read = np.array(task.read(number_of_samples_per_channel=num_samples))
               # words = np.bitwise_or.reduce(data_read * (1 << np.arange(16)[:, np.newaxis]), axis=0)
                self.unprocessed_data_batches.append(data_read)
                if len(self.unprocessed_data_batches) >= BATCH_SIZE:
                    # Convert each word to a 16-bit binary string, preserving leading zeros.
                    binary_words = ["'" + format(word, '016b') for batch in self.unprocessed_data_batches for word in np.bitwise_or.reduce(batch * (1 << np.arange(16)[:, np.newaxis]), axis=0)]
                    timestamps = [datetime.datetime.now().strftime("%M:%S.%f") for _ in range(len(binary_words))]
               
                    # Process the data 
                    x, y1, y2 = self.raw2coords(binary_words)
                    
                    # Optional: Handle the processed data (e.g., for plotting or saving)
                    processed_df = pd.DataFrame({'X': x, 'Y1': y1, 'Y2': y2})
                    # You could plot here or append to a DataFrame for later use
                    self.processed_data = pd.concat([self.processed_data, processed_df], ignore_index=True)
                    self.process()
                    self.unprocessed_data_batches = []
                    # Save the raw data chunk to CSV
                    df_chunk = pd.DataFrame(np.column_stack((timestamps, binary_words)), columns=columns)
                    df_chunk.to_csv(filename, mode='a', header=False, index=False)
        
    def raw2coords(self, rawdata):
        # Here, we strip the apostrophes before conversion
        data = [int(word.strip("'"), 2) for word in rawdata]

        k = 0
        cnt = 0
        many = len(data) // 3
        x = np.zeros(many, dtype=int)
        y1 = np.zeros(many, dtype=int)
        y2 = np.zeros(many, dtype=int)

        while k <= many * 3 - 3:
            y1t = (data[k] & 0x0003) == 0x0001
            y2t = (data[k+1] & 0x0003) == 0x0003
            xt = (data[k+2] & 0x0003) == 0x0002

            if xt and y1t and y2t:
                y1[cnt] = (data[k] >> 2) & 0x3fff
                y2[cnt] = (data[k+1] >> 2) & 0x3fff
                x[cnt] = (data[k+2] >> 2) & 0x3fff
                cnt += 1
                k += 3
            else:
                k += 1

        if cnt != 0:
            x = x[:cnt]
            y1 = y1[:cnt]
            y2 = y2[:cnt]

       

        return x, y1, y2
    
    def process(self):
        if not self.processed_data.empty:
            scale=2
            dsize=1024*scale #Pixel length of detector
            
        
            self.processed_data['den'] = self.processed_data['Y1'] + self.processed_data['Y2']
            no_zero = self.processed_data['den'] != 0
            self.processed_data = self.processed_data[no_zero]
            
            self.processed_data['ya'] = self.processed_data['Y1'] / self.processed_data['den']
            self.processed_data['yb'] = ((dsize * (4/3) * self.processed_data['Y1'] ) / self.processed_data['den'] - (dsize / 2) * (4/3) +512 * scale)
            
            
            self.processed_data['xra'] = self.processed_data['X'][::-1]
            self.processed_data['xrb'] = ((self.processed_data['X'] / 4) - 230 * scale)[::-1]
            # Save the processed data to CSV
            processed_filename = 'processed_data.csv'  # Name of the file for processed data
            self.processed_data.to_csv(processed_filename, mode='a', header=False, index=False)
           
        
    def init_plot(self):
        """Formats the plot."""
        self.ax[0].clear()
        self.ax[0].set_xlabel("X")
        self.ax[0].set_ylabel("Y")
        self.ax[0].set_title(filename)
        
        self.ax[1].tick_params(axis='x', labelsize=10)
        self.ax[1].tick_params(axis='y', labelsize=10)
        self.ax[1].tick_params(axis='both', length=10)
        #self.ax[1].set_title("2D Histogram Heatmap")
        self.ax[2].set_xlim(0, 2000)
        self.ax[2].set_xlabel('X', fontsize=10)
        self.ax[2].set_ylabel('Count', fontsize=10)
        #ax3.set_title('1D Projection along X-axis', fontsize=10)
        self.ax[2].tick_params(axis='x', labelsize=10)
        self.ax[2].tick_params(axis='y', labelsize=10)
        self.ax[2].tick_params(axis='both', length=10)
        #self.ax[2].set_title("1D Histogram")
        #initialize plots
        self.scatter = self.ax[0].scatter([],[], s=1)
        self.heatmap = self.ax[1].imshow(np.zeros((50, 50)), interpolation='nearest', origin='lower', cmap='plasma', aspect='auto')

        self.hist1d = self.ax[2].bar([], [])
    
        # Initialize colorbar here to avoid re-creation in update_plot
        self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax[1])
        
        return [self.scatter, self.heatmap, *self.hist1d]



    def update_plot(self, frame):
        """Updates the plot with new data."""
        if not self.processed_data.empty:
            xra = self.processed_data['xra'].values
            ya = self.processed_data['ya'].values
            xrb = self.processed_data['xrb'].values
            yb = self.processed_data['yb'].values
            #print(xrb)
            xbins= 500
            ybins= 100 # Bins adjusted to pixel size of detector
            self.ax[0].clear()
            # Update scatter plot of raw data
            self.scatter = self.ax[0].scatter(xra, ya, s=1)
            self.ax[1].clear()  # Clear the axis for the new heatmap
            # Update 2D histogram heatmap
            hist, xedges, yedges= self.ax[1].hist2d(xrb, yb, bins=(xbins, ybins), cmap='plasma')
            self.heatmap.set_data(hist.T)
            self.heatmap.set_extent([xedges[0], xedges[-1], yedges[0], yedges[-1]],interpolation='nearest', origin='lower', cmap='plasma', aspect='auto')
            #self.heatmap.autoscale()
            # Update color bar
            if not self.cbar:
                self.cbar = self.fig.colorbar(self.heatmap, ax=self.ax[1])

        
            # Update 1D histogram
            self.ax[2].clear()
            self.ax[2].hist(xrb, bins=(xbins))
            

    def run(self):
        """Run the acquisition and plotting."""
        stop_thread = threading.Thread(target=self.listen_for_stop)
        stop_thread.start()
        data_thread = threading.Thread(target=self.acquire_data, daemon=True)
        data_thread.start()

        self.plot()

        stop_thread.join()
        data_thread.join()
        
        
    def plot(self):
        """Sets up real-time plotting."""
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, blit=True, interval=500, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    daq_app = DataAcquisition()
    daq_app.run()    
