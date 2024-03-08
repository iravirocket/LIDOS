#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:53:17 2024

@author: Ravi
"""
import nidaqmx
import datetime
import threading
import time
import numpy as np
import pandas as pd

from nidaqmx.constants import Edge

# Create name of file to populate
date = '02_13_24'
identifier='dark9'
# Base filenam
base_filename = f'{date}_{identifier}'

class DataAcquisition:
    def __init__(self, base_filename):
        self.base_filename = base_filename
        self.filename = f'{base_filename}.csv'
        self.stop_acquisition = False
        self.columns = ['Timestamp', 'Binary Word']
        # Initialize CSV with headers
        pd.DataFrame(columns=self.columns).to_csv(self.filename, index=False)

    # Function to read user input in a separate thread
    def read_user_input():
        input("Press 'Enter' to end data acquisition.")
        stop_acquisition = True
    
    def take_data(self):
        with nidaqmx.Task() as task:
            # Read lines from port0 first
            # Then read lines from port1
            for line in reversed(range(8)):
                task.di_channels.add_di_chan("Dev1/port1/line" + str(line), line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            for line in reversed(range(8)):
                task.di_channels.add_di_chan("Dev1/port0/line" + str(line), line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        
            # Set the sample rate and number of samples for the task
            rate =20000
            num_samples = 10000
            # Set the sample rate for the task to change to continuous mode, change "finite" to "continuous"
            task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num_samples, active_edge=Edge.FALLING, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        
            # Export sample clock
            task.export_signals.export_signal(nidaqmx.constants.Signal.SAMPLE_CLOCK, "/Dev1/PFI5")
        
            # Start the task
            task.start()
        
        
            # Start a thread to read user input
            user_input_thread = threading.Thread(target=self.read_user_input)
            user_input_thread.start()
        
            # Read data until the user types 'stop'
            while not self.stop_acquisition:
                try:
                    # Read in the value of each line using the read function for each channel
                    data = np.array(task.read(number_of_samples_per_channel=num_samples))
            
                    # Combine the values of each line to form the 2-byte word
                    words = np.bitwise_or.reduce(data * (1 << np.arange(16)[:, np.newaxis]), axis=0)
            
                    # Convert the 2-byte words to binary strings
                    
                    binary_words = np.array(["'" + format(word, '016b') for word in words])
                    #binary_words=np.array([format(word, '016b') for word in words])
                    # Get the current date and time
                    timestamps = np.array([datetime.datetime.now().strftime("%M:%S.%f") for _ in range(num_samples)])
            
                    # Create a chunk of dataframe from the acquired data
                    df_chunk = pd.DataFrame(np.column_stack((timestamps, binary_words)), columns=self.columns)
                
                   # Append the chunk to the CSV file
                    df_chunk.to_csv(self.filename, mode='a', header=False, index=False)
                    
                except Exception as e:
                    print(f"Error during acquisition: {e}")
                    break
        
            # Stop the task
            task.stop()
    def populate_csv(self):
        # Consolidate the CSV (this step ensures data integrity but may not be strictly necessary)
        try:
            df_final = pd.read_csv(self.filename)
            df_final.to_csv(self.filename, index=False)
        except Exception as e:
            print(f"Error during CSV consolidation: {e}")
           
    
if __name__ == "__main__":
    daq = DataAcquisition(base_filename)
    daq.take_data()
    daq.populate_csv()
