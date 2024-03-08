#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:28:20 2024

@author: Ravi
"""
import threading

def orchestrate_acquisition_and_plotting(base_filename):
    # Initialize the data acquisition
    daq = DataAcquisition(base_filename)
    # Start data acquisition in a separate thread to allow it to run concurrently with processing/plotting
    daq_thread = threading.Thread(target=daq.run)  # Assuming `run` is a method that starts the acquisition
    daq_thread.start()

    # Assuming the processing and plotting module has a class named `DataProcessorAndPlotter`
    # and it has a method `run` that takes the filename to process and plot.
    processor = DataProcessorAndPlotter(f"{base_filename}.csv")
    # Wait for the data acquisition thread to finish
    daq_thread.join()
    # Now process and plot the data
    processor.run()

if __name__ == "__main__":
    base_filename = 'data_02_13_24_dark9'
    orchestrate_acquisition_and_plotting(base_filename)
