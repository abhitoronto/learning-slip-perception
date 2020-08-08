#!/usr/bin/env python2.7

"""
data_utils.py

Util file for handling takktile recorded data

Developed at UTIAS, Toronto.

author: Abhinav Grover
"""

##################### Error printing
from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
#####################

# Standard Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

# GLOBAL VARIABLES
SPEED_THRESH = 0.5              # TODO: tune this parameter
SLIP_STD_VALID_MULTIPLIER = 2 # TODO: tune this parameter

class takktile_dataloader(object):
    """
        Takktile data loading class (.mat files)

        This file processes the available tactile sensor data
        to follow certain conditions:
        1.  Create an iterable list of indices with valid data-series of length
            series_len by reading the dictionary data in data_dir
        2.  Slip speed above SPEED_THRESH is considered slipping, class creates a list
            of slip and non-slip indices. Each of the datapoints in the time-series
            data must be valid datapoint. # TODO: Add the valid field to data
        3.  Valid data: any datapoint where most flow vectors are in agreement
    """

    def __init__(self, data_dir, create_hist=True,  input_len = 20, mat_format = True):
        self.series_len = input_len
        self.create_hist = create_hist

        # Load Data
        self.__load_data_dir(data_dir, mat_format)

        if self.empty():
            eprint("ERROR: Not enough data in directory: {}".format(data_dir))
            return

        # Create histogram
        if self.create_hist:
            self.__create_slip_hist(show=False)

        # Process Data
        self.__process_data()

    ###########################################
    #  API FUNCTIONS
    ###########################################
    def save_slip_hist(self, location):
        """
        Save the 2D histogram of the slip data at location
            location: address including the name of the saved image
        """
        if self.empty():
            eprint("ERROR: Cannot save histogram: empty loader")
            return
        if not self.create_hist:
            eprint("ERROR: Cannot save histogram: histogram disabled")
            return
        self.slip_hist.savefig(location, dpi=self.slip_hist.dpi)

    def size(self):
        if self.empty():
            return 0
        return int(self.__data['num'])

    def empty(self):
        return self.__data['num'] <= self.series_len

    def get_slip_idx(self):
        """
        Return slip only data
        """
        if self.empty():
            return None
        return self.slip_idx

    def get_no_slip_idx(self):
        """
        Return no slip data
        """
        if self.empty():
            return None
        return self.no_slip_idx

    def get_slip_stream_idx(self):
        """
        Return slip only data
        """
        if self.empty():
            return None
        return self.slip_stream_idx

    def get_no_slip_stream_idx(self):
        """
        Return no slip data
        """
        if self.empty():
            return None
        return self.no_slip_stream_idx

    def get_valid_idx(self):
        """
        Return valid data
        """
        if self.empty():
            return None
        return self.valid_idx

    ###########################################
    #  PRIVATE FUNCTIONS
    ###########################################

    def __getitem__(self, idx):
        """
        Return learning data at index
            Data Format: [pressure], ([slip_dir], [slip_speed])
        """
        if self.empty() or not isinstance(idx, (int, long)):
            eprint("ERROR: Incorrect index access: {}".format(idx))
            return None

        if idx<0 or idx>=self.size():
            eprint("ERROR: Incorrect index access: {}".format(idx))
            return None

        return self.__get_pressure(idx), \
               (self.__get_slip_dir(idx), \
               self.__get_slip_speed(idx))

    def __len__(self):
        return self.size()

    def __load_data_dir(self, data_dir, mat_format):
        print("Loading data file {}".format(data_dir))
        self.__data_dir = data_dir
        self.mat_format = mat_format
        self.__data = sio.loadmat(data_dir + "/data.mat", appendmat=mat_format)
        # Check if the data contains all fields
        if 'num' not in self.__data or \
           'temp' not in self.__data or \
           'pressure' not in self.__data or \
           'time' not in self.__data or \
           'slip_speed' not in self.__data or \
           'slip_std' not in self.__data or \
           'slip_dir' not in self.__data:
            eprint("ERROR: {} has outdated data, skipping.....".format(data_dir))
            self.__data['num'] = 0
            return
        # calculate time period of data recording
        if self.size() >= 2:
            self.__data_period = self.__get_time(1) - self.__get_time(0)

    def __create_slip_hist(self, show=False):
        hist_data = self.__data['slip_dir'] * self.__data['slip_speed']
        self.slip_hist = plt.figure(figsize=(10,10))
        plt.hist2d(hist_data[:, 0], hist_data[:, 1], bins=100)
        t = np.linspace(0,np.pi*2,100)
        plt.plot(0.1*np.cos(t), 0.1*np.sin(t), linewidth=1)
        plt.title("Slip Histogram of data in {}".format(self.__data_dir))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        if show:
            self.__show_slip_hist()

    def __show_slip_hist(self):
        if not self.slip_hist:
            self.__create_slip_hist(show=True)
        else:
            plt.show(self.slip_hist)

    def __process_data(self):
        # define index lists
        self.valid_idx = []
        self.no_slip_idx = [] # No slip on current idx
        self.slip_idx = []    # Slip on current idx
        self.no_slip_stream_idx = [] #  No slip in data[idx-len+1]->data[idx]
        self.slip_stream_idx = []    #  Slip in data[idx-len+1]->data[idx]

        valid_counter = 0
        slip_counter = 0
        no_slip_counter = 0
        # Iterate over each datapoint
        for idx in range(self.size()):
            if self.__is_valid(idx):
                valid_counter += 1
                if self.__get_slip_speed(idx) < SPEED_THRESH:
                    no_slip_counter += 1
                    slip_counter = 0
                else:
                    no_slip_counter = 0
                    slip_counter += 1
            else:
                valid_counter = 0
                no_slip_counter = 0
                slip_counter = 0
            # data[i-s] to data[i] is the valid data for index i
            if valid_counter >= self.series_len:
                # Storing every valid index
                self.valid_idx.append(idx)
                # Storing slip and non slip index
                if no_slip_counter > 0:
                    self.no_slip_idx.append(idx)
                else:
                    self.slip_idx.append(idx)
                # Store slip and non-slip stream indices
                if no_slip_counter >= self.series_len:
                    self.no_slip_stream_idx.append(idx)
                if slip_counter >= self.series_len:
                    self.slip_stream_idx.append(idx)

    def __is_valid(self, idx):
        """
        Check for the current time step whether the flow vectors are in agreement
        or not, ie, is the current measurement valid or not

        Parameters
        ----------
        idx : int
            The index for the data to check the validity of

        Returns
        ----------
        valid : bool
            Is the data at this index valid?
        """
        if idx<0 or idx>=self.size():
            return False

        slip_speed = self.__get_slip_speed(idx)
        slip_vector = self.__get_slip_dir(idx) * slip_speed
        slip_std = self.__get_slip_std(idx)

        max_dim = [i for i in range(len(slip_vector)) if slip_vector[i]>SPEED_THRESH]

        if (slip_std*SLIP_STD_VALID_MULTIPLIER > slip_vector)[max_dim].any():
            return False

        return True

    def __get_pressure(self, idx):
        return self.__data['pressure'][idx]

    def __get_temp(self, idx):
        return self.__data['temp'][idx]

    def __get_slip_dir(self, idx):
        return self.__data['slip_dir'][idx]

    def __get_slip_speed(self, idx):
        return self.__data['slip_speed'][idx]

    def __get_slip_std(self, idx):
        return self.__data['slip_std'][idx]

    def __get_time(self, idx):
        return self.__data['time'][idx]

if __name__ == "__main__":
    data_base = "/home/abhinavg/data/takktile/"
    data_dirs = [os.path.join(data_base, o) for o in os.listdir(data_base)
                if os.path.isdir(os.path.join(data_base,o)) and "takktile_" in o ]
    dataloaders = []
    for dir in data_dirs:
        dataloaders.append(takktile_dataloader(dir))
        if not dataloaders[-1].empty():
            print("Number of datapoints that have a seq num sized trail of slip data")
            print(len(dataloaders[-1].get_slip_idx()))