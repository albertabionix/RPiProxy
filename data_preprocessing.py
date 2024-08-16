import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt


def plot(channel, signal, colour, x_min, x_max, label):
    plt.plot(np.arange(x_min, x_max, 1), signal, colour)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (mV)")
    plt.savefig('')


def filterdata(channel, sampling_rate):

    frequency = 1/np.mean(np.diff(np.linespace(0, len(channel)/sampling_rate, num=len(channel))))

    # Bandpass Butterworth Filter at 10-400Hz
    b, a = butter(2, ([10, 400]/(frequency/2)), btype='bandpass')
    dataf = filtfilt(b, a, channel)   

    # Signal Rectification (remove negative)
    dataRect = abs(dataf)

    # Notch Filter (60 Hz)
    low_cutoff_notch = 59 / (sampling_rate / 2)
    high_cutoff_notch = 61 / (sampling_rate / 2)

    bnotch, anotch = butter(4, [low_cutoff_notch, high_cutoff_notch], btype='stop')
    dataNotch = filtfilt(bnotch, anotch, dataRect)

    return dataNotch # returns the filtered data


if __name__ == "__main__":

    # 200ms window length, 50ms overlap
    sampling_rate = 2000 # Hz 
    window_length = 0.2 # sec
    next_window = 0.15

    sample_length = round(sampling_rate*window_length) # number of data points for the sample

    # Hard code the number of channels (electrodes)
    channel1 = []
    channel2 = []
    channel3 = []
    channel4 = []
    with open('C:\Documents\ABionix\EMG_Datasets\dataset2\one_raw.csv') as data:
        for line in data:
            line = line.split(",")
            channel1.append(float(line[0]))
            channel2.append(float(line[1]))
            channel3.append(float(line[2]))
            channel4.append(float(line[3]))

    filtered_data = filterdata(channel1, sampling_rate)


    ## Labelling ##
    # Strictly for sample data set

    cycle_time = 134 # seconds
    for rep in range(0, 5): # 5 repititions

        cycle_start = rep * cycle_time # changes based on the repetition

        segment_start = cycle_start * sampling_rate
        segment_end  = segment_start + sample_length
        count = 0

        while segment_end < (rep + 1) * cycle_time * sampling_rate:
            
            if segment_start >= 8000 + cycle_start and segment_start <= 20000 + cycle_start:
                label = 0
            elif segment_start >= 28000 + cycle_start and segment_start <= 40000 + cycle_start:
                label = 1
            elif segment_start >= 48000 + cycle_start and segment_start <= 60000 + cycle_start:
                label = 2
            elif segment_start >= 68000 + cycle_start and segment_start <= 80000 + cycle_start:
                label = 3
            elif segment_start >= 88000 + cycle_start and segment_start <= 100000 + cycle_start:
                label = 4
            elif segment_start >= 108000 + cycle_start and segment_start <= 120000 + cycle_start:
                label = 5
            elif segment_start >= 128000 + cycle_start and segment_start <= 140000 + cycle_start:
                label = 6
            elif segment_start >= 148000 + cycle_start and segment_start <= 160000 + cycle_start:
                label = 7
            elif segment_start >= 168000 + cycle_start and segment_start <= 180000 + cycle_start:
                label = 8
            elif segment_start >= 188000 + cycle_start and segment_start <= 200000 + cycle_start:
                label = 9
            else:
                label = 0

            plot(channel1, filtered_data, 'purple', segment_start, segment_end, label, count)

            segment_start = segment_start + next_window*sampling_rate
            segment_end = segment_start + sample_length

            count += 1









        



