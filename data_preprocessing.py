import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt


sampling_rate = 2000

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

"""
# Plot the Raw Data for Channel 1
plt.plot(np.arange(0, len(channel1), 1), channel1)
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.show()
plt.clf()
"""

frequency = 1/np.mean(np.diff(np.linspace(0, len(channel1)/sampling_rate, num=len(channel1))))

# Bandpass Butterworth Filter at 10-400Hz
b, a = butter(2, ([10, 400]/(frequency/2)), btype='bandpass')
dataf = filtfilt(b, a, channel1)

plt.plot(np.arange(0, len(channel1), 1), channel1, 'b')
plt.plot(np.arange(0, len(channel1), 1), dataf, 'r')
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.show()

# Wavelet Transform
"""
fftData = np.fft.fft(channel1)
freq = np.fft.fftfreq(len(channel1))*sampling_rate

fftData = fftData[0:int(len(fftData)/2)]
freq = freq[0:int(len(freq)/2)]

freq = freq[0:50000]
fftData = fftData[0:50000]

fftData = np.sqrt(fftData.real**2 + fftData.imag**2)

plt.plot(freq, fftData)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
plt.clf()
"""


## EMG Kaggle Dataset Sample Code
#data = pd.read_csv("C:\Documents\ABionix\EMG_data_for_gestures-master\EMG-data.csv") #kaggle
"""
plt.figure(figsize=(18, 6))
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")

plt.plot(data["time"], data["channel1"])
plt.xlim(0, max(data["time"]))
plt.ylim(-0.005, 0.005)
plt.show()



fftDataChannel1 = np.fft.fft(data["channel1"])
freq = np.fft.fftfreq(len(data["channel1"]))*1000

fftDataChannel1 = fftDataChannel1[0:int(len(fftDataChannel1)/2)]
freq = freq[0:int(len(freq)/2)]

#freq = freq[0:500000]
#fftDataChannel1 = fftDataChannel1[0:500000]


plt.plot(freq, fftDataChannel1.real**2 + fftDataChannel1.imag**2)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
plt.clf()
"""

#train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)