import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile


def Hamming_window(s):
    # for i in range(len(window)):
    w = []
    for i in range(len(s)):
        hw = 0.54 + 0.46 * np.cos((2 * np.pi * i)/len(s))
        w.append(hw)
    return w

def calculate_y(s,w):
    y = []
    for i in range(len(s)):
        yi = s[i]*w[i]
        y.append(yi)
    return y

def main():

    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("wavfile")
    args = parser.parse_args()

    # Read file
    samplingFreq, mySound = wavfile.read(args.wavfile)

    # sample rate = 16,000, sample per msec = 16, window length = 25ms
    sampleWindowSize = 400  #calculated
    sampleWindowStep = 160

    data = [mySound[i:i + sampleWindowSize] for i in range(0, len(mySound), sampleWindowStep)]
    samples = []
    hamming_samples = []
    y_samples = []
    for sample in data:
        if len(sample) == sampleWindowSize:
            samples.append(sample)
            hamming_samples.append(Hamming_window(sample))

    for i in range(len(samples)):
        y_samples.append(calculate_y(samples[i],hamming_samples[i]))

    # Fast Discrete Fourier Transform
    fft_y = np.fft.fft(y_samples)

    # Calculate Magnitude
    fft_y = np.abs(fft_y)

    # Convert to log scale
    fft_y = 10 * np.log10(fft_y)

    # Convert to 0 if negative
    fft_y[fft_y <0] = 0

    # Convert to 0-255 range
    fft_y_spec = 255 * (fft_y - np.min(fft_y))/(np.max(fft_y) - np.min(fft_y))

    # Pick only first 200 ks
    # 0 as white, 255 as black
    fft_y_spec = 255 - fft_y_spec[:,:200]

    # Convert float to int
    fft_y_spec = fft_y_spec.astype(np.int16)


    m, n = fft_y_spec.shape
    # fill matrix
    z = np.zeros((n,m))
    for i in range(m):
        for j in range(n):
            z[n-1-j,i] = fft_y_spec[i,j]


    # make plot
    fig = plt.figure(figsize = (10,8))
    plt.imshow(z, cmap = "gray")
    plt.axis('off')
    plt.draw()

    plt.pause(1)

    input("press Enter to close...")
    plt.close(fig)

if __name__ == "__main__":
    main()
