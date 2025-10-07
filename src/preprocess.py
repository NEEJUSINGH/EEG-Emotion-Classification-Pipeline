import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import argparse
import os

def bandpass_filter(data, lowcut=1, highcut=50, fs=128, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_fft_features(signal, fs=128):
    freq = np.fft.fftfreq(len(signal), 1/fs)
    fft_values = np.abs(fft(signal))
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    features = {}
    for band, (low, high) in bands.items():
        mask = (freq >= low) & (freq <= high)
        features[band] = np.mean(fft_values[mask])
    return features

def main(input_file, output_file):
    df = pd.read_csv(input_file)
    feature_rows = []

    for _, row in df.iterrows():
        signal = np.array([float(x) for x in row[1:].values])
        filtered = bandpass_filter(signal)
        features = extract_fft_features(filtered)
        features['label'] = row[0]
        feature_rows.append(features)

    out_df = pd.DataFrame(feature_rows)
    out_df.to_csv(output_file, index=False)
    print(f"✅ Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    main(args.input, args.output)
