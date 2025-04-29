from flask import Flask, jsonify
import struct
import socket
import time
import pickle

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Simple health check endpoint."""
    return jsonify({'status': 'Flask server is running'}), 200

TCP_IP = '127.0.0.1'
TCP_PORT = 8844
BUFFER_SIZE = 4096
SAVE_DURATION = 2  # seconds
PARSED_OUTPUT_FILE = "../parsed_eeg_data_10s.pkl"
COUNTER = 0

parsed_data_list = []  # Global list to collect parsed EEG entries
start_time = None      # Start time will be initialized when main starts
save_done = False      # To ensure we save only once

def parse_eeg_packet(packet):
    global parsed_data_list, start_time, save_done
    global COUNTER

    try:
        header = packet[:12]
        payload = packet[12:]
        packet_type = header[5]

        if packet_type != 1:
            print("Skipping packet type", packet_type)
            return
        COUNTER = COUNTER+1
        timestamp, = struct.unpack('>f', payload[0:4])

        eeg_data = []
        offset = 11
        for _ in range(24):
            ch_data, = struct.unpack('>f', payload[offset:offset+4])
            eeg_data.append(ch_data)
            offset += 4

        trigger, = struct.unpack('>f', payload[offset:offset+4])

        print(f"Time: {timestamp:.3f}, EEG[0:24]: {eeg_data[:24]}, Trigger: {trigger}")

        # Collect parsed data if within duration
        parsed_data_list.append({
            'timestamp': timestamp,
            'eeg': eeg_data,
            'trigger': trigger})

    except Exception as e:
        print("Failed to parse EEG packet:", e)

@app.route('/record', methods=['POST'])
def record():
    global start_time
    global COUNTER
    print(f"Connecting to DSI-24 stream on {TCP_IP}:{TCP_PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((TCP_IP, TCP_PORT))
        print("Connected. Reading EEG packets...")
        start_time = time.time()

        buffer = b''
        flag = False
        while True:
            data = s.recv(BUFFER_SIZE)
            buffer += data

            while b'@ABCD' in buffer:
                start = buffer.find(b'@ABCD')
                if len(buffer) < start + 12:
                    break

                packet_type = buffer[start + 5]
                packet_length = int.from_bytes(buffer[start + 6:start + 8], 'big')
                full_length = 12 + packet_length

                if len(buffer) < start + full_length:
                    break

                packet = buffer[start:start + full_length]
                parse_eeg_packet(packet)
                buffer = buffer[start + full_length:]
                if COUNTER == 1500:
                    flag = True
                    break
            if flag:
                break
        
        with open(PARSED_OUTPUT_FILE, 'wb') as f:
            pickle.dump(parsed_data_list, f)
            print(f"Saved parsed EEG data to '{PARSED_OUTPUT_FILE}'")


import pandas as pd
import numpy as np
from scipy.signal import iirnotch, filtfilt
from keras.models import load_model
    
def notch(channel_filtered_df):
    # Sampling frequency (Hz)
    fs = 300  # Replace with the actual sampling frequency of your data

    # Frequency to be removed from the signal (Hz)
    f0 = 50.0  # Notch frequency

    # Quality factor
    Q = 25.0  # Higher Q means a narrower notch

    # Design notch filter
    b, a = iirnotch(f0, Q, fs)

    # Initialize an empty DataFrame to store the filtered data
    filtered_df = pd.DataFrame(index=channel_filtered_df.index, columns=channel_filtered_df.columns)

    # Iterate over each column (channel) in the DataFrame
    for col in channel_filtered_df.columns:
        channel_data = channel_filtered_df[col]
        filtered_channel = np.zeros_like(channel_data)

        # Apply the notch filter on each 1500-sample segment
        for i in range(0, len(channel_data) // 1500):
            start_ind = i * 1500
            end_ind = start_ind + 1500
            
            # Extract the segment
            segment = channel_data[start_ind:end_ind]
            
            # Apply the notch filter to the segment
            filtered_segment = filtfilt(b, a, segment)
            
            # Store the filtered segment back in the array
            filtered_channel[start_ind:end_ind] = filtered_segment

        # Assign the filtered channel back to the DataFrame
        filtered_df[col] = filtered_channel

    return filtered_df

def car(df):
    # Initialize an empty DataFrame to store the CAR-applied data
    car_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Iterate over each 1500-sample segment
    for i in range(0, len(df) // 1500):
        start_ind = i * 1500
        end_ind = start_ind + 1500
        
        # Extract the segment
        segment = df.iloc[start_ind:end_ind]

        # Calculate the mean across all channels (columns) for each time point (row) in the segment
        channel_mean = segment.mean(axis=1)

        # Subtract the mean from each channel's data within the segment
        segment_car = segment.subtract(channel_mean, axis=0)

        # Store the CAR-applied segment back into the DataFrame
        car_df.iloc[start_ind:end_ind] = segment_car

    return car_df

import mne

def create_raw_object(df, selected_columns, n_channels):
    # Convert the DataFrame into a NumPy array
    eeg_data_array = df.values.T  # Transpose to make channels as rows

    ch_types = ['eeg'] * n_channels  # Define each channel as EEG

    # Create an info structure
    info = mne.create_info(ch_names=selected_columns, sfreq=300, ch_types=ch_types)

    # Create the Raw object
    raw = mne.io.RawArray(eeg_data_array, info)
    print(raw)
    return raw

def set_montage(raw):
    # Set a standard montage (10-20 system)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
def band_pass(raw,l_freq,h_freq):
    # Apply a band-pass filter (8-30 Hz, typical for EEG data) - for alpha and beta waves
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    return raw_filtered

def raw_to_array(raw):
    # Convert the Raw object to a NumPy array
    data_array = raw.get_data()
    return data_array


from mne.preprocessing import ICA

def apply_ica(raw, n_components=11, random_state=42):
    # Set up ICA
    ica = ICA(n_components=n_components, random_state=random_state, max_iter='auto')

    # Fit ICA on the raw data
    ica.fit(raw)

    # Plot ICA components to inspect them visually
    ica.plot_components()

    # You can plot properties of specific components as well (e.g., component 0)
    ica.plot_properties(raw, picks=[0])

    # Optionally, identify bad components (e.g., related to eye blinks)
    # You can do this manually by looking at the components, or automatically:
    # eog_indices, eog_scores = ica.find_bads_eog(raw)
    # ica.exclude = eog_indices

    # Apply the ICA to the raw object (removes the selected components)
    raw_cleaned = ica.apply(raw.copy())
    
    return raw_cleaned, ica

def convert_to_dataframe(data_tuple):
    # Extract the RawArray from the tuple (the first element of the tuple)
    raw = data_tuple[0]
    
    # Get the data from the RawArray (this will be a numpy array of shape (n_channels, n_times))
    raw_data = raw.get_data()
    
    # Transpose the raw data to get the desired shape (n_times, n_channels)
    raw_data = raw_data.T  # Now it's (n_times, n_channels)
    
    # Extract the number of samples and number of channels
    n_times, n_channels = raw_data.shape
    
    # Ensure the shape is 1500 x 13 as expected
    if n_times != 1500  or n_channels != 11:
        raise ValueError(f"Expected shape (1500, 11), but got {raw_data.shape}")
    
    # Create a pandas DataFrame from the numpy array
    df = pd.DataFrame(raw_data, columns=[f'Channel_{i}' for i in range(1, n_channels + 1)])
    
    return df

import numpy as np
from scipy.signal import hilbert, stft

# Define the parameters
FS = 300              # Sampling frequency
WINDOW_LENGTH = 300   # 1-second windows (128 samples)
OVERLAP = 150         # 50% overlap (64 samples)
STFT_NPERSEG = 75     # STFT window size (0.25s)
STFT_NOVERLAP = 37.5  # STFT overlap (50% of nperseg)

def preprocess_prediction(df):
    """Preprocess the prediction dataframe into Hilbert-STFT features"""
    n_samples, n_channels = df.shape
    X = []
    
    # Create sliding windows
    for start in range(0, n_samples - WINDOW_LENGTH, OVERLAP):
        end = start + WINDOW_LENGTH
        window = df.iloc[start:end].values  # (128, 12)
        
        channel_features = []
        for ch in range(n_channels):
            # 1. Compute analytic signal
            analytic_signal = hilbert(window[:, ch])
            
            # 2. Compute STFT of analytic signal
            _, _, Zxx = stft(analytic_signal, 
                            fs=FS, 
                            nperseg=STFT_NPERSEG, 
                            noverlap=STFT_NOVERLAP)
            
            # 3. Combine magnitude and phase
            mag = np.abs(Zxx)  # (17, 7)
            phase = np.angle(Zxx)
            channel_features.append(np.stack([mag, phase], axis=-1))  # (17, 7, 2)
        
        # Combine all channels: (17, 7, 12*2) = (17,7,24)
        combined = np.concatenate(channel_features, axis=-1)
        X.append(combined)
    
    return np.array(X)

@app.route('/predict', methods=['GET'])
def predict():
    # record()
    with open("../parsed_eeg_data_10s.pkl", "rb") as f:
        parsed_data = pickle.load(f)
        
    parsed_data = parsed_data[:1500]
    
    eeg_data = [entry['eeg'] for entry in parsed_data]
    df = pd.DataFrame(eeg_data)
    df.columns = [f'Channel_{i+1}' for i in range(df.shape[1])]
    
    df_subtracted = df.subtract(df['Channel_9'], axis=0)

    columns_to_drop = df_subtracted.columns[[16, 17, 20]]
    
    df_dropped = df_subtracted.drop(columns=columns_to_drop)
    df_dropped.columns = [f'Channel_{i+1}' for i in range(df_dropped.shape[1])]

    selected_channels = [1, 2, 3, 4, 5, 7, 8, 10, 11, 16, 17]
    selected_column_names = [f'Channel_{i+1}' for i in selected_channels]
    df_selected = df_dropped[selected_column_names]
    
    Filtered_df = notch(df_selected)
    
    CAR = car(Filtered_df)
    
    columns_to_extract = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz','Pz', 'A1', 'Fp1', 
                    'Fp2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'A2', 'T6', 'T4']
    selected_columns = [columns_to_extract[i] for i in selected_channels]
    raw = create_raw_object(CAR, selected_columns,CAR.shape[1])
    
    raw_filtered = band_pass(raw,8,30)
    df = pd.DataFrame(raw_to_array(raw_filtered).T)
    
    set_montage(raw_filtered)
    ICA = apply_ica(raw_filtered)
    df = convert_to_dataframe(ICA)

    model = load_model('../comlpex_stft_model.h5')

    X_pred = preprocess_prediction(df)
    X_pred = (X_pred - np.mean(X_pred, axis=(0,1,2))) / np.std(X_pred, axis=(0,1,2))    
    
    predictions = model.predict(X_pred)
    
    predicted_classes = np.argmax(predictions, axis=-1)
    
    # Get the most common class
    most_common_class = np.bincount(predicted_classes).argmax()
    dist ={0:'Left Movement', 1:'Left Imagery', 2:'Right Movement', 3:'Right Imagery'}
    with open('predicted_task.txt', 'w') as f:
        f.write(dist[most_common_class])
    return jsonify({'prediction': dist[most_common_class]})

if __name__ == '__main__':
    app.run(debug=True)
