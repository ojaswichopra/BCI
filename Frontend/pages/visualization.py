import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from scipy.signal import spectrogram, cwt, ricker, hilbert
import mne
sfreq = 128
frequencies = np.arange(8, 31, 2)
n_cycles = 2  # Number of cycles for each frequency
channel_to_plot = 'Channel3'  # Replace with the desired channel name
epoch_length = 640  # 5 seconds at 128 Hz

# Load the uploaded data from session state
def load_data():
    if "uploaded_data" not in st.session_state:
        st.error("No data uploaded. Please upload data on the Data Upload page.")
        return None
    return st.session_state["uploaded_data"]

def plot_time_domain(data, selected_channel):
    plt.figure(figsize=(10, 4))
    plt.plot(data[selected_channel])
    plt.title(f'Time-Domain Signal - {selected_channel}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    st.pyplot(plt)
    
def create_raw_object(df):
    channel_names = list(df.columns)
    data = df.to_numpy().T
    # Define channel names
    channel_types = ['eeg'] * len(channel_names)
    # Create MNE Info object
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    # Create RawArray
    raw = mne.io.RawArray(data, info)
    return raw

raw = create_raw_object(st.session_state["uploaded_data"])


def plot_stft(raw, channel_to_plot, epoch_length, window='hann', nperseg=128, noverlap=64):
    if channel_to_plot not in raw.info['ch_names']:
        raise ValueError(f"Channel '{channel_to_plot}' not found in raw.info['ch_names'].")

    picks = mne.pick_channels(raw.info['ch_names'], include=[channel_to_plot])
    data = raw.get_data(picks=picks)

    n_samples = data.shape[1]
    n_epochs = n_samples // epoch_length

    if n_epochs < 1:
        raise ValueError("Not enough data to create at least one epoch with the specified epoch_length.")

    epochs_data = data[:, :n_epochs * epoch_length].reshape((n_epochs, epoch_length))
    sfreq = raw.info['sfreq']

    for epoch_idx, epoch in enumerate(epochs_data):
        f, t, Sxx = spectrogram(epoch, fs=sfreq, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density')
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.colorbar(label='Power (dB)')
        plt.title(f"STFT - Epoch {epoch_idx + 1} - {channel_to_plot}")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        st.pyplot(plt)

def plot_cwt(raw, channel_to_plot, epoch_length):
    if channel_to_plot not in raw.info['ch_names']:
        raise ValueError(f"Channel '{channel_to_plot}' not found in raw.info['ch_names'].")

    picks = mne.pick_channels(raw.info['ch_names'], include=[channel_to_plot])
    data = raw.get_data(picks=picks)

    n_samples = data.shape[1]
    n_epochs = n_samples // epoch_length

    if n_epochs < 1:
        raise ValueError("Not enough data to create at least one epoch with the specified epoch_length.")

    epochs_data = data[:, :n_epochs * epoch_length].reshape((n_epochs, epoch_length))

    for epoch_idx, epoch in enumerate(epochs_data):
        widths = np.arange(1, 128)
        cwt_matrix = cwt(epoch, ricker, widths)

        plt.figure(figsize=(10, 5))
        plt.imshow(cwt_matrix, extent=[0, epoch_length / raw.info['sfreq'], 1, 128], aspect='auto', cmap='jet')
        plt.colorbar(label='Amplitude')
        plt.title(f"CWT - Epoch {epoch_idx + 1} - {channel_to_plot}")
        plt.ylabel("Scale")
        plt.xlabel("Time (s)")
        st.pyplot(plt)

def plot_hilbert_huang(raw, channel_to_plot, epoch_length):
    if channel_to_plot not in raw.info['ch_names']:
        raise ValueError(f"Channel '{channel_to_plot}' not found in raw.info['ch_names'].")

    picks = mne.pick_channels(raw.info['ch_names'], include=[channel_to_plot])
    data = raw.get_data(picks=picks)

    n_samples = data.shape[1]
    n_epochs = n_samples // epoch_length

    if n_epochs < 1:
        raise ValueError("Not enough data to create at least one epoch with the specified epoch_length.")

    epochs_data = data[:, :n_epochs * epoch_length].reshape((n_epochs, epoch_length))

    for epoch_idx, epoch in enumerate(epochs_data):
        analytic_signal = hilbert(epoch)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * raw.info['sfreq']

        plt.figure(figsize=(10, 5))
        plt.plot(amplitude_envelope, label="Amplitude Envelope")
        plt.plot(instantaneous_frequency, label="Instantaneous Frequency")
        plt.title(f"Hilbert-Huang Transform - Epoch {epoch_idx + 1} - {channel_to_plot}")
        plt.xlabel("Time (samples)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

def plot_psd(raw):
    fig = raw.plot_psd(fmax=60)  # Plot up to 60 Hz
    st.pyplot(fig)

def plot_jointplot(data, selected_channel):
    sns.set_context("notebook", font_scale=0.5)  # Shrinks font sizes (axis labels, ticks)
    fig = sns.jointplot(
        x=np.arange(len(data[selected_channel])),
        y=data[selected_channel],
        kind='scatter',
        height=3.5,              # Overall figure size (smaller than default 6)
        space=0.2,               # Space between main plot and marginal plots
        s=10                     # Size of scatter plot markers
    )
    st.pyplot(fig.figure)       # Use `.figure` to display correctly in Streamlit

# Streamlit Page Configuration
st.set_page_config(page_title="EEG Data Visualization", layout="wide")
st.title("Visualization")

data = load_data()
if data is not None:
    channels = list(data.columns)
    selected_channel = st.selectbox("Select EEG Channel:", channels)
    
    st.subheader("Time-Domain Signal")
    plot_time_domain(data, selected_channel)
    
    st.subheader("Hilbert-Huang Transform")
    plot_hilbert_huang(raw, selected_channel, epoch_length)
    
    st.subheader("STFT Spectrogram")
    plot_stft(raw, selected_channel, epoch_length)
    
    st.subheader("Continuous Wavelet Transform")
    plot_cwt(raw, selected_channel, epoch_length)
        
    st.subheader("Power Spectral Density")
    plot_psd(raw)
    
    st.subheader("Joint Plot")
    plot_jointplot(data, selected_channel)

st.sidebar.header("Navigation")
st.sidebar.page_link("pages/data_upload.py", label="Data Upload")
st.sidebar.page_link("pages/augmentation.py", label="Data Augmentation")
st.sidebar.page_link("pages/classification.py", label="CNN Classification")
