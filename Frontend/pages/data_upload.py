import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# Page Configuration
st.set_page_config(page_title="EEG Data Upload", layout="wide")

st.title("🧠 EEG Data Upload & Preprocessing")

# File Upload
uploaded_file = st.file_uploader("Upload EEG Dataset (CSV or MAT)", type=["csv", "mat"])

# List of columns to extract
columns_to_extract = [
    'EEG.Cz', 'EEG.Fz', 'EEG.Fp1', 'EEG.F7', 'EEG.F3',
    'EEG.FC1', 'EEG.C3', 'EEG.FC5', 'EEG.FT9', 'EEG.T7',
    'EEG.CP5', 'EEG.CP1', 'EEG.P3', 'EEG.P7', 'EEG.PO9',
    'EEG.O1', 'EEG.Pz', 'EEG.Oz', 'EEG.O2', 'EEG.PO10',
    'EEG.P8', 'EEG.P4', 'EEG.CP2', 'EEG.CP6', 'EEG.T8',
    'EEG.FT10', 'EEG.FC6', 'EEG.C4', 'EEG.FC2', 'EEG.F4',
    'EEG.F8', 'EEG.Fp2'
]

def load_eeg_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file, skiprows=1)
    elif file.name.endswith('.mat'):
        mat = scipy.io.loadmat(file)
        key = [k for k in mat.keys() if not k.startswith('__')][0]  # Extract the main data key
        data = pd.DataFrame(mat[key])
    else:
        return None
    extracted_df = data[columns_to_extract]
    extracted_df.reset_index(drop=True, inplace=True)
    experiment_duration = 2 + 10 + 2
    fs=128 
    valid_dataPoints = experiment_duration*fs
    
    # Removing the start lag
    col_start = 4*128
    extracted_df = extracted_df.drop(data.index[:512])
    extracted_df.reset_index(drop=True, inplace=True)
    
    # Removing the end lag
    extracted_df = extracted_df[:valid_dataPoints]
    
    # To get task EEG Signals only, removing the first 2 welcoming and last 2 closing seconds as well
    extracted_df = extracted_df.drop(data.index[:256])
    extracted_df.reset_index(drop=True, inplace=True)
    extracted_df = extracted_df[:1280]
    df = pd.DataFrame({})
    start_index = 3*128
    end_index = start_index + 5*128
    new_df = extracted_df.iloc[start_index:end_index].reset_index(drop=True)
    df = pd.concat([df, new_df], ignore_index=True)
    return df

if uploaded_file is not None:
    eeg_data = load_eeg_data(uploaded_file)
    st.session_state["uploaded_data"] = eeg_data
    if eeg_data is not None:
        st.write("## Dataset Overview")
        
        # Channel Selection
        st.write("#### Select EEG Channels")
        selected_channels = st.multiselect("Choose channels to visualize", eeg_data.columns.tolist(), default=eeg_data.columns[:5].tolist())
        
        if selected_channels:
            st.write("#### Time-Domain Signal Plot")
            fig, ax = plt.subplots(figsize=(10, 4))
            for ch in selected_channels:
                ax.plot(eeg_data[ch], label=ch)
            ax.legend()
            ax.set_title("EEG Signal over Time")
            st.pyplot(fig)
    else:
        st.error("Error loading data. Please upload a valid CSV or MAT file.")

# Navigation to Other Pages
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/visualization.py", label="Visualization")
st.sidebar.page_link("pages/augmentation.py", label="Data Augmentation (VAE)")
st.sidebar.page_link("pages/classification.py", label="CNN Classification & Results")
