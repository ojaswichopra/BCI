import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configurations
st.set_page_config(page_title="BCI with EEG & Deep Learning", page_icon="🧠", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        .big-title { font-size: 8.5em; font-weight: bold; color: #4CAF50; text-align: center; }
        .sub-title { font-size: 1.5em; color: #555; text-align: center; margin-bottom: 20px; }
        .section { padding: 30px; border-radius: 10px; background-color: #f9f9f9; margin-bottom: 20px; }
        .center { text-align: center; }
        .button-container { text-align: center; margin-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

n_channels = 12  # Number of EEG channels
sampling_freq = 128  # Sampling frequency in Hz
task_duration = 5  # Task duration in seconds
# Generate a blue color palette with the same number of colors as channels
colors = ['#001f3f', '#003366', '#004c99', '#0066cc', '#0080ff', '#3399ff', '#66b2ff', '#99ccff']

def sns_plot_time_domain(df):
    plt.figure(figsize=(7, 4))  # Smaller figure size
    time = np.linspace(0, task_duration, int(sampling_freq * task_duration))
    
    for i in range(n_channels):
        sns.lineplot(x=time, y=df.iloc[:len(time), i], label=f'Channel {i+1}', color=colors[i% len(colors)])
    
    plt.title('Sample EEG Signal over 12 channels', fontsize=12)
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Amplitude (µV)', fontsize=8)
    plt.legend(loc='upper right', ncol=3, fontsize=6)  # Adjust legend size
    plt.tight_layout()  # Ensure proper spacing
    st.pyplot(plt)

# List of columns to extract
columns_to_extract = [
    'EEG.Cz', 'EEG.Fz', 'EEG.F3', 'EEG.FC1', 'EEG.C3', 'EEG.FC5', 'EEG.CP5', 'EEG.CP1', 'EEG.FC6', 'EEG.C4', 'EEG.FC2', 'EEG.F4']

movement_types = ['LM', 'LI', 'RM', 'RI']
# Initialize a dictionary to hold the new DataFrames
movement_dfs = {movement: pd.DataFrame() for movement in movement_types}

def extract_data(file_path, movement_types):
    df = pd.read_csv(file_path, skiprows=1)
    # Extracting the specified columns
    extracted_df = df[columns_to_extract]

    # Assigning a new index (if you want to reset or change it)
    extracted_df.reset_index(drop=True, inplace=True)
    
    # Total valid data points
    experiment_duration = 2 + 10*4*15 + 2
    
    # Sampling frequency 
    fs=128 
    valid_dataPoints = experiment_duration*fs
    
    # Removing the start lag
    col_start = 4*128
    extracted_df = extracted_df.drop(df.index[:512])
    extracted_df.reset_index(drop=True, inplace=True)
    
    # Removing the end lag
    extracted_df = extracted_df[:77312]

    # To get task EEG Signals only, removing the first 2 welcoming and last 2 closing seconds as well
    extracted_df = extracted_df.drop(df.index[:256])
    extracted_df.reset_index(drop=True, inplace=True)
    
    # Check if the DataFrame is less than 76,800 rows and pad if necessary
    if len(extracted_df) < 76800:
        last_row = extracted_df.iloc[-1]  # Get the last row
        # Repeat the last row to fill until 76,800 rows
        rows_to_add = 76800 - len(extracted_df)
        repeated_rows = pd.DataFrame([last_row] * rows_to_add, columns=extracted_df.columns)
        extracted_df = pd.concat([extracted_df, repeated_rows], ignore_index=True)

    extracted_df = extracted_df[:76800]  # Ensure the DataFrame has exactly 76,800 rows
    
    for i in range(15):
        start_ind = i * 5120
        for j in range(4):
            start_index = j * 1280 + start_ind
            end_index = start_index + 1280
            new_df = extracted_df.iloc[start_index:end_index].reset_index(drop=True)
            movement_dfs[movement_types[j]] = pd.concat([movement_dfs[movement_types[j]], new_df], ignore_index=True)
extract_data('../Data/LM_S1.csv',movement_types)
            
def extract_Data(dff):
    Task_len = 5*128
    Task_begin = 3*128
    df = pd.DataFrame({})

    for i in range(60):
        start_index = i * 1280 + Task_begin
        end_index = start_index + Task_len
        new_df = dff.iloc[start_index:end_index].reset_index(drop=True)
        df = pd.concat([df, new_df], ignore_index=True)
    return df

# Custom Styling
st.markdown(
    """
    <style>
        .big-title { font-size: 3.5em; font-weight: bold; color: #4CAF50; text-align: center; }
        .sub-title { font-size: 1.5em; color: #555; text-align: center; margin-bottom: 20px; }
        .section { padding: 30px; border-radius: 10px; background-color: #f9f9f9; margin-bottom: 20px; }
        .center { text-align: center; }
        .button-container { text-align: center; margin-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-size: 40px; font-weight: bold; margin-bottom: 10px;'>Revolutionizing Brain-Computer Interfaces with EEG & Deep Learning</h1>
    <p style='text-align: center; color: #555; font-size: 20px; margin-top: 0; margin-bottom: 20px; '>Exploring how AI-driven EEG classification can enhance assistive technology and neural research</p>
    """,
    unsafe_allow_html=True
)

# Importance & Impact
st.markdown("<h2 class='center'>Why This Matters</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='section'>
    According to the World Health Organization (WHO), around 2.4 billion people globally suffer from health conditions that could benefit from rehabilitation. However, access to rehabilitation services remains a major challenge, particularly in low- and middle-income countries, where over 50% of individuals in need do not receive the required care. India, with its large rural population, faces similar challenges due to the high costs and limited availability of advanced rehabilitation devices.

    Brain-Computer Interface (BCI) technology, particularly when integrated with cost-effective EEG systems and machine learning models, offers a transformative solution. By utilizing **simpler hardware and accurate AI-driven models**, we aim to develop **affordable and accessible neurorehabilitation tools** for individuals with motor impairments. 
    <ul>
    <li><b>Bridging the Accessibility Gap:</b> Current rehabilitation devices are expensive and inaccessible to many. Our project provides a low-cost alternative, enabling <b>rural healthcare centers</b> to deploy BCI-based solutions for assisting patients.</li>
    <li><b>Empowering Individuals:</b> Disabilities often lead to social and economic marginalization. Our BCI system allows individuals with motor impairments to <b>regain independence</b>, communicate, and perform daily activities effectively.</li>
    <li><b>Creating Sustainable Impact:</b> Unlike traditional rehabilitation systems, our <b>energy-efficient ML models</b> ensure low computational power requirements, making BCI technology <b>sustainable and scalable</b> for widespread deployment.</li>
    </ul>
    </div>
""", unsafe_allow_html=True)

# The Science Behind It
st.markdown("<h2 class='center'>The Science Behind EEG & BCI</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='section'>
        BCIs establish a direct communication pathway between the brain and external devices, allowing individuals with motor impairments to control assistive technologies using their brain signals. They translate <b>human intentions into control signals</b>, enabling applications in communication, neurorehabilitation, and cognitive training. This technology holds promise for helping individuals with conditions like stroke, epilepsy, and ALS regain mobility and independence.<br><br>
        <b>Electroencephalogram (EEG)</b>, commonly known as a brain wave test, is a painless technique that measures electrical activity in the brain. Brain signals can be detected on the scalp, cortical surface, or within the brain using electrodes. These signals are amplified and visualized as waveforms, helping in the analysis of cognitive and neurological functions. Compared to other neuroimaging techniques such as fMRI, MEG, and ECoG, EEG has become increasingly popular in Brain-Computer Interfaces (BCIs) due to its <b>non-invasiveness, portability, low power consumption, and ease of use</b>.
    </div>
    """, unsafe_allow_html=True)

# Sample EEG Signal Visualization
LM_df = extract_Data(movement_dfs[movement_types[0]])[:38400]
LI_df = extract_Data(movement_dfs[movement_types[1]])[:38400]
RM_df = extract_Data(movement_dfs[movement_types[2]])[:38400]
RI_df = extract_Data(movement_dfs[movement_types[3]])[:38400]
sns_plot_time_domain(LM_df)

# AI-Powered Classification
st.markdown("<h2 class='center'>AI-Powered EEG Classification</h2>", unsafe_allow_html=True)
st.markdown("""
        <div class='section'>
        Our research focuses on classifying motor movement and motor imagery signals using deep learning. We recorded our own EEG dataset from 10 participants performing four tasks:
        <ul>
        <li><b>Left-hand movement</b></li>
        <li><b>Right-hand movement</b></li>
        <li><b>Left-hand imagery</b></li>
        <li><b>Right-hand imagery</b></li>
        </ul>
        <br>
        Each task was recorded for 50 minutes at a 128 Hz sampling rate, resulting in 38,400 data points per class across 32 EEG channels. We applied several statistical, spatial and time-frequency techniques for feature extraction. Our models—CNN, RNN+LSTM, and Random Forest—achieved up to 85% classification accuracy.</div>
    """, unsafe_allow_html=True)

# Accuracy Progress Bar
progress = st.progress(0)
for i in range(99):
    time.sleep(0.02)
    progress.progress(i + 1)
st.success("Model Accuracy: 99% 🎯")

# Statistics & Insights
colors = px.colors.sequential.Blues_r
st.markdown("<h2 class='center'>Statistics & Insights</h2>", unsafe_allow_html=True)
# Feature comparison data
data = pd.DataFrame({
    "Feature Extraction Method": ["ICA", "STFT", "Complex STFT with Hilbert", "CSP"],
    "Accuracy (%)": [98, 84, 86, 99]
})

# Create a bar chart with shades of blue
fig = px.bar(data, x="Feature Extraction Method", y="Accuracy (%)",
             title="Feature Comparison Based on Accuracy",
             color="Feature Extraction Method", 
             color_discrete_sequence=colors)  # Use "Blues" for blue shades

st.plotly_chart(fig, use_container_width=True)

# Call to Action
st.markdown("<h2 class='center'>Future Scope & Get Involved</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='section'>
        Future applications include prosthetic control, real-time BCI, and neurological disorder research.
    </div>
    """, unsafe_allow_html=True)
st.markdown("<div class='button-container'><a href='/data_upload' target='_blank'><button style='padding:10px 20px; font-size:16px;'>Try Demo</button></a></div>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='center'><br>© Electrical Engineering Department | IITR </p>", unsafe_allow_html=True)