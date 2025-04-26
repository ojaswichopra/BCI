import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Title ---
# Page Configuration
st.set_page_config(page_title="EEG Data Upload", layout="wide")

st.title("Data Augmentation with Variational Autoencoders (VAE)")

# --- Introduction to VAE ---
st.markdown("""
**Variational Autoencoders (VAE)** are generative models used to augment datasets by generating new, realistic data samples. 
In this project, We utilized a VAE to augment the STFT images of EEG signal data, which helped improve the diversity of the dataset and ultimately enhanced model performance.

VAE consists of two main components:
- **Encoder**: Compresses the input data (EEG signals) into a latent space.
- **Decoder**: Reconstructs the input data from the latent space and generates augmented data.
""")

# --- Original vs Augmented Images ---
st.header("Original vs Augmented Images")
st.markdown("""
Below are the comparisons between the original EEG signal images and the augmented images generated using the VAE.
""")
# Load and display original and augmented images (replace with actual file paths)
original_image_path = "../Assets/Original.png"  # Replace with the correct path
augmented_image_path = "../Assets/Augmented.png"  # Replace with the correct path

# --- Interactive Slider for Image Selection ---
option = st.selectbox("Select EEG Channel:", ["Original", "Augmented"])
if option == "Original":
    st.image(original_image_path, caption="Original EEG Signal", width=300)
else:
    st.image(augmented_image_path, caption="Augmented EEG Signal", width=300)

# --- Performance Comparison Before and After Augmentation ---
st.header("Model Performance Comparison")
st.markdown("""
Below is a comparison of model performance before and after using VAE for data augmentation. The model accuracy improves as a result of augmented data, which adds diversity to the dataset.
""")

# Example performance data before and after augmentation (replace with actual data)
# Example performance data
models = ['Original Data', 'Augmented Data']
accuracy = [0.84, 0.87]

# Define custom shades of blue
colors = ['#4A90E2', '#005BAC']  # Lighter and darker shades of blue

# Plotting the bar chart
fig, ax = plt.subplots()
bars = ax.bar(models, accuracy, color=colors)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}", ha='center', va='bottom')

ax.set_ylabel('Accuracy')
ax.set_ylim(0.8, 0.9)
ax.set_title('Model Performance Before and After Data Augmentation')

# Display the plot
st.pyplot(fig)

# --- VAE Architecture Diagram ---
st.header("VAE Architecture")
st.markdown("""
Below is the architecture of the Variational Autoencoder used in this project. It consists of two main components:
""")
col1, col2, col3 = st.columns([1, 3, 1])  # Center image using the middle column
# You can replace the path with your own VAE architecture diagram
vae_architecture_image = "../Assets/vae.png"  # Replace with the correct path
with col2:
    st.image(vae_architecture_image, caption="VAE Architecture", width=600)
    
# --- Code Snippet for VAE Model ---
st.header("VAE Model Code Snippet")
st.markdown("""
Below is the code snippet used to implement the Variational Autoencoder for EEG data augmentation. You can modify it based on your requirements.
""")

vae_code = """
# VAE Model Implementation (Simplified Example)
import tensorflow as tf
from tensorflow.keras import layers, models

# Encoder
def build_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(1, 50), strides=(1,25))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x) 
    x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    return models.Model(inputs, [z_mean, z_log_var])

# Decoder
def build_decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(6 * 33 * 64, activation='relu')(latent_inputs)
    x = Reshape((6, 33, 64))(x)
    x = Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=32, kernel_size=(1, 50), activation='relu', strides=(1,25), padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid', name='decoder_output')(x)

    # Ensure output matches input shape
    from keras.layers import Resizing
    outputs = Resizing(input_shape[0], input_shape[1])(x)
    return models.Model(latent_inputs, outputs)

# VAE model combining encoder and decoder
def build_vae(input_shape, latent_dim=32):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim)
    inputs = layers.Input(shape=input_shape)
    z_mean, z_log_var = encoder(inputs)
    z = z_mean  # Simplified for example; could be sampled from the latent space
    reconstructed = decoder(z)
    vae = models.Model(inputs, reconstructed)
    return vae
"""
# Display the code snippet in Streamlit
st.code(vae_code, language='python')

st.sidebar.header("Navigation")
st.sidebar.page_link("pages/data_upload.py", label="Data Upload")
st.sidebar.page_link("pages/visualization.py", label="Visualization")
st.sidebar.page_link("pages/classification.py", label="CNN Classification")
