# frontend.py
import streamlit as st
import requests
import time
import subprocess

FLASK_URL = "http://127.0.0.1:5000"
FLASK_FILE_PATH = "../backend.py"  # Path to your backend.py file

# Function to start Flask app using subprocess
def start_flask():
    """Start the Flask backend using subprocess."""
    try:
        # Start Flask server in the background
        subprocess.Popen(['python', FLASK_FILE_PATH])
        time.sleep(2)  # Give Flask a moment to start
        st.write("Flask server started.")
    except Exception as e:
        st.error(f"Error starting Flask server: {e}")

# Check if Flask is already running
def check_flask_running():
    """Check if Flask is already running by pinging the backend."""
    try:
        response = requests.get(f"{FLASK_URL}/ping")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Set the page configuration
st.set_page_config(
    page_title="EEG Task Classifier",
    page_icon="🧠",
    layout="wide"
)

# Main Title
st.title("🧠 EEG Task Classifier - Motor Imagery and Movement")
st.markdown(""" 
Please follow the instructions carefully:
- **Step 1:** Wear the EEG headset properly. Ensure all electrodes are making good contact.
- **Step 2:** Sit comfortably. Relax your muscles and minimize head movements.
- **Step 3:** Focus on performing one of the following tasks when instructed:
  - 🖐️ **Left-hand movement**
  - ✋ **Left-hand imagery** (imagine moving your left hand without actual motion)
  - 🤚 **Right-hand movement**
  - ✋ **Right-hand imagery** (imagine moving your right hand without actual motion)
- **Step 4:** Click the **Predict** button and immediately start the task. Hold it consistently for 5–10 seconds.

---

""")


# Predict Button
predict_button = st.button("🎯 Predict Task")

if predict_button:
    with st.spinner('Collecting EEG data... Please perform the task now!'):
        # Trigger the backend.py
        try:
        # Send a GET request to the Flask backend to trigger the prediction
            start_flask()  
            time.sleep(10)
            response = requests.get(f"{FLASK_URL}/predict")
            
            # Check if the response is successful
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Predicted Task: {prediction}")
            else:
                st.error(f"Error: {response.status_code}, Unable to get prediction")

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

# Navigation to Other Pages
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/data_upload.py", label="Data Upload")
st.sidebar.page_link("pages/visualization.py", label="Visualization")
st.sidebar.page_link("pages/augmentation.py", label="Data Augmentation (VAE)")