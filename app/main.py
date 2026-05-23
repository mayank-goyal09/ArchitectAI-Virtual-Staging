import streamlit as st
import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processor import process_room_image

st.set_page_config(page_title="AI Virtual Stager", page_icon="🏡", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏡 AI Virtual Stager")
st.write("Transform empty rooms into beautifully staged spaces using AI.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an empty room photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Original Room", use_container_width=True)

with col2:
    st.subheader("Output")
    if uploaded_file:
        if st.button("Generate Staging ✨"):
            st.info("Processing and generating... (Skeleton Maker & ControlNet)")
            # Simulated output
            st.warning("This is a placeholder. Logic will be implemented in src/generator.py")
    else:
        st.info("Upload a photo to see the results.")
