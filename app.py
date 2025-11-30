import streamlit as st
from PIL import Image 
import torch, base64
# from src.predict import load_model_and_predict

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: right; color: white; font-family: sans-serif;'> Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)

def set_backgroung(image_file):
    with open(image_file, 'rb') as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode()

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{encoded_img}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_backgroung('images/ui/ui_bg.png')

colA, colB = st.columns([2, 1])
with colA:
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image.', use_container_width=True)
        # model, prediction = load_model_and_predict(image)
        # st.write(f'Prediction: {prediction}')

with colB:
    model_choice = st.selectbox("Select Model", ("ResNet18", "Custom CNN", "EfficientNetB0"))

col1, col2, col3 = st.columns(3)
with col1:
    pass