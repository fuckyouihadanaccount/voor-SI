import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "https://mugshot-detector.onrender.com/predict"

st.title("Check of een foto een mugshot is")

file = st.file_uploader("Upload een foto", type=["jpg", "jpeg", "png"])

if file:
    if st.button("Analyseer foto"):
        with st.spinner("Bezig met analyseren..."):
            files = {"file": (file.name, file.getvalue())}
            
            try:
                response = requests.post(API_URL, files=files, timeout=15)
                response.raise_for_status()  # Raises HTTPError if not 200
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()
                
            try:
                data = response.json()
            except ValueError:
                st.error("API did not return valid JSON")
                st.write("Raw response from API:")
                st.write(response.text)
                st.stop()

        st.subheader("Resultaten")
        st.write("Reconstructie-error:", data.get("reconstruction_error"))
        st.write("Mugshot gedetecteerd:", data.get("is_mugshot"))

        img_data = base64.b64decode(data.get("image_base64", ""))
        img = Image.open(io.BytesIO(img_data))
        st.image(img, caption="Resultaat (geblurred indien mugshot)", use_container_width=True)

