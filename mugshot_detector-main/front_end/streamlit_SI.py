import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from PIL import Image
import random
import os

##blurring
def anonymize_face_simple(image, factor=3.0):
    (h, w) = image.shape[:2]
    kW, kH = int(w / factor), int(h / factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)

#crop
def crop_and_resize(img, size=300):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((size, size), Image.LANCZOS)
    return img_resized

#get images
BASE_URL = "https://raw.githubusercontent.com/fuckyouihadanaccount/voor-SI/main/mugshot_detector-main/fotos/images/"
def load_image_from_github(filename):
    url = BASE_URL + filename
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Could not load {url}")

    # Load from bytes → PIL
    img = Image.open(io.BytesIO(response.content)).convert("L")
    return img  # return raw PIL image, not resized yet

faceimages = ['00004_2_F.png','01546_1_F.png','01548_1_F.png','01549_1_F.png']
fimages = []
for filename in faceimages:
    img = load_image_from_github(filename)
    img = crop_and_resize(img, size=300)
    img = np.array(img)
    fimages.append(img)

#get face
# fig, ax = plt.subplots(1,4) 
# for i in range(0,4):
#     x = fimages[i]
#     ax[i].set_yticks([]),ax[i].set_xticks([])
#     ax[i].imshow(x, cmap='gray')

st.write('# Kies een afbeelding:')
if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0
if "show_blur" not in st.session_state:
    st.session_state.show_blur = False
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("Vorige"):
        st.session_state.show_blur = False
        st.session_state.img_idx = max(0, st.session_state.img_idx- 1)
with col2:
    if st.button("Volgende"):
        st.session_state.show_blur = False
        st.session_state.img_idx = min(3, st.session_state.img_idx + 1)
d= st.session_state.img_idx
image = fimages[d]
fd = {0:[50,260,75,230],
      1:[80,210,75,190],
      2:[40,250,60,200],
      3:[50,260,75,210]}

proportions = image[fd[d][0]:fd[d][1],fd[d][2]:fd[d][3]]
face = anonymize_face_simple(proportions, factor=3.0)
y, x = fd[d][0], fd[d][2] #begin x,y van face
h, w = face.shape[:2] #w/h van face
image2 = image.copy()
image2[y:y+h, x:x+w] = face

col4, col5 = st.columns([1, 1])
with col5:
    if st.button("Show Blur", key="blur_btn",type="primary"):
        st.session_state.show_blur = True
    st.markdown(f"<b>Afbeelding {d+1}</b> <br> Dimensies: {image.shape} <br> Gezicht: {fd[d]}", unsafe_allow_html=True)
with col4:
    if st.session_state.show_blur:
        st.image(image2, use_container_width=False)
    else:
        st.image(image, use_container_width=False)
    
# ---------- Plot function ----------
def plot(x,p=None, labels=False):
    fig, ax = plt.subplots(1, 5, figsize=(20, 2))
    for i in range(5):
        ax[i].imshow(x[i].reshape(28, 28), cmap='binary')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if labels:
            ax[i].set_xlabel(np.argmax(p[i]))
    st.pyplot(fig)
    plt.close(fig)

# ---------- Load images ----------
noimages = []
IMG_FOLDER = r"C:\\Users\\elija\\Downloads\\Data Science\\Smart industry\\College 4\\images"

@st.cache_data
def load_images(img_folder):
    aimages, cimages, dimages = [], [], []
    for filename in os.listdir(img_folder):
        if not filename.lower().endswith('.png'):
            continue  # skip non-PNG files

        full_path = os.path.join(img_folder, filename)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[Warning] Failed to load: {full_path}")
            continue
        image_pil = Image.fromarray(image)
        image_resized = np.array(image_pil.resize((28, 28), Image.LANCZOS))

        aimages.append(image_resized)
        if 'clean' in filename.lower():
            cimages.append(image_resized)
        elif 'distorted' in filename.lower():
            dimages.append(image_resized)
    for lst in [dimages, cimages, aimages]:
        random.shuffle(lst)
    aimages, dimages, cimages = np.array(aimages)/255.0, np.array(dimages)/255.0, np.array(cimages)/255.0
    return aimages, cimages, dimages

# Load images once (cached)
aimages, cimages, dimages = load_images(IMG_FOLDER)

# Plot clean images
# ---------- Add noise ----------
gaussian = (cv2.randn(np.zeros((28,28),dtype=np.uint8),128,20)).astype(np.uint8)
for image in cimages:
    img_uint8 = (image * 255).astype(np.uint8)
    noisy = cv2.add(img_uint8, gaussian)
    noimages.append(noisy / 255.0)
noimages = np.array(noimages)

# Show example images
imgs = [cimages[6], noimages[6], gaussian]
fig, ax = plt.subplots(1, 3, figsize=(10,4))
for i in range(3):
    ax[i].imshow(imgs[i], cmap='gray')
    ax[i].set_xticks([])
    ax[i].set_yticks([])


# ---------- Train autoencoder ----------
@st.cache_resource
def load_autoencoder():
    from tensorflow.keras.models import load_model
    return load_model("denoising_model.keras")

autoencoder = load_autoencoder()

denoised = dimages.reshape(-1, 784).astype('float32')
prediction = autoencoder.predict(denoised)

st.write('# Denoising model')

st.write("Onze'schone afbeeldingen':")
plot(cimages)

st.write('Verandering door noise:')
st.pyplot(fig)
plt.close(fig)

st.write('Output van het denoising model:')
plot(dimages)
plot(prediction.reshape(len(prediction), 28, 28))



