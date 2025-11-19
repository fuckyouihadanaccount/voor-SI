import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical
from PIL import Image
import random
import os

#dir
os.chdir('C:\\Users\\elija\\Downloads\\Data Science\\Smart industry\\College 4')

#plot function
def plot(x, p=None, labels=False):
    plt.figure(figsize=(20,2))
    for i in range(10): # refers to the first 10 images that we are plotting
        plt.subplot(1, 10, i+1) # 1 row and 10 columns; the index for the subplot 
        plt.imshow(x[i].reshape(28, 28), # this will show the image except we will have to reshape to 28 by 28 because we flattened it in the previous task.
                   cmap = 'binary') # "binary" so that we see the black and white images as they are.
        plt.xticks([])
        plt.yticks([])
        if labels: # if labels is true,
            plt.xlabel(np.argmax(p[i])) # then we also want to label our X axis.
    plt.show()
    return

##blurring
#blur function
def anonymize_face_simple(image, factor=3.0):
    (h, w) = image.shape[:2]
    kW, kH = int(w / factor), int(h / factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)

#get images
image_name = '00004_2_F.png'
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

#get face
proportions = image[200:800,200:600]
face = anonymize_face_simple(proportions, factor=3.0)

#blur to image
x, y = 200, 200 #begin x,y van face
h, w = face.shape[:2] #w/h van face
image[y:y+h, x:x+w] = face

#show image
plt.yticks([]),plt.xticks([])
plt.imshow(image,cmap='grey')


##denoising
#get images
noimages = []
dimages, cimages, aimages = [],[],[]
for name in os.listdir():
    if '.png' in name:
        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        image = np.array((image.resize((28,28), Image.LANCZOS)))
    aimages.append(image)
    if 'distorted' in name:
        dimages.append(image)
    if 'clean' in name:
        cimages.append(image)

for x in [dimages,cimages,aimages]:
    random.shuffle(x)
    x = np.stack(x) / 255
aimages,dimages,cimages = np.array(aimages),np.array(dimages),np.array(cimages)

#add noise
gaussian = (cv2.randn(np.zeros((28,28),dtype=np.uint8),128,20)).astype(np.uint8)
for image in cimages:
    noimages.append(cv2.add(image, gaussian))
noimages = np.array(noimages)


#denoising function
def train_autoencoder(X_train_noisy, X_train):
    """
    Deze functie heeft 2 datasets nodig, de afbeeldingen met ruis en die zonder ruis,
    De functie geeft de getrainde autoencoder, en print de training loss.
    de .predict methode kan gebruikt worden om de predictions te krijgen
    """
    input_image = Input(shape=(784,))
    encoded = Dense(64, activation='relu')(input_image)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_image, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    autoencoder.fit(
        X_train_noisy, X_train, 
        epochs=100,
        batch_size=512, 
        validation_split=0.2, # Use a validation split of 20%,
        verbose=False, # set verbose to false because we don't want to actually use any build logs. 
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            LambdaCallback(on_epoch_end=lambda e,l: print('{:.4f}'.format(l['val_loss']), end=' _ '))
        ]
    )
    print(' _ ')
    print('Training is complete!')
    return autoencoder

#train autoencoder
noise = noimages.reshape(-1, 784).astype('float32') / 255.0
images = cimages.reshape(-1, 784).astype('float32') / 255.0
autoencoder = train_autoencoder(noise, images)

#Prediction
denoised = dimages.reshape(-1, 784).astype('float32') / 255.0
prediction = autoencoder.predict(denoised)
plot(dimages)
plot(prediction.reshape(len(prediction), 28, 28))
