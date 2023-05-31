import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import requests
import pandas as pd
import os


class_names = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

model = tf.keras.models.load_model('tomatoleafDetection.hdf5')

# Get the total number of chunk files
# chunk_files = [file for file in os.listdir() if file.startswith('chunk_')]
# num_chunks = len(chunk_files)
#
# merged_data = []
#
# for i in range(num_chunks):
#     with open(f'chunk_{i}.pkl', 'rb') as file:
#         chunk_data = pickle.load(file)
#         merged_data.extend(chunk_data)
#
# similarity = merged_data

# split_size = 10000  # Number of elements per split
# # Split the data into smaller chunks
# chunks = [similarity[i:i+split_size] for i in range(0, len(similarity), split_size)]
# for i, chunk in enumerate(chunks):
#     with open(f'chunk_{i}.pkl', 'wb') as file:
#         pickle.dump(chunk, file)

st.title("Tomato Leaf Disease Prediction")

file = "hello"
file = st.file_uploader('Please upload an image', type=['jpg', 'png'])

import numpy as np


image = np.array(Image.open(BytesIO(file)))
images_batch = np.expand_dims(image, 0)
plt.imshow(images_batch[0].numpy().astype("uint8"))
first_image = images_batch[0].numpy().astype("uint8")
print("first image to predict")
plt.imshow(first_image)

batch_prediction = model.predict(images_batch)
print("predicted label: ", class_names[np.argmax(batch_prediction[0])])
st.write(class_names[np.argmax(batch_prediction[0])])






