from azureml.core import Model
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import requests
import os
import cv2 
import json

# Set search headers and URL
headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

def init():
    global model
    model_dir = Model.get_model_path('pokemon-classifier')
    model = load_model(os.path.join(model_dir, 'model.h5'))

def run(raw_data):
    image_dim = 128
    image_url = json.loads(raw_data)['image_url']
    with open('temp.jpg', 'wb') as file:
        download = requests.get(image_url, headers=headers)
        file.write(download.content)
    image = cv2.imread('temp.jpg')
    image = cv2.resize(image, (image_dim, image_dim))
    image = tf.cast(image, tf.float32)
    pred = model.predict(np.array([image])).tolist()
    return json.dumps({'prediction': pred})
