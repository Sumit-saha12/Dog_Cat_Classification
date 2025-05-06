import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
from PIL import Image
st.title('Dog vs Cat classifier')

image  = st.file_uploader('Upload a image')
if image:
    st.image(image)
    image_new = Image.open(image)
# image_read = cv2.imread(image)
    resize_image = cv2.resize(np.array(image_new),(150,150))
    new_resize_image = resize_image.reshape(1,150,150,3)

    model = keras.models.load_model('classification_model.h5')


    result = np.array(model.predict(new_resize_image))[0][0]

    if result>=0 and result<0.8:
        st.write('Prediction is: Cat')
    else:
        st.write('Prediction is: Dog')