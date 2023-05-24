import streamlit as st
from PIL import Image
import  numpy as np
import tensorflow as tf
import pickle

def main():
    st.title("Load Your Image")

    uploaded_file = st.file_uploader("Load Image", type=['png', 'jpg', 'jpeg'])

    result = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Loaded Image')
        image = image.resize((720, 720))
        image_array = np.array(image)
        #image_array = image_array / 255.0
        x_pred = np.expand_dims(image_array, axis=0)

        model_name = "model1.pkl"
        with open(model_name, 'rb') as archivo:
            model = pickle.load(archivo)

        prediction = model.predict(x_pred)
        predicted_class = np.argmax(prediction)
        result = f"Image corresponds to class {predicted_class}"

    st.write(result)

if __name__ == '__main__':
    main()
