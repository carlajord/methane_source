import streamlit as st
from PIL import Image
import numpy as np
import pickle
from streamlit_player import st_player
import time
from streamlit_print_layers import print_layers
import matplotlib.pyplot as plt
import tensorflow as tf

st.session_state['x_pred']= None
st.session_state['layer']=None
st.session_state['guess']= None

def load_image(uploaded_file):
    image_row= Image.open(uploaded_file).convert('RGB')
    st.image(image_row, caption='image loaded')
    image = image_row.resize((720, 720))
    image_array = np.array(image)
    normalized_image = image_array / 255.0
    x_pred = np.expand_dims(normalized_image, axis=0)
    return x_pred, image_row

def print_layers(x_pred, model):
    layer_names = []
    for layer in model.layers:
        if 'flatten' in layer.name:
            break
        layer_names.append(layer.name)
    def out_layer(model, x_pred, layer_name):
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        output = intermediate_model.predict(x_pred)
        return output
    for layer_name in layer_names:
        out=[]
        outputs = out_layer(model, x_pred, layer_name)
        for i in range(outputs.shape[3]):
                output_image = outputs[0, :, :, i]
                out.append(output_image)
    return out

def predict_image(x_pred):
    model_name = "model2.pkl"
    with open(model_name, 'rb') as archivo:
        model = pickle.load(archivo)
    prediction = model.predict(x_pred)
    predicted_class = np.argmax(prediction)
    predicted_probability = np.max(prediction)

    category_names={0: 'CAFOs',
                    1: 'Landfills',
                    2: 'Mines',
                    3: 'Negative',
                    4: 'ProcessingPlants',
                    5: 'RefineriesAndTerminals',
                    6: 'WWTreatment'}

    predicted_category = category_names[predicted_class]
    return predicted_category, predicted_class, predicted_probability, model

def main():
    st.markdown("""
        <style>
        body {
            background-color: #e8f5e9; /* A light green color */
        }
        .big-title {
            color: #1b5e20; /* A darker green color */
            font-size: 50px;
            text-align: center;
            margin-bottom: 50px;
            font-family: 'Courier New', Courier, monospace; /* A more eye-catching font */
        }
        .title {
            color: #388e3c; /* A darker green color */
            font-size: 32px;
            text-align: center;
            margin-bottom: 50px;
            font-family: 'Courier New', Courier, monospace; /* A more eye-catching font */
        }
        .section {
            margin-bottom: 30px;
            text-align: center; /* Center the section */
        }
        .result {
            margin: 30px auto; /* Center the result vertically and horizontally */
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
            text-align: center; /* Center the text */
            color: #1565c0; /* Blue text color */
        }
        </style>
        """, unsafe_allow_html=True)

    page = st.sidebar.radio("CHOOSE YOUR PAGE", ["input image", "inside the model", "Try to guess", "See the answer"])

    if page == "input image":
        st.title("METHANE SOURCE MAPPING")
        uploaded_file = st.file_uploader("please load image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            x_pred, image_row = load_image(uploaded_file)
            st.session_state['x_pred'] = x_pred
            if st.button('Show Prediction Result'):
                predicted_category, predicted_class, predicted_probability, model= predict_image(x_pred)
                st.markdown('<div class="result">', unsafe_allow_html=True)
                st.write(f"This image corresponds to: {predicted_category}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.session_state['predicted_category'] = predicted_category
                st.session_state['predicted_class'] = predicted_class
                st.session_state['predicted_probability'] = predicted_probability
                st.session_state['model'] = model

                if predicted_class != 3 and predicted_probability > 0.7:
                    image_path = "red.jpg"
                    image_result = Image.open(image_path)
                    st.image(image_result, caption='we will die')
                    st.markdown("""
                        <style>
                         { margin-bottom: 30px;
                         text-align: center; /* Center the section */
                        }
                        </style>
                        """, unsafe_allow_html=True)
                elif predicted_class != 3  and predicted_probability < 0.7:
                    image_path = "yellow.png"
                    image_result = Image.open(image_path)
                    st.image(image_result, caption='maybe we can save us')
                    st.markdown("""
                        <style>
                        body {
                            background-color: #FFFF00;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                elif predicted_class == 3:
                    image_path = "green.png"
                    image_result = Image.open(image_path)
                    st.image(image_result, caption='all good')
                    st.markdown("""
                        <style>
                        body {
                            background-color: #008000;
                        }
                        </style>
                        """, unsafe_allow_html=True)

    elif page == "inside the model":
        st.title("HOW OR MODEL LOOKS LIKE")

        if st.session_state['x_pred'] is not None:
            if 'button_pressed' not in st.session_state:
                st.session_state.button_pressed = False
            if st.button('show model layers') or st.session_state.button_pressed:
                st.session_state.button_pressed = True

                output_images = print_layers(x_pred=st.session_state['x_pred'], model=st.session_state['model'])
                def normalize_image(image):
                    min_val = np.min(image)
                    max_val = np.max(image)
                    normalized_image = (image - min_val) / (max_val - min_val)
                    if len(normalized_image.shape) == 2:
                        normalized_image = np.stack((normalized_image,)*3, axis=-1)
                    return normalized_image
                normalized_images = [normalize_image(img) for img in output_images]
                image_index = st.slider('Select image layer:', 0, len(normalized_images) - 1, 0)
                st.image(normalized_images[image_index], caption=f'Layer {image_index + 1}', use_column_width=True)
        else:
            st.write('we need the input for the model')

    elif page == "Try to guess":
        st.title("HERE SOME SIMILAR IMAGES TO HELP YOU")
        if st.session_state['layer'] is not None:
            pass
        else:
            st.write('we can not classify withow a model')

    elif page == "See the answer":
        st.title("THIS IS YOUR FINAL PREDICTION")
        if st.session_state['guess'] is not None:
            pass
        else:
            st.write('do not cheat go step by step')

if __name__ == "__main__":
    main()
