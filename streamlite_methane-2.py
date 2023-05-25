import streamlit as st
from PIL import Image
import numpy as np
import pickle
from streamlit_player import st_player
import time
from streamlit_print_layers import print_layers

## inputs##


def main():
    # page configuration
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


    # title
    st.markdown('<h1 class="title">METHANE SOURCE MAPPING </h1>', unsafe_allow_html=True)

    # load image
    st.markdown('<div class="section"><h3>load image</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("please load image", type=["jpg", "jpeg", "png"])

    #results
    if uploaded_file is not None:
        # image preprocessing
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='image loaded')
        image = image.resize((720, 720))
        image_array = np.array(image)
        normalized_image = image_array / 255.0
        # i need to expand the model to 1 image i nedd the batch_size (batch_size, height, width, channels)
        x_pred = np.expand_dims(normalized_image, axis=0)

        # load model
        model_name = "model2.pkl"
        with open(model_name, 'rb') as archivo:
            model = pickle.load(archivo)

        # predict
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

        if 'button_pressed' not in st.session_state:
                st.session_state.button_pressed = False

        if st.button('') or st.session_state.button_pressed:
                st.session_state.button_pressed = True
            # show results
                st.markdown('<div class="result">', unsafe_allow_html=True)
                st.write(f"this image corresponds to: {predicted_category}")

                # cat != 3
                if predicted_class != 3 and predicted_probability > 0.7:

                    image_path = "red.jpg"
                    image_result = Image.open(image_path)
                    st.image(image_result, caption='we will die')
                    st.markdown("""
                        <style>
                        body {
                            background-color: #ff0000;
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

                # cat ==3
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


                st.markdown('</div>', unsafe_allow_html=True)
                st.title('this is the print of your model:')

                ## we are going to show the leyers of the model for the new image
                st.title('How we know it??')
                if 'button_pressed' not in st.session_state:
                    st.session_state.button_pressed = False

                if st.button('Show model layers') or st.session_state.button_pressed:
                    st.session_state.button_pressed = True

                    output_images = print_layers(x_pred=x_pred, model=model)

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

if __name__ == '__main__':
    main()
