import streamlit as st
from PIL import Image
import numpy as np
import pickle
from streamlit_player import st_player

def main():
    # page configuraion
    st.markdown("""
        <style>
        .title {
            color: #3366ff;
            font-size: 32px;
            text-align: center;
            margin-bottom: 50px;
        }
        .section {
            margin-bottom: 30px;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
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
        # i need to expand the model to 1 image i nedd the batch_size (batch_size, height, width, channels)
        x_pred = np.expand_dims(image_array, axis=0)

        # load model
        model_name = "model1.pkl"
        with open(model_name, 'rb') as archivo:
            model = pickle.load(archivo)

        # predict
        prediction = model.predict(x_pred)
        predicted_class = np.argmax(prediction)
        predicted_probability = np.max(prediction)

        # cat name
        category_names = {
            0: 'CAFOs',
            1: 'Landfills',
            2: 'Mines',
            3: 'Negative',
            4: 'ProcessingPlants',
            5: 'RefineriesAndTerminals',
            6: 'WWTreatment'
        }
        predicted_category = category_names[predicted_class]

        # show results
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.write(f"this image corresponds to: {predicted_category}")

        # cat != 3
        if predicted_class != 3 and predicted_probability > 0.7:

            image_path = "red.jpg"
            image_result = Image.open(image_path)
            st.image(image_result, caption='we will die')

        elif predicted_class != 3  and predicted_probability < 0.7:

            image_path = "yellow.png"
            image_result = Image.open(image_path)
            st.image(image_result, caption='maybe we can save us')

        # cat ==3
        elif predicted_class == 3:
            image_path = "green.png"
            image_result = Image.open(image_path)
            st.image(image_result, caption='all good')


        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
