import pandas as pd
import folium
import base64
from PIL import Image
import io
import streamlit as st
import pydeck as pdk
import numpy as np
import pickle
from streamlit_player import st_player
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from streamlit_folium import folium_static, st_folium
import requests
import io
import time
from tensorflow import saved_model
import random
import plotly.express as px

# Function to reduce the size of the image
def reduce_image_size(image_path):
    response = requests.get(image_path)
    image_row = Image.open(io.BytesIO(response.content)).convert('RGB')
    with image_row as img:
        # Reduce the size of the image
        img.thumbnail((200, 200))
        # Convert the image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Return the base64 encoded image
        return f'data:image/png;base64,{img_str}'
# Function to edit the path
def edit_path(local=False):
    if local:
        df = pd.read_csv("./methane_224_data/smallsize224_all.csv")
    else:
        df = pd.read_csv('https://storage.googleapis.com/methane_source/test_images_demo.csv', engine='pyarrow')

    df = df.dropna()
    return df

def generate_map(df):
    # Create a map
    m = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()], zoom_start=5)

    for idx, row in df.iterrows():
        # Reduce the size of the image and convert it to base64
        #image_path = reduce_image_size(row.new_path)
        image_path = row.new_path

        # Create the html for the popup
        html = f'<img src="{image_path}" width=150>'
        iframe = folium.IFrame(html, width=200+20, height=200+20)

        # Add the marker to the map
        popup = folium.Popup(iframe, max_width=2650)
        folium.Marker([row.Latitude, row.Longitude], color='blue',
                            fill=True, fill_color='blue', popup=popup).add_to(m)

    # Convert the map to HTML
    #m_html = m._repr_html_()

    return m, image_path

def load_image(uploaded_file):
    response = requests.get(uploaded_file)
    image_row = Image.open(io.BytesIO(response.content)).convert('RGB')
    image = image_row.resize((224, 224))
    image_array = np.array(image).astype(float)
    normalized_image = image_array / 255.0
    x_pred = np.expand_dims(image_array, axis=0)
    return x_pred, image_row

def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model

def predict_image(x_pred, model):
    prediction = model.predict(x_pred)
    category_names = {
        0: 'CAFOs',
        1: 'Landfills',
        2: 'Mines',
        3: 'ProcessingPlants',
        4: 'RefineriesAndTerminals',
        5: 'WWTreatment'
    }

    predicted_labels = []
    threshold = {'CAFOs': 0.5,
             'Landfills': 0.5,
             'Mines': 0.5,
             'ProcessingPlants': 0.5,
             'RefineriesAndTerminals': 0.5,
             'WWTreatment': 0.5}

    for i, prob in enumerate(prediction[0]):
        if prob > threshold[category_names[i]]:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    df_predicted = pd.DataFrame({'Category': list(category_names.values()),
                                 'Predicted_Label': predicted_labels})
    df_predicted_prob = pd.DataFrame({'Category': list(category_names.values()),
                                 'Predicted_Label': prediction[0].tolist()})
    if all(label == 0 for label in predicted_labels):
        methane_source = 'no methane source'
    else:
        methane_source = ', '.join(category for category, label in zip(category_names.values(), predicted_labels) if label > 0)

    return df_predicted,category_names, df_predicted_prob, methane_source

def main():
    st.set_page_config(layout="wide")
    if 'df' not in st.session_state:
        st.session_state['df'] = edit_path(local=False)
    image_path=None

    if 'logo' not in st.session_state:
        logo_path = 'https://storage.googleapis.com/methane_source/slb_lewagon_logo.png'
        response_logo = requests.get(logo_path)
        logo = Image.open(io.BytesIO(response_logo.content))
        st.image(logo, use_column_width=False)

    # Display a spinner while the map is loading
    with st.spinner('Loading...'):
        # Create a map
        if 'map' not in st.session_state:
            m, image_path= generate_map(st.session_state['df'])
            model= load_model('original224-denseNet_Tuning_v5_image_normalize_add_images_tf_210.h5')
            st.session_state['map']= m
            st.session_state['model'] = model
            # call to render Folium map in Streamlit

    col1, col2= st.columns([2, 2])
    with col1:
        st.header("MAPPING")
        st.title("select a point")
        with st.spinner('Map Loading...'):
            st_data = st_folium(st.session_state['map'], height=500, width=600)
            d = st_data['last_object_clicked']
            if isinstance(d, dict) and 'lat' in d and 'lng' in d:
                lat = d['lat']
                lon = d['lng']
                df_subset = st.session_state['df'][(st.session_state['df']['Latitude'] == lat) & \
                                   (st.session_state['df']['Longitude'] == lon)]
                if not df_subset.empty:
                    image_path = df_subset.iloc[0]['new_path']
                else:
                    st.write("No matching data found.")

                matching_row = st.session_state['df'][(st.session_state['df']['Latitude'] == lat) & \
                                      (st.session_state['df']['Longitude'] == lon)][['State', 'Country', 'Latitude', 'Longitude']]

                matching_row_html = matching_row.to_html(index=False)
                st.markdown(matching_row_html, unsafe_allow_html=True)
    with col2:
        if st.button("Start Prediction"):
            st.header("Methane source prediction")
            with st.spinner('Predicting...'):
                #if st.session_state['image_path'] is not None:
                ##load image
                x_pred, image_row = load_image(image_path)
                ##do prediction
                df_predicted,category_names, df_predicted_prob, methane_source=predict_image(x_pred, st.session_state['model'])
                ##merge and plot prediction
                merged_df = pd.merge(df_predicted_prob, df_predicted[['Category', 'Predicted_Label']], on='Category', how='left')
                fig = px.bar(merged_df, y='Category', x='Predicted_Label_x',
                            hover_data=['Predicted_Label_y'], labels={'Predicted_Label_y':'Label'},
                            title='Prediction Probabilities', color='Category')
                fig.update_layout(height=400, width=500)
                st.plotly_chart(fig)
                ## select the methane source
                if df_predicted['Predicted_Label'].eq(0).all():
                    methane_source = "No methane source"
                else:
                    # If Predicted_Label is 1, set methane_source to corresponding category name
                    category_keys = {v: k for k, v in category_names.items()}
                    df_predicted['methane_source'] = df_predicted.apply(lambda row: row['Category']\
                        if row['Predicted_Label'] == 1 else None, axis=1)
                    methane_source = df_predicted['methane_source'].dropna().tolist()
                if methane_source == "No methane source":
                    color = "green"
                else:
                    color = "red"

                st.markdown(f"<div style='text-align: center; color: white; background-color: {color}; padding: 20px;'>"
                            f"<h3>Methane Source: {methane_source}</h3></div>", unsafe_allow_html=True)


    if st.button("Finish Presentation"):
        with st.expander("Imagen", expanded=True):
            image_path_ = "https://storage.googleapis.com/methane_source/methane_img.png"
            response = requests.get(image_path_)
            imagen = Image.open(io.BytesIO(response.content))
            # Crear el código CSS para la animación de la imagen
            css = f"""
            @keyframes moveImage {{
                0% {{ transform: translateY(-100%); }}
                100% {{ transform: translateY(100%); }}
            }}
            .moving-image {{
            position: fixed;
            left: 50;
            top: 0;
            animation: moveImage 5s linear infinite;
            }}
            """

            # Agregar el código CSS al markdown
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

            # Mostrar la imagen con la clase CSS para la animación
            st.markdown(f'<img src="{image_path_}" class="moving-image">', unsafe_allow_html=True)

            st.balloons()

if __name__ == "__main__":
    main()
