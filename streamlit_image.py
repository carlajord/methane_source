import pandas as pd
import folium
import base64
from PIL import Image
import io
import streamlit as st

# Function to reduce the size of the image
def reduce_image_size(image_path):
    with Image.open(image_path) as img:
        # Reduce the size of the image
        img.thumbnail((200, 200))

        # Convert the image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Return the base64 encoded image
        return f'data:image/png;base64,{img_str}'

# Function to edit the path
def edit_path():
    df = pd.read_csv("./methane_224_data/smallsize224_all.csv")
    df = df.dropna()
    root = 'methane_224_data/'
    df['new_path'] = df.img_dir.apply(lambda x: x.replace('F:\\\\CNOOC_testing\\\\Methane_dataset\\\\METHANE_PROJECT\\\\smallsize224\\\\', root)\
        .replace('F:\\\\CNOOC_testing\\\\Methane_dataset\\\\METHANE_PROJECT\\\\smallsize224_val\\\\', root)\
        .replace('F:\\\\CNOOC_testing\\\\Methane_dataset\\\\METHANE_PROJECT\\\\smallsize224_test\\\\', root)\
        .replace("\\\\", "/"))
    df['new_path'] = df.new_path.apply(lambda x: x.replace("\\","/"))
    df.Type = df.Type.astype(str)
    df_test = df.loc[df['dataset'] == "test"]
    return df_test

# Main function
def main():
    df = edit_path()

    # Create a map
    m = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()], zoom_start=2)

    for idx, row in df.iterrows():
        # Reduce the size of the image and convert it to base64
        image_path = reduce_image_size(row.new_path)

        # Create the html for the popup
        html = f'<img src="{image_path}" width=200>'
        iframe = folium.IFrame(html, width=200+20, height=200+20)

        # Add the marker to the map
        popup = folium.Popup(iframe, max_width=2650)
        folium.Marker([row.Latitude, row.Longitude], popup=popup).add_to(m)

    # Display the map
    m_html = m._repr_html_()
    st.components.v1.html(m_html, width=800, height=600)

if __name__ == "__main__":
    main()
