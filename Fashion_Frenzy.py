import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

st.title("Fashion Frenzy 🛍")
st.subheader("Welcome the our shopping website!!")
# Load the CSV files
csv_files = {
    'Hot selling🔥!!': '/Users/jiaheyu/Desktop/Data_analytics/cluster0.csv',
    'Bags👜': '/Users/jiaheyu/Desktop/Data_analytics/cluster1.csv',
    'Flats & Sneakers👟': '/Users/jiaheyu/Desktop/Data_analytics/cluster2.csv',
    'Bottoms & Dresses👗': '/Users/jiaheyu/Desktop/Data_analytics/cluster3.csv',
    'Sweats🥼': '/Users/jiaheyu/Desktop/Data_analytics/cluster4.csv',
    'Tops👚': '/Users/jiaheyu/Desktop/Data_analytics/cluster5.csv',
    'Boots & Heels👢': '/Users/jiaheyu/Desktop/Data_analytics/cluster6.csv',
}

dfs = {}
for category, file_name in csv_files.items():
    dfs[category] = pd.read_csv(file_name)

# Create a dropdown menu for the category keyword selection
category_query = st.selectbox('Select a category from the dropdown menu and click Seach:', list(csv_files.keys()), index=0)

# Create a search button to trigger the search
search_button = st.button('Search')
st.write('Want to see more? Keep Clicking Search!')

# Display the selected product and similar products
if search_button:
    if category_query in dfs:
        # Get a random image from the CSV file
        selected_image = dfs[category_query].sample(n=1).iloc[0]
        
        # Get the pixel values for the selected image
        pixels = np.array(selected_image.iloc[1:])
        
        # Reshape the pixels into a 28x28 array
        image_array = pixels.reshape((28, 28))
        
        # Convert the array to a PIL image
        pil_image = Image.fromarray(np.uint8(image_array * 255))
        
        # Display the selected image in Streamlit
        st.image(pil_image, caption=selected_image.iloc[0], width=200)
        
        # Display a message to show more recommendations
        st.write('Here are more recommendations:')
        
        # Sample up to 10 additional images from the CSV file
        filtered_images = dfs[category_query].sample(n=min(9, len(dfs[category_query])-1))
        
        # Display the recommended images in rows
        row1, row2, row3 = st.columns(3)
        for i in range(len(filtered_images)):
            # Get the pixel values for the current image
            pixels = np.array(filtered_images.iloc[i, 1:])
            
            # Reshape the pixels into a 28x28 array
            image_array = pixels.reshape((28, 28))
            
            # Convert the array to a PIL image
            pil_image = Image.fromarray(np.uint8(image_array * 255))
            
            # Display the image in Streamlit
            if i < 3:
                with row1:
                    st.image(pil_image, caption=filtered_images.iloc[i, 0], width=100)
            elif i < 6:
                with row2:
                    st.image(pil_image, caption=filtered_images.iloc[i, 0], width=100)
            else:
                with row3:
                    st.image(pil_image, caption=filtered_images.iloc[i, 0], width=100)
    else:
        st.write('No results found for the selected category.')