import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import parse_qs
from urllib.parse import parse_qs, urlencode

st.title('Fashion Frenzy')

# Load the CSV files
csv_files = {
    'Hotselling': '/Users/jiaheyu/Desktop/Data_analytics/cluster0.csv',
    'Bags': '/Users/jiaheyu/Desktop/Data_analytics/cluster1.csv',
    'Flats & Sneakers': '/Users/jiaheyu/Desktop/Data_analytics/cluster2.csv',
    'Bottoms & Dresses': '/Users/jiaheyu/Desktop/Data_analytics/cluster3.csv',
    'Sweats': '/Users/jiaheyu/Desktop/Data_analytics/cluster4.csv',
    'Tops': '/Users/jiaheyu/Desktop/Data_analytics/cluster5.csv',
    'Boots & Heels': '/Users/jiaheyu/Desktop/Data_analytics/cluster6.csv',
}

dfs = {}
for category, file_name in csv_files.items():
    dfs[category] = pd.read_csv(file_name)

# Create a text input field for the category keyword and a button to trigger the search
category_query = st.text_input('Search products by category:')
search_button = st.button('Search')

# Get the query parameters of the current URL
query_params_str = st.experimental_get_query_params()
query_params = parse_qs(query_params_str)

# If a clicked_image_key query parameter exists, this means that an image has been clicked
clicked_image_key = query_params.get('clicked_image_key', None)
query_params = parse_qs(query_params_str)

# Randomly show the images from the CSV file that matches the category query
if search_button:
    if category_query in dfs:
        # Sample up to 10 images from the CSV file
        filtered_images = dfs[category_query].sample(n=min(10, len(dfs[category_query])))
        
        # Loop over the filtered images and display them
        for i in range(len(filtered_images)):
            # Get the pixel values for the current image
            pixels = np.array(filtered_images.iloc[i, 1:])
            
            # Reshape the pixels into a 28x28 array
            image_array = pixels.reshape((28, 28))
            
            # Convert the array to a PIL image
            pil_image = Image.fromarray(np.uint8(image_array * 255))
            
            # Display the image in Streamlit with a unique key
            st.image(pil_image, caption=filtered_images.iloc[i, 0], width=200, use_column_width=True, metadata={'query_params': query_params_str})
        
       # If an image has been clicked, display a message and recommend more images
if clicked_image_key is not None:
    clicked_image_index = int(clicked_image_key.split('_')[1])
    query_params_str = st.image_clicked.metadata['query_params']

    # Compute the cosine similarity between the clicked image and all the other images in the same category
    clicked_image_pixels = np.array(dfs[category_query].iloc[clicked_image_index, 1:])
    similarity_scores = cosine_similarity([clicked_image_pixels], dfs[category_query].iloc[:, 1:])

    # Sort the similarity scores in descending order and get the top 3 images
    top_indices = np.argsort(similarity_scores[0])[::-1][1:4]
    
    # Display the top 3 similar images
    for index in top_indices:
        pixels = np.array(dfs[category_query].iloc[index, 1:])
        image_array = pixels.reshape((28, 28))




import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import parse_qs, urlencode

st.title('Fashion Frenzy')

# Load the CSV files
csv_files = {
    'Hotselling': '/Users/jiaheyu/Desktop/Data_analytics/cluster0.csv',
    'Bags': '/Users/jiaheyu/Desktop/Data_analytics/cluster1.csv',
    'Flats & Sneakers': '/Users/jiaheyu/Desktop/Data_analytics/cluster2.csv',
    'Bottoms & Dresses': '/Users/jiaheyu/Desktop/Data_analytics/cluster3.csv',
    'Sweats': '/Users/jiaheyu/Desktop/Data_analytics/cluster4.csv',
    'Tops': '/Users/jiaheyu/Desktop/Data_analytics/cluster5.csv',
    'Boots & Heels': '/Users/jiaheyu/Desktop/Data_analytics/cluster6.csv',
}

dfs = {}
for category, file_name in csv_files.items():
    dfs[category] = pd.read_csv(file_name)

# Create a text input field for the category keyword and a button to trigger the search
category_query = st.text_input('Search products by category:')
search_button = st.button('Search')

# Get the query parameters of the current URL
query_params_str = st.experimental_get_query_params()
query_params = parse_qs(query_params_str)

# If a clicked_image_key query parameter exists, this means that an image has been clicked
clicked_image_key = query_params.get('clicked_image_key', None)

# Randomly show the images from the CSV file that matches the category query
if search_button:
    if category_query in dfs:
        # Sample up to 10 images from the CSV file
        filtered_images = dfs[category_query].sample(n=min(10, len(dfs[category_query])))
        
        # Loop over the filtered images and display them
        for i in range(len(filtered_images)):
            # Get the pixel values for the current image
            pixels = np.array(filtered_images.iloc[i, 1:])
            
            # Reshape the pixels into a 28x28 array
            image_array = pixels.reshape((28, 28))
            
            # Convert the array to a PIL image
            pil_image = Image.fromarray(np.uint8(image_array * 255))
            
            # Display the image in Streamlit with a unique key
            st.image(pil_image, caption=filtered_images.iloc[i, 0], width=200, use_column_width=True)
        
# If an image has been clicked, display a message and recommend more images
if clicked_image_key is not None:
    clicked_image_index = int(clicked_image_key.split('_')[1])
    query_params_str = st.session_state.clicked_image_key

    # Compute the cosine similarity between the clicked image and all the other images in the same category
    clicked_image_pixels = np.array(dfs[category_query].iloc[clicked_image_index, 1:])
    similarity_scores
