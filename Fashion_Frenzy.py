import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import v_measure_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Fashion Frenzy 🛍", page_icon=":dress:")


# Add your logo and title
logo = Image.open("Logo")


# Create a two-column layout
col1, col2 = st.columns([1, 4])

# Add the logo to the left column
with col1:
    st.image(logo, width=100)

# Add the title to the right column
with col2:
    st.title("Fashion Frenzy 🛍")
    st.subheader("Welcome to our shopping website!!")

# Display the animation
video_file = open('animation.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

# Load the CSV files
csv_files = {
    'Hot selling🔥!!': 'cluster0.csv',
    'Bags👜': 'cluster1.csv',
    'Flats & Sneakers👟': 'cluster2.csv',
    'Bottoms & Dresses👗': 'cluster3.csv',
    'Sweats🥼': 'cluster4.csv',
    'Tops👚': 'cluster5.csv',
    'Boots & Heels👢': 'cluster6.csv',
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

# Define a function to calculate clustering metrics
def calculate_metrics(k, data):
    # Load true labels
    true_labels_df = pd.read_csv('true_label.csv')

    # Extract labels as a list
    true_labels = true_labels_df['label'].tolist()

    # Perform PCA
    pca = PCA(42)
    data1=pca.fit_transform(data)

    # Convert the data to a numpy array
    X = data1

    # Define number of clusters
    k = 7

    # Fit KMeans model and predict labels
    kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Compute the Silhouette Score for K-means clustering with k
    labels = kmeans.labels_
    sil_score = silhouette_score(X, labels)

    # Compute the Davies-Bouldin Index for K-means clustering with k
    Dav_score = davies_bouldin_score(X, labels)

    # Calculate V-measure
    v_measure = v_measure_score(true_labels, labels)

    # Display the metrics
    st.write('Silhouette Score:', sil_score)
    st.write('Davies-Bouldin Index:', Dav_score)
    st.write('V-measure Score:', v_measure)

# Load the data
data = pd.read_csv('product_images.csv')

# Call the function to calculate clustering metrics
calculate_metrics(7, data)
