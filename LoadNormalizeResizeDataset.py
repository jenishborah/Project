from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = "D:\Msc GU\Msc 4th Sem\Project\DataSet\\teadataset"

# Create empty lists for images and labels
images = []
labels = []

# Loop through each folder in the dataset directory 
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(folder_path):
        label = folder_name
        # Loop through each image in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Load the image using Pillow
            image = Image.open(file_path)
            # Append the image and label to the lists
            images.append(image)
            labels.append(label)
            # Define the target image size (e.g., 224x224x3 for RESTNET54  models)
target_size = (224, 224)

# Create an empty array to hold the preprocessed images
preprocessed_images = []

# Loop through each image and preprocess it
for image in images:
    # Resize the image
    resized_image = image.resize(target_size)
    
    # Convert the image to float32 and normalize the pixel values to have zero mean and unit variance
    normalized_image = (np.array(resized_image, dtype=np.float32) - 128.0) / 128.0
    
    # Append the preprocessed image to the list
    preprocessed_images.append(normalized_image)
    

