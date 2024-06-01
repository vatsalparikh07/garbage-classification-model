import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model("garbage_classification_model_inception.h5")

# Define image dimensions
img_height = 384
img_width = 512

# Define dustbin colors
dustbin_colors = {
    'Cardboard': 'brown',
    'Trash': 'grey',
    'Plastic': 'blue',
    'Metal': 'blue',
    'Glass': 'blue',
    'Paper': 'blue'
}

# Dictionary containing disposal information for each waste category
disposal_info = {
    'Cardboard': "Cardboard waste can be recycled. Please flatten cardboard boxes before recycling.",
    'Trash': "Trash should be disposed of in the general waste bin.",
    'Plastic': "Plastic waste can often be recycled. Please check with your local recycling program for guidance on recycling plastic.",
    'Metal': "Metal waste can often be recycled. Please check with your local recycling program for guidance on recycling metal.",
    'Glass': "Glass waste can often be recycled. Please check with your local recycling program for guidance on recycling glass.",
    'Paper': "Paper waste can be recycled. Please ensure paper is clean and dry before recycling."
}

# Function to predict waste category from an image
def predict_waste_category_from_image(image):
    # Preprocess the image
    img = image.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the category
    prediction = model.predict(img_array)
    waste_categories = ['Cardboard', 'Trash', 'Plastic', 'Metal', 'Glass', 'Paper']
    predicted_category_index = np.argmax(prediction)
    
    # Check if the predicted category index is valid
    if 0 <= predicted_category_index < len(waste_categories):
        predicted_category = waste_categories[predicted_category_index]
    else:
        predicted_category = "Unknown"

    return predicted_category

# Function to predict waste category from the 10th frame of a video
def predict_waste_category_from_video(video_bytes):
    try:
        # Write the video bytes to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # Open the video file
        video_cap = cv2.VideoCapture("temp_video.mp4")

        # Check if the video opened successfully
        if not video_cap.isOpened():
            raise RuntimeError("Error: Failed to open the video file.")

        for _ in range(20):
            ret, _ = video_cap.read()
            if not ret:
                raise RuntimeError("Error: Failed to read the 10th frame of the video.")

        ret, frame = video_cap.read()

        # Check if the frame was read successfully
        if ret:
            # Convert the frame to PIL Image format
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            predicted_category = predict_waste_category_from_image(pil_image)

            return predicted_category, frame
        else:
            raise RuntimeError("Error: Failed to read the frame of the video.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the video: {str(e)}")

# Function to get the path of the dustbin image based on color
def get_dustbin_image_path(color):
    dustbin_images_dir = "dustbin_images"
    return os.path.join(dustbin_images_dir, f"{color}.png")

# Streamlit app
def main():
    st.title("Waste Management AI")

    st.markdown("Upload an image or a video and let the AI classify the waste category.")

    # File uploader for image or video
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # Check if the uploaded file is an image or a video
        if uploaded_file.type.startswith('image'):
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Predict the waste category from the image
            predicted_category = predict_waste_category_from_image(image)
        elif uploaded_file.type.startswith('video'):
            # Predict the waste category from the first frame of the video
            try:
                predicted_category, frame = predict_waste_category_from_video(uploaded_file.read())

                # Display the video
                st.video(uploaded_file, start_time=0)

            except RuntimeError as e:
                st.error(str(e))
                return
        else:
            st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or a video (mp4).")
            return

        # Display the prediction
        st.subheader(f"Waste Category Predicted: {predicted_category}")

        # Display disposal information
        st.subheader("Disposal Information:")
        disposal_text = disposal_info.get(predicted_category, "Disposal information not available for this category.")
        st.write(disposal_text)

        # Display the corresponding dustbin image
        dustbin_color = dustbin_colors.get(predicted_category, 'blue')
        dustbin_image_path = get_dustbin_image_path(dustbin_color)
        if dustbin_image_path is not None:
            dustbin_image = Image.open(dustbin_image_path)
            st.image(dustbin_image, caption=f"Dustbin for {predicted_category}", width=200)

if __name__ == "__main__":
    main()
