import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import time

def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Initialize session state variables if not already set
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

# Define tabs and handle tab selection
tabs = st.tabs(["About", "Models"])

with tabs[0]:  # About Tab
    st.title("About")
    st.write("""
    This is an image classification project with six classes: 
             
        - 0: Buildings, 
        - 1: Forest, 
        - 2: Glacier, 
        - 3: Mountain, 
        - 4: Sea,  
        - 5: Street. 
    
    Choose a model on the 'Models' tab and upload an image to see the results.

    However, there is an issue with one of the models—it has overfitted. You will see which one has overfitted. 
    Why it has overfitted?

    The model overfitted because it wasn't complex enough for the data. When attempts were made to increase the model's complexity, the computer repeatedly restarted due to insufficient capacity. Therefore, the overfitted model was added, and work continued. While searching for a solution, the pretrained model InceptionV3 was found and decided upon for testing.

    After training the model, it was observed that choosing a pretrained model was a healthy and correct decision. The model can make predictions with a 91% success rate. When using a GPU or TPU, the following steps could be taken to address the overfitted model:
    
        -Data augmentation
        -Increasing model complexity
    These scenarios can be tested with a GPU or TPU, and if the model still overfits, one of the pretrained models can be selected to continue. This approach will save both time and resources.
             
    """)

class_label = {0:"Buildings",1:"Forest",2:"Glacier",3:"Mountain",4:"Sea",5:"Street"}

with tabs[1]:  # Models Tab
    st.title("Models Overview")
    st.write("""
    Select a model from the sidebar and upload an image to see the classification result.
    """)

    # Sidebar navigation for model selection
    st.sidebar.title("Navigation")
    st.session_state.selected_model = st.sidebar.selectbox(
        "Select a Model",
        ["", "CNN Model", "Pretrained CNN Model"],
        index=["", "CNN Model", "Pretrained CNN Model"].index(st.session_state.selected_model)
    )

    if st.session_state.selected_model == "CNN Model":
        st.title("CNN Model")
        st.write("""This model has overfitted. As you can observe, 
                 the model has become fixated on a single class—the only class it predicts is 'Forest.' 
                 This is a clear example of overfitting. 
                 One reason for this overfitting is the model's lack of sufficient complexity. """)
        
        model = load_model("cnn_model.h5")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            image_array = preprocess_image(image)
            
            if st.button("Predict"):
                start_time = time.time()
                prediction = model.predict(image_array)
                end_time = time.time()
                prediction_time = end_time - start_time

                
                predicted_class = np.argmax(prediction)
                class_name = class_label[predicted_class]
                
                st.write(f"Prediction Probabilities: {prediction}")

                
                st.write(f"Predicted Class: {predicted_class} --> {class_name}")
                st.write(f"Prediction Time: {prediction_time:.2f} seconds")

                threshold = 0.5  
                high_prob_indices = np.where(prediction[0] > threshold)[0]

                if len(high_prob_indices) > 0:
                    for i in high_prob_indices:
                        st.write(f"Class {i}: {prediction[0][i]:.4f} probability")
                else:
                    st.write("No class exceeded the specified threshold value.")
        
       


    elif st.session_state.selected_model == "Pretrained CNN Model":
        st.title("Pretrained CNN Modell")

        model = load_model("pretrained_cnn_model.h5")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_array = preprocess_image(image)
            if st.button("Predict"):
                
                start_time = time.time()
                prediction = model.predict(image_array)

                end_time = time.time()
                prediction_time = end_time - start_time

                predicted_class = np.argmax(prediction)
                class_name = class_label[predicted_class]

                st.write(f"Prediction Probabilities: {prediction}")
                st.write(f"Predicted Class: {predicted_class} --> {class_name}")
                st.write(f"Prediction Time: {prediction_time:.2f} seconds")

                threshold = 0.5  
                high_prob_indices = np.where(prediction[0] > threshold)[0]

                if len(high_prob_indices) > 0:
                    for i in high_prob_indices:
                        st.write(f"Class {i}: {prediction[0][i]:.4f} probability")
                else:
                    st.write("No class exceeded the specified threshold value.")