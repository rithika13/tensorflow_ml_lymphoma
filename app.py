import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to display the introduction section
def show_introduction():
    st.title('Lymphoma Classification using Ensemble of CNN Models')
    st.markdown("""
    
Explore our initiative aimed at simplifying the identification and comprehension of malignant lymphoma, a cancer originating in infection-fighting cells known as lymphocytes within the immune system. These cells are present in various parts of the body, including lymph nodes, spleen, thymus, bone marrow, and others.

Our project focuses on enhancing the accuracy and ease of identifying different cases of lymphoma, providing crucial insights for medical professionals in understanding the potential progression of the disease.

Utilizing an ensemble of Convolutional Neural Network (CNN) models, our application streamlines the classification of lymphoma. All it takes is a straightforward image upload, and the model promptly provides predictions regarding the specific type of lymphoma detected.

For a deeper dive into the intricacies of this project, we invite you to visit our [IEEE published paper](https://www.medicaps.ac.in/).
                This paper provides comprehensive information, highlighting the significance and advancements achieved in the field of lymphoma classification.
    """)

# Function to display the prediction section
def show_prediction():
    st.header("Lymphoma Classification System")
    # Load your model 
    model = load_model('Ensemble.h5')
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict'):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            class_names = ["Chronic Lymphocytic Leukemia", "Follicular Lymphoma", "Mantle Cell Lymphoma"]
            predicted_class_index = np.argmax(prediction)
            pred_class = class_names[predicted_class_index]
            pred_score = prediction[0][predicted_class_index]
            st.markdown(f"The uploaded image indicates the presence of **{pred_class}** and the prediction comes with a confidence score of  **{pred_score:.2f}**")

# Function to display the accuracy graphs section
def show_accuracy_graphs():
    st.header("Accuracy Graphs")

    # Define the path for each graph
    graph1 = 'Accuracy.png'
    graph2 = 'Loss.png'
    graph3 = 'confmatrix.png'

    # Display each image with the same fixed width
    st.image(graph1, caption='Training and Validation Accuracy', width=300)  
    st.image(graph2, caption='Training and Validation Loss', width=300)  
    st.image(graph3, caption='Confusion Matrix', width=300)  
    

# Function to check if the user is logged in
def is_user_logged_in():
    return 'logged_in' in st.session_state and st.session_state.logged_in

# Function to display the login form
def login_form():
    st.title("Lymphoma Classification System")
    form = st.form(key='login_form')
    username = form.text_input("Username")
    password = form.text_input("Password", type="password")
    login_button = form.form_submit_button("Login")
    if login_button:
        if username == "admin" and password == "demo@123":  
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Incorrect Username/Password")

# Main app
def main_app():
    st.sidebar.title("Lymphoma Classification")
    app_mode = st.sidebar.radio("Go to", ["Introduction", "Prediction", "Validation of Model"])

    if app_mode == "Introduction":
        show_introduction()
    elif app_mode == "Prediction":
        show_prediction()
    elif app_mode == "Validation of Model":
        show_accuracy_graphs()

# Main
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if is_user_logged_in():
    st.sidebar.title("Lymphoma Classification")
    main_app()
else:
    login_form()