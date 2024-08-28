from ultralytics import YOLO
import numpy as np
import streamlit as st
#we import the os module so we can use it to get the filepath of the 
import os

st.title("Lung Cancer Detection Application: ")
#we create an upload folder to save images that the user uploads.
upload_folder = '/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/uploads'
uploaded_file = st.file_uploader("Choose an image file: ", type = ['jpeg', 'png', 'jpg'])

#this function takes the uploaded image and finds the file path of it.
def get_filepath(uploaded_file):
    #since the function should only analyze uploaded_file if it is valid input, all the operational code is within an if statement.
    if uploaded_file is not None:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        #
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
if uploaded_file is not None: 
    #bytes of file equals t he number of bytes the uploaded file is.
    st.write("Uploaded Image: ")
    st.image(uploaded_file)
    model =  YOLO("/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/runs/classify/train3/weights/last.pt")
    #once model analyzes image, results are stored in results variable. We need to get the filepath of this image first before uploading.
    #getting the filepath of an image: 
    results = model(get_filepath(uploaded_file))
    #this gets you the classification names.
    names = results[0].names
    #we then get the classification probabilities and convert that data to a list.
    probList = results[0].probs.data.tolist()
    #we find the index of maximum probability using argmax function and then get the correct classification name based on the index.
    category = names[np.argmax(probList)]
    if not(category == 'normal'):
        #we display the category to the user, adding color to the category name for easier identification.
        st.write(f"The category of lung cancer shown in the image is: :blue[**{category}**]")
        st.write(f"The probabilities are: {probList}")
    else:
        st.write(f"The category is normal, meaning that this patient has no lung cancer.")

else: 
    st.warning("You have not uploaded a file yet, so there is nothing to display.")


