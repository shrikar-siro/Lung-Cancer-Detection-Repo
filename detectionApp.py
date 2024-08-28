import gradio as gr
from ultralytics import YOLO
import numpy as np

def prediction(image):
    model =  YOLO("/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/runs/classify/train3/weights/last.pt")
    results = model(image)
    #once model analyzes image, results are stored in results variable. We need to get the filepath of this image first before uploading.
    #getting the filepath of an image: 
    #this gets you the classification names.
    names = results[0].names
    #we then get the classification probabilities and convert that data to a list.
    probList = results[0].probs.data.tolist()
    #this for loop multiplies each number in probList with 100 to get percentages.
    names_with_prob = dict(zip(names.values(), probList))
    #we find the index of maximum probability using argmax function and then get the correct classification name based on the index.
    category = names[np.argmax(probList)]
    if not(category == 'normal'):
        #we display the category to the user, adding color to the category name for easier identification.
        #return f"The category of lung cancer shown in the image is: {category}"
        return f"This patient has {category}, with a probability of {(np.max(probList) * 100):.2f} %."
    else:
        return f"The patient has no lung cancer, as his lungs are normal."
    
project = gr.Interface(
    fn = prediction, 
    #by setting the type to filepath, we pass the filepath of the image uploaded into the function above.
    inputs= gr.Image("Image of Lungs: ", type="filepath"),
    outputs=["text"]
)


project.launch()