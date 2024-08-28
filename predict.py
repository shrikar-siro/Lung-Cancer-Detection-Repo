#we use this file to make predictions using our model. So we could upload an image, and that image could be fed into the model.
from ultralytics import YOLO
import numpy as np

#the models we can choose are either best.pt or last.pt.

#code taken from ultralytics documentation: 
#first step is we load the model we are going to use - in this case, last.pt. We take the absolute file path of the model.

model = YOLO("/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/runs/classify/train3/weights/last.pt")

#second step: set source equal to an image we will pass into the model. Again, use the absolute file path.
source = "/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/data/val/Large Cell Carcinoma/000111.png"

#third step: pass the image into the model.
results = model(source)

#print(results)

#we are only predicting an individual image, so we only want to access the first row of results.

names_dict = results[0].names
#print(names_dict)

#probs takes results[0] and shows the probabilities - however, this is not a list alone.
#we have to convert probs.data to a list by doing probs.data.tolist()
prob = results[0].probs.data.tolist()
#print(prob)

#category
#since names_dict is a dictionary with the number and the category name, we can get the max argument index by passing prob as an argument into np.argmax function.
#then we can pass that as an argument into names_dict. What we pass in will be one of the keys (0, 1, 2, or 3) and that depends on where the highest probability is located in prob.
category = names_dict[np.argmax(prob)]
print()
print()
print(f"Category: {category}")

#we can use this prediction in a web application, whether we use gradio, streamlit, flask, etc as our basis.



