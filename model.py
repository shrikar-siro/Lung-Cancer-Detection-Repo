#we will be creating a yoloV8 image classification model.
#step 1: import YOLO from ultralytics python library - this library allows us to use ML models.
from ultralytics import YOLO

#2: set model equal to yolo model.
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

#3: train the model on the dataset. The data parameter contains the absolute file path of your dataset.
#remember to structure your dataset into train and val folders for this to work.
results = model.train(data="/Users/shrikarstuff/Documents/MyOwnProjects/LungCancerML/data", epochs=100, imgsz=64)

#after training, we see that the model is getting better because accuracy is going up, while training loss is going down.
#more epochs should display the same pattern better. - current training loss is 1.0061

#train2 - we trained the model again, this time at 50 epochs - the training loss is considerably lower, at 0.3581.

#we can take a look at args.yaml file of train2 to see the parameters set in configuring this model.

#let's validate this model.