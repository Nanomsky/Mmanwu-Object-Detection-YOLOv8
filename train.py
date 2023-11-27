#https://docs.ultralytics.com/usage/python/
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')


#load a pretrained model
model = YOLO("yolov8n.pt") 


# Train the model 
results = model.train(data="config.yaml", epochs=100) 
 

#Export model
model.export(format='onnx', dynamic=True)