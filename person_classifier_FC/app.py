from facenet_pytorch import MTCNN
import facedetector_m
import torch
from person_cnn import CNN

#Initial model parameters
num_classes = 2

###Initiate Classifier###
model = CNN(num_classes=num_classes)
model.load_state_dict(torch.load('person_classify_2.pth'))
model.eval()

###Initiate Networks###
mtcnn = MTCNN()
fcd = facedetector_m.FaceDetector(mtcnn,classifier=model)
fcd.run(0)