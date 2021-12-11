# import libraries
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms


def is_it_me(self, frame):
    """
    Run Classifier on face
    """
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformed_img = transform(PIL_img)
    transformed_img = torch.unsqueeze(transformed_img, 0)  # fake batch dimension

    # Inference
    with torch.no_grad():
        outputs = self.classifier(transformed_img)
        prediction = torch.argmax(outputs, dim=1)

    return prediction
