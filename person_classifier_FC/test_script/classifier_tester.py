import torch
from face_recognition_projects.my_face_detector.person_classifier_FC.person_cnn import CNN
import torchvision.transforms as transforms
import PIL.Image as Image
import matplotlib.pyplot as plt

class_map = ['Abdullah', 'Not_Abdullah']
num_classes = len(class_map)

model = CNN(num_classes=num_classes)
# model = torch.load('pet_classify_0.pth')
model.load_state_dict(torch.load('person_classify.pth'))
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def image_classify(model, image_transforms, image_path, classes):
    image = Image.open(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    outputs = model(image)
    prediction = torch.argmax(outputs, dim=1)
    # print(classes[prediction.item()])
    # show image with prediction
    plt.title(f'This is a {class_map[prediction.item()]}')
    plt.axis('off')
    plt.show()

#Image Classifier


image_classify(model, image_transforms, 'me_2.jpg', class_map)

