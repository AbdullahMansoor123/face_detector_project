import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from face_recognition_projects.my_face_detector.person_classifier_FC.person_cnn import CNN  # sperated model file
import torch.optim as optim

# preprocessing data
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#train and test data path
train_dir = 'person_train_test_dataset/train'
test_dir = 'person_train_test_dataset/val'

# dataloader
train_dataset = datasets.ImageFolder(train_dir, transform=transform)  # 80%
test_dataset = datasets.ImageFolder(test_dir, transform=transform)  # 20%
# dataset_len = len(dataset)

# previously used data splitter
# train_len, test_len = dataset_len - 24, 24  # 24 represent the size of test data
# train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

# hyper-parameters
num_classes = 2
input_size = 53 * 53
batch_size = 32
learning_rate = 0.0001
num_epochs = 40

# train and test dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
print(train_loader)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('{} in use'.format(device))

# CNN Imported separately
# CNN Model
model = CNN(num_classes=num_classes)
model = model.to(device)
# print(model)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    print(f'epoch:{epoch + 1}/{num_epochs}...........\n')
    total_correct = 0.0
    running_loss = 0.0
    total = 0
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)

        # Forward Pass
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (labels == predictions).sum().item()

        # backward
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * images.size(0)
        loss.backward()
        # Optimizing
        optimizer.step()
    print(f'train loss: {(running_loss / total):.4f} train accuracy: {((total_correct / total) * 100):.2f}%')

    # test loop
    with torch.no_grad():
        model.eval()  # notify our layer we are in evaluation mode
        total_loss, total_correct = 0.0, 0.0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            # _, predictions = torch.max(outputs, dim=1)
            total_correct += (labels == predictions).sum().item()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
    print(f'test loss: {(total_loss / total):.4f} test accuracy: {((total_correct / total) * 100):.2f}%\n')
print('Training and testing completed!')

# saving the model
torch.save(model.state_dict(), 'person_classify_3.pth')
