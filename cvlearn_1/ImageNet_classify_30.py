import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models

data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), }

train_dataset = ImageFolder('D:/Code/Train_data/studywork_2023/data/train', transform=data_transforms['train'])
val_dataset = ImageFolder('D:/Code/Train_data/studywork_2023/data/val', transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)

num_classes = 30
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.fc = model.fc.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

loss_list = []
acc_list = []
tfacc_list = []  # top five acc
epoch_num = 50

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_list.append(running_loss / len(train_loader))
    print(f'Epoch {epoch + 1}, Loss: {loss_list[epoch]}')

    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels.to(device)).sum().item()
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            correct_top5 += torch.sum(predicted_top5 == labels.to(device).view(-1, 1)).item()

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    acc_list.append(top1_accuracy)
    tfacc_list.append(top5_accuracy)

    print(f'Top-1 Accuracy: {top1_accuracy * 100:.2f}%')
    print(f'Top-5 Accuracy: {top5_accuracy * 100:.2f}%')

plt.figure(figsize=(10, 5))

plt.plot(range(1, epoch_num + 1), loss_list, marker='o')
plt.title('Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('fig/epoch_loss.png')
plt.clf()

plt.plot(range(1, epoch_num + 1), acc_list, marker='o', color='green')
plt.plot(range(1, epoch_num + 1), tfacc_list, marker='o', color='red')
plt.title('Epoch Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('fig/epoch_acc.png')
plt.clf()
debug_point = 1
