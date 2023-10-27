import torch
import time
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
import scipy.io as scio

data_transforms = {x: transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) for x in
                   ['train', 'val']}

train_dataset = ImageFolder('D:/Code/Train_data/studywork_2023/data/train', transform=data_transforms['train'])
val_dataset = ImageFolder('D:/Code/Train_data/studywork_2023/data/val', transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model = model.to(device)

num_classes = 30
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.fc = model.fc.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

loss_list = []
acc_t1_list = []
acc_t5_list = []  # top five acc
time_list = []
itex_mat = 100

for itex_num in range(itex_mat):
    begin_time = time.time()
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
    print("itex---", itex_num, "Loss:", loss_list[itex_num])

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
    end_time = time.time()
    acc_t1_list.append(top1_accuracy)
    acc_t5_list.append(top5_accuracy)
    time_list.append((end_time - begin_time))
    print("Top-1 Accuracy:", acc_t1_list[itex_num])
    print("Top-5 Accuracy:", acc_t5_list[itex_num])
    print("Consume Time:", time_list[itex_num])
scio.savemat("./itex.mat",
             {'loss_list': loss_list, 'acc_t1_list': acc_t1_list, 'acc_t5_list': acc_t5_list, 'time_list': time_list})
debug_point = 1
